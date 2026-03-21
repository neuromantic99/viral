import os
from typing import List, Literal, Optional, Tuple, Dict
from matplotlib.lines import Line2D
import numpy as np
import sys
import time
import copy
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, confusion_matrix, f1_score
from deprecated import deprecated
from tqdm import tqdm

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.constants import CACHE_PATH, SERVER_PATH, TIFF_UMBRELLA, PLOT_PATH
from viral.models import (
    TrialInfo,
    Cached2pSession,
    GrosmarkConfig,
    BayesianDecodingConfig,
    DecodedEvent,
    BayesianDecodingResult,
    SSPConfig,
)

from viral.utils import get_session_type, get_genotype, mixed_effects, shaded_line_plot
from viral.imaging_utils import (
    split_fluoresence_online_freeze,
    trial_is_imaged,
)
from viral.grosmark_analysis import get_place_cells
from viral.ensemble_reactivation import get_ssp_vectors
from viral.sequence_utils import (
    detect_candidate_events,
    merge_close_events,
    filter_candidate_events_by_duration,
    filter_candidate_events_by_inter_event_time,
    additional_pc_check,
    construct_xy_by_bin,
    calculate_linear_weighted_correlation,
    calculate_circular_weighted_correlation,
    check_significance,
    bin_for_classification,
    calculate_radon_replay,
    get_cache_path,
    load_bayesian_cache,
    save_bayesian_cache,
)
from viral.sessions_keep import SESSIONS_KEEP


def get_population_vector(
    ssp_smoothed: np.ndarray,
    mouse_name: str,
    date: str,
    bayesian_config: BayesianDecodingConfig,
) -> np.ndarray:
    """
    Offline PSEs were detected by convolving each PC's (as assessed during that day's run) offline immobility firing rate vector
    Ssp with a 125-ms Gaussian kernel and z-scoring the smoothed firing rate vector. Subsequently, for each frame i, the population
    mean of the smoothed and z-scored vector was taken across PCs and subsequently z-scored.
    """
    z_scored = zscore(ssp_smoothed, axis=1)
    # Remove nans from silent neurons
    z_scored = np.nan_to_num(z_scored)

    population_vector = zscore(np.mean(z_scored, axis=0))
    population_vector_sd = np.std(population_vector)
    # TODO: remove after debugging
    plt.figure(figsize=(30, 20))
    plt.plot(population_vector)
    plt.hlines(
        population_vector_sd * bayesian_config.peak_threshold,
        xmin=0,
        xmax=len(population_vector),
        colors="r",
        linestyles="dashed",
        label="Peak threshold",
    )
    plt.hlines(
        population_vector_sd * bayesian_config.edge_threshold,
        xmin=0,
        xmax=len(population_vector),
        colors="g",
        linestyles="dashed",
        label="Edge threshold",
    )
    plt.title("Population vector")
    if not os.path.exists(PLOT_PATH / f"pse_events_{bayesian_config.epoch}"):
        os.makedirs(PLOT_PATH / f"pse_events_{bayesian_config.epoch}")
    plt.savefig(
        PLOT_PATH
        / f"pse_events_{bayesian_config.epoch}"
        / f"{mouse_name}_{date}_{bayesian_config.epoch}_population_vector.png",
        dpi=600,
    )
    plt.close()
    return population_vector


def find_pse_events(
    population_vector: np.ndarray,
    ssp: np.ndarray,
    config: BayesianDecodingConfig,
    duration_filter: bool = True,
) -> List[Tuple[int, int]]:
    """
    Putative PSEs were defined as epochs during which the z-scored population activity vector reached a peak of at least 3.5 s.d.
    above the mean with event-edges at 1 s.d. above the mean, with a minimum inter-event time of 0.2 s.
    Only PSE events lasting between 0.2 s (12 frames) and 1 s (60 frames), and during which at least 5 distinct PCs each fired at
    least one estimated spike, were kept for further analysis.
    """
    candidate_events = detect_candidate_events(population_vector, config)
    print(
        f"Found {len(candidate_events)} candidate events (before filtering for duration and additional PC check)"
    )
    if len(candidate_events) == 0:
        return []

    # merge events that are too close together (< 0.2s, i.e. 6 frames)
    # TODO: should we even merge them? or discard if the inter-event-time is too short?
    # merged_events = merge_close_events(candidate_events)
    # print(f"Merged into {len(merged_events)} events")
    # merged_events = candidate_events
    # print("For now, no measures to take care of close events")

    # TODO: if keeping this, then think about the threshold AND rename 'merged_events'
    # reject events that are too close together
    # in Grosmark, events closer than 0.2 seconds together were rejected (30 fps -> 0.2 * 30 = 6)
    # TODO: maybe -> I will try 0.1 seconds (i.e. 3 frames) to begin with, based on what I have seen in our data
    min_inter_event_time_frames = 6
    merged_events = filter_candidate_events_by_inter_event_time(
        candidate_events, min_inter_event_time_frames
    )
    print(
        f"Filtered to {len(merged_events)} events by inter-event time ({min_inter_event_time_frames/30} seconds)"
    )

    if duration_filter:
        # filter events by duration e.g. (0.2s - 1s) -> (6 - 30 frames)
        filtered_events = filter_candidate_events_by_duration(
            candidate_events=merged_events,
            event_duration_thresholds=config.event_duration,
        )
        print(f"Filtered to {len(filtered_events)} events by duration")
        if len(filtered_events) == 0:
            return filtered_events
    else:
        filtered_events = merged_events
        print("No duration filtering at the moment")

    # TODO: ssp is estimated spikes, right?
    # perform additional check: at least 5 distinct PCs each fired at least one estimated spike
    additionally_checked = additional_pc_check(filtered_events, ssp)
    print(f"{len(additionally_checked)} events remaining after additional PC check")

    return additionally_checked


def sequence_bayesian_decoding(
    offline_activity_binned: np.ndarray,
    place_fields: np.ndarray,
    config: BayesianDecodingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    "To perform offline sequence analysis, within-PSE PC activity was binned into non-overlapping two-frame time bins,
    and Bayesian decoding was performed on these two frame bins as described above."

    "Bayesian reconstruction of virtual position was performed utilizing a template comprising all the smoothed firing rate-by-position vectors of a PC as follows:
    (1)
    Where fi(pos) is the value of the firing rate-by-position vector of the ith PC at position pos, spi is the number of spikes fired by the ith PC in the time bin
    being decoded, τ is the duration of the time bin and n is the total number of PCs.
    Time bins (τ) of 20 frames (~333 ms) were used for Bayesian reconstruction of online activity, while time bins of 2 frames (~33 ms) were used for Bayesian
    reconstruction of offline activity, and only bins with non-zero firing rates were used for offline Bayesian decoding.
    Posterior probabilities were subsequently normalized to one:
    (2)
    Where Pn is the total number of positions (2-cm bins).
    When assessing the reconstruction accuracy, or reconstructed distance to the reward, the position corresponding to the maximal posterior probability was taken.
    For same-day decoding, a fivefold (applied by lap number) cross-validation was used.
    For cross-day position reconstruction, only registered cells that were found to be PCs on both the training and testing days were included,
    and only pairs of sessions containing at least five such PCs were included in the decoding analysis."

    Essentially, this is a Python implementation of https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/placeBayesLogBuffered.m

    Args:
        offline_activity_binned (np.ndarray):   Offline activity binned by time (shape: (n_cells, n_time_bins)).
        place_field (np.ndarray):               Place field array for place cells, template for Bayesian decoding (shape: (n_cells, n_spatial_bins)).
        config (BayesianDecodingConfig):        A config class containing parameters like bin sizes etc.
    Returns:
        np.ndarray:                             posterior_probability_matrix (shape: (n_time_bins, n_spatial_bins)).
        np.ndarray:                             pr_max (spatial bin with the highest posterior probability for that frame; shape: (n_time_bins)).

    Raises:
        AssertionError
    """
    assert (
        offline_activity_binned.shape[0] == place_fields.shape[0]
    ), "Numbers of place cells in offline activity and place field template do not match!"

    buffer = 12
    # TODO: or copy the frames bin_size rather than the time bin_size from Grosmark?
    # bin_length_time_online = config.bin_size_time_online / 30  # seconds
    bin_length_time = (
        config.bin_size_time_online / 30
        if config.epoch == "online"
        else config.bin_size_time_offline / 30
    )  # seconds
    n_time_bins = offline_activity_binned.shape[1]
    n_spatial_bins = place_fields.shape[1]

    Cr = offline_activity_binned.T  # shape: (n_time_bins, n_cells)
    rate_map = place_fields  # shape: (n_cells, n_spatial_bins)

    Cr = Cr * bin_length_time
    rate_map = place_fields.T + (10 ** (-10))

    term2 = (-bin_length_time) * np.sum(rate_map, axis=1)

    Pr = np.zeros((n_time_bins, n_spatial_bins))

    log_rate_map = np.log(rate_map)

    for S in range(0, n_time_bins, buffer):
        T = min(S + buffer, n_time_bins)
        c = Cr[S:T, :]  # [buffer x nCell]
        u = c @ log_rate_map.T + term2  # [buffer x nSpatial]
        u -= np.max(u, axis=1, keepdims=True)  # numerical stability
        Pr[S:T, :] = np.exp(u)
        Pr[S:T, :] /= np.sum(Pr[S:T, :], axis=1, keepdims=True)

    pr_max = np.argmax(Pr, axis=1)

    if np.isinf(Pr).any():
        raise ValueError("Infinite values in Pr")
    if np.isnan(Pr).any():
        raise ValueError("NaN values in Pr")

    # and only bins with non-zero firing rates were used for offline Bayesian decoding. Posterior probabilities were subsequently normalized to one:

    return Pr, pr_max
    # TODO: check the matlab implementation for same result
    # TODO: it is the same!!!


def plot_pse_event(
    posterior_probability_matrix: np.ndarray,
    idx: int,
    session: Cached2pSession,
    grosmark_config: GrosmarkConfig,
    bayesian_config: BayesianDecodingConfig,
    mode: Literal["linear", "circular"],
    corr_coeff: float,
    significance: Tuple[float, bool],
    do_radon_transform: Optional[bool] = True,
) -> None:
    n_time, n_pos = posterior_probability_matrix.shape

    plt.figure(figsize=(10, 8))
    if mode == "linear":
        plt.imshow(
            posterior_probability_matrix.T,
            vmin=0,
            vmax=np.max(posterior_probability_matrix) * 1.1,
            aspect="auto",
        )
        plt.title(
            f"Linear Weighted Correlation: {corr_coeff:.2f} (p={significance[0]:.4f})"
        )
    elif mode == "circular":
        plt.imshow(
            np.tile(posterior_probability_matrix.T, (2, 1)),
            vmin=0,
            vmax=np.max(posterior_probability_matrix) * 1.1,
            aspect="auto",
        )
        plt.title(
            f"Circular Weighted Correlation: {corr_coeff:.2f} (p={significance[0]:.4f})"
        )
    plt.colorbar()

    if do_radon_transform and mode == "circular":
        # trying the new Olafsdottir/Denovellis approach
        incorporate_nearby_positions = True
        if incorporate_nearby_positions:
            print("Using the 'nearby positions' approach by Denovellis et al.")
        # nearby_positions:
        # - 30 cm (Olafsdottir et al. 2016, https://doi.org/10.1038/nn.4291)
        # - 20 cm (Olafsdottir et al. 2015, https://doi.org/10.7554/eLife.06063)
        # - 15 cm (Denovellis et al. 2021, https://doi.org/10.7554/eLife.64505)
        nearby_positions = 30  # cm
        # min_n_bin_perc:
        # - 100.0 percent (Grosmark et al. 2021, https://doi.org/10.1038/s41593-021-00920-7)
        # - 1/3 = 33.33 percent (Olafsdottir et al. 2016 downloadable MATLAB code, https://doi.org/10.1038/nn.4291)
        min_n_bin_perc = 100
        radon_replay = calculate_radon_replay(
            posterior_probability_matrix=posterior_probability_matrix,
            bayesian_config=bayesian_config,
            incorporate_nearby_positions=incorporate_nearby_positions,
            nearby_positions=30,
            min_n_bin_perc=min_n_bin_perc,
        )

        y_range = int(
            nearby_positions / bayesian_config.bin_size_spatial
        )  # e.g. 30 cm band

        x = np.array([radon_replay.point1x, radon_replay.point2x])
        y = np.array([radon_replay.point1y, radon_replay.point2y])

        plt.plot(x, y, "r--")
        plt.plot(x, y + y_range, "w")
        plt.plot(x, y - y_range, "w")

        offset = (bayesian_config.total_length * 100) / bayesian_config.bin_size_spatial
        plt.plot(x, y + offset, "r--")
        plt.plot(x, y + y_range + offset, "w")
        plt.plot(x, y - y_range + offset, "w")
        plt.plot(x, y - offset, "r--")
        plt.plot(x, y + y_range - offset, "w")
        plt.plot(x, y - y_range - offset, "w")

        plt.title(
            f"Circular Weighted Correlation: {corr_coeff:.2f} (p={significance[0]:.4f}) \nRadon Slope: {radon_replay.slope_metres_per_sec:.2f} m/s ({radon_replay.replay_type} replay)"
        )

    plt.xlabel("Time (seconds)")
    xtick_bins = np.arange(0, n_time + 1, 15)
    xtick_labels = np.round(xtick_bins / 30, 2)
    plt.xticks(xtick_bins, xtick_labels)
    plt.ylabel("Position (centimetres)")
    if mode == "linear":
        plt.yticks(
            np.linspace(0, n_pos - 1, 5),
            [
                str(x)
                for x in np.linspace(
                    grosmark_config.start, grosmark_config.end, 5
                ).astype(int)
            ],
        )
        # plt.ylim(0, n_pos - 1)
        plt.ylim(n_pos - 1, 0)
    elif mode == "circular":
        plt.yticks(
            np.linspace(0, (n_pos * 2) - 1, 5),
            [
                str(x)
                for x in np.linspace(
                    grosmark_config.start, grosmark_config.end * 2, 5
                ).astype(int)
            ],
        )
        # plt.ylim(0, (n_pos * 2) - 1)
        plt.ylim((n_pos * 2) - 1, 0)
    plt.xlim(0, n_time - 1)

    plt.tight_layout()
    # radon_plot_root = PLOT_PATH / "pse_events_radon"
    radon_plot_root = PLOT_PATH / "pse_events_band"
    if not os.path.exists(radon_plot_root / session.mouse_name):
        os.makedirs(radon_plot_root / session.mouse_name)
    plt.savefig(
        radon_plot_root
        / session.mouse_name
        / f"{session.mouse_name}_{session.date}_{bayesian_config.epoch}_{mode}_{'chunk' if (bayesian_config.epoch == 'online') else 'event'}_{idx}.png"
    )
    plt.close()


def load_and_prepare_session_for_bayesian_decoding(
    mouse_name: str,
    date: str,
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
    use_train_test_split: bool = False,
    train_size: Optional[float] = 0.5,
) -> Tuple[Cached2pSession, np.ndarray, np.ndarray, List[TrialInfo] | None] | None:
    """
    Load and prepare session for Bayesian decoding (getting the cached session object and getting the place cells).
    A train-test split for the trials can be performed. (Makes sense if your checking the decoder's performance).
    If you're decoding online or offline PSE events, you shouldn't use the train-test split!
    The data should already be non-overlapping because of the speed filter for training (mobility) and testing (immobility), respectively.

    Returns:
        Cached2pSession:            The session object with the trials allotted according to the train-test split (online only).
        np.ndarray:                 The session's OASIS deconvoluted place cell activity.
        np.ndarray:                 Place fields of the place cells in the array above.
        list[TrialInfo] | None:     A list of the trials to test the decoder (online only).
        None:                       None if the session cannot be analysed (epoch is set to either of the offline but a wheel freeze does not exist)
    """
    with open(CACHE_PATH / f"{mouse_name}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())

    if bayesian_config.epoch != "online" and not session.wheel_freeze:
        # Skip session if intended to analyse offline activity but there was no wheel block
        print(f"Skipping {date} for mouse {mouse_name} as there was no wheel block")
        return None

    spks = np.load(
        TIFF_UMBRELLA
        / session.date
        / session.mouse_name
        / "suite2p"
        / "plane0"
        / "oasis_spikes.npy"
    )
    trials = [trial for trial in session.trials if trial_is_imaged(trial)]

    # TODO: this probably needs refactoring
    if use_train_test_split:
        # "training" the decoder (= place cell template) on a subset of trials
        trials_train, trials_test = train_test_split(
            trials, train_size=train_size, random_state=42
        )
    else:
        # otherwise just keep all imaged trials
        trials_train = trials
        trials_test = trials
        use_train_test_split = False

    # do the place cell template only on the training data!
    # the decoder is trained by supplying it with place cells and their respective place fields!
    session.trials = trials_train

    # TODO: double-check, but explicitly using BOD instead of speed thresholding now!
    t0 = time.time()
    pcs_mask, place_fields, _ = get_place_cells(
        session=session,
        spks=spks,
        rewarded=None,
        use_cache=True,
        config=grosmark_config,
        bin_occupancy_divide=True,
        plot=False,
        cache_file_additional_info={
            "train-test-split" if use_train_test_split else None
        },  # careful there when changing the train-test-split!!!
    )
    print(f"Time to get place cells: {time.time() - t0}")
    place_cells = spks[pcs_mask, :]
    pcs_place_fields = place_fields[pcs_mask, :]

    if bayesian_config.epoch == "online":
        return session, place_cells, pcs_place_fields, trials_test
    else:
        return session, place_cells, pcs_place_fields


def decode_online_epoch(
    session: Cached2pSession,
    trials_test: List[TrialInfo],
    place_cells: np.ndarray,
    place_fields: np.ndarray,
    bayesian_config: BayesianDecodingConfig,
) -> BayesianDecodingResult | None:
    """Perform the Bayesian decoding on phases of immobility within the online epoch of the session, i.e. using the test trials, and plot the events."""
    # It is a bit hidden, but Grosmark says in Fig. 5a: 'run epoch PSEs (occurring during immobility) '

    # get the ssp vector
    # TODO: is ssp test the correct one? (in terms of convolution)
    # phases of mobility
    # ssp_config_mobility = SSPConfig(mode="above", speed_threshold=5, n_consecutive_samples=3*30)

    # TODO: do we want to increase the speed threshold, and should it be below it for 3 consecutive seconds?
    # phases of immobility
    """Offline immobility epochs were defined as those in which the animal's velocity,
    smoothed with a half-second Gaussian kernel, was below 3 cm s-1 for at least 3 consecutive seconds.
    Online running epochs were defined as those in which the animal's smoothed velocity was above 5 cm s-1 for at least 3 consecutive seconds."""
    # smoothing is done in compute_grosmark_speed within get_ssp_vectors
    ssp_config_immobility = SSPConfig(
        mode="below",
        speed_threshold=3,
        n_consecutive_samples=3 * 30,
    )
    ssp_result = get_ssp_vectors(
        trials=trials_test,
        place_cells=place_cells,
        sigma=bayesian_config.sigma_online,
        mode=ssp_config_immobility.mode,
        speed_threshold=ssp_config_immobility.speed_threshold,
        n_consecutive_samples=ssp_config_immobility.n_consecutive_samples,
    )
    ssp_test, positions_test, _, _ = (
        ssp_result.ssp_vectors,
        ssp_result.position_vectors,
        ssp_result.trial_start_indices,
        ssp_result.chunk_start_indices,
    )
    # TODO: is the use of ssp correct?
    if ssp_test.size == 0:
        print("Empty ssp vector, returning None")
        return None
    assert ssp_test.shape[0] == place_cells.shape[0]

    population_vector = get_population_vector(
        ssp_smoothed=ssp_test,
        mouse_name=session.mouse_name,
        date=session.date,
        bayesian_config=bayesian_config,
    )
    pse_events = find_pse_events(
        population_vector=population_vector,
        ssp=ssp_test,
        config=bayesian_config,
    )

    if len(pse_events) == 0:
        print("No PSE events found, exiting")
        return
        # TODO: add back in!!!!
        # raise AssertionError("No PSE events found")

    pse_activity = [ssp_test[:, start:end] for start, end in pse_events]
    actual_positions = [positions_test[start:end] for start, end in pse_events]

    decoded_events: List[DecodedEvent] = list()
    for idx, event in enumerate(pse_activity):
        posterior_probability_matrix, pr_max = sequence_bayesian_decoding(
            event,
            place_fields=place_fields,
            config=bayesian_config,
        )
        linear_corr_coeff = calculate_linear_weighted_correlation(
            posterior_probability_matrix=posterior_probability_matrix,
            xy=construct_xy_by_bin(
                posterior_probability_matrix,
                mode="linear",
                total_length=bayesian_config.total_length,
            ),
        )
        linear_sign = check_significance(
            posterior_probability_matrix=posterior_probability_matrix,
            correlation=linear_corr_coeff,
            mode="linear",
            total_length=bayesian_config.total_length,
            n_shuffles=2000,
            significance=0.05,
        )
        circular_corr_coeff = calculate_circular_weighted_correlation(
            posterior_probability_matrix=posterior_probability_matrix,
        )
        circular_sign = check_significance(
            posterior_probability_matrix=posterior_probability_matrix,
            correlation=circular_corr_coeff,
            mode="circular",
            total_length=bayesian_config.total_length,
            n_shuffles=2000,
            significance=0.05,
        )
        plot_pse_event(
            posterior_probability_matrix=posterior_probability_matrix,
            idx=idx,
            session=session,
            grosmark_config=grosmark_config,
            bayesian_config=bayesian_config,
            mode="linear",
            corr_coeff=linear_corr_coeff,
            significance=linear_sign,
        )
        plot_pse_event(
            posterior_probability_matrix=posterior_probability_matrix,
            idx=idx,
            session=session,
            grosmark_config=grosmark_config,
            bayesian_config=bayesian_config,
            mode="circular",
            corr_coeff=circular_corr_coeff,
            significance=circular_sign,
            do_radon_transform=True,
        )
        print(circular_corr_coeff)
        decoded_events.append(
            DecodedEvent(
                posterior_probability_matrix=posterior_probability_matrix,
                pr_max=pr_max,
                actual_positions=actual_positions[idx],
                linear_weighted_r=linear_corr_coeff,
                linear_p_value=linear_sign[0],
                linear_rZ_score=linear_sign[2],
                circular_weighted_r=circular_corr_coeff,
                circular_p_value=circular_sign[0],
                circular_rZ_score=circular_sign[2],
            )
        )

    # TODO: add back in later!!!!!
    # TODO: wouldn't work like this anymore
    # assert len([corr for (corr, sign, _) in linear_corr_coeffs if sign]) + len(
    #     [corr for (corr, sign, _) in circular_corr_coeffs if sign]
    # )

    return BayesianDecodingResult(
        epoch=bayesian_config.epoch,
        decoded_events=decoded_events,
    )


def decode_offline_epoch(
    session: Cached2pSession,
    place_cells: np.ndarray,
    place_fields: np.ndarray,
    bayesian_config: BayesianDecodingConfig,
) -> BayesianDecodingResult:
    """Perform the Bayesian decoding on either of the offline epochs of the session and plot the events."""
    # get just the wheel freeze (either pre-run or post-run)
    if bayesian_config.epoch == "pre":
        # pre-training wheel freeze
        offline, _, _ = split_fluoresence_online_freeze(
            flu=place_cells, wheel_freeze=session.wheel_freeze
        )
    else:
        # post-training wheel freeze
        _, _, offline = split_fluoresence_online_freeze(
            flu=place_cells, wheel_freeze=session.wheel_freeze
        )
    # TODO: is this sigma correct?
    ssp_test = gaussian_filter1d(
        input=offline,
        sigma=bayesian_config.sigma_offline,
        axis=1,
    )

    # pre- or post-training wheel freeze
    population_vector = get_population_vector(
        ssp_smoothed=ssp_test,
        mouse_name=session.mouse_name,
        date=session.date,
        bayesian_config=bayesian_config,
    )
    pse_events = find_pse_events(
        population_vector=population_vector,
        ssp=ssp_test,
        config=bayesian_config,
    )

    if len(pse_events) == 0:
        # print("No PSE events found, exiting")
        # return
        raise AssertionError("No PSE events found")

    pse_activity = [ssp_test[:, start:end] for start, end in pse_events]

    decoded_events: List[DecodedEvent] = list()
    for idx, event in enumerate(pse_activity):
        posterior_probability_matrix, pr_max = sequence_bayesian_decoding(
            event,
            place_fields=place_fields,
            config=bayesian_config,
        )
        linear_corr_coeff = calculate_linear_weighted_correlation(
            posterior_probability_matrix=posterior_probability_matrix,
            xy=construct_xy_by_bin(
                posterior_probability_matrix,
                mode="linear",
                total_length=bayesian_config.total_length,
            ),
        )
        linear_sign = check_significance(
            posterior_probability_matrix=posterior_probability_matrix,
            correlation=linear_corr_coeff,
            mode="linear",
            total_length=bayesian_config.total_length,
            n_shuffles=2000,
            significance=0.05,
        )
        circular_corr_coeff = calculate_circular_weighted_correlation(
            posterior_probability_matrix=posterior_probability_matrix,
        )
        circular_sign = check_significance(
            posterior_probability_matrix=posterior_probability_matrix,
            correlation=circular_corr_coeff,
            mode="circular",
            total_length=bayesian_config.total_length,
            n_shuffles=2000,
            significance=0.05,
        )
        plot_pse_event(
            posterior_probability_matrix=posterior_probability_matrix,
            idx=idx,
            session=session,
            grosmark_config=grosmark_config,
            bayesian_config=bayesian_config,
            mode="linear",
            corr_coeff=linear_corr_coeff,
            significance=linear_sign,
        )
        plot_pse_event(
            posterior_probability_matrix=posterior_probability_matrix,
            idx=idx,
            session=session,
            grosmark_config=grosmark_config,
            bayesian_config=bayesian_config,
            mode="circular",
            corr_coeff=circular_corr_coeff,
            significance=circular_sign,
            do_radon_transform=True,
        )
        print(circular_corr_coeff)
        decoded_events.append(
            DecodedEvent(
                posterior_probability_matrix=posterior_probability_matrix,
                pr_max=pr_max,
                actual_positions=None,
                linear_weighted_r=linear_corr_coeff,
                linear_p_value=linear_sign[0],
                linear_rZ_score=linear_sign[2],
                circular_weighted_r=circular_corr_coeff,
                circular_p_value=circular_sign[0],
                circular_rZ_score=circular_sign[2],
            )
        )

    # TODO: if wanting to keep it, has to be changed
    # sanity check that the correlations in the events aren't all just artifactual
    # except if it is the 'pre' epoch (that would probably even be a good sign)
    # if bayesian_config.epoch != "pre":
    #     assert (
    #         len([corr for (corr, sign, _) in linear_corr_coeffs if sign])
    #         + len([corr for (corr, sign, _) in circular_corr_coeffs if sign])
    #         > 0
    #     ), "No significant event found (in neither linear nor circular weighted correlation)!"

    return BayesianDecodingResult(
        epoch=bayesian_config.epoch, decoded_events=decoded_events
    )


def decode_for_performance_check(
    mouse_name: str,
    date: str,
    bayesian_config: BayesianDecodingConfig,
) -> Dict:
    """
    Perform the Bayesian decoding on phases of mobility within the online epoch of the session, i.e. using the test trials.

    Use this decoding function for assessing the decoder's accuracy/performance (e.g. f1 score, r2, ...).
    Critical differences to `decode_online_epoch`:
    Here, a train-test split of the trials has to be done to prevent data leakage!
    Here, the decoding is done on MOBILITY ssp vectors!!! Also, is not trying to detect PSE events!
    """
    train_size = 0.5  # fraction of trials used to get place cells

    session, place_cells, place_fields, trials_test = (
        load_and_prepare_session_for_bayesian_decoding(
            mouse_name=mouse_name,
            date=date,
            bayesian_config=bayesian_config,
            grosmark_config=grosmark_config,
            use_train_test_split=True,
            train_size=train_size,
        )
    )
    print(f"Working on {session.mouse_name}: {session.date} - {session.session_type}")
    print(f"Checking Bayesian decoder performance")

    # get the ssp vector
    # TODO: is ssp test the correct one? (in terms of convolution)
    # phases of mobility
    """Offline immobility epochs were defined as those in which the animal's velocity,
    smoothed with a half-second Gaussian kernel, was below 3 cm s-1 for at least 3 consecutive seconds.
    Online running epochs were defined as those in which the animal's smoothed velocity was above 5 cm s-1 for at least 3 consecutive seconds."""
    # smoothing is done in compute_grosmark_speed within get_ssp_vectors
    ssp_config_mobility = SSPConfig(
        mode="above",
        speed_threshold=5,
        n_consecutive_samples=3 * 30,
    )
    ssp_result = get_ssp_vectors(
        trials=trials_test,
        place_cells=place_cells,
        sigma=bayesian_config.sigma_online,
        mode=ssp_config_mobility.mode,
        speed_threshold=ssp_config_mobility.speed_threshold,
        n_consecutive_samples=ssp_config_mobility.n_consecutive_samples,
    )
    ssp_test, positions_test = (
        ssp_result.ssp_vectors,
        ssp_result.position_vectors,
    )
    # TODO: is the use of ssp correct?
    if ssp_test.size == 0:
        print("Empty ssp vector, returning None")
        return None
    assert ssp_test.shape[0] == place_cells.shape[0]

    # do the actual Bayesian decoding
    # TODO: is by-trial for correlation correct??? Or would I have to do the decoding on each trial individually???
    _, pr_max = sequence_bayesian_decoding(
        ssp_test,
        place_fields=place_fields,
        config=bayesian_config,
    )
    plot_decoded_vs_actual_position(
        positions=positions_test,
        pr_max=pr_max,
        session=session,
        bayesian_config=bayesian_config,
    )

    assert positions_test.shape == pr_max.shape
    # TODO: are we ok with this binning here?
    y_true_bins = bin_for_classification(
        positions_test, bayesian_config=bayesian_config
    )
    y_pred_bins = pr_max  # .astype(int)
    n_bins = int(
        (bayesian_config.end_spatial - bayesian_config.start_spatial)
        / bayesian_config.bin_size_spatial
    )
    assert np.max(y_true_bins) < n_bins and np.max(y_pred_bins) < n_bins

    plot_confusion_matrix_actual_vs_decoded_position(
        y_true_bins=y_true_bins,
        y_pred_bins=y_pred_bins,
        session=session,
        bayesian_config=bayesian_config,
    )

    # TODO: think about the averaging method
    f1 = f1_score(y_true=y_true_bins, y_pred=y_pred_bins, average="weighted")
    # "weighted average of the F1 scores of each class for the multiclass task"
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    f1_by_position = f1_score(y_true=y_true_bins, y_pred=y_pred_bins, average=None)
    r_square = r2_score(y_true=y_true_bins, y_pred=y_pred_bins)
    print("Done checking Bayesian decoder performance")
    return {"f1": f1, "f1_by_position": f1_by_position, "r2": r_square}


def main(
    mouse_name: str,
    date: str,
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
) -> BayesianDecodingResult | None:
    """
    'Offline PSEs were detected by convolving each PC's (as assessed during that day's run) offline immobility firing rate vector
    Ssp with a 125-ms Gaussian kernel and z-scoring the smoothed firing rate vector. Subsequently, for each frame i, the population
    mean of the smoothed and z-scored vector was taken across PCs and subsequently z-scored.
    Putative PSEs were defined as epochs during which the z-scored population activity vector reached a peak of at least 3.5 s.d.
    above the mean with event-edges at 1 s.d. above the mean, with a minimum inter-event time of 0.2 s.
    Only PSE events lasting between 0.2 s (12 frames) and 1 s (60 frames), and during which at least 5 distinct PCs each fired at least one estimated spike,
    were kept for further analysis.'
    """
    use_cache = False
    train_size = 0.5  # fraction of trials used to get place cells (online only)

    print(f"Doing {mouse_name} on {date} - analysing {bayesian_config.epoch}")
    prepared_session = load_and_prepare_session_for_bayesian_decoding(
        mouse_name=mouse_name,
        date=date,
        train_size=train_size,
        bayesian_config=bayesian_config,
        grosmark_config=grosmark_config,
    )
    if not prepared_session:
        return None

    cache_path = get_cache_path(
        mouse_name=mouse_name,
        date=date,
        bayesian_config=bayesian_config,
    )

    if use_cache and os.path.exists(cache_path):
        print("Loading decoded cache")
        result = load_bayesian_cache(cache_path)
    else:
        # if cache doesn't exist, create it
        os.makedirs(cache_path.parent, exist_ok=True)
        if bayesian_config.epoch == "online":
            session, place_cells, place_fields, trials_test = prepared_session
            print(
                f"Working on {session.mouse_name}: {session.date} - {session.session_type}"
            )
            print(f"Analysing online activity")
            result = decode_online_epoch(
                session=session,
                trials_test=trials_test,
                place_cells=place_cells,
                place_fields=place_fields,
                bayesian_config=bayesian_config,
            )
        else:
            session, place_cells, place_fields = prepared_session
            print(
                f"Working on {session.mouse_name}: {session.date} - {session.session_type}"
            )
            print(f"Analysing {bayesian_config.epoch} activity")
            if not session.wheel_freeze:
                print(f"No wheel freeze for this session, cannot decode any offline")
                return None
            result = decode_offline_epoch(
                session=session,
                place_cells=place_cells,
                place_fields=place_fields,
                bayesian_config=bayesian_config,
            )
        if result:
            save_bayesian_cache(cache_path=cache_path, result=result)

    print(f"Done for {session.mouse_name} on {session.date}")
    return result


def plot_decoded_and_actual_position(positions: np.ndarray, pr_max: np.ndarray) -> None:
    assert positions.shape == pr_max.shape
    plt.figure(figsize=(10, 4))
    plt.plot(
        positions,
        label="Actual Position",
        color="b",
        linestyle="",
        marker="o",
        markersize=1,
    )
    plt.plot(
        pr_max * 5,
        label="Decoded Position",
        color="r",
        alpha=0.8,
        linestyle="",
        marker="o",
        markersize=1,
    )
    plt.savefig("positions.png")


def plot_decoded_vs_actual_position(
    positions: np.ndarray,
    pr_max: np.ndarray,
    session: Cached2pSession,
    bayesian_config: BayesianDecodingConfig,
) -> None:
    assert positions.shape == pr_max.shape
    plt.figure(figsize=(5, 5))
    plt.scatter(positions, pr_max * bayesian_config.bin_size_spatial, s=5, alpha=0.5)
    plt.xlabel("Actual Position")
    plt.ylabel("Decoded Position")
    plt.title(
        f"{session.mouse_name} {session.date} - {session.session_type} (online) \n({bayesian_config.bin_size_time_offline} frames per bin, {bayesian_config.bin_size_spatial} cm per bin)"
    )
    plot_root = PLOT_PATH / "decoded_vs_actual"
    if not os.path.exists(plot_root / session.mouse_name):
        os.makedirs(plot_root / session.mouse_name)
    plt.savefig(
        plot_root / f"{session.mouse_name}_{session.date}_decoded_vs_actual.png",
        dpi=300,
    )


@deprecated(
    "By eye it is yielding the same result as using sklearn's f1 score when setting average='weighted'"
)
def f1_score_by_position(
    y_true_bins: np.ndarray, y_pred_bins: np.ndarray, n_bins: int
) -> List[float]:
    """f1 score by actual position (position bins in ascending order)"""
    # TODO: at the moment, this is super dependent on the spatial bin size (I have seen some predictions being off by one bin, hence still getting an f1 score of 0.0)
    f1_scores = list()
    unique_true_bins = np.unique(y_true_bins)
    for bin_value in range(n_bins):
        if not bin_value in unique_true_bins:
            f1_scores.append(np.nan)
            continue
        else:
            # get f1 score for all the values of a unique y_true_bin
            mask = y_true_bins == bin_value
            assert np.count_nonzero(mask) > 0
            # TODO: again, which averaging method?
            # -> like in sklearn, it should be weighted average I think
            f1_scores.append(
                f1_score(
                    y_true=y_true_bins[mask],
                    y_pred=y_pred_bins[mask],
                    average="weighted",
                ),
            )
    return f1_scores


def get_statistics_correlation(
    genotype: str,
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
) -> pd.DataFrame:
    result = {
        "stage": [],
        "epoch": [],
        "circular_weighted_r": [],
        "circular_p_values": [],
        "circular_rZ_scores": [],
        "linear_weighted_r": [],
        "linear_p_values": [],
        "linear_rZ_scores": [],
        "mouse_id": [],
        "genotype": [],
    }
    if bayesian_config.epoch == "online":
        result["f1"] = []
        result["f1_score_by_position"] = []
        result["r2"] = []
    for mouse_name in SESSIONS_KEEP.keys():
        if get_genotype(mouse_name) != genotype:
            continue
        for stage in ["unsupervised", "learning", "learned"]:
            print(f"Doing {mouse_name} at {stage} stage")
            try:
                date = SESSIONS_KEEP[mouse_name][stage]
            except KeyError:
                print("No session found for this stage in SESSIONS_KEEP, skip")
                continue
            if date is None:
                continue
            # use_cache = False
            use_cache = True
            cache_path = get_cache_path(
                mouse_name=mouse_name,
                date=date,
                bayesian_config=bayesian_config,
            )
            if os.path.exists(cache_path) and use_cache:
                print("Found cache, will load from cache")
                bayesian = load_bayesian_cache(cache_path)
            else:
                try:
                    # load_and_prepare_session_for_bayesian_decoding(
                    #     mouse_name=mouse_name,
                    #     date=date,
                    #     bayesian_config=bayesian_config,
                    #     grosmark_config=grosmark_config,
                    # )
                    # bayesian = None
                    bayesian = main(
                        mouse_name=mouse_name,
                        date=date,
                        bayesian_config=bayesian_config,
                        grosmark_config=grosmark_config,
                    )
                except (ValueError, FileNotFoundError, KeyError) as e:
                    # except (ValueError, FileNotFoundError, KeyError, AssertionError) as e:
                    print(f"Error processing {mouse_name} at {stage} stage: {e}")
                    continue

                if not bayesian:
                    print("Session invalid for further analysis, skip")
                    continue
            # TODO: think about what we'll do with the p_values/significance
            # TODO: did it change the result taking the mean of the absolute values?
            result["circular_weighted_r"].append(
                np.mean(
                    [
                        np.abs(decoded_event.circular_weighted_r)
                        for decoded_event in bayesian.decoded_events
                    ]
                )
            )
            result["linear_weighted_r"].append(
                np.mean(
                    [
                        np.abs(decoded_event.linear_weighted_r)
                        for decoded_event in bayesian.decoded_events
                    ]
                )
            )
            result["circular_p_values"].append(
                [
                    decoded_event.circular_p_value
                    for decoded_event in bayesian.decoded_events
                ]
            )  # need to .explode() later
            result["linear_p_values"].append(
                [
                    decoded_event.linear_p_value
                    for decoded_event in bayesian.decoded_events
                ]
            )  # need to .explode() later
            if bayesian_config.epoch == "online":
                decoder_performance = decode_for_performance_check(
                    mouse_name=mouse_name, date=date, bayesian_config=bayesian_config
                )
                result["f1"].append(decoder_performance["f1"])
                result["f1_score_by_position"].append(
                    decoder_performance["f1_by_position"]
                )
                result["r2"].append(decoder_performance["r2"])
            # TODO: are those ones correct?
            result["linear_rZ_scores"].append(
                np.mean(
                    [
                        np.abs(decoded_event.linear_rZ_score)
                        for decoded_event in bayesian.decoded_events
                    ]
                )
            )
            result["circular_rZ_scores"].append(
                np.mean(
                    [
                        np.abs(decoded_event.circular_rZ_score)
                        for decoded_event in bayesian.decoded_events
                    ]
                )
            )
            result["mouse_id"].append(mouse_name)
            result["genotype"].append(genotype)
            result["stage"].append(stage)
            result["epoch"].append(bayesian.epoch)
    return pd.DataFrame(result)


def plot_decoded_vs_actual_position_rsquare(
    bayesian_config: BayesianDecodingConfig, grosmark_config: GrosmarkConfig
) -> None:
    wt = get_statistics_correlation("WT", bayesian_config, grosmark_config)
    nlgf = get_statistics_correlation("NLGF", bayesian_config, grosmark_config)
    all_data = pd.concat([wt, nlgf], ignore_index=True)

    fig = plt.figure()
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    # p_values = {}

    # for stage in ["Baseline", "Trained"]:
    #     subset = all_data[all_data["stage"] == stage]
    #     assert len(subset) > 100, "make sure nothing weird happend"
    #     p_value = mixed_effects(
    #         df=subset,
    #         dependent_var="correlation",
    #         independent_var="genotype",
    #         group_name="mouse_id",
    #     ).filter(like="C(genotype)")
    #     p_values[f"{stage}"] = p_value

    sns.boxplot(
        data=all_data,
        x="stage",
        y="r2",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        showfliers=False,
    )

    plt.tight_layout()
    sns.despine()
    plt.ylim(None, 1.49)
    ax = plt.gca()
    ymin_plot, ymax_plot = ax.get_ylim()
    plot_range = ymax_plot - ymin_plot
    text_y = ymax_plot - plot_range * 0.1  # place text just below the top of the axis
    # for i, stage in enumerate(["Baseline", "Trained"]):
    #     p_text = f"P = {round(p_values[stage].values[0], 2)}"
    #     ax.text(i, text_y, p_text, ha="center", va="top")

    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        # place legend centered relative to the axes (not the whole figure)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)

    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "decoded_vs_actual_positions"
        / f"actual_vs_decoded_rsquare.png"
    )


def plot_decoded_vs_actual_position_f1(
    bayesian_config: BayesianDecodingConfig, grosmark_config: GrosmarkConfig
) -> None:
    wt = get_statistics_correlation(
        "WT", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    nlgf = get_statistics_correlation(
        "NLGF", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    all_data = pd.concat([wt, nlgf], ignore_index=True)

    fig = plt.figure()
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    p_values = {}

    for stage in ["unsupervised", "learning", "learned"]:
        subset = all_data[all_data["stage"] == stage]
        # assert len(subset) > 100, "make sure nothing weird happend"
        assert len(subset) > 4, "make sure nothing weird happend"
        p_value = mixed_effects(
            df=subset,
            dependent_var="f1",
            independent_var="genotype",
            group_name="mouse_id",
        ).filter(like="C(genotype)")
        p_values[f"{stage}"] = p_value

    sns.boxplot(
        data=all_data,
        x="stage",
        y="f1",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        showfliers=False,
    )

    plt.tight_layout()
    sns.despine()
    # plt.ylim(None, 1.49)
    ax = plt.gca()
    # ymin_plot, ymax_plot = ax.get_ylim()
    # plot_range = ymax_plot - ymin_plot
    # text_y = ymax_plot - plot_range * 0.1  # place text just below the top of the axis
    # for i, stage in enumerate(["Baseline", "Trained"]):
    #     p_text = f"P = {round(p_values[stage].values[0], 2)}"
    #     ax.text(i, text_y, p_text, ha="center", va="top")

    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        # place legend centered relative to the axes (not the whole figure)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)

    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "decoded_vs_actual_positions"
        / "actual_vs_decoded_f1score.png"
    )
    # plt.savefig(Path("plots") / "actual_vs_decoded_f1score_macro.png")
    # plt.savefig(Path("plots") / "actual_vs_decoded_f1score_weighted.png")


def plot_confusion_matrix_actual_vs_decoded_position(
    y_true_bins: np.ndarray,
    y_pred_bins: np.ndarray,
    session: Cached2pSession,
    bayesian_config: BayesianDecodingConfig,
    plot_landmarks: bool = True,
) -> None:
    landmarks_cm = [45, 90, 135]
    landmarks = [l / bayesian_config.bin_size_spatial for l in landmarks_cm]

    cm = confusion_matrix(y_true=y_true_bins, y_pred=y_pred_bins)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, cmap="viridis", square=True)
    if plot_landmarks:
        plt.vlines(
            landmarks, ymin=0, ymax=cm.shape[0], colors="r", linestyles="--", alpha=0.7
        )
    plt.xlabel("Decoded Bin")
    plt.ylabel("True Bin")
    plt.title(
        f"{session.mouse_name} {session.date} - {get_session_type(session.session_type)} (online) \n({bayesian_config.bin_size_time_offline} frames per bin, {bayesian_config.bin_size_spatial} cm per bin)"
    )
    plt.tight_layout()
    plot_root = PLOT_PATH / "decoded_vs_actual"
    if not os.path.exists(plot_root / session.mouse_name):
        os.makedirs(plot_root / session.mouse_name)
    plt.savefig(
        plot_root / f"{session.mouse_name}_{session.date}_confusion_matrix.png",
        dpi=300,
    )


def plot_correlation_across_stages(
    mode: Literal["linear", "circular"],
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
) -> None:
    wt = get_statistics_correlation(
        "WT", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    nlgf = get_statistics_correlation(
        "NLGF", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    all_data = pd.concat([wt, nlgf], ignore_index=True)

    fig, ax = plt.subplots()
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    p_values = {}

    # for stage in ["Baseline", "Trained"]:
    for stage in ["unsupervised", "learning", "learned"]:
        # for stage in ["learning", "learned"]:
        subset = all_data[all_data["stage"] == stage]
        # assert len(subset) > 100, "make sure nothing weird happend"
        p_value = mixed_effects(
            df=subset,
            dependent_var=f"{mode}_weighted_r",
            independent_var="genotype",
            group_name="mouse_id",
        ).filter(like="C(genotype)")
        p_values[f"{stage}"] = p_value

    sns.boxplot(
        data=all_data,
        x="stage",
        y=f"{mode}_weighted_r",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        showfliers=False,
    )
    offset = {
        "WT": -0.2,
        "NLGF": +0.2,
    }

    x_positions = {
        stage: i for i, stage in enumerate(["unsupervised", "learning", "learned"])
    }

    for genotype in ["WT", "NLGF"]:
        sub = all_data[all_data["genotype"] == genotype]
        xs = [x_positions[s] + offset[genotype] for s in sub["stage"]]

        ax.scatter(
            xs,
            sub[f"{mode}_weighted_r"],
            alpha=1,
            s=40,
            color=palette[genotype],
            edgecolor="black",
            label=None,
            zorder=10,
        )

    plt.tight_layout()
    sns.despine()
    # plt.ylim(None, 1.49)
    ax = plt.gca()
    ymin_plot, ymax_plot = ax.get_ylim()
    plot_range = ymax_plot - ymin_plot
    text_y = ymax_plot - plot_range * 0.1  # place text just below the top of the axis
    # for i, stage in enumerate(["Baseline", "Trained"]):
    for i, stage in enumerate(["unsupervised", "learning", "learned"]):
        p_text = f"P = {round(p_values[stage].values[0], 2)}"
        ax.text(i, text_y, p_text, ha="center", va="top")

    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        # place legend centered relative to the axes (not the whole figure)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)

    plt.savefig(
        SERVER_PATH / "viral_plots" / f"{bayesian_config.epoch}_{mode}_weighted_r.png"
    )


def plot_correlation_across_stages_trajectories(
    mode: Literal["linear", "circular"],
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
) -> None:
    wt = get_statistics_correlation(
        "WT", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    nlgf = get_statistics_correlation(
        "NLGF", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    all_data = pd.concat([wt, nlgf], ignore_index=True)

    stages = ["unsupervised", "learning", "learned"]
    genotypes = ["WT", "NLGF"]
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    fig, ax = plt.subplots(figsize=(7, 5))

    all_data["stage"] = pd.Categorical(
        all_data["stage"], categories=stages, ordered=True
    )
    all_data = all_data.sort_values(["mouse_id", "stage"])

    for genotype in genotypes:
        sub = all_data[all_data["genotype"] == genotype]
        for mouse_id, df_mouse in sub.groupby("mouse_id"):
            ax.plot(
                df_mouse["stage"],
                df_mouse[f"{mode}_weighted_r"],
                marker="o",
                linewidth=1.5,
                alpha=0.6,
                color=palette[genotype],
            )

    ax.set_xlabel("Stage")
    ax.set_ylabel(f"{mode} weighted r")

    legend_lines = [
        Line2D([0], [0], color=palette["WT"], lw=2),
        Line2D([0], [0], color=palette["NLGF"], lw=2),
    ]
    ax.legend(legend_lines, ["WT", "NLGF"], title="Genotype")

    sns.despine()
    plt.tight_layout()
    plt.savefig(SERVER_PATH / "viral_plots" / f"{mode}_weighted_r_trajectories.png")


def plot_correlation_against_f1_score_across_stages(
    mode: Literal["linear", "circular"],
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
) -> None:
    wt = get_statistics_correlation(
        "WT", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    nlgf = get_statistics_correlation(
        "NLGF", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    all_data = pd.concat(
        [wt, nlgf],
        ignore_index=True,
    )

    fig, ax = plt.subplots()
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    p_values = {}

    # for stage in ["Baseline", "Trained"]:
    for stage in ["unsupervised", "learning", "learned"]:
        subset = all_data[all_data["stage"] == stage]
        # assert len(subset) > 100, "make sure nothing weird happend"
        p_value = mixed_effects(
            df=subset,
            dependent_var=f"{mode}_weighted_r",
            independent_var="genotype",
            group_name="mouse_id",
        ).filter(like="C(genotype)")
        p_values[f"{stage}"] = p_value

    sns.boxplot(
        data=all_data,
        x="stage",
        y=f"{mode}_weighted_r",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        showfliers=False,
    )
    offset = {
        "WT": -0.2,
        "NLGF": +0.2,
    }

    x_positions = {
        stage: i for i, stage in enumerate(["unsupervised", "learning", "learned"])
    }

    for genotype in ["WT", "NLGF"]:
        sub = all_data[all_data["genotype"] == genotype]
        xs = [x_positions[s] + offset[genotype] for s in sub["stage"]]

        ax.scatter(
            xs,
            sub[f"{mode}_weighted_r"],
            alpha=1,
            s=40,
            color=palette[genotype],
            edgecolor="black",
            label=None,
            zorder=10,
        )

    plt.tight_layout()
    sns.despine()
    # plt.ylim(None, 1.49)
    ax = plt.gca()
    ymin_plot, ymax_plot = ax.get_ylim()
    plot_range = ymax_plot - ymin_plot
    text_y = ymax_plot - plot_range * 0.1  # place text just below the top of the axis
    # for i, stage in enumerate(["Baseline", "Trained"]):
    for i, stage in enumerate(["unsupervised", "learning", "learned"]):
        p_text = f"P = {round(p_values[stage].values[0], 2)}"
        ax.text(i, text_y, p_text, ha="center", va="top")

    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        # place legend centered relative to the axes (not the whole figure)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)

    plt.savefig(SERVER_PATH / "viral_plots" / f"{mode}_weighted_r.png")


def plot_f1_score_by_position_per_session(
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
) -> None:
    wt = get_statistics_correlation(
        "WT", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    nlgf = get_statistics_correlation(
        "NLGF", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    all_data = pd.concat(
        [wt, nlgf],
        ignore_index=True,
    )

    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    # p_values = {}

    # for stage in ["Baseline", "Trained"]:
    stages = ["unsupervised", "learning", "learned"]

    for genotype in ["WT", "NLGF"]:
        genotype_data = all_data[all_data["genotype"] == genotype]
        for mouse in genotype_data["mouse_id"].unique():
            mouse_data = genotype_data[genotype_data["mouse_id"] == mouse]
            fig, axes = plt.subplots(
                nrows=1, ncols=len(mouse_data), sharex=True, sharey=True
            )
            axes = np.atleast_1d(axes)
            for ax, stage in zip(axes, stages):
                subset = mouse_data[mouse_data["stage"] == stage][
                    "f1_score_by_position"
                ]
                if len(subset) == 0:
                    print(f"No data for {mouse} - {stage} stage")
                    continue
                values = subset.iloc[0]
                plot_df = pd.DataFrame(
                    {
                        "position": range(len(values)),
                        "f1_score": values,
                    }
                )
                if len(subset) == 0:
                    continue
                sns.scatterplot(
                    data=plot_df,
                    x="position",
                    y="f1_score",
                    # colors=palette[genotype],
                    ax=ax,
                )
                ax.set_title(f"{stage}")
            plt.suptitle(f"{mouse}")
            plt.tight_layout()
            sns.despine()
            plt.savefig(
                SERVER_PATH
                / "viral_plots"
                / "f1_score"
                / f"{mouse}_f1_score_by_position.png"
            )


def plot_f1_score_by_position(
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
) -> None:
    wt = get_statistics_correlation(
        "WT", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    nlgf = get_statistics_correlation(
        "NLGF", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    all_data = pd.concat(
        [wt, nlgf],
        ignore_index=True,
    )

    stages = ["unsupervised", "learning", "learned"]
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    fig, axes = plt.subplots(
        1,
        len(stages),
        figsize=(8 * len(stages), 8),
        sharey=True,
    )

    axes = np.atleast_1d(axes)

    tidy = all_data.copy()

    tidy["position"] = tidy["f1_score_by_position"].apply(lambda x: list(range(len(x))))

    tidy = tidy.explode(["f1_score_by_position", "position"])
    tidy = tidy.reset_index(drop=True)

    tidy["f1_score_by_position"] = tidy["f1_score_by_position"].astype(float)
    tidy["position"] = tidy["position"].astype(int)

    for ax, stage in zip(axes, stages):
        stage_data = tidy[tidy["stage"] == stage]
        for genotype in ["WT", "NLGF"]:
            genotype_data = stage_data[stage_data["genotype"] == genotype]
            pivot_data = genotype_data.pivot_table(
                index="mouse_id",
                columns="position",
                values="f1_score_by_position",
            )
            arr = pivot_data.values
            x_axis = pivot_data.columns.values
            shaded_line_plot(
                arr=arr,
                x_axis=x_axis,
                axis=ax,
                color=palette[genotype],
                label=genotype,
            )
        x_ticks = np.arange(
            0,
            int(
                (bayesian_config.end_spatial - bayesian_config.start_spatial)
                / bayesian_config.bin_size_spatial
            ),
            1,
        )
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks * bayesian_config.bin_size_spatial)
        # sns.boxplot(
        #     data=stage_data,
        #     x="position",
        #     y="f1_score_by_position",
        #     hue="genotype",
        #     hue_order=["WT", "NLGF"],
        #     palette=palette,
        #     showfliers=False,
        #     ax=ax,
        # )

        ax.set_title(stage)
        ax.set_xlabel("Position")

    axes[0].set_ylabel("F1 score")

    handles, labels = axes[0].get_legend_handles_labels()
    # for ax in axes:
    #     ax.legend_.remove()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        # bbox_to_anchor=(0.5, 1.05),
    )

    # plt.tight_layout()
    sns.despine()
    plt.savefig(SERVER_PATH / "viral_plots" / "f1_score" / "f1_score_by_position.png")


def plot_rZ_scores(
    bayesian_configs: Dict[str, BayesianDecodingConfig],
    grosmark_config: GrosmarkConfig,
    mode: Literal["linear", "circular"] = "circular",
) -> None:
    wt_pre = get_statistics_correlation(
        "WT", bayesian_config=bayesian_configs["pre"], grosmark_config=grosmark_config
    )
    nlgf_pre = get_statistics_correlation(
        "NLGF", bayesian_config=bayesian_configs["pre"], grosmark_config=grosmark_config
    )

    wt_online = get_statistics_correlation(
        "WT",
        bayesian_config=bayesian_configs["online"],
        grosmark_config=grosmark_config,
    )
    nlgf_online = get_statistics_correlation(
        "NLGF",
        bayesian_config=bayesian_configs["online"],
        grosmark_config=grosmark_config,
    )

    wt_post = get_statistics_correlation(
        "WT", bayesian_config=bayesian_configs["post"], grosmark_config=grosmark_config
    )
    nlgf_post = get_statistics_correlation(
        "NLGF",
        bayesian_config=bayesian_configs["post"],
        grosmark_config=grosmark_config,
    )

    all_data = pd.concat(
        [wt_pre, nlgf_pre, wt_online, nlgf_online, wt_post, nlgf_post],
        ignore_index=True,
    )

    variable = f"{mode}_rZ_scores"

    all_data_exploded = all_data.explode(variable)
    all_data_exploded = all_data.explode(variable).reset_index(drop=True)

    stages = ["unsupervised", "learning", "learned"]
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    pre_values = (
        all_data_exploded[all_data_exploded["epoch"] == "pre"]
        .groupby(["mouse_id", "stage"], as_index=False)[variable]
        .mean()
        .rename(columns={variable: "pre_rZ"})
    )
    all_data_exploded = all_data_exploded.merge(
        pre_values,
        on=["mouse_id", "stage"],
        how="left",
    )
    all_data_exploded["rZ_norm"] = (
        all_data_exploded[variable] / all_data_exploded["pre_rZ"]
    )

    fig, axes = plt.subplots(1, len(stages), figsize=(12, 4), sharey=True)

    for ax, stage in zip(axes, stages):
        stage_data = all_data_exploded[all_data_exploded["stage"] == stage]
        sns.boxplot(
            data=stage_data,
            x="epoch",
            y="rZ_norm",
            hue="genotype",
            palette=palette,
            ax=ax,
        )
        ax.set_title(stage)
        ax.set_ylabel("Normalised rZ score (vs pre)")

        epochs = stage_data["epoch"].unique()
        y_max = stage_data["rZ_norm"].max()
        y_offset = 0.1 * y_max
        for i, epoch in enumerate(sorted(epochs)):
            epoch_data = stage_data[stage_data["epoch"] == epoch]
            wt = epoch_data[epoch_data["genotype"] == "WT"]["rZ_norm"]
            nlgf = epoch_data[epoch_data["genotype"] == "NLGF"]["rZ_norm"]

            if len(wt) < 2 or len(nlgf) < 2:
                continue

            stat, p = ttest_ind(wt, nlgf, equal_var=False)
            # wt_mean, nlgf_mean = wt.mean(), nlgf.mean()

            text = f"p={p:.3f}\n"

            ax.text(
                i,
                y_max + y_offset,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.legend_.remove()

    fig.legend(handles[:2], labels[:2], loc="upper right")

    plt.tight_layout()
    plt.savefig(SERVER_PATH / "viral_plots" / f"{variable}.png")


def plot_n_significant_events(
    bayesian_configs: Dict[str, BayesianDecodingConfig],
    grosmark_config: GrosmarkConfig,
    mode: Literal["linear", "circular"] = "circular",
    significance: float = 0.05,
) -> None:
    wt_pre = get_statistics_correlation(
        "WT", bayesian_config=bayesian_configs["pre"], grosmark_config=grosmark_config
    )
    nlgf_pre = get_statistics_correlation(
        "NLGF", bayesian_config=bayesian_configs["pre"], grosmark_config=grosmark_config
    )

    wt_online = get_statistics_correlation(
        "WT",
        bayesian_config=bayesian_configs["online"],
        grosmark_config=grosmark_config,
    )
    nlgf_online = get_statistics_correlation(
        "NLGF",
        bayesian_config=bayesian_configs["online"],
        grosmark_config=grosmark_config,
    )

    wt_post = get_statistics_correlation(
        "WT", bayesian_config=bayesian_configs["post"], grosmark_config=grosmark_config
    )
    nlgf_post = get_statistics_correlation(
        "NLGF",
        bayesian_config=bayesian_configs["post"],
        grosmark_config=grosmark_config,
    )

    all_data = pd.concat(
        [wt_pre, nlgf_pre, wt_online, nlgf_online, wt_post, nlgf_post],
        ignore_index=True,
    )

    variable = f"{mode}_p_values"

    all_data_exploded = all_data.explode(variable)
    all_data_exploded = all_data.explode(variable).reset_index(drop=True)

    stages = ["unsupervised", "learning", "learned"]
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    pre_values = (
        all_data_exploded[all_data_exploded["epoch"] == "pre"]
        .groupby(["mouse_id", "stage"], as_index=False)[variable]
        .mean()
        .rename(columns={variable: "pre_rZ"})
    )
    all_data_exploded = all_data_exploded.merge(
        pre_values,
        on=["mouse_id", "stage"],
        how="left",
    )
    all_data_exploded["significant"] = all_data_exploded[variable] < significance
    counts = (
        all_data_exploded.groupby(
            ["mouse_id", "stage", "epoch", "genotype"], as_index=False
        )["significant"]
        .sum()
        .rename(columns={"significant": "n_significant"})
    )

    fig, axes = plt.subplots(1, len(stages), figsize=(12, 4), sharey=True)

    for ax, stage in zip(axes, stages):
        stage_data = counts[counts["stage"] == stage]
        sns.boxplot(
            data=stage_data,
            x="epoch",
            y="n_significant",
            hue="genotype",
            palette=palette,
            ax=ax,
        )
        ax.set_title(stage)
        ax.set_ylabel(f"Number of significant events (p < {significance})")

        epochs = stage_data["epoch"].unique()
        y_max = stage_data["n_significant"].max()
        y_offset = 0.1 * y_max
        for i, epoch in enumerate(sorted(epochs)):
            epoch_data = stage_data[stage_data["epoch"] == epoch]
            wt = epoch_data[epoch_data["genotype"] == "WT"]["n_significant"]
            nlgf = epoch_data[epoch_data["genotype"] == "NLGF"]["n_significant"]

            if len(wt) < 2 or len(nlgf) < 2:
                continue

            stat, p = ttest_ind(wt, nlgf, equal_var=False)
            # wt_mean, nlgf_mean = wt.mean(), nlgf.mean()

            text = f"p={p:.3f}\n"

            ax.text(
                i,
                y_max + y_offset,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.legend_.remove()

    fig.legend(handles[:2], labels[:2], loc="upper right")

    plt.tight_layout()
    plt.savefig(SERVER_PATH / "viral_plots" / f"{variable}.png")


if __name__ == "__main__":
    # did show a little bit
    # mouse = "JB036"
    # date = "2025-07-05"
    # date = "2025-07-11"

    # mouse = "JB030"
    # date = "2025-03-25"

    # mouse = "JB034"
    # date = "2025-07-08"

    # saw "landmarks"?
    # mouse = "JB035"
    # date = "2025-07-11"

    # looks like landmarks
    # mouse = "JB034"
    # date = "2025-07-04"

    # TODO: time bin 2, spatial bin 5 or 10 cm
    # we recorded @30 fps, i.e. 0.2 sec = 6 frames, 1 sec = 30 frames
    # so, 33 ms would be 1 frame for offline (changed it to 2 frames) and 333 ms would be 10 frames for online decoding
    bayesian_config_online = BayesianDecodingConfig(
        epoch="online",
        # TODO: or is the 125 ms sigma also just offline and 1 s for online????
        sigma_offline=(125 / 1000) * 30,  # "125 ms Gaussian kernel"
        sigma_online=30,  # "1 s Gaussian kernel"
        # peak_threshold=3.5,
        peak_threshold=2,
        edge_threshold=1,
        # event_duration=(6, 30),
        # event_duration=(6, 120),
        # event_duration=(6, 90),
        event_duration=(6, 45),
        # bin_size_time_offline=2,
        bin_size_time_offline=10,
        bin_size_time_online=10,
        start_spatial=0,
        end_spatial=180,
        # bin_size_spatial=5,
        # bin_size_spatial=10,
        bin_size_spatial=15,
    )
    grosmark_config = GrosmarkConfig(
        bin_size=bayesian_config_online.bin_size_spatial,
        start=bayesian_config_online.start_spatial,
        end=bayesian_config_online.end_spatial,
    )

    bayesian_config_pre = copy.deepcopy(bayesian_config_online)
    bayesian_config_pre.epoch = "pre"

    bayesian_config_post = copy.deepcopy(bayesian_config_online)
    bayesian_config_post.epoch = "post"

    plot_rZ_scores(
        bayesian_configs={
            "pre": bayesian_config_pre,
            "online": bayesian_config_online,
            "post": bayesian_config_post,
        },
        grosmark_config=grosmark_config,
        mode="circular",
    )
    plot_n_significant_events(
        bayesian_configs={
            "pre": bayesian_config_pre,
            "online": bayesian_config_online,
            "post": bayesian_config_post,
        },
        grosmark_config=grosmark_config,
        mode="circular",
        significance=0.05,
    )

    plot_correlation_across_stages(
        mode="linear",
        bayesian_config=bayesian_config_online,
        grosmark_config=grosmark_config,
    )
    plot_correlation_across_stages(
        mode="circular",
        bayesian_config=bayesian_config_online,
        grosmark_config=grosmark_config,
    )
    plot_correlation_across_stages(
        mode="linear",
        bayesian_config=bayesian_config_pre,
        grosmark_config=grosmark_config,
    )
    plot_correlation_across_stages(
        mode="circular",
        bayesian_config=bayesian_config_pre,
        grosmark_config=grosmark_config,
    )

    plot_correlation_across_stages(
        mode="linear",
        bayesian_config=bayesian_config_post,
        grosmark_config=grosmark_config,
    )
    plot_correlation_across_stages(
        mode="circular",
        bayesian_config=bayesian_config_post,
        grosmark_config=grosmark_config,
    )

    # TODO: just on the online at the moment unfortunately
    # plot_correlation_across_stages_trajectories(
    #     mode="linear", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    # )
    # plot_correlation_across_stages_trajectories(
    #     mode="circular",
    #     bayesian_config=bayesian_config,
    #     grosmark_config=grosmark_config,
    # )

    plot_decoded_vs_actual_position_f1(
        bayesian_config=bayesian_config_online, grosmark_config=grosmark_config
    )
    plot_correlation_against_f1_score_across_stages(
        "circular",
        bayesian_config=bayesian_config_online,
        grosmark_config=grosmark_config,
    )
    plot_decoded_vs_actual_position_rsquare(bayesian_config_online, grosmark_config)
    plot_decoded_vs_actual_position_f1(bayesian_config_online, grosmark_config)
    plot_f1_score_by_position_per_session(bayesian_config_online, grosmark_config)
    plot_f1_score_by_position(bayesian_config_online, grosmark_config)
