import os
from typing import List, Literal, Optional, Tuple
from matplotlib.lines import Line2D
import numpy as np
import sys
import time
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, confusion_matrix, f1_score
from deprecated import deprecated
from tqdm import tqdm

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.constants import CACHE_PATH, SERVER_PATH, TIFF_UMBRELLA, PLOT_PATH
from viral.models import (
    Cached2pSession,
    GrosmarkConfig,
    BayesianDecodingConfig,
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
    additional_pc_check,
    construct_xy_by_bin,
    calculate_linear_weighted_correlation,
    calculate_circular_weighted_correlation,
    check_significance,
    bin_for_classification,
    calculate_radon_replay,
    get_cache_path,
)
from viral.sessions_keep import SESSIONS_KEEP


def get_population_vector(
    ssp_smoothed: np.ndarray, mouse_name: str, date: str
) -> np.ndarray:
    """
    Offline PSEs were detected by convolving each PC's (as assessed during that day's run) offline immobility firing rate vector
    Ssp with a 125-ms Gaussian kernel and z-scoring the smoothed firing rate vector. Subsequently, for each frame i, the population
    mean of the smoothed and z-scored vector was taken across PCs and subsequently z-scored.
    """
    # TODO: check axis?
    z_scored = zscore(ssp_smoothed, axis=1)
    # Remove nans from silent neurons
    z_scored = np.nan_to_num(z_scored)

    # TODO: check axis?
    # return zscore(np.mean(z_scored, axis=0))
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
    plt.savefig(
        PLOT_PATH
        / f"{'pse_events_online' if bayesian_config.online else 'pse_events_offline'}"
        / f"{mouse_name}_{date}_{'online' if bayesian_config.online else 'offline'}_population_vector.png",
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

    # TODO: is the z_scored population activity vector the one with the z_scored means????
    candidate_events = detect_candidate_events(population_vector, config)
    print(
        f"Found {len(candidate_events)} candidate events (before filtering for duration and additional PC check)"
    )
    if len(candidate_events) == 0:
        return []

    # merge events that are too close together (< 0.2s, i.e. 6 frames)
    # TODO: should we even merge them? or discard if the inter-event-time is too short?
    merged_events = merge_close_events(candidate_events)
    print(f"Merged into {len(merged_events)} events")

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


def offline_sequence_bayesian_decoding(
    offline_activity_binned: np.ndarray,
    place_fields: np.ndarray,
    config: BayesianDecodingConfig,
):
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
        # TODO: leave here or move elsewhere?
        line = "multi"
        n_lines = 50

        radon_replay = calculate_radon_replay(
            posterior_probability_matrix=posterior_probability_matrix,
            bayesian_config=bayesian_config,
            line=line,
            n_lines=n_lines,
        )

        offset = (bayesian_config.total_length * 100) / bayesian_config.bin_size_spatial

        for line in radon_replay:
            x = [line.point1x, line.point2x]
            y = [line.point1y, line.point2y]

            # if len(radon_replay) == 1:
            if line == "single":
                # TODO: is it correct to only plot that if it is a single line?
                plt.plot(x, y, color="r", linestyle="--")

            plt.plot(x, [y + offset for y in y], color="r", linestyle="--", linewidth=2)
            plt.plot(x, [y - offset for y in y], color="r", linestyle="--", linewidth=2)

        if line == "single":
            plt.title(
                f"Circular Weighted Correlation: {corr_coeff:.2f} (p={significance[0]:.4f}) \nRadon Slope: {radon_replay[0].slope_metres_per_sec:.2f} m/s ({radon_replay[0].replay_type} replay)"
            )
        elif line == "multi":
            plt.title(
                f"Circular Weighted Correlation: {corr_coeff:.2f} (p={significance[0]:.4f})"
            )

    n_time, n_pos = posterior_probability_matrix.shape

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

    if not os.path.exists(
        PLOT_PATH / "pse_events_radon_multiline" / session.mouse_name
    ):
        os.makedirs(PLOT_PATH / "pse_events_radon_multiline" / session.mouse_name)
    plt.savefig(
        PLOT_PATH
        # / "pse_events_radon"
        / "pse_events_radon_multiline"
        / session.mouse_name
        / f"{session.mouse_name}_{session.date}_{bayesian_config.epoch}_{mode}_{'chunk' if (bayesian_config.epoch == 'online') else 'event'}_{idx}.png"
    )
    plt.close()


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

    en bloc decoding:       Decode the entire period (online or offline) at once
    per PSE event decoding: Detect PSE events, then decode each detected PSE event separately
    """
    use_cache = False

    train_size = 0.5  # fraction of trials used to get place cells (online only)

    with open(CACHE_PATH / f"{mouse_name}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())

    print(f"Working on {session.mouse_name}: {session.date} - {session.session_type}")

    if not bayesian_config.epoch == "online" and not session.wheel_freeze:
        # Skip session if intended to analyse offline activity but there was no wheel block
        print(f"Skipping {date} for mouse {mouse_name} as there was no wheel block")
        return

    print(f"Analysing {bayesian_config.epoch} activity")
    print(
        f"Decoding {'en bloc' if bayesian_config.en_bloc else 'per PSE event/per trial chunk'}"
    )

    cache_path = get_cache_path(
        mouse_name=session.mouse_name,
        date=session.date,
        bayesian_config=bayesian_config,
    )

    spks = np.load(
        TIFF_UMBRELLA
        / session.date
        / session.mouse_name
        / "suite2p"
        / "plane0"
        / "oasis_spikes.npy"
    )
    trials = [trial for trial in session.trials if trial_is_imaged(trial)]

    if bayesian_config.epoch == "online":
        # "training" the decoder (= place cell template) on a subset of trials
        trials_train, trials_test = train_test_split(
            trials, train_size=train_size, random_state=42
        )
        use_train_test_split = True

        # all trial frames
        # ssp_test, positions_test = get_ssp_vectors(trials_test, place_cells, bayesian_config.sigma_online, "all", 5, 3 * 30)

        # TODO: change back!
        # phases of mobility
        # ssp_config = SSPConfig(mode="above", speed_threshold=5, n_consecutive_samples=3*30)

        # # TODO: do we want to increase the speed threshold, and should it be below it for 3 consecutive seconds?
        # # phases of immobility
        # ssp_config = SSPConfig(mode="below", speed_threshold=2, n_consecutive_samples=3*30)
    else:
        # otherwise just keep all imaged trials
        trials_train = trials
        use_train_test_split = False

    # do the place cell template only on the training data!
    session.trials = trials_train

    t0 = time.time()
    pcs_mask, place_fields, _ = get_place_cells(
        session=session,
        spks=spks,
        rewarded=None,
        use_cache=True,
        config=grosmark_config,
        plot=False,
        use_train_test_split=use_train_test_split,  # careful there when changing the train-test-split!!!
    )
    print(f"Time to get place cells: {time.time() - t0}")

    place_cells = spks[pcs_mask, :]

    # TODO: is ssp test the correct one? (in terms of convolution)

    if use_cache and os.path.exists(cache_path):
        print("Loading decoded cache")
        loaded_cache = np.load(cache_path, allow_pickle=True)
        posterior_probability_matrices = loaded_cache["posterior_probability_matrices"]
        pr_max_matrices = loaded_cache["pr_max_matrices"]
        linear_weighted_r = (
            loaded_cache["linear_weighted_r"].astype(float)
            if loaded_cache["linear_weighted_r"] is not None
            else None
        )
        circular_weighted_r = (
            loaded_cache["circular_weighted_r"].astype(float)
            if loaded_cache["circular_weighted_r"] is not None
            else None
        )
        actual_positions = (
            loaded_cache["actual_positions"]
            if loaded_cache["actual_positions"] is not None
            else None
        )
        linear_rZ_scores = (
            loaded_cache["linear_rZ_scores"]
            if loaded_cache["linear_rZ_scores"] is not None
            else None
        )
        circular_rZ_scores = (
            loaded_cache["circular_rZ_scores"]
            if loaded_cache["circular_rZ_scores"] is not None
            else None
        )

        result = BayesianDecodingResult(
            posterior_probability_matrices=posterior_probability_matrices,
            pr_max_matrices=pr_max_matrices,
            linear_weighted_r=linear_weighted_r,
            circular_weighted_r=circular_weighted_r,
            actual_positions=actual_positions,
            linear_rZ_scores=linear_rZ_scores,
            circular_rZ_scores=circular_rZ_scores,
        )
    else:
        # if cache doesn't exist, create it
        os.makedirs(cache_path.parent, exist_ok=True)
        if bayesian_config.epoch == "online":
            # use the test trials for decoding
            # TODO: do we want to increase the speed threshold, and should it be below it for 3 consecutive seconds?
            # phases of immobility
            # ssp_config_immobility = SSPConfig(
            #     mode="below", speed_threshold=2, n_consecutive_samples=3 * 30
            # )
            ssp_config_immobility = SSPConfig(
                mode="below", speed_threshold=1, n_consecutive_samples=3 * 30
            )

            ssp_result = get_ssp_vectors(
                trials=trials_test,
                place_cells=place_cells,
                sigma=bayesian_config.sigma_online,
                mode=ssp_config_immobility.mode,
                speed_threshold=ssp_config_immobility.speed_threshold,
                n_consecutive_samples=ssp_config_immobility.n_consecutive_samples,
            )
            ssp_test, positions_test, _, chunk_start_indices_test = (
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
        else:
            # getting just the wheel freeze
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

        if not bayesian_config.en_bloc:
            if not bayesian_config.epoch == "online":
                # pre- or post-training wheel freeze
                population_vector = get_population_vector(
                    ssp_smoothed=ssp_test, mouse_name=mouse_name, date=date
                )
                pse_events = find_pse_events(
                    population_vector=population_vector,
                    ssp=ssp_test,
                    config=bayesian_config,
                )

                if len(pse_events) == 0:
                    print("No PSE events found, exiting")
                    return

                pse_activity = [ssp_test[:, start:end] for start, end in pse_events]

                # TODO: why can place_fields contain NaNs???
                events_ppm = list()
                events_pr_maxs = list()
                linear_corr_coeffs: List[Tuple[float, bool]] = list()
                circular_corr_coeffs: List[Tuple[float, bool]] = list()
                for idx, event in enumerate(pse_activity):
                    posterior_probability_matrix, pr_max = (
                        offline_sequence_bayesian_decoding(
                            event,
                            place_fields=place_fields[pcs_mask, :],
                            config=bayesian_config,
                        )
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
                    events_ppm.append(posterior_probability_matrix)
                    events_pr_maxs.append(pr_max)
                    linear_corr_coeffs.append((linear_corr_coeff, linear_sign[1]))
                    circular_corr_coeffs.append((circular_corr_coeff, circular_sign[1]))
                # sanity check that the correlations in the events aren't all just artifactual
                assert (
                    len([corr for (corr, sign) in linear_corr_coeffs if sign]) > 0
                ), "No significant PSE events found (linear weighted correlation)!"
                assert (
                    len([corr for (corr, sign) in circular_corr_coeffs if sign]) > 0
                ), "No significant PSE events found (circular weighted correlation)!"
                result = BayesianDecodingResult(
                    posterior_probability_matrices=events_ppm,
                    pr_max_matrices=events_pr_maxs,
                    linear_weighted_r=[corr for (corr, _, _), in linear_corr_coeffs],
                    circular_weighted_r=[corr for (corr, _, _) in circular_corr_coeffs],
                    actual_positions=positions_test if bayesian_config.online else None,
                )
            else:
                # TODO: is by-trial for correlation correct??? Or would I have to do the decoding on each trial individually???
                posterior_probability_matrix, pr_max = (
                    offline_sequence_bayesian_decoding(
                        ssp_test,
                        place_fields=place_fields[pcs_mask, :],
                        config=bayesian_config,
                    )
                )
                # online training epoch
                current_chunk_idx = 0
                all_chunks = [x for xs in chunk_start_indices_test for x in xs]
                chunk_actual_positions = list()
                chunk_ppms = list()
                chunk_pr_maxs = list()
                linear_corr_coeffs: List[Tuple[float, bool]] = list()
                circular_corr_coeffs: List[Tuple[float, bool]] = list()
                # chunk_start_indices_test is a nested list with the outer list being the trial and the inner list being the chunk_start_indices within
                for trial_idx in tqdm(range(len(chunk_start_indices_test))):
                    for _, start in enumerate(chunk_start_indices_test[trial_idx]):
                        if current_chunk_idx < len(all_chunks) - 1:
                            end = all_chunks[current_chunk_idx + 1]
                        else:
                            end = posterior_probability_matrix.shape[0]
                        chunk_actual_positions.append(positions_test[start:end])
                        chunk_ppm = posterior_probability_matrix[start:end, :]
                        chunk_ppms.append(chunk_ppm)
                        chunk_pr_maxs.append(pr_max[start:end])
                        xy = construct_xy_by_bin(
                            chunk_ppm,
                            mode="linear",
                            total_length=bayesian_config.total_length,
                        )
                        linear_corr_coeff = calculate_linear_weighted_correlation(
                            posterior_probability_matrix=chunk_ppm,
                            xy=xy,
                        )
                        linear_sign = check_significance(
                            chunk_ppm,
                            linear_corr_coeff,
                            "linear",
                            bayesian_config.total_length,
                            2000,
                            0.05,
                        )
                        linear_corr_coeffs.append(
                            (linear_corr_coeff, linear_sign[1], linear_sign[2])
                        )
                        circular_corr_coeff = calculate_circular_weighted_correlation(
                            chunk_ppm
                        )
                        circular_sign = check_significance(
                            chunk_ppm,
                            circular_corr_coeff,
                            "circular",
                            bayesian_config.total_length,
                            2000,
                            0.05,
                        )
                        circular_corr_coeffs.append(
                            (circular_corr_coeff, circular_sign[1], circular_sign[2])
                        )
                        plot_pse_event(
                            posterior_probability_matrix=chunk_ppm,
                            idx=current_chunk_idx,
                            session=session,
                            grosmark_config=grosmark_config,
                            bayesian_config=bayesian_config,
                            mode="linear",
                            corr_coeff=linear_corr_coeff,
                            significance=linear_sign,
                        )
                        plot_pse_event(
                            posterior_probability_matrix=chunk_ppm,
                            idx=current_chunk_idx,
                            session=session,
                            grosmark_config=grosmark_config,
                            bayesian_config=bayesian_config,
                            mode="circular",
                            corr_coeff=circular_corr_coeff,
                            significance=circular_sign,
                            do_radon_transform=True,
                        )
                        current_chunk_idx += 1
                # sanity check that the correlations in the trials aren't all just artifactual
                assert (
                    len([corr for (corr, sign, _) in linear_corr_coeffs if sign]) > 0
                ), "No significant trial found (linear weighted correlation)!"
                assert (
                    len([corr for (corr, sign, _) in circular_corr_coeffs if sign]) > 0
                ), "No significant trial found (circular weighted correlation)!"
                result = BayesianDecodingResult(
                    posterior_probability_matrices=chunk_ppms,
                    pr_max_matrices=chunk_pr_maxs,
                    actual_positions=chunk_actual_positions,
                    linear_weighted_r=[corr for (corr, _, _) in linear_corr_coeffs],
                    circular_weighted_r=[corr for (corr, _, _) in circular_corr_coeffs],
                    linear_rZ_scores=[score for (_, _, score) in linear_corr_coeffs],
                    circular_rZ_scores=[
                        score for (_, _, score) in circular_corr_coeffs
                    ],
                )
                # plot_decoded_vs_actual_position(
                #     positions=positions_test,
                #     pr_max=pr_max,
                #     session=session,
                #     bayesian_config=bayesian_config,
                # )
                plot_confusion_matrix_actual_vs_decoded_position(
                    actual_position=positions_test,
                    decoded_position=pr_max * bayesian_config.bin_size_spatial,
                    session=session,
                    bayesian_config=bayesian_config,
                )
        else:
            posterior_probability_matrix, pr_max = offline_sequence_bayesian_decoding(
                ssp_test,
                place_fields=place_fields[pcs_mask, :],
                config=bayesian_config,
            )
            result = BayesianDecodingResult(
                posterior_probability_matrices=posterior_probability_matrix,
                pr_max_matrices=pr_max,
            )
            plt.figure(figsize=(25, 4))
            plt.imshow(
                posterior_probability_matrix.T,
                vmin=0,
                vmax=np.max(posterior_probability_matrix) * 1.1,
                aspect="auto",
            )
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(
                PLOT_PATH
                / "sequence_en_bloc"
                / f"{session.mouse_name}_{session.date}_{bayesian_config.epoch}_en_bloc.png"
            )
    np.savez(
        cache_path,
        posterior_probability_matrices=np.array(
            result.posterior_probability_matrices, dtype=object
        ),
        pr_max_matrices=np.array(result.pr_max_matrices, dtype=object),
        linear_weighted_r=np.array(result.linear_weighted_r, dtype=float),
        circular_weighted_r=np.array(result.circular_weighted_r, dtype=float),
        actual_positions=(
            np.array(result.actual_positions, dtype=object)
            if result.actual_positions is not None
            else None
        ),
        linear_rZ_scores=(
            np.array(result.linear_rZ_scores, dtype=float)
            if result.linear_rZ_scores is not None
            else None
        ),
        circular_rZ_scores=(
            np.array(result.circular_rZ_scores, dtype=float)
            if result.circular_rZ_scores is not None
            else None
        ),
    )
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
    plt.savefig(
        f"plots/{session.mouse_name}_{session.date}_decoded_vs_actual.png", dpi=300
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
        # "circular_weighted_r": [],
        # "linear_weighted_r": [],
        "f1": [],
        "f1_score_by_position": [],
        "r2": [],
        "mouse_id": [],
        "genotype": [],
    }
    for mouse_name in SESSIONS_KEEP.keys():
        if get_genotype(mouse_name) != genotype:
            continue
        for stage in ["unsupervised", "learning", "learned"]:
            print(f"Doing {mouse_name} at {stage} stage")
            try:
                date = SESSIONS_KEEP[mouse_name][stage]
            except KeyError:
                print("No session found in SESSIONS_KEEP, skip")
                continue
            if date is None:
                print("No valid date, skip")
                continue
            # use_cache = True
            use_cache = False
            cache_path = get_cache_path(
                mouse_name=mouse_name,
                date=date,
                bayesian_config=bayesian_config,
            )
            if os.path.exists(cache_path) and use_cache:
                npz = np.load(
                    cache_path,
                    allow_pickle=True,
                )
                bayesian = BayesianDecodingResult(
                    posterior_probability_matrices=npz[
                        "posterior_probability_matrices"
                    ].tolist(),
                    pr_max_matrices=npz["pr_max_matrices"].tolist(),
                    linear_weighted_r=npz["linear_weighted_r"].tolist(),
                    circular_weighted_r=npz["circular_weighted_r"].tolist(),
                    actual_positions=(
                        npz["actual_positions"].tolist()
                        if "actual_positions" in npz.files
                        else None
                    ),
                    linear_rZ_scores=npz["linear_rZ_scores"].tolist(),
                    circular_rZ_scores=npz["circular_rZ_scores"].tolist(),
                )
            else:
                try:
                    bayesian = main(mouse_name, date, bayesian_config, grosmark_config)
                except (ValueError, FileNotFoundError, KeyError) as e:
                    print(f"Error processing {mouse_name} at {stage} stage: {e}")
                    continue

            actual_positions_flattened = np.array([])
            pr_max_flattened = np.array([])
            for trial_position in bayesian.actual_positions:
                actual_positions_flattened = np.concatenate(
                    (actual_positions_flattened, np.array(trial_position))
                )
            for trial_pr_max in bayesian.pr_max_matrices:
                pr_max_flattened = np.concatenate(
                    (pr_max_flattened, np.array(trial_pr_max))
                )

            assert actual_positions_flattened.shape == pr_max_flattened.shape

            # TODO: are we ok with this binning here?
            y_true_bins = bin_for_classification(
                actual_positions_flattened, bayesian_config=bayesian_config
            )
            y_pred_bins = pr_max_flattened.astype(int)

            n_bins = int(
                (bayesian_config.end_spatial - bayesian_config.start_spatial)
                / bayesian_config.bin_size_spatial
            )
            assert np.max(y_true_bins) < n_bins and np.max(y_pred_bins) < n_bins

            # TODO: think about the averaging method
            f1 = f1_score(y_true=y_true_bins, y_pred=y_pred_bins, average="weighted")
            # "weighted average of the F1 scores of each class for the multiclass task"
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
            f1_by_position = f1_score(
                y_true=y_true_bins, y_pred=y_pred_bins, average=None
            )
            r_square = r2_score(y_true=y_true_bins, y_pred=y_pred_bins)
            # TODO: this function will unfortunately ave to be specific to en_bloc and per_event or entail both
            # result["circular_weighted_r"].append(np.mean(bayesian.circular_weighted_r))
            # result["linear_weighted_r"].append(np.mean(bayesian.linear_weighted_r))
            result["f1"].append(f1)
            result["f1_score_by_position"].append(f1_by_position)
            result["r2"].append(r_square)
            result["mouse_id"].append(mouse_name)
            result["genotype"].append(genotype)
            result["stage"].append(stage)
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


def plot_decoded_vs_actual_position_f1(bayesian_config: BayesianDecodingConfig) -> None:
    wt = get_statistics_correlation("WT", bayesian_config)
    nlgf = get_statistics_correlation("NLGF", bayesian_config)
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
    actual_position: np.ndarray,
    decoded_position: np.ndarray,
    session: Cached2pSession,
    bayesian_config: BayesianDecodingConfig,
    plot_landmarks: bool = True,
) -> None:
    landmarks_cm = [45, 90, 135]
    landmarks = [l / bayesian_config.bin_size_spatial for l in landmarks_cm]

    y_true_bins = bin_for_classification(actual_position, bayesian_config)

    y_pred_bins = bin_for_classification(decoded_position, bayesian_config)

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
    plt.savefig(
        PLOT_PATH
        / "decoded_vs_actual_positions"
        / f"{session.mouse_name}_{session.date}_confusion_matrix.png",
        dpi=300,
    )


def plot_correlation_across_stages(mode: Literal["linear", "circular"]) -> None:
    wt = get_statistics_correlation("WT")
    nlgf = get_statistics_correlation("NLGF")
    all_data = pd.concat([wt, nlgf], ignore_index=True)

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

    plt.savefig(
        SERVER_PATH / "viral_plots" / "sequence_en_bloc" / f"{mode}_weighted_r.png"
    )


def plot_correlation_across_stages_trajectories(
    mode: Literal["linear", "circular"],
) -> None:
    wt = get_statistics_correlation("WT")
    nlgf = get_statistics_correlation("NLGF")
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
    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "sequence_en_bloc"
        / f"{mode}_weighted_r_trajectories.png"
    )


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

    plt.savefig(
        SERVER_PATH / "viral_plots" / "sequence_en_bloc" / f"{mode}_weighted_r.png"
    )


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


def plot_rZ_score(
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
    bayesian_config = BayesianDecodingConfig(
        en_bloc=False,
        epoch="online",
        # TODO: or is the 125 ms sigma also just offline and 1 s for online????
        sigma_offline=(125 / 1000) * 30,  # "125 ms Gaussian kernel"
        sigma_online=30,  # "1 s Gaussian kernel"
        peak_threshold=3.5,
        edge_threshold=1,
        # event_duration=(6, 30),
        event_duration=(6, 120),
        bin_size_time_offline=2,
        bin_size_time_online=10,
        start_spatial=0,
        end_spatial=180,
        # bin_size_spatial=5,
        bin_size_spatial=10,
        # bin_size_spatial=15,
    )
    grosmark_config = GrosmarkConfig(
        bin_size=bayesian_config.bin_size_spatial,
        start=bayesian_config.start_spatial,
        end=bayesian_config.end_spatial,
    )
    plot_f1_score_by_position_per_session(bayesian_config, grosmark_config)
    plot_f1_score_by_position(bayesian_config, grosmark_config)
    plot_correlation_across_stages(mode="linear")
    plot_correlation_across_stages(mode="circular")

    # plot_correlation_across_stages(mode="linear")
    # plot_correlation_across_stages(mode="circular")
    # plot_correlation_across_stages_trajectories(mode="linear")
    # plot_correlation_across_stages_trajectories(mode="circular")
    # plot_decoded_vs_actual_position_f1(bayesian_config)
    # plot_correlation_against_f1_score_across_stages(
    #     "circular", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    # )
    # plot_decoded_vs_actual_position_rsquare(bayesian_config, grosmark_config)
    # plot_decoded_vs_actual_position_f1(bayesian_config)
