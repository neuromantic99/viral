from typing import List, Tuple
import numpy as np
import sys
import os
import time
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.constants import CACHE_PATH, SERVER_PATH, TIFF_UMBRELLA
from viral.models import (
    Cached2pSession,
    GrosmarkConfig,
    BayesianDecodingConfig,
)

from viral.utils import (
    above_threshold_for_n_consecutive_samples,
    shuffle_rows,
)
from viral.imaging_utils import split_fluoresence_online_freeze, trial_is_imaged
from viral.grosmark_analysis import get_place_cells
from viral.ensemble_reactivation import get_ssp_vectors


# TODO: is this the "right Ssp"? This is hard to understand
def get_population_vector(ssp: np.ndarray) -> np.ndarray:
    """
    Offline PSEs were detected by convolving each PC's (as assessed during that day's run) offline immobility firing rate vector
    Ssp with a 125-ms Gaussian kernel and z-scoring the smoothed firing rate vector. Subsequently, for each frame i, the population
    mean of the smoothed and z-scored vector was taken across PCs and subsequently z-scored.
    """

    # TODO: which Ssp?
    sigma = 125 / 1000 * 30  # 125 ms kernel
    smoothed = np.apply_along_axis(gaussian_filter1d, axis=1, arr=ssp, sigma=sigma)

    # TODO: which axis?
    z_scored = zscore(smoothed, axis=1)
    # Remove nans from silent neurons
    z_scored = np.nan_to_num(z_scored)

    # TODO: which axis?
    return zscore(np.mean(z_scored, axis=0))


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

    # we recorded @30 fps, i.e. 0.2 sec = 6 frames, 1 sec = 30 frames
    # TODO: is this always one??
    population_vector_sd = np.std(population_vector)
    # find all peaks above 3.5 SD
    peaks = above_threshold_for_n_consecutive_samples(
        arr=population_vector,
        threshold=config.peak_threshold * population_vector_sd,
        n_samples=1,
    )

    peak_indices = np.where(peaks)[0]

    # find event edges above 1 SD
    events = list()
    for peak_idx in peak_indices:
        # look for start (go backwards until below 1 SD)
        start_idx = peak_idx
        while (
            start_idx > 0
            and population_vector[start_idx]
            > config.edge_threshold * population_vector_sd
        ):
            start_idx -= 1
        # look for end (go forwards until below 1 SD)
        end_idx = peak_idx
        while (
            end_idx < len(population_vector) - 1
            and population_vector[end_idx]
            > config.edge_threshold * population_vector_sd
        ):
            end_idx += 1
        events.append((start_idx, end_idx))

    print(f"Found {len(events)} events")
    if len(events) == 0:
        return events

    # merge events that are too close together (< 0.2s, i.e. 6 frames)
    events = sorted(events, key=lambda x: x[0])  # events sorted by start time

    merged_events = [events[0]]  # need the first event as a starting point
    for current_start, current_end in events:
        last_start, last_end = merged_events[-1]
        if current_start - last_end < 6:  # less than 0.2s apart
            merged_events[-1] = (
                last_start,
                current_end,
            )  # merge two events that are too close to each other
        else:
            # not too close, keep the event
            merged_events.append((current_start, current_end))

    print(f"Merged into {len(merged_events)} events")

    if duration_filter:
        # filter events by duration e.g. (0.2s - 1s) -> (6 - 30 frames)
        filtered_events = list()
        for start_idx, end_idx in merged_events:
            duration = end_idx - start_idx + 1
            if config.event_duration[0] <= duration <= config.event_duration[1]:
                filtered_events.append((start_idx, end_idx))

        print(f"Filtered to {len(filtered_events)} events by duration")
        if len(filtered_events) == 0:
            return filtered_events
    else:
        filtered_events = merged_events
        print("No duration filtering at the moment")

    # TODO: ssp is estimated spikes, right?
    # perform additional check: at least 5 distinct PCs each fired at least one estimated spike
    additionally_checked = list()
    for start_idx, end_idx in filtered_events:
        pcs_with_spikes = np.where(np.sum(ssp[:, start_idx:end_idx], axis=1) >= 1)[0]
        if len(pcs_with_spikes) >= 5:
            additionally_checked.append((start_idx, end_idx))

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
    bin_length_time_offline = config.bin_size_time_offline / 30  # seconds
    n_time_bins = offline_activity_binned.shape[1]
    n_spatial_bins = place_fields.shape[1]

    Cr = offline_activity_binned.T  # shape: (n_time_bins, n_cells)
    rate_map = place_fields  # shape: (n_cells, n_spatial_bins)

    Cr = Cr * bin_length_time_offline
    rate_map = place_fields.T + (10 ** (-10))

    term2 = (-bin_length_time_offline) * np.sum(rate_map, axis=1)

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


def shuffle_pse_event(
    posterior_probability_matrix: np.ndarray, n_shuffles: int = 5
) -> np.ndarray:
    """
    "While several shuffle approaches were used (Extended Data Fig. 6),
    the principal shuffle used in the main figures involved the random re-ordering (resampling without replacement)
    of the bins observed within a given event."
    "'timeBinPermutation': permutes (resamples without replacement)"
    # TODO Essentially, this is a Python implementation of https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/shufflePopulationEvents.m time bin permutation
    """
    # TODO: check actual shape
    # TODO: make this an empirical P value function?
    return shuffle_rows(posterior_probability_matrix)


def plot_pse_event(
    posterior_probability_matrix: np.ndarray,
    pr_max: np.ndarray,
    idx: int,
    session: Cached2pSession,
    grosmark_config: GrosmarkConfig,
    bayesian_config: BayesianDecodingConfig,
) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(posterior_probability_matrix.T, vmin=0, vmax=0.07, aspect="auto")
    plt.colorbar()
    # plt.xlabel("Time (seconds)")
    # xtick_bins = np.linspace(0, posterior_probability_matrix.shape[0] - 1, 2)
    # xtick_labels = np.round(xtick_bins * bayesian_config.bin_size_time_offline / 30, 2)
    # plt.xticks(xtick_bins, xtick_labels)
    # plt.ylabel("Position (centimetres)")
    # plt.yticks(
    #     np.linspace(0, posterior_probability_matrix.shape[1], 5),
    #     [
    #         str(x)
    #         for x in np.linspace(grosmark_config.start, grosmark_config.end, 5).astype(
    #             int
    #         )
    #     ],
    # )
    plt.tight_layout()
    plt.savefig(
        f"plots/pse_events/{session.mouse_name}_{session.date}_{"online" if bayesian_config.online else "offline"}_event{idx}.png"
    )


def main() -> None:
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

    # did show a little bit
    mouse = "JB036"
    # date = "2025-07-05"
    date = "2025-07-11"

    # mouse = "JB030"
    # date = "2025-03-25"

    # mouse = "JB034"
    # date = "2025-07-08"

    # saw "landmarks"?
    # mouse = "JB035"
    # date = "2025-07-11"

    # TODO: done this
    # looks like landmarks???????
    # mouse = "JB034"
    # date = "2025-07-04"

    # TODO: implement doing this on ITI as well
    # use_ITI = True

    # TODO: time bin 2, spatial bin 5 or 10 cm
    # CAUTION: spatial bin size in GrosmarkConfig!
    bayesian_config = BayesianDecodingConfig(
        en_bloc=True,
        online=True,
        # peak_threshold=3.5,
        peak_threshold=2,
        edge_threshold=0,
        event_duration=(6, 30),
        bin_size_time_online=10,
        bin_size_time_offline=2,
        bin_size_spatial=5,
    )

    # can't use cached place cell threshold when doing a train/test split
    # use_cache = False if bayesian_config.online else True

    train_size = 0.5  # fraction of trials used to get place cells (online only)

    grosmark_config = GrosmarkConfig(
        bin_size=bayesian_config.bin_size_spatial,
        start=0,  # 0
        end=180,
    )

    with open(CACHE_PATH / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())

    print(f"Working on {session.mouse_name}: {session.date} - {session.session_type}")

    if not session.wheel_freeze:
        print(f"Skipping {date} for mouse {mouse} as there was no wheel block")
        return

    print(f"Analysing {"online" if bayesian_config.online else "offline"} activity")
    print(f"Decoding {"en bloc" if bayesian_config.en_bloc else "per PSE event"}")

    spks = np.load(
        TIFF_UMBRELLA
        / session.date
        / session.mouse_name
        / "suite2p"
        / "plane0"
        / "oasis_spikes.npy"
    )
    trials = [trial for trial in session.trials if trial_is_imaged(trial)]

    if bayesian_config.online:
        # "training" the decoder (= place cell template) on a subset of trials
        trials_train, trials_test = train_test_split(
            trials, train_size=train_size, random_state=42
        )
    else:
        # otherwise just keep all imaged trials
        trials_train = trials

    # do the place cell template only on the training data!
    session.trials = trials_train

    cache_file = Path(
        f"{SERVER_PATH}/viral_caches/sequence_detection/{session.mouse_name}_{session.date}_{f"online" if bayesian_config.online else "offline"}_{f"en_bloc" if bayesian_config.en_bloc else "per_event"}_place_cells.npz"
    )

    if cache_file.exists():
        npz = np.load(cache_file)
        pcs_mask = npz["pcs_mask"]
        place_fields = npz["place_fields"]
        place_threshold = npz["place_threshold"]
        print("Loaded place cells from cache")

    else:
        t0 = time.time()
        pcs_mask, place_fields, place_threshold = get_place_cells(
            session=session,
            spks=spks,
            rewarded=None,
            use_cache=use_cache,
            config=grosmark_config,
            plot=False,
        )
        print(f"Time to get place cells: {time.time() - t0}")
        np.savez(
            cache_file,
            pcs_mask=pcs_mask,
            place_fields=place_fields,
            place_threshold=place_threshold,
        )

    place_cells = spks[pcs_mask, :]

    if bayesian_config.online:
        # use the test trials for decoding

        # phases of mobility
        ssp_test, positions_test = get_ssp_vectors(trials_test, place_cells)

        # phases of immobility
        # ssp_test = get_ssp_vectors(
        #     trials=trials_test,
        #     place_cells=spks,
        #     above=False,
        #     speed_threshold=1,
        #     n_consecutive_samples=3 * 30,
        # )
    else:
        # getting just the post-session wheel freeze
        _, _, offline = split_fluoresence_online_freeze(
            flu=place_cells, wheel_freeze=session.wheel_freeze
        )
        sigma = 30
        ssp_test = gaussian_filter1d(
            input=offline,
            sigma=sigma,
            axis=1,
        )

    if not bayesian_config.en_bloc:
        population_vector = get_population_vector(ssp_test)
        pse_events = find_pse_events(population_vector, ssp_test, bayesian_config)

        if len(pse_events) == 0:
            print("No PSE events found, exiting")
            return

        # TODO: just for debugging, remove eventually
        plt.figure()
        plt.plot(population_vector)
        for event_start, event_end in pse_events:
            plt.vlines(
                event_start,
                ymin=min(population_vector),
                ymax=max(population_vector),
                colors="r",
            )
            plt.vlines(
                event_end,
                ymin=min(population_vector),
                ymax=max(population_vector),
                colors="b",
            )
        plt.savefig(
            f"plots/{session.date}_{session.mouse_name}_population_vector_{"online" if bayesian_config.online else "offline"}.png"
        )

        pse_activity = [ssp_test[:, start:end] for start, end in pse_events]

        # TODO: why can place_fields contain NaNs???

        for idx, event in enumerate(pse_activity):
            posterior_probability_matrix, pr_max = offline_sequence_bayesian_decoding(
                event,
                place_fields=place_fields[pcs_mask, :],
                config=bayesian_config,
            )
            plot_pse_event(
                posterior_probability_matrix,
                pr_max,
                idx,
                session,
                grosmark_config,
                bayesian_config,
            )
    else:
        posterior_probability_matrix, pr_max = offline_sequence_bayesian_decoding(
            ssp_test,
            place_fields=place_fields[pcs_mask, :],
            config=bayesian_config,
        )
        plt.figure(figsize=(10, 4))
        plt.imshow(posterior_probability_matrix.T, vmin=0, vmax=0.05, aspect="auto")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(
            f"plots/pse_events/{session.mouse_name}_{session.date}_{"online" if bayesian_config.online else "offline"}_en_bloc.png"
        )

        plot_decoded_vs_actual_position(
            positions=positions_test,
            pr_max=pr_max,
            session=session,
            bayesian_config=bayesian_config,
        )

    print(f"Done for {session.mouse_name} on {session.date}")


def test_against_matlab() -> None:
    python_result = np.genfromtxt("JB030_2025-03-25_result.txt")
    import pandas as pd

    df = pd.read_csv("matlab_result.csv", header=None)
    matlab_result = df.to_numpy()

    plt.figure()
    plt.imshow(python_result.T, aspect="auto", vmin=0, vmax=0.07)
    plt.colorbar()
    plt.savefig("python_result")

    plt.figure()
    plt.imshow(matlab_result.T, aspect="auto", vmin=0, vmax=0.07)
    plt.colorbar()
    plt.savefig("matlab_result")

    assert np.all(np.isclose(python_result, matlab_result, rtol=1e-04))


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
    plt.scatter(positions, pr_max * 5, s=5, alpha=0.5)
    plt.xlabel("Actual Position")
    plt.ylabel("Decoded Position")
    plt.title(
        f"{session.mouse_name} {session.date} - online ({bayesian_config.bin_size_time_offline} frames per bin, {bayesian_config.bin_size_spatial} cm per bin)"
    )
    plt.savefig(
        f"plots/{session.mouse_name}_{session.date}_decoded_vs_actual.png", dpi=300
    )


if __name__ == "__main__":
    main()
    # test_against_matlab()
    # decode_en_bloc()
