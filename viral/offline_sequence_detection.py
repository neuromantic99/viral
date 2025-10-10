from typing import List, Tuple
import numpy as np
import sys
import os
import time
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.constants import CACHE_PATH, TIFF_UMBRELLA
from viral.models import (
    Cached2pSession,
    GrosmarkConfig,
    BayesianDecodingConfig,
)

from viral.utils import (
    above_threshold_for_n_consecutive_samples,
    shuffle_rows,
    array_bin_mean,
)
from viral.imaging_utils import (
    split_fluoresence_online_freeze,
)
from viral.grosmark_analysis import get_place_cells


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
    population_vector: np.ndarray, ssp: np.ndarray, config: BayesianDecodingConfig
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

    # TODO: test, is the edge 1 SD or below then??
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
        print("No events found")
        exit()

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

    # filter events by duration e.g. (0.2s - 1s) -> (6 - 30 frames)
    filtered_events = list()
    for start_idx, end_idx in merged_events:
        duration = end_idx - start_idx
        if config.event_duration[0] <= duration <= config.event_duration[1]:
            filtered_events.append((start_idx, end_idx))

    print(f"Filtered to {len(filtered_events)} events by duration")

    # TODO: ssp is estimated spikes, right?
    # perform additional check: at least 5 distinct PCs each fired at least one estimated spike
    additionally_checked = list()
    for start_idx, end_idx in filtered_events:
        pcs_with_spikes = np.where(np.sum(ssp[:, start_idx:end_idx], axis=1) >= 1)[0]
        if len(pcs_with_spikes) >= 5:
            additionally_checked.append((start_idx, end_idx))

    print(f"{len(additionally_checked)} events remaining after additional PC check")

    return additionally_checked


# TODO: whoops, where is online_activity being binned??
# def bin_offline_activity(
#     offline_activity: np.ndarray, config: BayesianDecodingConfig
# ) -> np.ndarray:
#     """To perform offline sequence analysis, within-PSE PC activity was binned into non-overlapping two-frame time bins [...]"""
#     # TODO: unittest this!
#     n_frames = offline_activity.shape[1]
#     n_bins = n_frames // config.bin_size_time_offline
#     bins = np.arange(
#         0, n_bins + config.bin_size_time_offline, config.bin_size_time_offline
#     )
#     binned_activity = list()
#     for i in range(len(bins) - 1):
#         bin_start = bins[i]
#         bin_end = bins[i + 1]
#         if i == len(bins) - 2:
#             # Include upper bound in last bin
#             print("Hey tehre")
#         else:
#             binned_activity.append(offline_activity[:, bin_start:bin_end])


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

    #Essentially, this is a Python implementation of https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/placeBayesLogBuffered.m

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

    # TODO start looking here!!
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
    Essentially, this is a Python implementation of https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/shufflePopulationEvents.m time bin permutation
    """
    # TODO: check actual shape
    # TODO: make this an empirical P value function?
    return shuffle_rows(posterior_probability_matrix)


def plot_pse_event(
    posterior_probability_matrix: np.ndarray,
    pr_max: np.ndarray,
    idx: int,
    bayesian_config: BayesianDecodingConfig,
) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(posterior_probability_matrix.T, vmin=0, vmax=0.05, aspect="auto")
    plt.colorbar()
    plt.xlabel("Time (seconds)")
    xtick_bins = np.linspace(0, posterior_probability_matrix.shape[0] - 1, 2)
    xtick_labels = np.round(xtick_bins * bayesian_config.bin_size_time_offline / 30, 2)
    plt.xticks(xtick_bins, xtick_labels)
    plt.ylabel("Position (centimetres)")
    ytick_bins = np.linspace(0, posterior_probability_matrix.shape[1] - 1, 15)
    ytick_labels = np.round(ytick_bins * bayesian_config.bin_size_spatial, 0)
    plt.yticks(ytick_bins, ytick_labels)
    plt.tight_layout()
    plt.savefig(f"plots/pse_events/event{idx}.svg")
    print(pr_max)


def main() -> None:
    """
    Offline PSEs were detected by convolving each PC’s (as assessed during that day’s run) offline immobility firing rate vector
    Ssp with a 125-ms Gaussian kernel and z-scoring the smoothed firing rate vector. Subsequently, for each frame i, the population
    mean of the smoothed and z-scored vector was taken across PCs and subsequently z-scored.
    Putative PSEs were defined as epochs during which the z-scored population activity vector reached a peak of at least 3.5 s.d.
    above the mean with event-edges at 1 s.d. above the mean, with a minimum inter-event time of 0.2 s.
    Only PSE events lasting between 0.2 s (12 frames) and 1 s (60 frames), and during which at least 5 distinct PCs each fired at least one estimated spike,
    were kept for further analysis.
    """
    # mouse = "JB036"
    # date = "2025-07-05"
    # mouse = "JB030"
    # date = "2025-03-25"

    # mouse = "JB034"
    # date = "2025-07-08"

    mouse = "JB035"
    date = "2025-07-11"

    # TODO: implement doing this on ITI as well
    # use_ITI = True

    bayesian_config = BayesianDecodingConfig(
        peak_threshold=3.5,
        edge_threshold=1,
        event_duration=(6, 30),
        bin_size_time_online=10,
        bin_size_time_offline=1,
        bin_size_spatial=2,
    )

    with open(CACHE_PATH / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())

    print(f"Working on {session.mouse_name}: {session.date} - {session.session_type}")

    if not session.wheel_freeze:
        print(f"Skipping {date} for mouse {mouse} as there was no wheel block")
        return

    cache_file = (
        HERE.parent
        / f"{session.mouse_name}suite2p_{session.date}_offline_sequence_detection.npz"
    )

    grosmark_config = GrosmarkConfig(
        bin_size=2,
        start=0,
        end=170,
    )
    spks = np.load(
        TIFF_UMBRELLA
        / session.date
        / session.mouse_name
        / "suite2p"
        / "plane0"
        / "oasis_spikes.npy"
    )

    if os.path.exists(cache_file):
        print("Loading from cache")
        pcs_mask = np.load(cache_file)["pcs_mask"]
        preactivation = np.load(cache_file)["preactivation"]
        reactivation = np.load(cache_file)["reactivation"]
        place_fields = np.load(cache_file)["place_fields"]
        place_cells = spks[pcs_mask, :]
    else:
        t0 = time.time()
        pcs_mask, place_fields = get_place_cells(
            session=session,
            spks=spks,
            rewarded=None,
            config=grosmark_config,
            plot=False,
        )
        place_cells = spks[pcs_mask, :]
        print(f"Time to get place cells: {time.time() - t0}")
        preactivation, _, reactivation = split_fluoresence_online_freeze(
            flu=place_cells, wheel_freeze=session.wheel_freeze
        )
        np.savez(
            cache_file,
            pcs_mask=pcs_mask,
            preactivation=preactivation,
            reactivation=reactivation,
            place_fields=place_fields,
        )

    population_vector = get_population_vector(reactivation)
    pse_events = find_pse_events(population_vector, reactivation, bayesian_config)

    pse_activity = [
        array_bin_mean(
            arr=reactivation[:, start:end],
            bin_size=bayesian_config.bin_size_time_offline,
            axis=1,
        )
        for start, end in pse_events
    ]

    # TODO: should we do this?
    # "[...] and only bins with non-zero firing rates were used for offline Bayesian decoding.""
    # offline_activity_binned = [
    #     np.delete(
    #         event_activity,
    #         np.where(~event_activity.any(axis=0))[0],
    #         axis=1,
    #     )
    #     for event_activity in pse_activity
    # ]

    # TODO: why can place_fields contain NaNs???

    for idx, event in enumerate(pse_activity):
        posterior_probability_matrix, pr_max = offline_sequence_bayesian_decoding(
            event,
            place_fields=place_fields[pcs_mask, :],
            config=bayesian_config,
        )
        plot_pse_event(posterior_probability_matrix, pr_max, idx, bayesian_config)
        if idx == 0:
            np.savetxt(f"{mouse}_{date}_Cr.txt", event)
            np.savetxt(f"{mouse}_{date}_rateMap.txt", place_fields[pcs_mask, :])
            np.savetxt(f"{mouse}_{date}_result.txt", posterior_probability_matrix)


def test_against_matlab() -> None:
    python_result = np.genfromtxt("JB030_2025-03-25_result.txt")
    import pandas as pd

    df = pd.read_csv("matlab_result.csv", header=None)
    matlab_result = df.to_numpy()

    plt.figure()
    plt.imshow(python_result.T, aspect="auto", vmin=0, vmax=0.05)
    plt.colorbar()
    plt.savefig("python_result")

    plt.figure()
    plt.imshow(matlab_result.T, aspect="auto", vmin=0, vmax=0.05)
    plt.colorbar()
    plt.savefig("matlab_result")

    assert np.all(np.isclose(python_result, matlab_result, rtol=1e-04))


if __name__ == "__main__":
    # main()
    test_against_matlab()
