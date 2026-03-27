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
from scipy.stats import zscore, ttest_ind, ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, confusion_matrix, f1_score, mean_absolute_error
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
    BayesianDecoderPerformance,
    SSPConfig,
    SSPVectorData,
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
    make_position_bins,
    filter_candidate_events_by_duration,
    filter_candidate_events_by_inter_event_time,
    additional_pc_check,
    construct_xy_by_bin,
    calculate_linear_weighted_correlation,
    calculate_circular_weighted_correlation,
    check_significance_of_correlation,
    bin_for_classification,
    calculate_radon_replay,
    check_significance_radon_fit,
    get_cache_path,
    load_bayesian_cache,
    save_bayesian_cache,
)
from viral.sessions_keep import SESSIONS_KEEP


def get_population_vector(
    ssp_smoothed: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    "Offline PSEs were detected by convolving each PC's (as assessed during that day's run) offline immobility firing rate vector
    Ssp with a 125-ms Gaussian kernel and z-scoring the smoothed firing rate vector. Subsequently, for each frame i, the population
    mean of the smoothed and z-scored vector was taken across PCs and subsequently z-scored." (Grosmark et al., 2021)

    Get the z-scored population activity vector, its mean and standard deviation.
    """
    z_scored = zscore(ssp_smoothed, axis=1)
    # Remove nans from silent neurons
    z_scored = np.nan_to_num(z_scored)

    population_vector = zscore(np.mean(z_scored, axis=0))
    population_vector_mean = np.mean(population_vector)
    population_vector_sd = np.std(population_vector)
    return population_vector, population_vector_mean, population_vector_sd


def compute_pse_thresholds_session(
    trials: List[TrialInfo],
    place_cells: np.ndarray,
    ssp_config_immobility: SSPConfig,
    bayesian_config: BayesianDecodingConfig,
    threshold_path: Path,
) -> Tuple[float, float]:
    """
    "Offline PSEs were detected by convolving each PC's (as assessed during that day's run) offline immobility firing rate vector
    Ssp with a 125-ms Gaussian kernel and z-scoring the smoothed firing rate vector. Subsequently, for each frame i, the population
    mean of the smoothed and z-scored vector was taken across PCs and subsequently z-scored.
    [...]
    Putative PSEs were defined as epochs during which the z-scored population activity vector reached a peak of at least 3.5 s.d.
    above the mean with event-edges at 1 s.d. above the mean, with a minimum inter-event time of 0.2 s." (Grosmark et al., 2021)

    The peak and edge thresholds have to be computed on the running/online epoch immobility ssp and kept for all the epochs within the same session.
    This function has to be run before decoding any epoch and should be called with the BayesianDecodingConfig's epoch set to "online"

    Arguments:
        trials (List[TrialInfo]):                   A list of the trials to test the decoder (online only).
        place_cells (np.ndarray):                   An array of place cell spiking activity (shape=(n_place_cells, n_frames)).
        mouse_name (str):                           The mouse's ID.
        date (str):                                 The session date.
        ssp_config_immobility (SSPConfig):          A SSPConfig object, with the settings for getting the immobility ssp vector.
        bayesian_config (BayesianDecodingConfig):   A BayesianDecodingConfig, with settings for the thresholds.
        threshold_path (Path):                      The path to which the result should be cached to.

    Returns:
        Tuple[float, float]:                        A tuple of peak threshold and edge threshold for the entire session.

    Raises:
        AssertionError:                             When trying to compute the peak and edge threshold on the wrong epoch!
    """
    # TODO: perhaps on the ITI as well?
    assert (
        bayesian_config.epoch == "online"
    ), "The thresholds for PSEs in the session should be computed on the online/run epoch!"
    print("Computing PSE thresholds for this session")

    # TODO: you need to make sure this is the same as in the online decoding!!!
    ssp = get_ssp_vectors(
        trials=trials,
        place_cells=place_cells,
        sigma=bayesian_config.sigma_online,
        mode=ssp_config_immobility.mode,
        speed_threshold=ssp_config_immobility.speed_threshold,
        n_consecutive_samples=ssp_config_immobility.n_consecutive_samples,
        take_iti_out=False if bayesian_config.epoch == "online_ITI" else True,
    ).ssp_vectors

    _, population_vector_mean, population_vector_sd = get_population_vector(
        ssp_smoothed=ssp,
    )

    peak_threshold = (
        population_vector_mean + population_vector_sd * bayesian_config.peak_threshold
    )
    edge_threshold = (
        population_vector_mean + population_vector_sd * bayesian_config.edge_threshold
    )

    np.savez(
        threshold_path,
        peak_threshold=peak_threshold,
        edge_threshold=edge_threshold,
    )
    return peak_threshold, edge_threshold


def load_pse_thresholds_session(threshold_path: Path) -> Tuple[float, float]:
    print("Loading PSE thresholds from cache")
    data = np.load(threshold_path)
    return data["peak_threshold"], data["edge_threshold"]


def find_pse_events(
    population_vector: np.ndarray,
    peak_threshold: float,
    edge_threshold: float,
    ssp: np.ndarray,
    bayesian_config: BayesianDecodingConfig,
    mouse_name: str,
    date: str,
    duration_filter: bool = True,
) -> List[Tuple[int, int]]:
    """
    Putative PSEs were defined as epochs during which the z-scored population activity vector reached a peak of at least 3.5 s.d.
    above the mean with event-edges at 1 s.d. above the mean, with a minimum inter-event time of 0.2 s.
    Only PSE events lasting between 0.2 s (12 frames) and 1 s (60 frames), and during which at least 5 distinct PCs each fired at
    least one estimated spike, were kept for further analysis.
    """
    plot_population_vectors = True
    if plot_population_vectors:
        plt.figure(figsize=(30, 20))
        plt.plot(population_vector)
        plt.hlines(
            peak_threshold,
            xmin=0,
            xmax=len(population_vector),
            colors="r",
            linestyles="dashed",
            label="Peak threshold",
        )
        plt.hlines(
            edge_threshold,
            xmin=0,
            xmax=len(population_vector),
            colors="g",
            linestyles="dashed",
            label="Edge threshold",
        )
        plt.title("Population vector")
        if not os.path.exists(
            PLOT_PATH / "bayesian" / f"pse_events_{bayesian_config.epoch}"
        ):
            os.makedirs(PLOT_PATH / "bayesian" / f"pse_events_{bayesian_config.epoch}")
        plt.savefig(
            PLOT_PATH
            / "bayesian"
            / f"pse_events_{bayesian_config.epoch}"
            / f"{mouse_name}_{date}_{bayesian_config.epoch}_population_vector.png",
            dpi=600,
        )
        plt.close()

    candidate_events = detect_candidate_events(
        population_vector=population_vector,
        peak_threshold=peak_threshold,
        edge_threshold=edge_threshold,
    )
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
    min_inter_event_time_frames = 6
    merged_events = filter_candidate_events_by_inter_event_time(
        candidate_events, min_inter_event_time_frames
    )
    print(
        f"Filtered to {len(merged_events)} events by inter-event time ({min_inter_event_time_frames/30} seconds)"
    )

    # TODO: temporary for debugging
    duration_filter = False

    if duration_filter:
        # filter events by duration e.g. (0.2s - 1s) -> (6 - 30 frames)
        filtered_events = filter_candidate_events_by_duration(
            candidate_events=merged_events,
            event_duration_thresholds=bayesian_config.event_duration,
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


def grosmark_bayesian_decoder(
    activity: np.ndarray,
    positions_test: Optional[np.ndarray],
    place_fields: np.ndarray,
    bayesian_config: BayesianDecodingConfig,
    bin_size_time: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        offline_activity_binned: (np.ndarray)       Offline activity (shape: (n_cells, n_frames)).
        place_fields: (np.ndarray)                  Place field array for place cells, template for Bayesian decoding (shape: (n_cells, n_spatial_bins)).
        bayesianconfig: (BayesianDecodingConfig)    A config class containing parameters like bin sizes etc.
    Returns:
        np.ndarray:                                 posterior_probability_matrix (shape: (n_time_bins, n_spatial_bins)).
        np.ndarray:                                 pr_max (spatial bin with the highest posterior probability for that frame; shape: (n_time_bins)).

    Raises:
        AssertionError
    """
    # TODO: amend docstring
    T = activity.shape[1]
    assert (
        activity.shape[0] == place_fields.shape[0]
    ), "Numbers of place cells in offline activity and place field template do not match!"

    buffer = 12
    bin_length_time = bin_size_time / 30  # seconds

    frames_per_bin = bin_size_time
    # Build time bins of size frames_per_bin
    n_time_bins = T // frames_per_bin
    # TODO: this is a bit redundant and doesnt apply for the performance check, but if bin_size_time is 2 frames,
    # then a minimum of 10 bins would mean an event length of at least 20 frames, i.e. 0.67 seconds
    # and as per Grosmark, we want e.g. 0.2 seconds
    if (
        # n_time_bins < 10
        n_time_bins
        < (bayesian_config.event_duration[0]) / bin_size_time
    ):  # had to change this as the temporal bin size and settings for event size should allow events with less than 3 temporal bins
        raise RuntimeError("Too few decoding time bins. Check fps/dt or data length.")

    # reshape into (M, n_timebins, frames_per_bin) and sum -> n_j per timebin
    activity = activity[:, : n_time_bins * frames_per_bin]
    # n_counts = E_trim.reshape(M, n_timebins, frames_per_bin).sum(
    #     axis=2
    # )  # (M, n_timebins)

    # "true position" for each timebin: mean position across frames
    if positions_test is not None:
        pos_trim = positions_test[: n_time_bins * frames_per_bin]
        true_pos = pos_trim.reshape(n_time_bins, frames_per_bin).mean(axis=1)
    else:
        true_pos = None

    n_spatial_bins = place_fields.shape[1]

    Cr = activity.T  # shape: (n_time_bins, n_cells)
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
    # TODO: where did this part go???

    _, centres = make_position_bins(
        bayesian_config.start_spatial,
        bayesian_config.end_spatial,
        bayesian_config.bin_size_spatial,
    )
    decoded_pos = centres[pr_max]

    return Pr, pr_max, decoded_pos, true_pos
    # TODO: check the matlab implementation for same result
    # TODO: it is the same!!!
    # TODO: but function was amended 27/03/2026


def compute_g_rates(
    E_train: np.ndarray, pos_train: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Courtesy of Daniel Goodwin.

    E_train: (M, T) binary events per frame
    pos_train: (T,) position cm
    edges: position bin edges
    Returns:
      g: (n_pos_bins, M)  rate of significant frames per second in each spatial bin
      pX: (n_pos_bins,)   occupancy prior (normalised)
    """
    M, T = E_train.shape
    pos = np.asarray(pos_train, float)

    # restrict to track range
    m = np.isfinite(pos) & (pos >= edges[0]) & (pos <= edges[-1])
    pos = pos[m]
    E = E_train[:, m]
    if E.shape[1] < 50:
        raise RuntimeError("Too few valid training frames in track range.")

    n_pos_bins = len(edges) - 1
    b = np.digitize(pos, edges) - 1
    b = np.clip(b, 0, n_pos_bins - 1)

    # occupancy in FRAMES
    occ_frames = np.bincount(b, minlength=n_pos_bins).astype(float)

    # count events per (posbin, neuron) in FRAMES
    # g wants rate per second: (events / time_in_bin_seconds)
    g = np.zeros((n_pos_bins, M), float)

    for i in range(n_pos_bins):
        mi = b == i
        if not np.any(mi):
            continue
        # events per neuron in this spatial bin (frames)
        ev = np.sum(E[:, mi], axis=1).astype(float)  # (M,)
        time_sec = mi.sum() / 30
        if time_sec > 0:
            g[i, :] = ev / time_sec

    # occupancy prior
    pX = (
        occ_frames / np.sum(occ_frames)
        if np.sum(occ_frames) > 0
        else np.ones(n_pos_bins) / n_pos_bins
    )
    return g, pX


def goodwin_bayesian_decoder(
    E_test: np.ndarray,
    pos_test: np.ndarray,
    g: np.ndarray,
    pX: np.ndarray,
    *,
    edges: np.ndarray,
    centers: np.ndarray,
    frames_per_bin: int,
    n_samples: int,
    n_neurons: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Courtesy of Daniel Goodwin.
    Implementation of Climer et al., 2025 (https://doi.org/10.1038/s41586-025-09245-y).

    Returns:
      decoded_pos_avg: (n_timebins,) averaged decoded position across subsamples
      true_pos_bin:    (n_timebins,) true position (mean pos within timebin)
      t_sec:           (n_timebins,) time in seconds for each timebin (starts at 0)
    """
    # TODO: if returning the posterior probability matrix, amend docstring
    rng_seed = 123
    rng = np.random.default_rng(rng_seed)

    M, T = E_test.shape
    n_pos_bins = g.shape[0]
    assert g.shape == (n_pos_bins, M)

    pos = np.asarray(pos_test, float)

    # Build time bins of size frames_per_bin
    n_timebins = T // frames_per_bin
    # if n_timebins < 10:
    if (
        n_timebins < 3
    ):  # had to change this as the temporal bin size and settings for event size should allow events with less than 3 temporal bins
        raise RuntimeError("Too few decoding time bins. Check fps/dt or data length.")

    # reshape into (M, n_timebins, frames_per_bin) and sum -> n_j per timebin
    E_trim = E_test[:, : n_timebins * frames_per_bin]
    n_counts = E_trim.reshape(M, n_timebins, frames_per_bin).sum(
        axis=2
    )  # (M, n_timebins)

    # "true position" for each timebin: mean position across frames
    pos_trim = pos[: n_timebins * frames_per_bin]
    true_pos = pos_trim.reshape(n_timebins, frames_per_bin).mean(axis=1)

    # Time axis in seconds for each bin
    t_sec = np.arange(n_timebins) * dt

    # mask out timebins whose true pos is out of range (optional)
    if true_pos:
        in_rng = (
            np.isfinite(true_pos) & (true_pos >= edges[0]) & (true_pos <= edges[-1])
        )
    else:
        in_rng = None

    # Precompute log(pX)
    log_pX = np.log(np.maximum(pX, 1e-12))

    # For each subsample, decode argmax_i log p(x_i|n)
    decoded_pos_samples = np.full((n_samples, n_timebins), np.nan, float)

    post_accumulated = np.zeros((n_pos_bins, n_timebins))
    for s in range(n_samples):
        idx = rng.choice(M, size=n_neurons, replace=False)

        # Pull sub-matrices
        g_sub = g[:, idx]  # (n_pos_bins, n_neurons)
        n_sub = n_counts[idx, :]  # (n_neurons, n_timebins)

        # log likelihood:
        # log p(x_i|n) = log pX(x_i) + sum_j n_j log g_{i,j} - dt * sum_j g_{i,j}  (+ const)
        g_safe = np.maximum(g_sub, 1e-12)
        log_g = np.log(g_safe)

        # termA: sum_j n_j log g_{i,j}
        termA = log_g @ n_sub

        # termB: -dt * sum_j g_{i,j}
        termB = -dt * np.sum(g_sub, axis=1)[:, None]  # (n_pos_bins, 1)

        log_post = log_pX[:, None] + termA + termB

        decoded_bins = np.argmax(log_post, axis=0)  # (n_timebins,)
        decoded_pos_samples[s, :] = centers[decoded_bins]

        # normalise
        log_post -= np.max(log_post, axis=0, keepdims=True)
        post = np.exp(log_post)
        post /= np.sum(post, axis=0, keepdims=True)

        post_accumulated += post

    post_mean = post_accumulated / n_samples

    decoded_pos_avg = np.nanmean(decoded_pos_samples, axis=0)

    # apply in-range mask as NaN for clean error calc
    if in_rng:
        decoded_pos_avg[~in_rng] = np.nan
    # true_pos[~in_rng] = np.nan

    # TODO: whoops we'll need true pos again I think
    assert post_mean.T.shape[0] == n_timebins
    # return decoded_pos_avg, true_pos, t_sec, post_mean.T
    return decoded_pos_avg, t_sec, post_mean.T


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
        scoring = "Denovellis"
        radon_replay = calculate_radon_replay(
            posterior_probability_matrix=posterior_probability_matrix,
            bayesian_config=bayesian_config,
            incorporate_nearby_positions=incorporate_nearby_positions,
            nearby_positions=nearby_positions,
            min_n_bin_perc=min_n_bin_perc,
            scoring=scoring,
        )
        # radon_replay_significance = check_significance_radon_fit(
        #     real_radon=radon_replay,
        #     posterior_probability_matrix=posterior_probability_matrix,
        #     bayesian_config=bayesian_config,
        #     incorporate_nearby_positions=incorporate_nearby_positions,
        #     nearby_positions=nearby_positions,
        #     min_n_bin_perc=min_n_bin_perc,
        #     scoring=scoring,
        #     n_shuffles=2000,
        #     significance=0.05,
        # )

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
        # plt.title(
        #     f"Circular Weighted Correlation: {corr_coeff:.2f} (p={significance[0]:.4f}) \nRadon Slope: {radon_replay.slope_metres_per_sec:.2f} m/s ({radon_replay.replay_type} replay, p={radon_replay_significance[0]:.4f})"
        # )

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

    radon_plot_root = PLOT_PATH / "bayesian" / "pse_events_band"
    if not os.path.exists(radon_plot_root / session.mouse_name):
        os.makedirs(radon_plot_root / session.mouse_name)
    plt.savefig(
        radon_plot_root
        / session.mouse_name
        / f"{session.mouse_name}_{session.date}_{bayesian_config.epoch}_{mode}_event_{idx}.png"
    )
    plt.close()


def load_and_prepare_session_for_bayesian_decoding(
    mouse_name: str,
    date: str,
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
    use_train_test_split: bool = False,
    train_size: Optional[float] = 0.5,
) -> Tuple[Cached2pSession, np.ndarray, np.ndarray, List[TrialInfo]] | None:
    """
    Load and prepare session for Bayesian decoding (getting the cached session object and getting the place cells).
    A train-test split for the trials can be performed. (Makes sense if your checking the decoder's performance).
    If you're decoding online or offline PSE events, you shouldn't use the train-test split!
    The data should already be non-overlapping because of the speed filter for training (mobility) and testing (immobility), respectively.

    Returns:
        Cached2pSession:            The session object with the trials allotted according to the train-test split (online only).
        np.ndarray:                 The session's OASIS deconvoluted place cell activity.
        np.ndarray:                 Place fields of the place cells in the array above.
        list[TrialInfo] | None:     A list of the trials to test the decoder (returned for all epochs, but used in online only).
        None:                       None if the session cannot be analysed (epoch is set to either of the offline but a wheel freeze does not exist)
    """
    # TODO: if using Goodwin approach then amend docstring and tidy up func calls
    print("Loading and preparing session")
    with open(CACHE_PATH / f"{mouse_name}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())

    if not "online" in bayesian_config.epoch and not session.wheel_freeze:
        # Skip session as intended to analyse offline activity but there was no wheel block
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

    if not use_train_test_split:
        return session, place_cells, pcs_place_fields, trials_test
    else:
        return session, place_cells, pcs_place_fields, trials_test, trials_train


def decode_events(
    session: Cached2pSession,
    place_fields: np.ndarray,
    ssp_test: SSPVectorData,
    ssp_train: SSPVectorData,
    pse_events: List[Tuple[int, int]],
    bayesian_config: BayesianDecodingConfig,
    # positions_test=Optional[np.ndarray],
    significance: float = 0.05,
    only_plot_significant_events: bool = False,
) -> List[DecodedEvent]:
    """Decode and plot events."""
    decoded_events: List[DecodedEvent] = list()
    pse_activity = [ssp_test.ssp_vectors[:, start:end] for start, end in pse_events]
    if "online" in bayesian_config.epoch:
        positions_test = ssp_test.position_vectors
        actual_positions = [positions_test[start:end] for start, end in pse_events]
    else:
        # in the offline wheel freezes, there is no real position as the mouse is forced to sit, hence return None
        actual_positions = None
    for idx, event in enumerate(pse_activity):
        # dt = (
        #     bayesian_config.bin_size_time_online / 30
        # )  # e.g. 10 frames/bin / 30 frames/sec = 1/3 sec/bin

        # edges, centers = make_position_bins(
        #     bayesian_config.start_spatial,
        #     bayesian_config.end_spatial,
        #     bayesian_config.bin_size_spatial,
        # )
        # # Train: compute g_{i,j} and prior pX
        # g, pX = compute_g_rates(
        #     E_train=ssp_train.ssp_vectors,
        #     pos_train=ssp_train.position_vectors,
        #     edges=edges,
        # )  # g: (n_pos_bins, M)

        # pr_max, _, posterior_probability_matrix = goodwin_bayesian_decoder(
        #     E_test=event,
        #     pos_test=actual_positions[idx] if actual_positions else None,
        #     g=g,
        #     pX=pX,
        #     edges=edges,
        #     centers=centers,
        #     frames_per_bin=bayesian_config.bin_size_time_online,
        #     n_samples=bayesian_config.n_samples,
        #     n_neurons=bayesian_config.n_neurons,
        #     dt=dt,
        # )

        posterior_probability_matrix, pr_max, _, _ = grosmark_bayesian_decoder(
            event,
            positions_test=None,
            place_fields=place_fields,
            bin_size_time=bayesian_config.bin_size_time_offline,
            bayesian_config=bayesian_config,
        )
        linear_corr_coeff = calculate_linear_weighted_correlation(
            posterior_probability_matrix=posterior_probability_matrix,
            xy=construct_xy_by_bin(
                posterior_probability_matrix,
                mode="linear",
                total_length=bayesian_config.total_length,
            ),
        )
        linear_sign = check_significance_of_correlation(
            posterior_probability_matrix=posterior_probability_matrix,
            correlation=linear_corr_coeff,
            mode="linear",
            total_length=bayesian_config.total_length,
            n_shuffles=2000,
            significance=significance,
        )
        circular_corr_coeff = calculate_circular_weighted_correlation(
            posterior_probability_matrix=posterior_probability_matrix,
        )
        circular_sign = check_significance_of_correlation(
            posterior_probability_matrix=posterior_probability_matrix,
            correlation=circular_corr_coeff,
            mode="circular",
            total_length=bayesian_config.total_length,
            n_shuffles=2000,
            significance=significance,
        )
        if not only_plot_significant_events:
            # plotting all events regardless of significance
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
        else:
            # only plotting events that are significant in either the linear or circular correlation, or both
            if linear_sign[1] or circular_sign[1]:
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
        decoded_events.append(
            DecodedEvent(
                posterior_probability_matrix=posterior_probability_matrix,
                pr_max=pr_max,
                actual_positions=actual_positions[idx] if actual_positions else None,
                linear_weighted_r=linear_corr_coeff,
                linear_p_value=linear_sign[0],
                linear_rZ_score=linear_sign[2],
                circular_weighted_r=circular_corr_coeff,
                circular_p_value=circular_sign[0],
                circular_rZ_score=circular_sign[2],
            )
        )
    return decoded_events


def decode_online_epoch(
    session: Cached2pSession,
    trials: List[TrialInfo],
    place_cells: np.ndarray,
    place_fields: np.ndarray,
    pse_thresholds: Tuple[float, float],
    bayesian_config: BayesianDecodingConfig,
) -> BayesianDecodingResult | None:
    """Perform the Bayesian decoding on phases of immobility within the online epoch of the session, i.e. using the test trials, and plot the events."""
    # It is a bit hidden, but Grosmark says in Fig. 5a: 'run epoch PSEs (occurring during immobility) '

    # get the ssp vector
    # TODO: is ssp test the correct one? (in terms of convolution)
    # phases of mobility
    # ssp_config_mobility = SSPConfig(mode="above", speed_threshold=5, n_consecutive_samples=3*30)
    ssp_config_mobility = SSPConfig(
        mode="above",
        speed_threshold=5,
        n_consecutive_samples=3 * 30,
    )
    ssp_train = get_ssp_vectors(
        trials=trials,
        place_cells=place_cells,
        sigma=bayesian_config.sigma_online,
        mode=ssp_config_mobility.mode,
        speed_threshold=ssp_config_mobility.speed_threshold,
        n_consecutive_samples=ssp_config_mobility.n_consecutive_samples,
        take_iti_out=False if bayesian_config.epoch == "online_ITI" else True,
    )
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
    ssp_test = get_ssp_vectors(
        trials=trials,
        place_cells=place_cells,
        sigma=bayesian_config.sigma_online,
        mode=ssp_config_immobility.mode,
        speed_threshold=ssp_config_immobility.speed_threshold,
        n_consecutive_samples=ssp_config_immobility.n_consecutive_samples,
        take_iti_out=False if bayesian_config.epoch == "online_ITI" else True,
    )
    # ssp_test, positions_test, _, _ = (
    #     ssp_result.ssp_vectors,
    #     ssp_result.position_vectors,
    #     ssp_result.trial_start_indices,
    #     ssp_result.chunk_start_indices,
    # )
    # TODO: is the use of ssp correct?
    # if ssp_test.size == 0:
    if ssp_test.ssp_vectors.size == 0:
        print("Empty ssp vector, returning None")
        return None
    # assert ssp_test.shape[0] == place_cells.shape[0]
    assert ssp_test.ssp_vectors.shape[0] == place_cells.shape[0]

    population_vector, _, _ = get_population_vector(
        ssp_smoothed=ssp_test.ssp_vectors,
    )
    peak_threshold, edge_threshold = pse_thresholds
    pse_events = find_pse_events(
        population_vector=population_vector,
        peak_threshold=peak_threshold,
        edge_threshold=edge_threshold,
        ssp=ssp_test.ssp_vectors,
        bayesian_config=bayesian_config,
        mouse_name=session.mouse_name,
        date=session.date,
    )
    if len(pse_events) > 0:
        decoded_events = decode_events(
            session=session,
            place_fields=place_fields,
            ssp_test=ssp_test,
            ssp_train=ssp_train,
            pse_events=pse_events,
            bayesian_config=bayesian_config,
            # positions_test=ssp_test.position_vectors,
            significance=0.05,
            only_plot_significant_events=True,
        )
    else:
        decoded_events = []
        print("No valid PSE events found")

    return BayesianDecodingResult(
        epoch=bayesian_config.epoch,
        decoded_events=decoded_events,
    )


def decode_offline_epoch(
    session: Cached2pSession,
    place_cells: np.ndarray,
    place_fields: np.ndarray,
    pse_thresholds: Tuple[float, float],
    bayesian_config: BayesianDecodingConfig,
) -> BayesianDecodingResult:
    """Perform the Bayesian decoding on either of the offline epochs of the session and plot the events."""
    ssp_config_mobility = SSPConfig(
        mode="above",
        speed_threshold=5,
        n_consecutive_samples=3 * 30,
    )
    ssp_train = get_ssp_vectors(
        trials=session.trials,
        place_cells=place_cells,
        sigma=bayesian_config.sigma_online,
        mode=ssp_config_mobility.mode,
        speed_threshold=ssp_config_mobility.speed_threshold,
        n_consecutive_samples=ssp_config_mobility.n_consecutive_samples,
        take_iti_out=False if bayesian_config.epoch == "online_ITI" else True,
    )
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
    ssp_test = SSPVectorData(
        ssp_vectors=gaussian_filter1d(
            input=offline,
            sigma=bayesian_config.sigma_offline,
            axis=1,
        )
    )

    # pre- or post-training wheel freeze
    population_vector, _, _ = get_population_vector(ssp_smoothed=ssp_test.ssp_vectors)
    peak_threshold, edge_threshold = pse_thresholds
    pse_events = find_pse_events(
        population_vector=population_vector,
        peak_threshold=peak_threshold,
        edge_threshold=edge_threshold,
        ssp=ssp_test.ssp_vectors,
        bayesian_config=bayesian_config,
        mouse_name=session.mouse_name,
        date=session.date,
    )

    if len(pse_events) > 0:
        decoded_events = decode_events(
            session=session,
            place_fields=place_fields,
            ssp_test=ssp_test,
            ssp_train=ssp_train,
            pse_events=pse_events,
            bayesian_config=bayesian_config,
            # positions_test=ssp_test.position_vectors,
            significance=0.05,
            only_plot_significant_events=True,
        )
    else:
        decoded_events = []
        print("No valid PSE events found")

    return BayesianDecodingResult(
        epoch=bayesian_config.epoch, decoded_events=decoded_events
    )


def decode_for_performance_check(
    mouse_name: str,
    date: str,
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
    use_cache: bool = False,
) -> BayesianDecoderPerformance:
    """
    Perform the Bayesian decoding on phases of mobility within the online epoch of the session, i.e. using the test trials.

    Use this decoding function for assessing the decoder's accuracy/performance (e.g. f1 score, r2, ...).
    Critical differences to `decode_online_epoch`:
    Here, a train-test split of the trials has to be done to prevent data leakage!
    Here, the decoding is done on MOBILITY ssp vectors!!! Also, is not trying to detect PSE events!
    """
    train_size = 0.7  # fraction of trials used to get place cells

    cache_path = (
        SERVER_PATH
        / "viral_caches"
        / "bayesian"
        / f"{mouse_name}_{date}_decoder_performance_bin_size_spatial-{bayesian_config.bin_size_spatial}_train_size-{train_size}.npz"
    )

    if os.path.exists(cache_path) and use_cache:
        print("Loading from cache")
        loaded = np.load(cache_path, allow_pickle=True)
        data = loaded["data"].item()
        result = BayesianDecoderPerformance(**data)
    else:
        print("Checking Bayesian decoder performance")
        session, place_cells, place_fields, trials_test, trials_train = (
            load_and_prepare_session_for_bayesian_decoding(
                mouse_name=mouse_name,
                date=date,
                bayesian_config=bayesian_config,
                grosmark_config=grosmark_config,
                use_train_test_split=True,
                train_size=train_size,
            )
        )
        print(
            f"Working on {session.mouse_name}: {session.date} - {session.session_type}"
        )

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
        ssp_train_result = get_ssp_vectors(
            trials=trials_train,
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
        ssp_train, positions_train = (
            ssp_train_result.ssp_vectors,
            ssp_train_result.position_vectors,
        )
        # TODO: is the use of ssp correct?
        # TODO: would have to be amended for train ssp as well
        if ssp_test.size == 0:
            # if ssp_test.ssp_vectors.size == 0:
            print("Empty ssp vector, returning None")
            return None
        assert ssp_test.shape[0] == place_cells.shape[0]
        # assert ssp_test.ssp_vectors.shape[0] == place_cells.shape[0]

        # do the actual Bayesian decoding
        _, pr_max, decoded_positions, true_positions = grosmark_bayesian_decoder(
            ssp_test,
            positions_test=positions_test,
            place_fields=place_fields,
            bin_size_time=bayesian_config.bin_size_time_online,
            bayesian_config=bayesian_config,
        )
        # TODO: add more comments / explanations
        # n_neurons = 50
        # n_samples = 100
        # dt = (
        #     bayesian_config.bin_size_time_online / 30
        # )  # e.g. 10 frames/bin / 30 frames/sec = 1/3 sec/bin

        # edges, centers = make_position_bins(
        #     bayesian_config.start_spatial,
        #     bayesian_config.end_spatial,
        #     bayesian_config.bin_size_spatial,
        # )
        # # Train: compute g_{i,j} and prior pX
        # g, pX = compute_g_rates(
        #     E_train=ssp_train, pos_train=positions_train, edges=edges
        # )  # g: (n_pos_bins, M)

        # true_positions = positions_test
        # decoded_positions, _, _ = goodwin_bayesian_decoder(
        #     E_test=ssp_test,
        #     pos_test=positions_test,
        #     g=g,
        #     pX=pX,
        #     edges=edges,
        #     centers=centers,
        #     frames_per_bin=bayesian_config.bin_size_time_online,
        #     n_samples=n_samples,
        #     n_neurons=n_neurons,
        #     dt=dt,
        # )

        plot_decoded_vs_actual_position(
            positions=true_positions,
            decoded_positions=decoded_positions,
            # positions=true_positions,
            # pr_max=decoded_positions,
            session=session,
            bayesian_config=bayesian_config,
        )

        # assert positions_test.shape == pr_max.shape
        # TODO: are we ok with this binning here?? Like really?
        y_true_bins = bin_for_classification(
            true_positions, bayesian_config=bayesian_config
        )
        # y_pred_bins = pr_max  # .astype(int)
        # assert true_positions.shape == decoded_positions.shape
        # y_true_bins = bin_for_classification(true_positions, bayesian_config)
        y_pred_bins = bin_for_classification(decoded_positions, bayesian_config)
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
        # was confused why in the plots this was often negative
        # from the docstring: "Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse)."
        # See this for reference: https://towardsdatascience.com/explaining-negative-r-squared-17894ca26321/
        # The definiton of R² has one critical assumption: "aforementioned equality is defined for models trained on the same data"!
        # "In short, R² is only the square of correlation if we happen to be
        # (1) using linear regression models, and (2) are evaluating them on the same data they are fitted (as established previously)."
        # i.e. two things: 1. its use outside of linear regression can be problematic, 2. in fact, it would be more correct to not do the train-test split?
        # TODO: do we care, should I implement that correctly?
        r_square = r2_score(y_true=y_true_bins, y_pred=y_pred_bins)

        # mae = mean_absolute_error(y_true=y_true_bins, y_pred=y_pred_bins)
        decoded_positions = pr_max * bayesian_config.bin_size_spatial
        abs_err = np.abs(decoded_positions - true_positions)
        mae = np.nanmean(abs_err)

        result = BayesianDecoderPerformance(
            f1_score=f1,
            f1_score_by_position=f1_by_position,
            r2=r_square,
            mean_absolute_error=mae,
        )

        data = result.model_dump()
        np.savez(
            cache_path,
            data=np.array(data, dtype=object),
        )
    print("Done checking Bayesian decoder performance")
    return result


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

    print(f"Doing {mouse_name} on {date} - analysing {bayesian_config.epoch}")
    prepared_session = load_and_prepare_session_for_bayesian_decoding(
        mouse_name=mouse_name,
        date=date,
        bayesian_config=bayesian_config,
        grosmark_config=grosmark_config,
    )
    if not prepared_session:
        return None
    session, place_cells, place_fields, trials_test = prepared_session

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

        # thresholds for peak and edges of candidate PSE events
        # TODO: perhaps on the ITI as well?
        threshold_path = (
            SERVER_PATH
            / "viral_caches"
            / "bayesian"
            / f"{mouse_name}_{date}_online_pse_thresholds.npz"
        )
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
        if os.path.exists(threshold_path):
            pse_thresholds = load_pse_thresholds_session(threshold_path=threshold_path)
        else:
            pse_thresholds = compute_pse_thresholds_session(
                trials=trials_test,
                place_cells=place_cells,
                ssp_config_immobility=ssp_config_immobility,
                bayesian_config=bayesian_config,
                threshold_path=threshold_path,
            )
        if "online" in bayesian_config.epoch:
            print(
                f"Working on {session.mouse_name}: {session.date} - {session.session_type}"
            )
            print("Analysing online activity")
            result = decode_online_epoch(
                session=session,
                trials=trials_test,
                place_cells=place_cells,
                place_fields=place_fields,
                pse_thresholds=pse_thresholds,
                bayesian_config=bayesian_config,
            )
        else:
            print(
                f"Working on {session.mouse_name}: {session.date} - {session.session_type}"
            )
            print(f"Analysing {bayesian_config.epoch} activity")
            if not session.wheel_freeze:
                print("No wheel freeze for this session, cannot decode any offline")
                return None
            pse_thresholds = load_pse_thresholds_session(threshold_path=threshold_path)
            result = decode_offline_epoch(
                session=session,
                place_cells=place_cells,
                place_fields=place_fields,
                pse_thresholds=pse_thresholds,
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
    decoded_positions: np.ndarray,
    session: Cached2pSession,
    bayesian_config: BayesianDecodingConfig,
) -> None:
    """Plot the actual position on the x-axis and the decoded position on the y-axis for tested decoder."""
    assert positions.shape == decoded_positions.shape
    plt.figure(figsize=(5, 5))
    plt.scatter(positions, decoded_positions, s=5, alpha=0.5)
    plt.xlabel("Actual Position")
    plt.ylabel("Decoded Position")
    plt.title(
        f"{session.mouse_name} {session.date} - {session.session_type} (online) \n({bayesian_config.bin_size_time_offline} frames per bin, {bayesian_config.bin_size_spatial} cm per bin)"
    )
    plot_root = PLOT_PATH / "bayesian" / "decoded_vs_actual"
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
                print("No session found for this stage in SESSIONS_KEEP, skip")
                continue
            use_cache = False
            # use_cache = True
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
            result["circular_weighted_r"].append(
                [
                    decoded_event.circular_weighted_r
                    for decoded_event in bayesian.decoded_events
                ]
            )
            result["linear_weighted_r"].append(
                [
                    decoded_event.linear_weighted_r
                    for decoded_event in bayesian.decoded_events
                ]
            )
            result["circular_p_values"].append(
                [
                    decoded_event.circular_p_value
                    for decoded_event in bayesian.decoded_events
                ]
            )
            result["linear_p_values"].append(
                [
                    decoded_event.linear_p_value
                    for decoded_event in bayesian.decoded_events
                ]
            )
            result["linear_rZ_scores"].append(
                [
                    decoded_event.linear_rZ_score
                    for decoded_event in bayesian.decoded_events
                ]
            )
            result["circular_rZ_scores"].append(
                [
                    decoded_event.circular_rZ_score
                    for decoded_event in bayesian.decoded_events
                ]
            )
            result["mouse_id"].append(mouse_name)
            result["genotype"].append(genotype)
            result["stage"].append(stage)
            result["epoch"].append(bayesian.epoch)
    return pd.DataFrame(result)


def get_statistics_decoder_performance(
    genotype: str,
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
) -> pd.DataFrame:
    result = {
        "f1": [],
        "f1_score_by_position": [],
        "r2": [],
        "mean_absolute_error": [],
        "stage": [],
        "epoch": [],
        "mouse_id": [],
        "genotype": [],
    }
    use_cache = False
    # use_cache = True

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
                print("No session found for this stage in SESSIONS_KEEP, skip")
                continue

            decoder_performance = decode_for_performance_check(
                mouse_name=mouse_name,
                date=date,
                bayesian_config=bayesian_config,
                grosmark_config=grosmark_config,
                use_cache=use_cache,
            )
            result["f1"].append(decoder_performance.f1_score)
            result["f1_score_by_position"].append(
                decoder_performance.f1_score_by_position
            )
            result["r2"].append(decoder_performance.r2)
            result["mean_absolute_error"].append(
                decoder_performance.mean_absolute_error
            )
            result["mouse_id"].append(mouse_name)
            result["genotype"].append(genotype)
            result["stage"].append(stage)
            result["epoch"].append(bayesian_config.epoch)
    return pd.DataFrame(result)


def plot_decoded_vs_actual_position_rsquare(
    bayesian_config: BayesianDecodingConfig, grosmark_config: GrosmarkConfig
) -> None:
    """Plot the decoder's r2."""
    wt = get_statistics_decoder_performance("WT", bayesian_config, grosmark_config)
    nlgf = get_statistics_decoder_performance("NLGF", bayesian_config, grosmark_config)
    all_data = pd.concat([wt, nlgf], ignore_index=True)
    fig = plt.figure()
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}
    sns.boxplot(
        data=all_data,
        x="stage",
        y="r2",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        showfliers=False,
    )
    stages = ["unsupervised", "learning", "learned"]
    p_values = dict()
    for stage in stages:
        # TODO: should this be mixed_effects or ttest?
        stage_data = all_data[all_data["stage"] == stage]
        wt_data = stage_data[stage_data["genotype"] == "WT"]["r2"]
        nlgf_data = stage_data[stage_data["genotype"] == "NLGFT"]["r2"]
        _, p_value = ttest_ind(wt_data, nlgf_data, equal_var=False)
        p_values[f"{stage}"] = p_value
    ax = plt.gca()
    ymin_plot, ymax_plot = ax.get_ylim()
    plot_range = ymax_plot - ymin_plot
    text_y = ymax_plot - plot_range * 0.1  # place text just below the top of the axis
    for i, stage in enumerate(stages):
        p_text = f"P = {round(p_values[stage], 2)}"
        ax.text(i, text_y, p_text, ha="center", va="top")
    plt.tight_layout()
    sns.despine()
    # plt.ylim(None, 1.49)

    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        # place legend centered relative to the axes (not the whole figure)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)

    plt.savefig(PLOT_PATH / "bayesian" / "decoded_vs_actual" / f"rsquare.png")


def plot_decoded_vs_actual_position_f1(
    bayesian_config: BayesianDecodingConfig, grosmark_config: GrosmarkConfig
) -> None:
    wt = get_statistics_decoder_performance(
        "WT", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    nlgf = get_statistics_decoder_performance(
        "NLGF", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    all_data = pd.concat([wt, nlgf], ignore_index=True)

    fig = plt.figure()
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    stages = ["unsupervised", "learning", "learned"]
    p_values = {}
    for stage in stages:
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
    ymin_plot, ymax_plot = ax.get_ylim()
    plot_range = ymax_plot - ymin_plot
    text_y = ymax_plot - plot_range * 0.1  # place text just below the top of the axis
    for i, stage in enumerate(stages):
        p_text = f"P = {round(p_values[stage].values[0], 2)}"
        ax.text(i, text_y, p_text, ha="center", va="top")

    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        # place legend centered relative to the axes (not the whole figure)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)

    plt.savefig(PLOT_PATH / "bayesian" / "decoded_vs_actual" / "f1_score.png")


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
    plot_root = PLOT_PATH / "bayesian" / "decoded_vs_actual"
    plt.savefig(
        plot_root / f"{session.mouse_name}_{session.date}_confusion_matrix.png",
        dpi=300,
    )


def plot_mean_decoding_error(
    bayesian_config: BayesianDecodingConfig, grosmark_config: GrosmarkConfig
) -> None:
    wt = get_statistics_decoder_performance(
        "WT", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    nlgf = get_statistics_decoder_performance(
        "NLGF", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    all_data = pd.concat([wt, nlgf], ignore_index=True)

    fig = plt.figure()
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    stages = ["unsupervised", "learning", "learned"]
    p_values = {}
    for stage in stages:
        subset = all_data[all_data["stage"] == stage]
        p_value = mixed_effects(
            df=subset,
            dependent_var="mean_absolute_error",
            independent_var="genotype",
            group_name="mouse_id",
        ).filter(like="C(genotype)")
        p_values[f"{stage}"] = p_value

    sns.boxplot(
        data=all_data,
        x="stage",
        y="mean_absolute_error",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        showfliers=False,
    )
    plt.title("Bayesian decoding error \n(each datapoint = 1 session)")
    plt.tight_layout()
    sns.despine()
    # plt.ylim(None, 1.49)
    ax = plt.gca()
    ymin_plot, ymax_plot = ax.get_ylim()
    plot_range = ymax_plot - ymin_plot
    text_y = ymax_plot - plot_range * 0.1  # place text just below the top of the axis
    for i, stage in enumerate(stages):
        p_text = f"P = {round(p_values[stage].values[0], 2)}"
        ax.text(i, text_y, p_text, ha="center", va="top")

    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        # place legend centered relative to the axes (not the whole figure)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)

    plt.savefig(
        PLOT_PATH / "bayesian" / "decoded_vs_actual" / "mean_absolute_error.png"
    )


def plot_correlation_across_stages(
    mode: Literal["linear", "circular"],
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
    significance: float = 0.05,
) -> None:
    wt = get_statistics_correlation(
        "WT", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    nlgf = get_statistics_correlation(
        "NLGF", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )

    variable = f"{mode}_weighted_r"
    variable_p_values = f"{mode}_p_values"

    all_data = pd.concat([wt, nlgf], ignore_index=True)
    all_data_exploded = all_data.explode(variable).reset_index(drop=True)
    # TODO: whomp, super dangerous, try and see what to do about NaNs!!!
    # all_data_exploded[variable] = pd.to_numeric(
    #     all_data_exploded[variable], errors="coerce"
    # )
    # all_data_exploded = all_data_exploded.dropna(subset=variable).reset_index(drop=True)

    fig, ax = plt.subplots()
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    p_values = {}

    significance_text_lines = dict()
    for stage in ["unsupervised", "learning", "learned"]:
        subset = all_data_exploded[all_data_exploded["stage"] == stage].reset_index(
            drop=True
        )
        # # assert len(subset) > 100, "make sure nothing weird happend"
        # TODO: did not converge?
        # p_value = mixed_effects(
        #     df=subset,
        #     dependent_var=variable,
        #     independent_var="genotype",
        #     group_name="mouse_id",
        # ).filter(like="C(genotype)")
        # p_values[f"{stage}"] = p_value
        significance_text_lines[stage] = get_significance_text_lines(
            data=subset,
            variable_p_values=variable_p_values,
            significance=significance,
        )

    # TODO: should the mean be plotted?
    sns.boxplot(
        data=all_data_exploded,
        x="stage",
        y=variable,
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
        sub = all_data_exploded[all_data_exploded["genotype"] == genotype]
        xs = [x_positions[s] + offset[genotype] for s in sub["stage"]]

        ax.scatter(
            xs,
            sub[variable],
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
    # for i, stage in enumerate(["unsupervised", "learning", "learned"]):
    #     # p_text = f"P = {round(p_values[stage], 2)}"
    #     # ax.text(i, text_y, p_text, ha="center", va="top")
    #     p_values_text = "\n".join(significance_text_lines[stage])
    #     ax.text(
    #         0.02,
    #         0.98,
    #         p_values_text,
    #         transform=ax.transAxes,
    #         ha="left",
    #         va="top",
    #         fontsize=7,
    #     )

    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        # place legend centered relative to the axes (not the whole figure)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)

    plt.savefig(
        PLOT_PATH / "bayesian" / f"{bayesian_config.epoch}_{mode}_weighted_r.png"
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
    plt.savefig(
        PLOT_PATH
        / "bayesian"
        / f"{bayesian_config.epoch}_{mode}_weighted_r_trajectories.png"
    )


# TODO: won't work anymore
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

    plt.savefig(PLOT_PATH / "bayesian" / f"{mode}_weighted_r.png")


def plot_f1_score_by_position_per_session(
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
) -> None:
    wt = get_statistics_decoder_performance(
        "WT", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    nlgf = get_statistics_decoder_performance(
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
                PLOT_PATH
                / "bayesian"
                / "decoded_vs_actual"
                / f"{mouse}_f1_score_by_position.png"
            )


def plot_f1_score_by_position(
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
) -> None:
    wt = get_statistics_decoder_performance(
        "WT", bayesian_config=bayesian_config, grosmark_config=grosmark_config
    )
    nlgf = get_statistics_decoder_performance(
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
        wt_stage_data = (
            stage_data[stage_data["genotype"] == "WT"]
            .groupby("mouse_id")["f1_score_by_position"]
            .mean()
            .values
        )
        nlgf_stage_data = (
            stage_data[stage_data["genotype"] == "NLGF"]
            .groupby("mouse_id")["f1_score_by_position"]
            .mean()
            .values
        )
        p = ks_2samp(
            data1=wt_stage_data,
            data2=nlgf_stage_data,
            alternative="two-sided",
        ).pvalue
        text = f"p={p:.3f}\n(two-sample Kolmogorov-Smirnov test)\n"
        ax.text(
            0.02,
            0.98,
            text,
            # ha="center",
            # va="bottom",
            fontsize=8,
            transform=ax.transAxes,
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

        ax.set_title(stage)
        ax.set_xlabel("Position")

    axes[0].set_ylabel("F1 score")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        # bbox_to_anchor=(0.5, 1.05),
    )

    # plt.tight_layout()
    sns.despine()
    plt.savefig(
        PLOT_PATH / "bayesian" / "decoded_vs_actual" / "f1_score_by_position.png"
    )


def plot_rZ_scores(
    bayesian_config: BayesianDecodingConfig,
    grosmark_config: GrosmarkConfig,
    mode: Literal["linear", "circular"] = "circular",
    significance: float = 0.05,
) -> None:
    """Plot rZ scores."""
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

    variable = f"{mode}_rZ_scores"
    variable_p_values = f"{mode}_p_values"

    # TODO: is this what we want? per mouse mean of the absolute values per event
    # all_data[variable] = all_data[variable].apply(lambda x: np.mean(np.abs(x)))
    all_data = all_data.explode(variable)

    all_data[variable] = pd.to_numeric(all_data[variable], errors="coerce")
    # TODO: dangerous, think about i
    all_data = all_data.dropna(subset=[variable])

    stages = ["unsupervised", "learning", "learned"]
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    fig, axes = plt.subplots(1, len(stages), figsize=(12, 4), sharey=True)

    for ax, stage in zip(axes, stages):
        stage_data = all_data[all_data["stage"] == stage]
        stage_data = stage_data.reset_index(drop=True)
        sns.boxplot(
            data=stage_data,
            x="genotype",
            y=variable,
            hue="genotype",
            palette=palette,
            ax=ax,
        )
        ax.set_title(stage)
        ax.set_ylabel("rZ score")

        y_max = stage_data[variable].max()
        y_offset = 0.1 * y_max

        # wt_data = stage_data[stage_data["genotype"] == "WT"][variable]
        # nlgf_data = stage_data[stage_data["genotype"] == "NLGF"][variable]

        if len(stage_data["mouse_id"].unique()) > 1:
            try:
                stage_data = stage_data.reset_index(drop=True)
                p = mixed_effects(
                    df=stage_data,
                    dependent_var=variable,
                    independent_var="genotype",
                    group_name="mouse_id",
                ).filter(like="C(genotype)")
                text = f"p={p.iloc[0]:.3f}\n"
            except AssertionError:
                text = "cannot perform mixed effects"
        else:
            text = "cannot perform mixed effects"

        significance_text_lines = get_significance_text_lines(
            data=stage_data,
            variable_p_values=variable_p_values,
            significance=significance,
        )

        # text = f"p={p:.3f}\n"
        ax.text(
            0.5,
            y_max + y_offset,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
        )

        # p_values_text = "\n".join(significance_text_lines)
        # ax.text(
        #     0.02,
        #     0.98,
        #     p_values_text,
        #     transform=ax.transAxes,
        #     ha="left",
        #     va="top",
        #     fontsize=7,
        # )

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles[:2], labels[:2], loc="upper right")

    plt.tight_layout()
    plt.savefig(PLOT_PATH / "bayesian" / f"{variable}_{bayesian_config.epoch}.png")


def get_significance_text_lines(
    data: pd.DataFrame, variable_p_values: str, significance: float = 0.05
):
    significance_text_lines = list()
    for genotype in ["WT", "NLGF"]:
        significance_text_lines.append(f"{genotype}:")
        for _, row in data[data["genotype"] == genotype].iterrows():
            pvals = row[variable_p_values]
            n_sig = sum(p < significance for p in pvals)
            n_tot = len(pvals)
            significance_text_lines.append(f"{row['mouse_id']}: {n_sig}/{n_tot}")
    return significance_text_lines


def plot_normalised_rZ_scores(
    bayesian_configs: Dict[str, BayesianDecodingConfig],
    grosmark_config: GrosmarkConfig,
    mode: Literal["linear", "circular"] = "circular",
    significance: float = 0.05,
) -> None:
    """Plot rZ scores normalised against the pre-run epoch."""
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
    variable_p_values = f"{mode}_p_values"

    # TODO: is this what we want? per mouse mean of the absolute values per event
    # all_data[variable] = all_data[variable].apply(lambda x: np.mean(np.abs(x)))
    all_data_exploded = all_data.explode(variable)

    all_data_exploded[variable] = pd.to_numeric(
        all_data_exploded[variable], errors="coerce"
    )
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
    # mice have to have a pre value
    valid_mice = pre_values["mouse_id"].unique()
    all_data_exploded = all_data_exploded[
        all_data_exploded["mouse_id"].isin(valid_mice)
    ]
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

            # TODO: mixed effects instead?
            # stat, p = ttest_ind(wt, nlgf, equal_var=False)
            try:
                p = mixed_effects(
                    df=epoch_data,
                    dependent_var=variable,
                    independent_var="genotype",
                    group_name="mouse_id",
                ).filter(like="C(genotype)")
                text = f"p={p:.3f}\n"
            except AssertionError:
                text = "cannot perform mixed effects"

            ax.text(
                i,
                y_max + y_offset,
                text,
                ha="center",
                va="bottom",
                fontsize=8,
            )

            # significance_text_lines = get_significance_text_lines(
            #     data=epoch_data,
            #     variable_p_values=variable_p_values,
            #     significance=significance,
            # )
            # p_values_text = "\n".join(significance_text_lines)
            # ax.text(
            #     0.02,
            #     0.98,
            #     p_values_text,
            #     transform=ax.transAxes,
            #     ha="left",
            #     va="top",
            #     fontsize=7,
            # )

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        ax.legend_.remove()

    fig.legend(handles[:2], labels[:2], loc="upper right")

    plt.tight_layout()
    plt.savefig(PLOT_PATH / "bayesian" / f"normalised_{variable}.png")


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

    fig, axes = plt.subplots(1, len(stages), figsize=(12, 8), sharey=True)

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

            # stat, p = ttest_ind(wt, nlgf, equal_var=False)
            try:
                p = mixed_effects(
                    df=epoch_data,
                    dependent_var="n_significant",
                    independent_var="genotype",
                    group_name="mouse_id",
                ).filter(like="C(genotype)")
                text = f"p={p.iloc[0]:.3f}\n"
            except AssertionError:
                text = "cannot perform mixed effects"

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
    plt.savefig(PLOT_PATH / "bayesian" / f"{variable}.png")


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
        event_duration=(6, 90),
        # event_duration=(6, 45),
        # bin_size_time_offline=2,
        bin_size_time_offline=2,  # originally, 2 frames @ 60 fps, now 2 frames @ 30 fps
        bin_size_time_online=20,  # originally, 20 frames @ 60 fps, now 20 frames @ 30 fps
        start_spatial=0,
        end_spatial=180,
        # bin_size_spatial=5,
        # bin_size_spatial=10,
        bin_size_spatial=15,
        n_samples=100,
        n_neurons=50,
    )
    grosmark_config = GrosmarkConfig(
        bin_size=bayesian_config_online.bin_size_spatial,
        start=bayesian_config_online.start_spatial,
        end=bayesian_config_online.end_spatial,
    )

    # TODO: well, this isn't amazing engineering but are we ok with this?
    bayesian_config_online_ITI = copy.deepcopy(bayesian_config_online)
    bayesian_config_online_ITI.epoch = "online_ITI"
    # TODO: is this ok? place fields on 0-180 cm, decoding for 0-180 cm, but using testing data from 0-inf cms?

    bayesian_config_pre = copy.deepcopy(bayesian_config_online)
    bayesian_config_pre.epoch = "pre"

    bayesian_config_post = copy.deepcopy(bayesian_config_online)
    bayesian_config_post.epoch = "post"

    bayesian_config_decoder = copy.deepcopy(bayesian_config_online)
    bayesian_config_decoder.epoch = "online"
    bayesian_config_decoder.bin_size_time_offline = 10
    bayesian_config_decoder.bin_size_time_online = 10

    plot_mean_decoding_error(
        bayesian_config=bayesian_config_decoder, grosmark_config=grosmark_config
    )

    plot_f1_score_by_position_per_session(bayesian_config_decoder, grosmark_config)
    plot_f1_score_by_position(bayesian_config_decoder, grosmark_config)
    plot_decoded_vs_actual_position_rsquare(bayesian_config_decoder, grosmark_config)
    plot_decoded_vs_actual_position_f1(bayesian_config_decoder, grosmark_config)

    plot_rZ_scores(
        bayesian_config=bayesian_config_online,
        grosmark_config=grosmark_config,
        mode="circular",
        significance=0.05,
    )

    plot_rZ_scores(
        bayesian_config=bayesian_config_pre,
        grosmark_config=grosmark_config,
        mode="circular",
        significance=0.05,
    )

    plot_rZ_scores(
        bayesian_config=bayesian_config_online_ITI,
        grosmark_config=grosmark_config,
        mode="circular",
        significance=0.05,
    )

    plot_rZ_scores(
        bayesian_config=bayesian_config_post,
        grosmark_config=grosmark_config,
        mode="circular",
        significance=0.05,
    )

    plot_normalised_rZ_scores(
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

    plot_correlation_across_stages_trajectories(
        mode="linear",
        bayesian_config=bayesian_config_online_ITI,
        grosmark_config=grosmark_config,
    )
    plot_correlation_across_stages_trajectories(
        mode="circular",
        bayesian_config=bayesian_config_online_ITI,
        grosmark_config=grosmark_config,
    )
