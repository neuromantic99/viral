import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Literal

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.models import BayesianDecodingConfig
from viral.imaging_utils import shuffle_rows


def detect_candidate_events(
    population_vector: np.ndarray, config: BayesianDecodingConfig
) -> List[Tuple[int, int]]:
    """based on detect_candidate_spindles from James & Jana"""

    candidate_events: List[Tuple[int, int]] = list()
    peak_threshold = np.mean(population_vector) + config.peak_threshold * np.std(
        population_vector
    )
    edge_threshold = np.mean(population_vector) + config.edge_threshold * np.std(
        population_vector
    )

    in_event = False
    peak_exceeded = False
    peak_value = -np.inf

    for idx, value in enumerate(population_vector):

        # look for candidate event start
        if value > edge_threshold and not in_event:
            start_event = idx
            in_event = True

        # update peak amplitude
        if in_event and value > peak_value:
            peak_value = value

        # check if peak value exceeded peak threshold
        if in_event and peak_value >= peak_threshold:
            peak_exceeded = True

        # check for candidate event end
        if in_event and peak_exceeded and (value < edge_threshold):
            in_event = False
            peak_exceeded = False
            candidate_events.append((start_event, idx))
            peak_value = -np.inf
            continue

        # discard candidate events if edge threshold is never crossed
        if in_event and not peak_exceeded and value < edge_threshold:
            in_event = False
            peak_exceeded = False
            peak_value = -np.inf

    return candidate_events


def merge_close_events(
    candidate_events: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    events = sorted(candidate_events, key=lambda x: x[0])  # events sorted by start time
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
    return merged_events


def filter_candidate_events_by_duration(
    candidate_events: List[Tuple[int, int]], event_duration_thresholds: Tuple[int]
) -> List[Tuple[int, int]]:
    filtered_events = list()
    for start_idx, end_idx in candidate_events:
        duration = end_idx - start_idx + 1
        # TODO: just for debugging, remove later
        if duration < event_duration_thresholds[0]:
            print("Event too short:", duration)
        elif duration > event_duration_thresholds[1]:
            print("Event too long:", duration)
        if event_duration_thresholds[0] <= duration <= event_duration_thresholds[1]:
            filtered_events.append((start_idx, end_idx))
    return filtered_events


def additional_pc_check(
    filtered_events: List[Tuple[int, int]], ssp: np.ndarray
) -> List[Tuple[int, int]]:
    additionally_checked = list()
    for start_idx, end_idx in filtered_events:
        pcs_with_spikes = np.where(np.sum(ssp[:, start_idx:end_idx], axis=1) >= 1)[0]
        if len(pcs_with_spikes) >= 5:
            additionally_checked.append((start_idx, end_idx))
    return additionally_checked


def construct_xy_by_bin(
    posterior_probability_matrix: np.ndarray,
    mode: Literal["linear", "circular"],
    total_length: Optional[float] = None,
) -> np.ndarray:
    """
    We need to precompute a xy-by-bin grid for linear/circular weighted correlation.

    Essentially, this is a Python implementation of the section in
    https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/demo_LinearReplayAnalysis.m
    and https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/demo_CircularReplayAnalysis.m, respectively.
    """
    ## In matlab this is done for all unique event lengths and on a fixed 'maze length' and number of spatial bins (at least I think so, 100)
    ## x is time bin, y is spatial bin

    # h = histc(synthEvents.eventFiringRateMatrixID, unique(synthEvents.eventFiringRateMatrixID));
    # uBins = unique(h);

    # ySpatial = linspace(0, totalMazeLength,  synthEvents.params.nSpatialBins + 1);
    # ySpatial = ySpatial(1:synthEvents.params.nSpatialBins);
    # xyByBin = {};
    # for i = 1:length(uBins)
    #     x = repmat((1:uBins(i))', [1, 100]);
    #     y = repmat(ySpatial, [uBins(i), 1]);
    #     xyByBin{uBins(i)} = [reshape(x, [], 1), reshape(y, [], 1)];
    # end

    n_time, n_pos = posterior_probability_matrix.shape

    x = np.arange(1, n_time + 1)

    if mode == "linear":
        if total_length is None:
            raise ValueError("total_length must be provided for linear mode")
        # ySpatial = linspace(0, totalMazeLength,  synthEvents.params.nSpatialBins + 1);
        # ySpatial = ySpatial(1:synthEvents.params.nSpatialBins);
        y_spatial = np.linspace(0, total_length, n_pos + 1)[:-1]
    elif mode == "circular":
        # % for circular weighted correlatioon analysis the circular (spatial)
        # % variable must be encoded in radians:
        # ySpatial = linspace(0, 2*pi, synthEvents.params.nSpatialBins + 1);
        # ySpatial = ySpatial(1:synthEvents.params.nSpatialBins);
        y_spatial = np.linspace(0, 2 * np.pi, n_pos + 1)[:-1]
        # y_spatial = np.linspace(0, 2 * 3.14, n_pos + 1)[:-1] # just for checking against MATLAB (pi precision errrors!)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    xx, yy = np.meshgrid(x, y_spatial, indexing="ij")
    return np.column_stack([xx.ravel(order="F"), yy.ravel(order="F")])


def calculate_linear_weighted_correlation(
    posterior_probability_matrix: np.ndarray, xy: np.ndarray
) -> float:
    """
    "Where posj is the jth spatial bin, bini is the ith temporal (two frame) bin in the event,
    Prij is the Bayesian posterior probability for that spatial bin at that temporal bin,
    M is the total number of temporal bins and N is the total number of spatial bins."

    Essentially, this is a Python implementation of https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/calcWeightedLinearCorr.m
    """
    w = posterior_probability_matrix.flatten(
        order="F"
    )  # column-major flattening like matlab
    if np.isnan(w).any():
        raise ValueError("Encountered NaN in posterior probability matrix")

    w = w / np.sum(w)

    assert np.isclose(np.sum(w), 1.0), "Weights do not sum to 1!"

    # weighted means
    mxy = np.sum(xy * w[:, None], axis=0)

    # weighted covariance terms
    covxy = np.sum(w * (xy[:, 0] - mxy[0]) * (xy[:, 1] - mxy[1]))
    covxx = np.sum(w * (xy[:, 0] - mxy[0]) ** 2)
    covyy = np.sum(w * (xy[:, 1] - mxy[1]) ** 2)

    return covxy / np.sqrt(covxx * covyy)
    # TODO: check against Matlab!
    # TODO: for perfect diagonal?
    # TODO: for slightly off-diagonal: Python 0.9708761195036539 MATLAB 0.9709 -> AssertionError -> but fine? what tolerance?


def calculate_circular_weighted_correlation(
    posterior_probability_matrix: np.ndarray,
) -> float:
    """
    "Where posj is the jth spatial bin, bini is the ith temporal (two frame) bin in the event,
    Prij is the Bayesian posterior probability for that spatial bin at that temporal bin,
    M is the total number of temporal bins and N is the total number of spatial bins.
    However, this measurement only accounts for linear relationships and is therefore not sufficient
    to detect sequences of the circular run belt that may span the artificially defined belt 'edges'.
    Generally, the circo-linear correlation coefficient, rcl, between a circular variable a, and a linear variable x is defined as follows:"
    "Where corr, sin and cos denote the Pearson's (linear) correlation, sine and cosine operators, respectively.
    Therefore, the circo-linear-weighted correlation coefficient between time (the linear variable) and position (the circular variable)
    weighted by the posterior probability of position in each time bin was derived by combining equations (5)-(7) and (8)-(11) as follows:"

    Essentially, this is a Python implementation of https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/calcWeightedCircCorr.m
    """
    # xy = construct_xy_by_bin(posterior_probability_matrix, mode="circular")

    # rxs = calculate_linear_weighted_correlation(
    #     posterior_probability_matrix=posterior_probability_matrix,
    #     xy=np.column_stack((xy[:, 0], np.sin(xy[:, 1]))),
    # )
    # rxc = calculate_linear_weighted_correlation(
    #     posterior_probability_matrix=posterior_probability_matrix,
    #     xy=np.column_stack((xy[:, 0], np.cos(xy[:, 1]))),
    # )
    # rcs = calculate_linear_weighted_correlation(
    #     posterior_probability_matrix=posterior_probability_matrix,
    #     xy=np.column_stack((np.sin(xy[:, 1]), np.cos(xy[:, 1]))),
    # )

    # return np.sqrt((rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / (1 - rcs**2))

    xy = construct_xy_by_bin(posterior_probability_matrix, mode="circular")

    # w = posterior_probability_matrix.flatten(
    #     order="F"
    # )  # column-major flattening like matlab
    # if np.isnan(w).any():
    #     raise ValueError("Encountered NaN in posterior probability matrix")
    # DON'T: This is done in calculate_linear_weighted_correlation already!

    # rxs = calcWeightedLinearCorr([xy(:, 1), sin(xy(:, 2))], w);
    # rxc = calcWeightedLinearCorr([xy(:, 1), cos(xy(:, 2))], w);
    # rcs = calcWeightedLinearCorr([sin(xy(:, 2)),cos(xy(:, 2))], w);
    x = xy[:, 0]
    y = xy[:, 1]

    rxs = calculate_linear_weighted_correlation(
        posterior_probability_matrix, np.column_stack((x, np.sin(y)))
    )
    rxc = calculate_linear_weighted_correlation(
        posterior_probability_matrix, np.column_stack((x, np.cos(y)))
    )
    rcs = calculate_linear_weighted_correlation(
        posterior_probability_matrix, np.column_stack((np.sin(y), np.cos(y)))
    )

    return np.sqrt((rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / (1 - rcs**2))

    # TODO: check against Matlab
    # TODO: for perfect diagonal?
    # TODO: for slightly off-diagonal: Python 0.7583601138637951 MATLAB 0.7584 (set pi to 3.14 to prevent pi-precision errors) -> AssertionError -> but fine? what tolerance?


def check_significance(
    posterior_probability_matrix: np.ndarray,
    correlation: float,
    mode: Literal["linear", "circular"] = "circular",
    total_length: float | None = None,
    n_shuffles: int = 2000,
    significance: float = 0.05,
) -> Tuple[float, bool]:
    """
    "For each event, the observed weighted circo-linear correlation coefficients weightedr(circular), hereafter referred to as weighted-r,
    was compared to a distribution of 2,000 null weighted-r values either by z-scoring the observed value by the null values (rZ score) or
    as the empirical P value.
    While several shuffle approaches were used (Extended Data Fig. 6),
    the principal shuffle used in the main figures involved the random re-ordering (resampling without replacement)
    of the bins observed within a given event."
    "'timeBinPermutation': permutes (resamples without replacement)"

    Essentially, this is a Python implementation of https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/shufflePopulationEvents.m
    """
    if mode == "linear":
        shuffled_weighted_rs = list()
        xy = construct_xy_by_bin(
            posterior_probability_matrix, mode="linear", total_length=total_length
        )
        for i in range(n_shuffles):
            shuffled = shuffle_rows(posterior_probability_matrix)
            shuffled_weighted_rs.append(
                calculate_linear_weighted_correlation(shuffled, xy)
            )
    elif mode == "circular":
        shuffled_weighted_rs = list()
        for i in range(n_shuffles):
            shuffled = shuffle_rows(posterior_probability_matrix)
            shuffled_weighted_rs.append(
                calculate_circular_weighted_correlation(shuffled)
            )
    r_real = np.abs(correlation)
    # TODO: it isn't clear, should the raw pse activity or the ppm be shuffled?
    r_shuffled = np.abs(shuffled_weighted_rs)
    empirical_p = np.mean(r_shuffled >= r_real)
    rz_score = (r_real - np.mean(r_shuffled)) / np.std(r_shuffled)
    print(f"empirical p-value: {empirical_p:.2f}, rZ score: {rz_score:.2f}")
    return empirical_p, empirical_p < significance


def bin_for_classification(
    position_array: np.ndarray,
    bayesian_config: BayesianDecodingConfig,
) -> np.ndarray:
    bin_edges = np.arange(
        bayesian_config.start_spatial,
        bayesian_config.end_spatial + bayesian_config.bin_size_spatial,
        bayesian_config.bin_size_spatial,
    )

    bin_edges = np.arange(
        bayesian_config.start_spatial,
        bayesian_config.end_spatial + bayesian_config.bin_size_spatial,
        bayesian_config.bin_size_spatial,
    )

    bin_indices = np.digitize(position_array, bin_edges) - 1

    return np.clip(bin_indices, 0, len(bin_edges) - 2)


def test_construct_xy_by_bin_against_matlab() -> None:
    # matlab_xy = np.genfromtxt("xy_matlab.csv", delimiter=",")
    # python_xy = np.genfromtxt("xyByBin_linear.txt", delimiter=" ")
    matlab_xy = np.genfromtxt("xy_circular_matlab copy.csv", delimiter=",")
    python_xy = np.round(np.genfromtxt("xyByBin_circular.txt", delimiter=" "), 5)
    # https://uk.mathworks.com/help/matlab/ref/pi.html
    # TODO: figure out the pi precision needed
    assert np.all(np.isclose(matlab_xy, python_xy))


def create_dummy_ppm_perfect_diagonal() -> np.ndarray:
    ppm = np.zeros(shape=(10, 10))
    np.fill_diagonal(ppm, 0.5)
    return ppm


def create_dummy_ppm_more_complex() -> np.ndarray:
    ppm = np.zeros(shape=(10, 10))
    np.fill_diagonal(ppm, 0.5)
    ppm[3, 5] = 0.7
    return ppm
