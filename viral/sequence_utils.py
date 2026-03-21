import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Literal, cast
from scipy.ndimage import gaussian_filter1d
from scipy.io import savemat, loadmat
from scipy.signal import correlate2d
from skimage.transform import radon

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.constants import SERVER_PATH
from viral.models import (
    BayesianDecodingConfig,
    BayesianDecodingResult,
    RadonLUT,
    RadonReplayResult,
)
from viral.utils import shuffle_rows


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
    prev_value = population_vector[0]
    for idx, value in enumerate(population_vector):

        # look for candidate event start (i.e. crossed the edge threshold between the previous and current value)
        if (prev_value <= edge_threshold) and (value > edge_threshold) and not in_event:
            start_event = idx
            in_event = True

        # update peak amplitude
        if in_event and value > peak_value:
            peak_value = value

        # check if peak value exceeded peak threshold
        if in_event and peak_value >= peak_threshold:
            peak_exceeded = True

        # check for candidate event end (i.e. crossed the edge threshold between the previous and current value)
        if (
            in_event
            and peak_exceeded
            and (prev_value >= edge_threshold)
            and (value < edge_threshold)
        ):
            in_event = False
            peak_exceeded = False
            candidate_events.append((start_event, idx))
            peak_value = -np.inf
            continue

        # discard candidate events if peak threshold is never crossed
        if in_event and not peak_exceeded and value < edge_threshold:
            in_event = False
            peak_exceeded = False
            peak_value = -np.inf

        prev_value = value
    return candidate_events


def merge_close_events(
    candidate_events: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    events = sorted(candidate_events, key=lambda x: x[0])  # events sorted by start time
    merged_events = [events[0]]  # need the first event as a starting point
    # start to compare the second event onwards
    for current_start, current_end in events[1:]:
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


def filter_candidate_events_by_inter_event_time(
    candidate_events: List[Tuple[int, int]], min_inter_event_time_frames: int
) -> List[Tuple[int, int]]:
    """
    "Putative PSEs were defined as epochs during which the z-scored population activity vector reached a peak of at least 3.5 s.d.
    above the mean with event-edges at 1 s.d. above the mean, with a minimum inter-event time of 0.2 s."

    This function returns all events which have a minimum inter-event time of no less than the given variable in frames.
    Short inter-event times will result in the following event being rejected.
    """
    events = sorted(candidate_events, key=lambda x: x[0])  # events sorted by start time
    filtered_events = [
        events[0]
    ]  # the first event cannot be invalid and is needed as a reference point
    # start to compare the second event onwards
    for idx, (current_start, current_end) in enumerate(events[1:]):
        # TODO: which event should be rejected then? Just the following, or both?
        # TODO: also, should we compare to the last filtered one, or the last one before filtering?
        # _, last_end = events[idx]
        _, last_end = filtered_events[-1]
        if (
            current_start - last_end < min_inter_event_time_frames
        ):  # events are too close together, skip
            continue
        else:
            # not too close, keep the event
            filtered_events.append((current_start, current_end))
    return filtered_events


def filter_candidate_events_by_duration(
    candidate_events: List[Tuple[int, int]], event_duration_thresholds: Tuple[int, int]
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
) -> Tuple[float, bool, float]:
    """
    "For each event, the observed weighted circo-linear correlation coefficients weightedr(circular), hereafter referred to as weighted-r,
    was compared to a distribution of 2,000 null weighted-r values either by z-scoring the observed value by the null values (rZ score) or
    as the empirical P value.
    While several shuffle approaches were used (Extended Data Fig. 6),
    the principal shuffle used in the main figures involved the random re-ordering (resampling without replacement)
    of the bins observed within a given event."
    "'timeBinPermutation': permutes (resamples without replacement)
    % randomly permute bins within each event
    "

    Essentially, this is a Python implementation of https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/shufflePopulationEvents.m
    Computes an empirical p-value to check for significance.

    Arguments:
        posterior_probability_matrix (np.ndarray):  The posterior probability matrix of the decoded event/chunk.
        correlation (float):                        The computed linear/circular correlation coefficient.
        mode (Literal):                             Whether to use linear or circular correlation (use the same mode as the correlation coefficient above!). Defaults to "circular".
        total_length (float):                       Total length of the corridor.
        n_shuffles (int):                           Number of shuffles to perform. Defaults to 2,000 shuffles.
        significance (float):                       Significance level. Defaults to p < 0.05.

    Returns:
        Tuple[float, bool, float]:                  Tuple of empirical p-value, significant (bool) and rZ score.

    """
    if mode == "linear":
        shuffled_weighted_rs = list()
        xy = construct_xy_by_bin(
            posterior_probability_matrix, mode="linear", total_length=total_length
        )
        for _ in range(n_shuffles):
            shuffled = shuffle_rows(posterior_probability_matrix)
            shuffled_weighted_rs.append(
                calculate_linear_weighted_correlation(shuffled, xy)
            )
    elif mode == "circular":
        shuffled_weighted_rs = list()
        for _ in range(n_shuffles):
            shuffled = shuffle_rows(posterior_probability_matrix)
            shuffled_weighted_rs.append(
                calculate_circular_weighted_correlation(shuffled)
            )
    r_real = np.abs(correlation)
    # TODO: it isn't clear, should the raw pse activity or the ppm be shuffled?
    r_shuffled = np.abs(shuffled_weighted_rs)
    empirical_p = np.mean(r_shuffled >= r_real)
    assert empirical_p >= 0
    rz_score = (r_real - np.mean(r_shuffled)) / np.std(r_shuffled)
    print(f"empirical p-value: {empirical_p:.2f}, rZ score: {rz_score:.2f}")
    # TODO: can probably think about not returning significance bool
    return empirical_p, empirical_p < significance, rz_score


def bin_for_classification(
    position_array: np.ndarray,
    bayesian_config: BayesianDecodingConfig,
) -> np.ndarray:
    bin_edges = np.arange(
        bayesian_config.start_spatial,
        bayesian_config.end_spatial + bayesian_config.bin_size_spatial,
        bayesian_config.bin_size_spatial,
    )

    bin_indices = np.digitize(position_array, bin_edges) - 1

    return np.clip(bin_indices, 0, len(bin_edges) - 2)


def pol2cart(rho, phi):
    """
    Based on https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    and pol2cart.m (MATLAB).
    """
    # x = r.*cos(th);
    # y = r.*sin(th);
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


# TODO: how could we confirm this is correct??
def compute_xp(image_shape: Tuple[int, int]):
    """
    xp is radial coordinate in radon transformation but never explicitly defined in the MATLAB code.
    This matches padding logic in skimage.transform.radon.
    """
    # xp = np.arange(padded_image.shape[0]) - center
    diagonal = np.sqrt(2) * max(image_shape)
    padded_size = int(np.ceil(diagonal))
    centre = padded_size // 2
    xp = np.arange(padded_size) - centre
    return xp


def create_radon_lut(n_spatial_bins: int, n_time_bins: int) -> RadonLUT:
    """
    Essentially a Python implementation of https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/makeRadonLookupTable.m.
    """

    # (spatial_bins, time_bins)
    template = np.ones((n_spatial_bins, n_time_bins), float)

    # Annoyingly, Grosmark changes theta used back and forth from degrees to radian and vice versa.
    # From my understanding
    # (1) all_slopes use radian
    # (2) radon transform uses degrees
    # (3) point computation uses radian
    # (4) the saved theta array uses degrees (which makes sense, because it is later only used to do a radon transform again)
    theta_degrees = np.arange(0, 179.5, 0.5)
    theta_radian = np.deg2rad(theta_degrees)
    all_slopes = 1 / np.tan(theta_radian)

    # In Grosmark's MATLAB implementation, they manually set the offsets for the radon transform to try.
    # As this would make it integral to have our own implementation of the radon transform and/or mess with the input to the radon function here,
    # I have decided to not match the MATLAB code exactly but to use the skimage radon implementation instead.
    # By default, MATLAB is enforcing an odd offset count. I.e., you radon trabsform here could have one more offset than in MATLAB.
    # radon_transform = radon(template, theta=theta, circle=False)
    radon_transform = radon(template, theta=theta_degrees, circle=False)
    # savemat("template_python.mat", {"template": template})
    # savemat("lut_radon_transform.mat", {"radon_transform": radon_transform})

    # xp = np.arange(-(n_offsets // 2), n_offsets // 2 + n_offsets % 2)
    xp = compute_xp(image_shape=template.shape)
    savemat("lut_radon_transform.mat", {"RO": radon_transform, "xp": xp})
    n_radon_points = radon_transform.shape[0]
    assert n_radon_points == len(xp)
    # TODO: in test: assert here not more than one offset more compared to matlab
    # TODO: perhaps test independently for some posterior probability matrices?

    # % centerX = floor((nTemporalBins(S) + 1)/2);
    # % centerY = floor((nSpatialBins + 1)/2);
    center_x = (n_time_bins + 1) // 2
    center_y = (n_spatial_bins + 1) // 2

    point1x = np.full(shape=(len(xp), len(theta_radian)), fill_value=np.nan)
    point1y = np.full(shape=(len(xp), len(theta_radian)), fill_value=np.nan)
    point2x = np.full(shape=(len(xp), len(theta_radian)), fill_value=np.nan)
    point2y = np.full(shape=(len(xp), len(theta_radian)), fill_value=np.nan)
    for b_idx, b in enumerate(xp):
        for t_idx, t in enumerate(theta_radian):
            # pi precision ladies and gentlemen!
            x, y = pol2cart(b, t)

            x2 = center_x + x
            y2 = center_y - y

            m = all_slopes[t_idx]
            intercept = y2 - m * x2

            # % calculate the 4 possible points which may intersect the edge
            # % of the rectangle defined by the event size:
            top_edge = n_spatial_bins + 0.5
            right_edge = n_time_bins + 0.5
            left_point = np.array([0.0, intercept])
            right_point = np.array([right_edge, right_edge * m + intercept])
            bottom_point = np.array([-intercept / m, 0.0])
            top_point = np.array([(top_edge - intercept) / m, top_edge])

            possible_points = np.vstack(
                [left_point, right_point, bottom_point, top_point]
            )

            # % then pick the two points which actually intersect the
            # % event-rectangle
            valid = np.isfinite(possible_points).all(axis=1)
            inside = (
                (possible_points[:, 0] >= 0)
                & (possible_points[:, 0] <= right_edge)
                & (possible_points[:, 1] >= 0)
                & (possible_points[:, 1] <= top_edge)
            )
            k = np.where(valid & inside)[0]

            if k.size >= 2:
                point1x[b_idx, t_idx] = possible_points[k[0], 0]
                point1y[b_idx, t_idx] = possible_points[k[0], 1]
                point2x[b_idx, t_idx] = possible_points[k[1], 0]
                point2y[b_idx, t_idx] = possible_points[k[1], 1]

    path_length_from_points = np.hypot(point2x - point1x, point2y - point1y)
    space_offset = point2y - point1y
    temp_offset = point2x - point1x
    space_offset_round = np.round(space_offset)
    temp_offset_round = np.round(temp_offset)
    temp_offset_round_perc = 100 * np.abs(np.floor(temp_offset)) / n_time_bins
    # temp_offset_round_perc = 100 * (abs(temp_offset) / n_time_bins)

    # repeat the slope vector for all possible intersections
    # out{nTemporalBins(S)}.slope = repmat(allSlopes(:)', [length(xp), 1]);
    slope = np.tile(all_slopes.reshape(1, -1), (len(xp), 1))

    # set all slopes to NaN for which the point1 x is NaN (i.e., set slopes to NaN which did not intersect the image properly)
    # out{nTemporalBins(S)}.slope(isnan(out{nTemporalBins(S)}.point1X)) = NaN;
    invalid = np.where(np.isnan(point1x))
    slope[invalid] = np.nan

    radon_lut = RadonLUT(
        path_length=radon_transform,
        xp=xp,
        theta=theta_degrees,
        n_radon_points=n_radon_points,
        point1x=point1x,
        point1y=point1y,
        point2x=point2x,
        point2y=point2y,
        slope=slope,
        path_length_from_points=path_length_from_points,
        space_offset=space_offset,
        temp_offset=temp_offset,
        space_offset_round=space_offset_round,
        temp_offset_round=temp_offset_round,
        temp_offset_round_perc=temp_offset_round_perc,
    )

    # 'pathLength', 'xp', 'theta', 'nRadonPoints', 'point1X', 'point1Y', 'point2X', 'point2Y',
    # 'size', 'slope', 'pathLengthFromPoints', 'spaceOffset', 'tempOffset', 'spaceOffsetRound',
    # 'tempOffsetRound', 'tempOffsetRoundPerc'
    savemat(
        "radon_lut_python.mat",
        {
            "pathLength": radon_lut.path_length,
            "xp": radon_lut.xp,
            "theta": radon_lut.theta,
            "nRadonPoints": radon_lut.n_radon_points,
            "point1X": radon_lut.point1x,
            "point1Y": radon_lut.point1y,
            "point2X": radon_lut.point2x,
            "point2Y": radon_lut.point2y,
            "size": radon_lut.path_length.shape,
            "slope": radon_lut.slope,
            "pathLengthFromPoints": radon_lut.path_length_from_points,
            "spaceOffset": radon_lut.space_offset,
            "tempOffset": radon_lut.temp_offset,
            "spaceOffsetRound": radon_lut.space_offset_round,
            "tempOffsetRound": radon_lut.temp_offset_round,
            "tempOffsetRoundPerc": radon_lut.temp_offset_round_perc,
        },
    )
    return radon_lut


def calculate_radon_replay(
    posterior_probability_matrix: np.ndarray,
    bayesian_config: BayesianDecodingConfig,
    incorporate_nearby_positions: bool = True,
    nearby_positions: float = 30,
    min_n_bin_perc: float = 100,
) -> RadonReplayResult:
    """
    "To determine the precise trajectory content of each sequence, a modified 'line casting' or Radon transformation approach was employed.
    Briefly, for each event, the posterior probabilities were tiled twice by position to account for 'edge' spanning sequences and smoothed
    with a 5-cm Gaussian kernel across positions within each time bin. Subsequently, lines, restricted to those crossing all bins,
    were densely cast along this matrix and the mean of the posterior probability for each of these lines was calculated.
    The trajectory was defined as the casted line with the highest mean posterior probability value, and the sign of the slope
    of the trajectory line defined whether it was a forward or reverse sequence." (Grosmark)

    Essentially, this is mostly a Python implementation of https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/calcRadonReplay.m.
    Saw a method to pick the best line depending on the max posterior probability of the line and a nearby band in Olafsdottir et al. 2016, https://www.nature.com/articles/nn.4291#Sec2.
    Unfortunately, the MATLAB code was only partly available and the approach is using an algorithm close to the Radon transform, but differing from Grosmark's approach.
    Denovellis et al. 2021, https://doi.org/10.7554/eLife.64505, used a Radon line-fitting algorithm close to Grosmark.
    In https://github.com/Eden-Kramer-Lab/replay_trajectory_paper/blob/master/src/standard_decoder.py they share code to max the posterior probability matrix
    along a line plus a band of nearby positions.
    Hence, I used their 'nearby positions' approach as an example to modify my Python implementation of Grosmark's MATLAB code where indicated.

    Arguments:
        posterior_probability_matrix (np.ndarray):      The Posterior probability matrix of the event.
        bayesian_config (BayesianDecodingConfig):       The BayesianDecodingConfig to use.
        incorporate_nearby_positions (bool):            Whether to use the "nearby positions"/"band" approach (Olafsdottir, Denovellis). Defaults to True.
        nearby_positions (float):                       If using the aforementioned "nearby positions"/"band" approach, give the y_range in cms. Defaults to 30 cm.
        min_n_bin_perc (float):                         Percentage of time bins that the best fit line has to cross. Defaults to 100 percent.
    """
    # TODO: careful: axes??!
    n_time, n_pos = posterior_probability_matrix.shape

    # TODO: check: is it really like matlab?
    tiled_posterior_probability_matrix = np.tile(posterior_probability_matrix.T, (2, 1))

    sigma = 5 / bayesian_config.bin_size_spatial  # 5 cm
    smoothed_tiled_posterior_probability_matrix = gaussian_filter1d(
        input=tiled_posterior_probability_matrix, sigma=sigma, axis=0
    )

    # TODO: check if that works with our data, and/or change the docstring of this function
    min_spatial_disp = 0  # % minimum spatial displacement of valid lines (in bins)
    # TODO: this was set to 100 in MATLAB, but will that work?
    # min_n_bin_perc = (
    #     100.0  # % minimum percentage of temporal bins that valid lines must cross
    # )
    # min_n_bin_perc = (
    #     15.0  # % minimum percentage of temporal bins that valid lines must cross
    # )

    # TODO: is that right? -> pretty sure this is correct, check
    # https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/demo_CircularReplayAnalysis.m line 107
    radon_lut = create_radon_lut(n_spatial_bins=n_pos * 2, n_time_bins=n_time)

    # checked this against MATLAB, it works
    # % minPLength = sqrt((floor(uNS(U).*(minNBinPerc/100)) + 1)^2 + minSpatialDisp^2);
    # CAREFUL: it has to be n_pos multiplied by 2 as in MATLAB this is done on the tiled posterior probability matrix
    min_path_length = np.sqrt(
        (np.floor(2 * n_pos * (min_n_bin_perc / 100.0)) + 1) ** 2 + min_spatial_disp**2
    )

    good_lines = (
        (np.abs(radon_lut.space_offset) >= min_spatial_disp)
        & (np.abs(radon_lut.temp_offset_round_perc) >= min_n_bin_perc)
        & (radon_lut.path_length_from_points >= min_path_length)
    )

    theta = radon_lut.theta
    n_radon_points = radon_lut.n_radon_points

    # using Denovellis code
    if incorporate_nearby_positions:
        n_nearby_bins = int(nearby_positions / 2 // bayesian_config.bin_size_spatial)
        kernel = np.ones(2 * n_nearby_bins + 1)
        # convolve along the spatial axis
        # TODO: Denovellis is convolving the untiled ppm, is that goign to make a difference if the axis is adjusted in our code?
        tiled_convolved_posterior_probability_matrix = np.apply_along_axis(
            lambda time_bin: np.convolve(time_bin, kernel, mode="same"),
            axis=0,
            arr=tiled_posterior_probability_matrix,
        )
        # end of Denovellis

        # TODO: is there something we are not thinking about that would make the Denovellis approach inviable somewhere downstream?
        radon_transform = radon(
            tiled_convolved_posterior_probability_matrix, theta=theta, circle=False
        )
        assert radon_transform.shape[0] == n_radon_points
        # normalise by path length like Grosmark but trying to account for the "band"
        # 'score = np.max(sinogram) / (n_time * n_nearby_bins)' from Denovellis
        # TODO: hence, I think Grosmark is picking the best line based on normalised Radon score along the path
        # TODO: whilst Denovellis is scoring them by the sum/max
        # TODO: what approach do we want to stay faithful to?
        # radon_transform_mean = radon_transform / (
        #     radon_lut.path_length * (2 * n_nearby_bins + 1)
        # )
        # TODO: stay faithful to Grosmark for now?
        radon_transform_mean = radon_transform / radon_lut.path_length

    else:
        radon_transform = radon(
            smoothed_tiled_posterior_probability_matrix, theta=theta, circle=False
        )
        assert radon_transform.shape[0] == n_radon_points
        # normalise by path length
        radon_transform_mean = radon_transform / radon_lut.path_length

    savemat("python_radon_transform.mat", {"radon_transform": radon_transform})

    radon_transform[~good_lines] = 0

    # TODO: wait, how is the MATLAB implementation dealing with this? not changing it!
    # No this won't work with np.max because it will take NaN as max
    # radon_transform_mean = np.nan_to_num(radon_transform_mean, nan=0)

    # find the line with the highest mean posterior probability
    # path lenghts are often, i.e. leading to zero divisions with NaNs as result, hence preventing them from being taken into consideration
    pos_mean_max = np.nanmax(radon_transform_mean)
    linear_index = np.nanargmax(radon_transform_mean)

    # convert linear index to 2D indices
    line_idx, theta_idx = np.unravel_index(linear_index, radon_transform.shape)

    # extract properties of the best line
    slope = radon_lut.slope[line_idx, theta_idx]
    path_length = radon_lut.path_length[line_idx, theta_idx]
    point1x = radon_lut.point1x[line_idx, theta_idx]
    point1y = radon_lut.point1y[line_idx, theta_idx]
    point2x = radon_lut.point2x[line_idx, theta_idx]
    point2y = radon_lut.point2y[line_idx, theta_idx]

    # ensure point1 comes before point2 along the x-axis
    if point2x < point1x:
        point1x, point2x = point2x, point1x
        point1y, point2y = point2y, point1y

    # TODO: this seems off: check units
    bin_size_time_frames = (
        bayesian_config.bin_size_time_online
        if bayesian_config.epoch == "online"
        else bayesian_config.bin_size_time_offline
    )
    # bin_size_time_seconds = bin_size_time_frames / 30  # imaging @30 fps
    # CircReplayOutput.Radon.slope = CircReplayOutput.Radon.slope...
    # *(totalMazeLength/synthEvents.params.nSpatialBins)/synthEvents.params.eventBinDuration;
    # slope_metres_per_sec = (
    #     slope * (bayesian_config.total_length / n_pos) / (bin_size_time_seconds)
    # )
    # n_spatial_bins, n_time_bins = smoothed_tiled_posterior_probability_matrix.shape
    # slope_metres_per_sec = slope * (
    #     (bayesian_config.total_length * 2 / n_spatial_bins) / (bin_size_time_seconds)
    # )  # might be correct as I tile the ppm to span two lengths of the track, right?

    # slope is done using radian -> i.e., slope is in radian/time bin
    # angular slope to linear speed -> v = omega * r
    # omega = slope
    # r = maze radius
    # r = bayesian_config.total_length * 2  # we tiled the posterior_probability matrix
    # v = slope * r
    # v = v * bin_size_time_seconds

    # slope_metres_per_sec = v  # I am extremely confused, this looks correct in the plot but is really not what Grosmark does

    # convert slope in spatial bins/temporal bins to metres/seconds
    metres_per_spatial_bin = bayesian_config.total_length / n_pos
    seconds_per_temporal_bin = bin_size_time_frames / 30  # imaging @30 fps
    slope_metres_per_sec = slope * metres_per_spatial_bin / seconds_per_temporal_bin

    replay_type = cast(
        Literal["forward", "reverse"], "forward" if slope >= 0 else "reverse"
    )

    return RadonReplayResult(
        pos_mean=pos_mean_max,
        path_length=path_length,
        point1x=point1x,
        point1y=point1y,
        point2x=point2x,
        point2y=point2y,
        slope=slope,
        slope_metres_per_sec=slope_metres_per_sec,
        replay_type=replay_type,
    )


def test_construct_xy_by_bin_against_matlab() -> None:
    # matlab_xy = np.genfromtxt("xy_matlab.csv", delimiter=",")
    # python_xy = np.genfromtxt("xyByBin_linear.txt", delimiter=" ")
    matlab_xy = np.genfromtxt("xy_circular_matlab copy.csv", delimiter=",")
    python_xy = np.round(np.genfromtxt("xyByBin_circular.txt", delimiter=" "), 5)
    # https://uk.mathworks.com/help/matlab/ref/pi.html
    # TODO: figure out the pi precision needed
    assert np.all(np.isclose(matlab_xy, python_xy))


def create_dummy_ppm_perfect_diagonal(
    n_spatial_bins: int, n_time_bins: int
) -> np.ndarray:
    # (n_time_bins, n_spatial_bins)
    ppm = np.zeros(shape=(n_time_bins, n_spatial_bins))
    np.fill_diagonal(ppm, 0.5)
    return ppm


def create_dummy_ppm_more_complex(n_spatial_bins: int, n_time_bins: int) -> np.ndarray:
    # (n_time_bins, n_spatial_bins)
    ppm = np.zeros(shape=(n_time_bins, n_spatial_bins))
    np.fill_diagonal(ppm, 0.5)
    ppm[3, 5] = 0.7
    return ppm


def compare_radon_against_matlab() -> None:
    bayesian_config = BayesianDecodingConfig(
        en_bloc=True,  #
        epoch="pre",
        sigma_offline=1,  #
        sigma_online=1,  #
        peak_threshold=3.5,  #
        edge_threshold=1,  #
        event_duration=(6, 120),  #
        bin_size_time_offline=2,
        bin_size_time_online=2,
        start_spatial=0,  #
        end_spatial=100,  #
        bin_size_spatial=5,
    )
    n_spatial_bins = 20
    n_time_bins = 30
    ppm = create_dummy_ppm_perfect_diagonal(n_spatial_bins, n_time_bins)

    ### compare makeRadonLookupTable
    create_radon_lut(n_spatial_bins=n_spatial_bins * 2, n_time_bins=n_time_bins)
    savemat("ppm_python.mat", {"ppm": ppm})

    ### OUTDATED
    ## TESTING RADON TRANSFORM AGAINST MATLAB
    # even after giving the same number of radon points, radon won't generate the same result
    # radTrans = RO, xp
    # RO equals path_lenght!
    # radon_transform_python = loadmat("lut_radon_transform.mat")["radon_transform"]
    # radon_transform_matlab = loadmat("radTrans.mat")["radTrans"]
    # assert np.all(np.isclose(radon_lut.path_length, radon_transform_matlab))
    # # theta is equal! (see below)
    # template_python = loadmat("template_python.mat")["template"]
    # template_matlab = loadmat("templateMATLAB.mat")["template"]
    # assert np.array_equal(template_python, template_matlab)
    # # templates are equal!
    # assert np.all(np.isclose(radon_transform_python, radon_transform_matlab))

    # WARNING: As inputs were equal but outputs differed, probably inconsistency between MATLAB's internal C compiled radon transformation and Skimage's implementation
    # Hence using the Python result downstream in the test

    python_lut = loadmat("radon_lut_python.mat")
    matlab_lut = loadmat("matlabLUT.mat")["saveRadonLUT"]
    matlab_radonLookupTable = {
        name: np.squeeze(matlab_lut[name][0, 0]) for name in matlab_lut.dtype.names
    }

    # 'pathLength', 'xp', 'theta', 'nRadonPoints', 'point1X', 'point1Y', 'point2X', 'point2Y',
    # 'size', 'slope', 'pathLengthFromPoints', 'spaceOffset', 'tempOffset', 'spaceOffsetRound',
    # 'tempOffsetRound', 'tempOffsetRoundPerc

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    assert np.all(
        np.isclose(
            python_lut["pathLength"],
            matlab_radonLookupTable["pathLength"],
            # rtol=rtol,
            # atol=atol,
        )
    )  # pass

    # MATLAB's radon transform will use the some number of radon points as Python (when given as an argument) but will choose
    # different positions on the detector function, i.e., the xp arrays will never be close.
    # assert python_lut["xp"][0, 0].shape[1] == matlab_radonLookupTable["xp"].shape[0]
    assert python_lut["xp"].shape[1] == matlab_radonLookupTable["xp"].shape[0]
    assert np.all(
        np.isclose(
            python_lut["xp"],
            matlab_radonLookupTable["xp"],
        )
    )  # pass

    assert np.all(
        np.isclose(
            python_lut["theta"],
            matlab_radonLookupTable["theta"],
        )  # pass
    )

    # This will always work as it is given by the Python radon transform.
    assert (
        python_lut["nRadonPoints"][0, 0] == matlab_radonLookupTable["nRadonPoints"]
    )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    assert np.all(
        np.isclose(
            python_lut["slope"],
            matlab_radonLookupTable["slope"],
            equal_nan=True,
        )
    )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    assert np.all(
        np.isclose(
            python_lut["point1X"],
            matlab_radonLookupTable["point1X"],
            equal_nan=True,
        )
    )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    assert np.all(
        np.isclose(
            python_lut["point1Y"],
            matlab_radonLookupTable["point1Y"],
            equal_nan=True,
        )
    )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    assert np.all(
        np.isclose(
            python_lut["point2X"],
            matlab_radonLookupTable["point2X"],
            equal_nan=True,
        )
    )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    assert np.all(
        np.isclose(
            python_lut["point2Y"],
            matlab_radonLookupTable["point2Y"],
            equal_nan=True,
        )
    )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    assert np.all(
        np.isclose(
            python_lut["pathLengthFromPoints"],
            matlab_radonLookupTable["pathLengthFromPoints"],
            equal_nan=True,
        )
    )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    assert np.all(
        np.isclose(
            python_lut["spaceOffset"],
            matlab_radonLookupTable["spaceOffset"],
            equal_nan=True,
            # rtol=rtol,
            # atol=atol,
        )
    )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    assert np.all(
        np.isclose(
            python_lut["tempOffset"],
            matlab_radonLookupTable["tempOffset"],
            equal_nan=True,
            # rtol=rtol,
            # atol=atol,
        )
    )  # pass

    # TODO: atol or rtol and which level?
    # TODO: as it is rounded, perhaps atol of 1 unit?
    # (e.g. np.round(20.5) will equal tp 20 whereas MATLAB round(20.5) will equal to 21)
    # TODO: or, we could change our rounding logic if we want to match it exactly
    # TODO: this looks ok:
    # np.max(np.abs(np.nan_to_num(python_lut["space_offset_round"][0, 0] - matlab_radonLookupTable["spaceOffsetRound"])))
    # np.float64(1.0)
    assert np.all(
        np.isclose(
            python_lut["spaceOffsetRound"],
            matlab_radonLookupTable["spaceOffsetRound"],
            equal_nan=True,
            atol=1,
        )
    )  # pass

    # TODO: atol or rtol and which level?
    # TODO: as it is rounded, perhaps atol of 1 unit?
    # (e.g. np.round(20.5) will equal 20 whereas MATLAB round(20.5) will equal 21)
    # TODO: or, we could change our rounding logic if we want to match it exactly
    # TODO: this looks ok
    # np.max(np.abs(np.nan_to_num(python_lut["temp_offset_round"][0, 0] - matlab_radonLookupTable["tempOffsetRound"])))
    # np.float64(1.0)
    assert np.all(
        np.isclose(
            python_lut["tempOffsetRound"],
            matlab_radonLookupTable["tempOffsetRound"],
            equal_nan=True,
            atol=1,
        )
    )  # pass

    # TODO: atol or rtol and which level?
    # TODO: this looks ok:
    # np.max(np.abs(np.nan_to_num(python_lut["temp_offset_round_perc"][0, 0] - matlab_radonLookupTable["tempOffsetRoundPerc"])))
    # np.float64(0.0)
    assert np.all(
        np.isclose(
            python_lut["tempOffsetRoundPerc"],
            matlab_radonLookupTable["tempOffsetRoundPerc"],
            equal_nan=True,
            atol=1e-4,
        )
    )  # pass

    ## TODO: the radon lut Python vs MATLAB looks ok for now, but definitely do a second and final check together with the actual radon replay check!!
    # TODO: did another check using the prepared MATLAB scripts, looks ok too

    ### compare calcRadonReplay
    radon_replay = calculate_radon_replay(ppm, bayesian_config)
    savemat("radon_python.mat", {"radon": radon_replay})

    # TODO: same radon transform issue as above, took the radon lut from Python and the second radon transform from Python as well
    matlab_radon = loadmat("matlabRadonReplay.mat")["saveRadonReplay"]
    matlab_radonReplay = {
        name: np.squeeze(matlab_radon[name][0, 0]) for name in matlab_radon.dtype.names
    }
    # 'posMean', 'id', 'slope', 'pathLength', 'pointsXYXY'
    assert np.isclose(
        matlab_radonReplay["posMean"],
        radon_replay.pos_mean,
    )  # pass
    assert np.isclose(
        matlab_radonReplay["slope"],
        radon_replay.slope,
    )  # pass
    assert np.isclose(
        matlab_radonReplay["pathLength"],
        radon_replay.path_length,
    )  # pass
    assert np.all(
        np.isclose(
            matlab_radonReplay["pointsXYXY"],
            np.array(
                [
                    radon_replay.point1x,
                    radon_replay.point1y,
                    radon_replay.point2x,
                    radon_replay.point2y,
                ]
            ),
        )
    )  # pass

    ## TODO:radon replay Python vs MATLAB looks ok for now, but clean up and test again before asking for PR review!!!
    # TODO: did another check using the prepared MATLAB scripts, looks ok too


def compare_radon_functions() -> None:
    n_spatial_bins = 10
    n_temporal_bins = 30
    # posteriors = create_dummy_ppm_perfect_diagonal(
    #     n_spatial_bins=n_spatial_bins, n_time_bins=n_temporal_bins
    # )
    posteriors = create_dummy_ppm_more_complex(
        n_spatial_bins=n_spatial_bins, n_time_bins=n_temporal_bins
    )
    savemat("posteriors.mat", {"posteriors": posteriors})
    theta = np.deg2rad(np.arange(0, 179.5, 0.5))
    print(theta)
    radon_transform = radon(posteriors, theta, circle=False)
    radon_transform_circle = radon(posteriors, theta, circle=True)
    savemat("radon_transform.mat", {"radon_transform": radon_transform})
    savemat(
        "radon_transform_circle.mat", {"radon_transform_circle": radon_transform_circle}
    )


def compare_pol2cart_against_matlab() -> None:
    """
    Checked our Python implementation of MATLAB's pol2cart:
    https://github.com/josefbitzenhofer/Grosmark_match_MATLAB/blob/main/pythonPol2Cart.m.
    Used example from Mathwork: https://uk.mathworks.com/help/releases/R2025b/matlab/ref/pol2cart.html.
    """
    theta = [0, np.pi / 4, np.pi / 2, np.pi]
    rho = [
        5,
        5,
        10,
        10,
    ]
    x, y = pol2cart(rho=rho, phi=theta)

    matlab_pol2cart = loadmat("pol2cart.mat")
    matlab_x = matlab_pol2cart["x"].squeeze()
    matlab_y = matlab_pol2cart["y"].squeeze()

    assert np.array_equal(x, matlab_x)
    assert np.array_equal(y, matlab_y)
    print("Checked")


def get_cache_path(
    mouse_name: str, date: str, bayesian_config: BayesianDecodingConfig
) -> Path:
    return (
        SERVER_PATH
        / "viral_caches"
        / "sequence_detection"
        / "bayesian"
        / f"{mouse_name}_{date}_{bayesian_config.epoch}_bin_size_spatial-{bayesian_config.bin_size_spatial}.npz"
    )


def save_bayesian_cache(cache_path: Path, result: BayesianDecodingResult) -> None:
    """Save Bayesian decoding result to cache."""
    data = result.model_dump()
    np.savez(
        cache_path,
        data=np.array(data, dtype=object),
    )


def load_bayesian_cache(cache_path: Path) -> BayesianDecodingResult:
    """Load Bayesian decoding result from cached file."""
    loaded = np.load(cache_path, allow_pickle=True)
    data = loaded["data"].item()
    return BayesianDecodingResult(**data)


if __name__ == "__main__":
    # compare_radon_against_matlab()
    compare_pol2cart_against_matlab()
