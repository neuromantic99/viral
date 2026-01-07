import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Literal, cast
from scipy.ndimage import gaussian_filter1d
from scipy.io import savemat, loadmat
from skimage.transform import radon

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.models import BayesianDecodingConfig, RadonLUT, RadonReplayResult
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

        # discard candidate events if peak threshold is never crossed
        if in_event and not peak_exceeded and value < edge_threshold:
            in_event = False
            peak_exceeded = False
            peak_value = -np.inf

        # TODO: in an older commit, there was broken code that wanted to check whether both edges were crossed, do we want that now?

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
    xp is radial coordinate in radon transformation but never explicitly defined.
    This matches padding logic in skimage.transform.radon.
    """
    # xp = np.arange(padded_image.shape[0]) - center
    diagonal = np.sqrt(2) * max(image_shape)
    padded_size = int(np.ceil(diagonal))
    centre = padded_size // 2
    xp = np.arange(padded_size) - centre
    return xp


# TODO: TEST ALL THIS against MATLAB implementation!!!!!!
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

    # return RadonLUT(
    #     path_length=radon_transform,
    #     xp=xp,
    #     theta=theta,
    #     n_radon_points=n_radon_points,
    #     point1x=point1x,
    #     point1y=point1y,
    #     point2x=point2x,
    #     point2y=point2y,
    #     slope=slope,
    #     path_length_from_points=path_length_from_points,
    #     space_offset=space_offset,
    #     temp_offset=temp_offset,
    #     space_offset_round=space_offset_round,
    #     temp_offset_round=temp_offset_round,
    #     temp_offset_round_perc=temp_offset_round_perc,
    # )

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
    # 'tempOffsetRound', 'tempOffsetRoundPerc
    savemat("radon_lut_python.mat", {"radon_lut": radon_lut})
    # savemat(
    #     "radon_lut_python.mat",
    #     {
    #         "pathLength": radon_lut.path_length,
    #         "xp": radon_lut.xp,
    #         "theta": radon_lut.theta,
    #         "nRadonPoints": radon_lut.n_radon_points,
    #         "point1X": radon_lut.point1x,
    #         "point1Y": radon_lut.point1y,
    #         "point2X": radon_lut.point2x,
    #         "point2Y": radon_lut.point2y,
    #         "size": radon_lut.path_length.shape,
    #         "slope": radon_lut.slope,
    #         "pathLengthFromPoints": radon_lut.path_length_from_points,
    #         "spaceOffset": radon_lut.space_offset,
    #         "tempOffset": radon_lut.temp_offset,
    #         "spaceOffsetRound": radon_lut.space_offset_round,
    #         "tempOffsetRound": radon_lut.temp_offset_round,
    #         "tempOffsetRoundPerc": radon_lut.temp_offset_round_perc,
    #     },
    # )
    return radon_lut


def calculate_radon_replay(
    posterior_probability_matrix: np.ndarray, bayesian_config: BayesianDecodingConfig
) -> RadonReplayResult:
    """
    "To determine the precise trajectory content of each sequence, a modified 'line casting' or Radon transformation approach was employed.
    Briefly, for each event, the posterior probabilities were tiled twice by position to account for 'edge' spanning sequences and smoothed
    with a 5-cm Gaussian kernel across positions within each time bin. Subsequently, lines, restricted to those crossing all bins,
    were densely cast along this matrix and the mean of the posterior probability for each of these lines was calculated.
    The trajectory was defined as the casted line with the highest mean posterior probability value, and the sign of the slope
    of the trajectory line defined whether it was a forward or reverse sequence."

    Essentially, this is a Python implementation of https://github.com/losonczylab/Grosmark_NatNeuro_2021/blob/main/calcRadonReplay.m.
    """
    # TODO: careful: axes??!
    n_time, n_pos = posterior_probability_matrix.shape

    # TODO: check: is it really like matlab?
    tiled_posterior_probability_matrix = np.tile(posterior_probability_matrix.T, (2, 1))

    # TODO: careful axes?!
    sigma = 5 / bayesian_config.bin_size_spatial  # 5 cm
    smoothed_tiled_posterior_probability_matrix = gaussian_filter1d(
        input=tiled_posterior_probability_matrix, sigma=sigma, axis=0
    )

    # TODO: check if that works with our data
    min_spatial_disp = 0  # % minimum spatial displacement of valid lines (in bins)
    # TODO: this was set to 100 in MATLAB, but will that work?
    min_n_bin_perc = (
        100.0  # % minimum percentage of temporal bins that valid lines must cross
    )

    # TODO: is that right?
    radon_lut = create_radon_lut(n_spatial_bins=n_pos * 2, n_time_bins=n_time)

    # checked this against MATLAB, it works
    min_path_length = np.sqrt(
        (np.floor(n_time * (min_n_bin_perc / 100.0)) + 1) ** 2 + min_spatial_disp**2
    )

    good_lines = (
        (np.abs(radon_lut.space_offset) >= min_spatial_disp)
        & (np.abs(radon_lut.temp_offset_round_perc) >= min_n_bin_perc)
        & (radon_lut.path_length_from_points >= min_path_length)
    )

    theta = radon_lut.theta
    n_radon_points = radon_lut.n_radon_points

    radon_transform = radon(
        smoothed_tiled_posterior_probability_matrix, theta=theta, circle=False
    )
    assert radon_transform.shape[0] == n_radon_points

    radon_transform[~good_lines] = 0
    # savemat("python_radon_transform.mat", {"radon_transform": radon_transform})

    # normalise by path length
    radon_transform_mean = radon_transform / radon_lut.path_length

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

    # TODO: this seems off: check units
    bin_size_time_frames = (
        bayesian_config.bin_size_time_online
        if bayesian_config.online
        else bayesian_config.bin_size_time_offline
    )
    bin_size_time_seconds = bin_size_time_frames / 30  # imaging @30 fps
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
    r = bayesian_config.total_length * 2  # we tiled the posterior_probability matrix
    v = slope * r
    v = v * bin_size_time_seconds

    slope_metres_per_sec = v  # I am extremely confused, this looks correct in the plot but is really not what Grosmark does

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
    from scipy.io import savemat, loadmat

    bayesian_config = BayesianDecodingConfig(
        en_bloc=True,  #
        online=False,
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
    n_time_bins = 20
    ppm = create_dummy_ppm_perfect_diagonal(n_spatial_bins, n_time_bins)
    radon_lut = create_radon_lut(
        n_spatial_bins=n_spatial_bins * 2, n_time_bins=n_time_bins
    )
    savemat("ppm_python.mat", {"ppm": ppm})
    radon_replay = calculate_radon_replay(ppm, bayesian_config)
    savemat("radon_python.mat", {"radon": radon_replay})

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

    python_lut = loadmat("radon_lut_python.mat")["radon_lut"]
    # python_radon_lookup_table = {
    #     name: np.squeeze(python_lut[name][0, 0]) for name in python_lut.dtype.names
    # }
    matlab_lut = loadmat("matlabLUT.mat")["saveRadonLUT"]
    matlab_radonLookupTable = {
        name: np.squeeze(matlab_lut[name][0, 0]) for name in matlab_lut.dtype.names
    }

    # 'pathLength', 'xp', 'theta', 'nRadonPoints', 'point1X', 'point1Y', 'point2X', 'point2Y',
    # 'size', 'slope', 'pathLengthFromPoints', 'spaceOffset', 'tempOffset', 'spaceOffsetRound',
    # 'tempOffsetRound', 'tempOffsetRoundPerc

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    # assert np.all(
    #     np.isclose(
    #         python_lut["path_length"][0, 0], matlab_radonLookupTable["pathLength"]
    #     )
    # )  # pass

    # MATLAB's radon transform will use the some number of radon points as Python (when given as an argument) but will choose
    # different positions on the detector function, i.e., the xp arrays will never be close.
    assert python_lut["xp"][0, 0].shape[1] == matlab_radonLookupTable["xp"].shape[0]
    # assert np.all(
    #     np.isclose(python_lut["xp"][0, 0], matlab_radonLookupTable["xp"])
    # )  # pass

    assert np.all(
        np.isclose(python_lut["theta"][0, 0], matlab_radonLookupTable["theta"])  # pass
    )

    # This will always work as it is given by the Python radon transform.
    assert (
        python_lut["n_radon_points"][0, 0] == matlab_radonLookupTable["nRadonPoints"]
    )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    # assert np.all(
    #     np.isclose(
    #         python_lut["slope"][0, 0],
    #         matlab_radonLookupTable["slope"],
    #         equal_nan=True,
    #     )
    # )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    # assert np.all(
    #     np.isclose(
    #         python_lut["point1x"][0, 0],
    #         (matlab_radonLookupTable["point1X"]),
    #         equal_nan=True,
    #     )
    # )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    # assert np.all(
    #     np.isclose(
    #         python_lut["point1y"][0, 0],
    #         matlab_radonLookupTable["point1Y"],
    #         equal_nan=True,
    #     )
    # )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    # assert np.all(
    #     np.isclose(
    #         python_lut["point2x"][0, 0],
    #         matlab_radonLookupTable["point2X"],
    #         equal_nan=True,
    #     )
    # )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    # assert np.all(
    #     np.isclose(
    #         python_lut["point2y"][0, 0],
    #         matlab_radonLookupTable["point2Y"],
    #         equal_nan=True,
    #     )
    # )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    # assert np.all(
    #     np.isclose(
    #         python_lut["path_length_from_points"][0, 0],
    #         matlab_radonLookupTable["pathLengthFromPoints"],
    #         equal_nan=True,
    #     )
    # )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    # assert np.all(
    #     np.isclose(
    #         python_lut["space_offset"][0, 0],
    #         matlab_radonLookupTable["spaceOffset"],
    #         equal_nan=True,
    #     )
    # )  # pass

    # As the radon transform works differently in MATLAB, for some elements in this array it will never be close (see plot).
    # It does work fine if you use the Python radon transform for it instead.
    # assert np.all(
    #     np.isclose(
    #         python_lut["temp_offset"][0, 0],
    #         matlab_radonLookupTable["tempOffset"],
    #         equal_nan=True,
    #     )
    # )  # pass

    # TODO: atol or rtol and which level?
    # TODO: as it is rounded, perhaps atol of 1 unit?
    # (e.g. np.round(20.5) will equal tp 20 whereas MATLAB round(20.5) will equal to 21)
    # TODO: or, we could change our rounding logic if we want to match it exactly
    # TODO: this looks ok:
    # np.max(np.abs(np.nan_to_num(python_lut["space_offset_round"][0, 0] - matlab_radonLookupTable["spaceOffsetRound"])))
    # np.float64(1.0)
    # assert np.all(
    #     np.isclose(
    #         python_lut["space_offset_round"][0, 0],
    #         matlab_radonLookupTable["spaceOffsetRound"],
    #         equal_nan=True,
    #         atol=1,
    #     )
    # )  #

    # TODO: atol or rtol and which level?
    # TODO: as it is rounded, perhaps atol of 1 unit?
    # (e.g. np.round(20.5) will equal tp 20 whereas MATLAB round(20.5) will equal to 21)
    # TODO: or, we could change our rounding logic if we want to match it exactly
    # TODO: this looks ok
    # np.max(np.abs(np.nan_to_num(python_lut["temp_offset_round"][0, 0] - matlab_radonLookupTable["tempOffsetRound"])))
    # np.float64(1.0)
    # assert np.all(
    #     np.isclose(
    #         python_lut["temp_offset_round"][0, 0],
    #         matlab_radonLookupTable["tempOffsetRound"],
    #         equal_nan=True,
    #         atol=1,
    #     )
    # )  #

    # TODO: atol or rtol and which level?
    # TODO: this looks ok:
    # np.max(np.abs(np.nan_to_num(python_lut["temp_offset_round_perc"][0, 0] - matlab_radonLookupTable["tempOffsetRoundPerc"])))
    # np.float64(0.0)
    # assert np.all(
    #     np.isclose(
    #         python_lut["temp_offset_round_perc"][0, 0],
    #         matlab_radonLookupTable["tempOffsetRoundPerc"],
    #         equal_nan=True,
    #         atol=1e-4,
    #     )
    # )  #

    ### TODO: the radon lut Python vs MATLAB looks ok for now, but definitely do a second and final check together with the actual radon replay check!!

    # TODO: same radon transform issue as above, took the radon lut from Python and the second radon transform from Python as well
    matlab_radon = loadmat("matlabRadonReplay.mat")["saveRadonReplay"]
    matlab_radonReplay = {
        name: np.squeeze(matlab_radon[name][0, 0]) for name in matlab_radon.dtype.names
    }
    # 'posMean', 'id', 'slope', 'pathLength', 'pointsXYXY'
    assert np.isclose(matlab_radonReplay["posMean"], radon_replay.pos_mean)  # pass
    assert np.isclose(matlab_radonReplay["slope"], radon_replay.slope)  # pass
    assert np.isclose(
        matlab_radonReplay["pathLength"], radon_replay.path_length
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

    ### TODO:radon replay Python vs MATLAB looks ok for now, but clean up and test again before asking for PR review!!!


def compare_radon_functions() -> None:
    n_spatial_bins = 20
    n_temporal_bins = 20
    posteriors = create_dummy_ppm_perfect_diagonal(
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


if __name__ == "__main__":
    compare_radon_against_matlab()
