from itertools import product
from typing import Any, Dict, List, Tuple, Literal
import numpy as np
import sys
import concurrent.futures
import os
import time
from pathlib import Path
from matplotlib import pyplot as plt
from deprecated import deprecated
from scipy.linalg import fractional_matrix_power, subspace_angles
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon, zscore, rankdata
from opt_einsum import contract
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from itertools import product

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.gsheets_importer import gsheet2df
from viral.constants import (
    CACHE_PATH,
    LOCAL_DFF_PATH,
    SERVER_PATH,
    SPREADSHEET_ID,
    TIFF_UMBRELLA,
    grosmark_config,
)
from viral.models import (
    Cached2pSession,
    EnsembleSessionResult,
    GrosmarkConfig,
    ReactivationSummary,
    SortedPlaceCells,
    TrialInfo,
    SSPVectorData,
)
from viral.sessions_keep import SESSIONS_KEEP
from viral.rastermap_utils import (
    align_validate_data,
    process_trials_data,
    filter_speed_position,
)
from viral.utils import (
    above_threshold_for_n_consecutive_samples,
    circularly_permute_rows,
    get_movement_bool,
    below_threshold_for_n_consecutive_samples,
    degrees_to_cm,
    get_genotype,
    get_wheel_circumference_from_rig,
    permute_row_order,
    shaded_line_plot,
    shuffle_rows,
    split_continuous_chunks,
    threshold_detect,
    threshold_detect_continuous,
    threshold_detect_edges,
)
from viral.imaging_utils import (
    compute_speed_grosmark,
    load_imaging_data,
    restrict_to_immobility,
    split_fluoresence_online_freeze,
    trial_is_imaged,
)
from viral.grosmark_analysis import get_place_cells
from viral.multiple_sessions import get_iti_still_frames, iti_masks_by_trial_type

# 150 ms Gaussian kernel at 30 Hz, as in Grosmark's offline activity matrix Z
OFFLINE_SIGMA = 150 / 1000 * 30


def process_behaviour(
    session: Cached2pSession,
    wheel_circumference: float,
    spks: np.ndarray,
    speed_bin_size: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    trials = [trial for trial in session.trials if trial_is_imaged(trial)]
    aligned_trial_frames, neural_data_trials = align_validate_data(spks, trials)
    imaged_trials_infos = process_trials_data(
        trials,
        aligned_trial_frames,
        neural_data_trials,
        wheel_circumference,
        speed_bin_size,
    )
    corridor_widths = np.array([trial.corridor_width for trial in imaged_trials_infos])
    positions = np.vstack([trial.frames_positions for trial in imaged_trials_infos])
    speed = np.vstack([trial.frames_speed for trial in imaged_trials_infos])
    neural_data = np.concatenate(
        [trial.signal for trial in imaged_trials_infos], axis=1
    )
    assert len(corridor_widths) == len(
        aligned_trial_frames
    ), "Number of corridor widths and aligned_trial_frames do not match"
    assert positions.shape[0] == speed.shape[0] == neural_data.shape[1]
    # Checking that neither positions nor speed have values that exceed the trial boundaries
    assert np.min(positions[:, 0]) == aligned_trial_frames[0, 0]
    assert np.max(positions[:, 0]) == aligned_trial_frames[-1, 1]
    assert np.min(speed[:, 0]) == aligned_trial_frames[0, 0]
    assert np.max(speed[:, 0]) == aligned_trial_frames[-1, 1]
    valid_frames = set()
    for start, end, _ in aligned_trial_frames:
        valid_frames.update(np.arange(start, end + 1))
    assert np.all(np.isin(positions[:, 0], list(valid_frames)))
    assert np.all(np.isin(speed[:, 0], list(valid_frames)))
    return positions, speed, aligned_trial_frames


def get_running_bouts(
    place_cells: np.ndarray,
    speed: np.ndarray,
    frames_positions: np.ndarray,
    aligned_trial_frames: np.ndarray,
    config: GrosmarkConfig,
) -> np.ndarray:
    # TODO: think about the speed_threshold, arbitrarily set it to 5 cm/s
    trial_activities = list()
    for start, end, _ in aligned_trial_frames:
        trial_activity = place_cells[:, start : end + 1]
        trial_activities.append(trial_activity)
    place_cells_behaviour = np.concatenate(trial_activities, axis=1)
    behaviour_mask = filter_speed_position(
        speed=speed,
        frames_positions=frames_positions,
        speed_threshold=-100,  # cm/s
        position_threshold=(config.start, config.end),
        use_or=False,
    )
    return place_cells_behaviour[:, behaviour_mask]


@deprecated("Use fast_ica_sklearn instead, tested and its the same")
def fast_ica_significant_components(X: np.ndarray, n_components: int) -> np.ndarray:
    """
    Port of github.com/tortlab/Cell-Assembly-Detection/blob/master/fast_ica.m to python

    X: (n_cells, n_timepoints) z-scored
    n_components: int

    Returns:
    Significant components (n_cells, n_components)
    """
    # demean X, (does this make sense for the z-scored data?)
    X = X - np.mean(X, axis=1, keepdims=True)
    X1 = X.copy()

    # covariance matrix
    C = X @ X.T / X.shape[1]

    # eigenvalues are sorted from largest to smallest
    # but in the documentation it says they are not necessarily sorted
    # so make sure
    eigenvalues, eigenvectors = np.linalg.eig(C)
    sorted_order = np.argsort(eigenvalues)[::-1]
    # Take only siginificant eigenvectors / values
    eigenvectors = eigenvectors[:, sorted_order[:n_components]]
    eigenvalues = eigenvalues[sorted_order[:n_components]]
    D = np.diag(eigenvalues ** (-1 / 2))

    X = D @ eigenvectors.T @ X
    whitening_matrix = D @ eigenvectors.T
    dewhitening_matrix = eigenvectors @ np.linalg.inv(D)

    ## ICA ##
    # X.shape 0 and n_components are the same so bit weird

    B = np.random.normal(size=(X.shape[0], n_components))

    ortho = lambda x: x @ fractional_matrix_power(x.T @ x, -0.5)
    B = ortho(B)

    W = np.random.uniform(size=(B.T @ whitening_matrix).shape)

    N = X.shape[1]

    for _ in tqdm(range(500)):
        hyp_tan = np.tanh(X.T @ B)
        B = (
            X @ hyp_tan / N
            - np.ones((B.shape[0], 1))
            @ np.expand_dims(np.mean(1 - hyp_tan**2, axis=0), axis=0)
            * B
        )
        B = ortho(B)
        W = B.T @ whitening_matrix

    return W.T


def fast_ica_sklearn(X: np.ndarray, n_components: int) -> np.ndarray:
    from sklearn.decomposition import FastICA

    # Don't need to do this
    # X_centered = X - np.mean(X, axis=1, keepdims=True)

    X_centered = X

    # Transpose to (n_samples, n_features) for sklearn
    X_centered = X_centered.T

    ica = FastICA(
        n_components=n_components,
        whiten="unit-variance",
        fun="logcosh",  # logcosh with alpha = 1 is the same as tanh used in matlab
        fun_args={"alpha": 1.0},
        max_iter=500,
        random_state=0,  # Optional: for reproducibility
    )
    S = ica.fit_transform(X_centered)  # Shape: (n_timepoints, n_components)
    W = ica.components_  # Shape: (n_components, n_cells)

    return W.T


def n_significant_components_circular_shift(
    ssp_vectors_z: np.ndarray,
    eigenvalues: np.ndarray,
    n_shuffles: int = 200,
    percentile: float = 99,
) -> int:
    """Eigenvalue threshold from a null that respects the smoothing.

    Marcenko-Pastur assumes the columns of the activity matrix are independent samples.
    get_ssp_vectors convolves with a 1-second Gaussian before the ICA, so adjacent frames
    are heavily correlated and the effective sample count is roughly n_frames / 106
    rather than n_frames. With 150 cells and 9000 frames the nominal q is 60 but the
    effective q is about 0.6, below the regime where the formula is even defined, and the
    threshold lands at 1.28 when the true null maximum is near 4.9.

    Measured on synthetic data with the same smoothing: on activity with ZERO planted
    assemblies, MP declared 45 significant components. This null declared 0, and
    recovered 3, 5 and 8 exactly when that many were planted.

    Circularly shifting each cell independently preserves its firing rate and its
    temporal autocorrelation while destroying cross-cell timing, which is precisely the
    null for "are these cells co-active beyond chance". The threshold is a high
    percentile of the largest null eigenvalue, so it controls the family-wise error
    across components rather than testing each one separately.

    One thing this does NOT separate: a population-wide co-fluctuation from arousal or
    brain state produces a large leading eigenvalue, and circular shifts destroy that too,
    so it counts as signal here exactly as it does under MP. If the leading component has
    near-uniform positive weights across all cells, that is what you are looking at.

    Re-z-scoring after the shift is unnecessary - a circular shift only reindexes, so
    each row's mean and variance are unchanged.
    """
    n_cols = ssp_vectors_z.shape[1]
    null_max = [
        np.linalg.eigvalsh(
            (lambda shifted: shifted @ shifted.T / n_cols)(
                circularly_permute_rows(ssp_vectors_z)
            )
        ).max()
        for _ in range(n_shuffles)
    ]
    return int(np.sum(eigenvalues > np.percentile(null_max, percentile)))


def compute_ICA_components(
    ssp_vectors: np.ndarray,
    n_component_method: Literal[
        "marcenko_pastur", "circular_shift"
    ] = "marcenko_pastur",
    n_shuffles: int = 200,
    percentile: float = 99,
) -> np.ndarray:
    """ICA cell assemblies, with the number of components chosen by n_component_method.

    "marcenko_pastur" is what Grosmark and Lopes-dos-Santos specify and stays the default
    so existing results are reproducible. It is invalid on data smoothed with a 1-second
    kernel and over-declares badly - see n_significant_components_circular_shift, which
    is the alternative. Reporting both is stronger than either: if a conclusion holds at
    60 components and at 12, the component criterion is not carrying it.
    """
    # 'the elements of M (in our case σ2 = 1 due to z-score normalization), Ncolumns is the number of columns and Nrows the number of rows.'
    n_rows, n_cols = ssp_vectors.shape

    # 'spiking matrix' (Lopes-dos-Santos):
    # (neurons, time) ('each matrix entry denotes the number of spikes of a given neuron (rows) in a given time bin (columns)')
    # ssp_vectors is (n_place_cells, n_frames)

    # 'Next, the spike count of each neuron (i.e., each row of the matrix) is normalized by z-score transformation'
    ssp_vectors_z = zscore(ssp_vectors, axis=1)

    # 'in our case the covariance matrix is equal to the correlation matrix, and can be calculated as:
    # C = Z*Z.T / Ncolumns
    # where Z is the (z-scored) spike matrix, T the transpose operator, and Ncolumns is the number of time bins of Z.
    # Thus, the element at the i-th column and j-th row of C is the linear correlation between neurons i and j.'
    covariance_matrix = (ssp_vectors_z @ ssp_vectors_z.T) / n_cols

    # 'Since C is necessarily real and symmetric, it follows from the spectral theorem that it can be decomposed'
    # eigvalsh, not eig: the covariance matrix is symmetric, so this returns real, sorted
    # eigenvalues. np.linalg.eig returns them unsorted and can return a complex dtype,
    # which then makes the comparison against the threshold raise.
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)

    # 'where σ2 is the variance of the elements of M (in our case σ2 = 1 due to z-score normalization)'
    assert np.isclose(np.var(ssp_vectors_z), 1)

    if n_component_method == "marcenko_pastur":
        q = n_cols / n_rows

        # 'with q = Ncolumns/Nrows ≥ 1'
        assert q >= 1

        # 'λmax and λmin are the maximum and minimum bounds, respectively, and are calculated as:'
        lambda_max = (1 + np.sqrt(1 / q)) ** 2

        # 'Thus, if the rows of M are statistically independent, the probability of finding an eigenvalue outside these bounds is zero.
        #  In other words, the variance of the data in any axis cannot be larger than λmax when neurons are uncorrelated.
        #  Therefore, λmax can be used as a statistical threshold for detecting cell assembly activity'
        n_significant_components = int(np.sum(eigenvalues > lambda_max))
    elif n_component_method == "circular_shift":
        n_significant_components = n_significant_components_circular_shift(
            ssp_vectors_z,
            eigenvalues,
            n_shuffles=n_shuffles,
            percentile=percentile,
        )
    else:
        raise ValueError(f"Unknown n_component_method: {n_component_method}")

    if n_significant_components < 1:
        return np.zeros((ssp_vectors_z.shape[0], 0))

    return normalise_ensemble_matrix(
        fast_ica_sklearn(ssp_vectors_z, n_significant_components)
    )


def normalise_ensemble_matrix(ensemble_matrix: np.ndarray) -> np.ndarray:
    """Scale each ICA component to unit L2 norm, and fix its sign.

    sklearn's FastICA returns components_ carrying the scale of the whitening, which is
    arbitrary. Reactivation strength goes as the squared norm of the weight vector, so
    without this R is in per-session units and cannot be compared across sessions or
    genotypes. Within a session the pre/post contrast still cancels, since it uses the
    same w, which is why this does not invalidate a paired result - but it does make any
    pooled or between-group comparison of absolute strength meaningless, and it also
    means sorting components by "total strength" partly sorts by ICA scaling.

    Unit-norming is standard in this literature (Lopes-dos-Santos et al.,
    van de Ven et al.) and is what makes a shared y-axis across animals possible.

    The sign flip puts the largest-magnitude weight positive. R is quadratic so it is
    unaffected, but classify_and_sort_place_cells thresholds raw weights and would pick
    the wrong cells from an all-negative component.
    """
    norms = np.linalg.norm(ensemble_matrix, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    normalised = ensemble_matrix / norms

    if normalised.shape[1] == 0:
        return normalised

    largest = np.argmax(np.abs(normalised), axis=0)
    signs = np.sign(normalised[largest, np.arange(normalised.shape[1])])
    signs[signs == 0] = 1.0
    return normalised * signs


def rescale_strength_to_unit_norm(
    strength: np.ndarray, ensemble_matrix: np.ndarray
) -> np.ndarray:
    """Convert an R computed with unnormalised weights to its unit-norm equivalent.

    R_b = Z.T (w_b w_b.T - diag) Z is exactly quadratic in w_b, so dividing component b
    by the squared norm of its weight vector gives precisely the R that normalised
    weights would have produced. Cached sessions therefore do not need recomputing.

    Idempotent: applied to a strength computed from already-normalised weights the
    divisor is 1 and nothing changes, so old and new caches can share a code path.

    The shuffled moments rescale identically, because permute_row_order permutes the
    rows of w and so leaves every column norm untouched. That also means the
    shuffle-referenced z is already scale-invariant and needs no correction - only raw
    strength does.
    """
    assert strength.shape[0] == ensemble_matrix.shape[1], (
        f"strength has {strength.shape[0]} components but ensemble_matrix has "
        f"{ensemble_matrix.shape[1]}"
    )
    squared_norms = np.linalg.norm(ensemble_matrix, axis=0) ** 2
    squared_norms[squared_norms == 0] = 1.0
    return strength / squared_norms[:, np.newaxis]


def get_offline_activity_matrix(
    reactivation: np.ndarray, smooth: bool = True
) -> np.ndarray:
    """Offline reactivation was assessed from the 150-ms Gaussian kernel convolved offline activity matrix Z.

    Smooth first, then z-score. The order matters: convolution removes variance by a
    factor set by the cell's temporal autocorrelation, so z-scoring first leaves Z
    with a per-cell variance well below 1 that depends on how clustered in time that
    cell's events are. Measured on matched spike counts, isolated events retain 0.06
    of the variance while events in bursts of eight retain 0.44 - a sevenfold spread.
    Reactivation strength goes as Z squared, so cells or groups that differ in
    burstiness score differently for reasons unrelated to assembly structure, which
    matters here because burstiness is exactly the kind of thing that differs between
    genotypes. Smoothing first and normalising after removes that: Z has unit variance
    per cell regardless. Firing rate alone does not drive this - for temporally
    independent events the factor is constant across rates.

    This also matches the online path, where get_ssp_vectors applies the 1-s kernel
    and compute_ICA_components z-scores afterwards.

    Silent cells smooth to all-zero and z-score to NaN, so they are zeroed out.

    Pass smooth=False when the input has already been convolved on the continuous
    recording, as build_offline_matrix_from_mask does for a non-contiguous selection of
    frames. Smoothing a concatenation of separate ITIs would blend the end of one into
    the start of the next.
    """
    if smooth:
        reactivation = gaussian_filter1d(reactivation, sigma=OFFLINE_SIGMA, axis=1)
    return np.nan_to_num(zscore(reactivation, axis=1))


def offline_reactivation(
    reactivation: np.ndarray,
    ensemble_matrix: np.ndarray,
    do_shuffle: bool = False,
    presmoothed: bool = False,
) -> np.ndarray:
    """
    For each component, b, of ICA ensemble matrix w, a
    square projection matrix, P, was computed from wb as follows:
    Pb = wb * wbT
    Where T denotes the transpose operator. Subsequently, the diagonal of the
    projection matrix P was set to zero to exclude each cell's individual firing rate
    variance.
    Offline reactivation was assessed from the 150-ms Gaussian kernel convolved offline activity matrix Z.
    For the ith time point (frame) in Z, the reactivation strength Rb,i
    of the bth ICA component was calculated as the square of the projection length of Zi on Pb as follows:
    Rbi = ZiT * Pb * Zb
    """

    if do_shuffle:
        # """ICA components were shuffled by randomly permuting the weight matrix w across
        # PCs and recalculating the reactivation strength."""
        # Across PCs means across the rows of w, which are the cells. See
        # permute_row_order for why shuffle_rows is the wrong function here.
        ensemble_matrix = permute_row_order(ensemble_matrix)

    offline_activity_matrix = get_offline_activity_matrix(
        reactivation=reactivation, smooth=not presmoothed
    )

    n_timepoints = reactivation.shape[1]
    n_cells = ensemble_matrix.shape[0]
    n_components = ensemble_matrix.shape[1]

    # Einstein summation convention
    # components -> b
    # frames -> i
    # cells -> k
    # j (place holder)
    # P -> (components, cells, cells) -> P[b, j, k]
    # w -> (cells, components) -> w[b, j]
    # Z -> (cells, timepoints) -> Z[c, i]
    # R -> (components, timepoints) -> R[b, i]

    # outer product for each component b with itself
    # shape (components, cells, cells)
    projection_matrices = contract("kb,jb->bkj", ensemble_matrix, ensemble_matrix)
    assert projection_matrices.shape == (
        n_components,
        n_cells,
        n_cells,
    )

    # set diagonal to zero
    for b in range(projection_matrices.shape[0]):
        np.fill_diagonal(projection_matrices[b], 0)

    reactivation_strength = contract(
        "ik,bkj,ji->bi",
        offline_activity_matrix.T,  # Zi.T
        projection_matrices,  # Pb
        offline_activity_matrix,  # Zi
    )

    assert reactivation_strength.shape == (
        n_components,
        n_timepoints,
    )

    return reactivation_strength


class RunningMoments:
    """Welford accumulator for the mean and s.d. of the shuffled null.

    Holding all 500 shuffles to take a percentile costs
    n_shuffles x n_components x n_frames floats, which is well over a gigabyte for a
    normal session. We only need the first two moments, so accumulate them instead.
    """

    def __init__(self, shape: Tuple[int, ...]) -> None:
        self.n = 0
        self.mean = np.zeros(shape)
        self._m2 = np.zeros(shape)

    def update(self, x: np.ndarray) -> None:
        assert x.shape == self.mean.shape
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (x - self.mean)

    @property
    def std(self) -> np.ndarray:
        assert self.n > 1, "Need at least two shuffles for a standard deviation"
        return np.sqrt(self._m2 / (self.n - 1))


def reactivation_zscore(
    reactivation_strength: np.ndarray,
    shuffle_mean: np.ndarray,
    shuffle_std: np.ndarray,
) -> np.ndarray:
    """Express reactivation strength as a z against its own shuffled null.

    R = Z.T @ w @ w.T @ Z scales with the squared norm of the ICA weight vector, and
    sklearn does not return unit-norm components, so raw R is in arbitrary per-session
    units. Within a session the pre versus post contrast still cancels that, but no
    comparison across sessions or genotypes does. Dividing through by the null removes
    the scale entirely.

    The null is evaluated per component AND per timepoint, so it is matched to the
    instantaneous population activity: the shuffles reuse the real Z_i and only
    reassign which cell owns which ensemble weight. The z therefore asks whether the
    co-activation pattern matches the run ensemble beyond what the overall level of
    activity at that moment would produce, which also removes any global excitability
    difference between groups.
    """
    assert reactivation_strength.shape == shuffle_mean.shape == shuffle_std.shape

    # A component that is dead flat across every shuffle carries no information;
    # keep it finite rather than propagating infinities into the event detection.
    safe_std = np.where(shuffle_std > 0, shuffle_std, np.nan)
    return np.nan_to_num((reactivation_strength - shuffle_mean) / safe_std, nan=0.0)


def detect_reactivation_events(
    reactivation_z: np.ndarray,
    offline_spks: np.ndarray,
    peak_z: float = 3.0,
    edge_z: float = 1.0,
    min_frames: int = 2,
    min_participating_cells: int = 5,
) -> List[List[Tuple[int, int]]]:
    """Find reactivation events per component as excursions of the shuffle-referenced z.

    An event is a stretch above edge_z that reaches at least peak_z somewhere inside
    it. Thresholds are additive on a z, not multiplicative on a raw value: the old
    "greater than 1.5 times the shuffled threshold" rule inverts wherever that
    threshold is negative, which R frequently is once the projection diagonal is
    zeroed.

    Excursions are returned once each. above_threshold_events maps every peak back to
    its surrounding excursion, so a doubly-peaked event was previously counted twice,
    inflating counts as a function of how peaky the trace is, which differs between
    epochs and between groups.

    Events must also have at least min_participating_cells distinct place cells each
    firing at least one estimated spike inside them, which is Grosmark's own criterion
    for PSEs ("at least 5 distinct PCs each fired at least one estimated spike"). It is
    needed here because the z divides by the null's per-timepoint s.d., and at frames
    where almost nothing is active every permutation of w gives nearly the same R, so
    that s.d. collapses and a trivial fluctuation is scaled into a large z. Measured on
    synthetic data, 27% of z>3 events fell in the quietest 10% of frames against a 10%
    chance expectation. Requiring real population participation removes them.

    offline_spks is the binarised sparsified spike estimate (Ssp) for this epoch,
    shape (n_place_cells, n_timepoints) - the same array the z was computed from,
    before smoothing. Its cell count need not match the number of components.
    """
    assert peak_z > edge_z, "Peak threshold must be above the edge threshold"
    assert offline_spks.shape[1] == reactivation_z.shape[1], (
        f"offline_spks has {offline_spks.shape[1]} frames but reactivation_z has "
        f"{reactivation_z.shape[1]}; they must be the same epoch"
    )

    fired = offline_spks > 0

    events_per_component = []
    for component in reactivation_z:
        deduplicated = sorted(
            set(
                above_threshold_events(
                    component, upper_threshold=peak_z, lower_threshold=edge_z
                )
            )
        )
        events_per_component.append(
            [
                (on, off)
                for on, off in deduplicated
                if off - on >= min_frames
                and np.count_nonzero(fired[:, on:off].any(axis=1))
                >= min_participating_cells
            ]
        )

    return events_per_component


def summarise_reactivation(
    reactivation_z: np.ndarray,
    offline_spks: np.ndarray,
    peak_z: float = 3.0,
    edge_z: float = 1.0,
    fs: int = 30,
    min_participating_cells: int = 5,
) -> ReactivationSummary:
    """Reduce one offline epoch to a per-component event rate and event amplitude.

    Rate is the primary measure and amplitude the secondary one; see
    ReactivationSummary for why they are not collapsed into a single number.
    """
    events_per_component = detect_reactivation_events(
        reactivation_z,
        offline_spks=offline_spks,
        peak_z=peak_z,
        edge_z=edge_z,
        min_participating_cells=min_participating_cells,
    )

    # How many excursions the participation criterion removed, to make it visible
    # whether it is doing anything on real data.
    events_before_participation = detect_reactivation_events(
        reactivation_z,
        offline_spks=offline_spks,
        peak_z=peak_z,
        edge_z=edge_z,
        min_participating_cells=0,
    )
    n_events_rejected = np.array(
        [
            len(before) - len(after)
            for before, after in zip(events_before_participation, events_per_component)
        ]
    )

    immobility_seconds = reactivation_z.shape[1] / fs

    n_events = np.array([len(events) for events in events_per_component])

    mean_peak_z = np.array(
        [
            (
                np.mean([np.max(component[on:off]) for on, off in events])
                if events
                else np.nan
            )
            for component, events in zip(reactivation_z, events_per_component)
        ]
    )

    return ReactivationSummary(
        event_rate_hz=n_events / immobility_seconds,
        mean_peak_z=mean_peak_z,
        n_events=n_events,
        n_events_rejected=n_events_rejected,
        immobility_seconds=immobility_seconds,
    )


def build_offline_matrix_from_mask(
    place_cells: np.ndarray, frame_mask: np.ndarray
) -> np.ndarray:
    """Build the offline activity matrix Z from a non-contiguous selection of frames.

    Smooth across the continuous recording FIRST, then select. The ITI still-frames mask
    picks scattered frames out of the session - separate ITIs, with gaps inside them
    wherever the mouse moved - so smoothing the concatenation would blend the end of one
    ITI into the start of the next and manufacture co-activation across trial
    boundaries.

    Z-scoring happens after selection, over the retained frames only, so the
    normalisation reflects the offline epoch rather than the whole session.
    """
    assert (
        place_cells.shape[1] == frame_mask.size
    ), f"place_cells has {place_cells.shape[1]} frames, mask has {frame_mask.size}"
    smoothed = gaussian_filter1d(place_cells, sigma=OFFLINE_SIGMA, axis=1)
    return get_offline_activity_matrix(smoothed[:, frame_mask], smooth=False)


def build_run_ensembles(
    session: Cached2pSession,
    place_cells: np.ndarray,
    rewarded: bool | None,
    n_component_method: Literal[
        "marcenko_pastur", "circular_shift"
    ] = "marcenko_pastur",
) -> np.ndarray:
    """ICA ensembles from the running bouts of one trial type, on a FIXED cell set.

    place_cells must be the same array whatever the value of rewarded. Defining place
    cells separately per trial type would change which rows are in the offline matrix
    too, and the two templates would then be scored on different cells - which destroys
    the point of the comparison, since the whole appeal is that only the template
    differs.
    """
    trials = [
        trial
        for trial in session.trials
        if trial_is_imaged(trial)
        and (rewarded is None or trial.texture_rewarded == rewarded)
    ]
    if not trials:
        return np.zeros((place_cells.shape[0], 0))

    ssp_result = get_ssp_vectors(trials=trials, place_cells=place_cells)
    if ssp_result.ssp_vectors.size == 0:
        return np.zeros((place_cells.shape[0], 0))

    return compute_ICA_components(
        ssp_vectors=ssp_result.ssp_vectors,
        n_component_method=n_component_method,
    )


def offline_correlation_matrix(offline_z: np.ndarray) -> np.ndarray:
    """Pairwise correlation matrix of the offline activity matrix Z."""
    return offline_z @ offline_z.T / offline_z.shape[1]


def mean_reactivation_strength(
    correlation: np.ndarray, ensemble_matrix: np.ndarray
) -> np.ndarray:
    """Mean over time of R, computed exactly from the correlation matrix.

    R_b,i = Z_i.T (w_b w_b.T - diag) Z_i, and Z is z-scored, so averaging over time turns
    (1/T) sum_i Z_ki Z_ji into the correlation C_kj and the whole thing collapses to a
    bilinear form:

        mean_t R_b = sum_{k != j} w_kb w_jb C_kj = w_b.T C w_b - sum_k w_kb^2 C_kk

    There is no time dimension left. Verified exact to 3e-14 against evaluating R frame
    by frame and averaging, and it makes the shuffled null ~260x cheaper: the
    correlation matrix is built once and each permutation is then a quadratic form on an
    n_cells square matrix rather than a pass over every frame.

    C_kk is subtracted explicitly rather than assumed to be 1, because silent cells are
    zeroed by build_offline_matrix_from_mask and so have zero variance, not unit.
    """
    weighted = np.einsum("kb,kj,jb->b", ensemble_matrix, correlation, ensemble_matrix)
    self_terms = (ensemble_matrix**2 * np.diag(correlation)[:, np.newaxis]).sum(axis=0)
    return weighted - self_terms


def score_offline_reactivation(
    offline_z: np.ndarray,
    ensemble_matrix: np.ndarray,
    n_shuffles: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean reactivation strength per component, raw and against the shuffled null.

    Mean over every offline frame - no threshold, no event detection, no participation
    criterion. This is what Grosmark's Extended Data Fig 6a plots, and with only a few
    minutes of ITI stillness per session it is the only measure with enough data behind
    it: an event rate would rest on single-digit counts per component.

    The z here is the observed mean referenced to the distribution of SHUFFLED means,
    one number per component. That differs from the per-timepoint z used for event
    detection in main(), where normalising each frame against its own null matters
    because a moment of high population activity can cross a threshold by chance. For a
    mean over every frame that variation averages out, and referencing the summary
    statistic to its own null distribution is the more direct question: is this
    component reactivated more than a template built from the same weights on the wrong
    cells?

    Also returns the null's mean and s.d. per component. observed - null_mean is the
    quantity to use for BETWEEN-GROUP comparisons: it is the numerator of the z, so it
    is the same contrast, but without dividing by a term that shrinks as cells are
    added. Measured on a fixed assembly with only the number of other place cells
    varying, observed - null_mean held at 9.2-10.0 while the z ran from 15 to 122. If
    two genotypes differ in place cell yield, the z alone would show a difference that
    is entirely an artefact of that yield.

    offline_z must already be the smoothed, z-scored activity matrix, so pass the output
    of build_offline_matrix_from_mask.
    """
    correlation = offline_correlation_matrix(offline_z)
    observed = mean_reactivation_strength(correlation, ensemble_matrix)

    null = np.array(
        [
            mean_reactivation_strength(correlation, permute_row_order(ensemble_matrix))
            for _ in range(n_shuffles)
        ]
    )

    null_mean = null.mean(axis=0)
    null_std = null.std(axis=0, ddof=1)
    safe_std = np.where(null_std == 0, np.nan, null_std)
    return (
        observed,
        np.nan_to_num((observed - null_mean) / safe_std),
        null_mean,
        null_std,
    )


def iti_ensemble_reactivation(
    session: Cached2pSession,
    spks: np.ndarray,
    min_retained_seconds: float = 120.0,
    n_shuffles: int = 500,
    use_place_cell_cache: bool = True,
    n_component_method: Literal[
        "marcenko_pastur", "circular_shift"
    ] = "marcenko_pastur",
    min_subset_frames: int = 1800,
) -> pd.DataFrame:
    """Score ITI quiescence against run ensembles, for sessions with no wheel freeze.

    Substitutes for the pre/post freeze contrast by scoring the SAME offline frames
    against three templates: all running bouts, rewarded-texture bouts only, and
    unrewarded-texture bouts only. Because it is the same frames and the same cells,
    arousal, movement, SNR, brightness and time in session all cancel exactly - which
    the pre/post contrast cannot claim, since those epochs differ in every one of them.

    Place cells are defined once from all trials so the offline matrix is identical
    across templates. Only the ensemble identity varies.

    Sessions below min_retained_seconds are skipped and reported. Retention is dominated
    by movement - mice run through the ITI - so a substantial fraction of sessions
    yields too little to score.

    use_place_cell_cache reuses the cached shuffled place-field threshold, which is the
    expensive part of get_place_cells. Note the cache key encodes the mouse, date,
    rewarded flag and GrosmarkConfig but NOT the version of oasis_spikes.npy it was
    derived from, so re-running the deconvolution silently invalidates it while leaving
    the filename looking valid. Clear viral_caches/place_cells whenever you re-run
    OASIS, or pass False here.
    """
    mask, per_trial = get_iti_still_frames(session)
    retained_seconds = mask.sum() / 30

    if retained_seconds < min_retained_seconds:
        print(
            f"Skipping {session.mouse_name} {session.date}: {retained_seconds:.0f} s of "
            f"ITI stillness, below {min_retained_seconds:.0f} s"
        )
        return pd.DataFrame()

    # rewarded=None so the cell set is defined from all trials, and stays fixed across
    # the three templates scored below
    pcs_mask, _, _ = get_place_cells(
        session=session,
        spks=spks,
        rewarded=None,
        config=grosmark_config,
        plot=False,
        use_cache=use_place_cell_cache,
    )
    if pcs_mask is None:
        return pd.DataFrame()
    place_cells = spks[pcs_mask, :]

    # The offline frames are split by the trial they follow, as well as being scored as
    # a whole. Splitting the frames rather than the templates keeps every component,
    # and a difference between subsets cannot come from static online-offline coupling,
    # which is identical in both.
    frame_masks = {"all": mask}
    frame_masks.update(iti_masks_by_trial_type(per_trial, n_frames=mask.size))

    offline_matrices = {}
    for iti_after, subset_mask in frame_masks.items():
        if subset_mask.sum() < min_subset_frames:
            print(
                f"  skipping iti_after={iti_after}: {subset_mask.sum() / 30:.0f} s, "
                f"below {min_subset_frames / 30:.0f} s"
            )
            continue
        offline_matrices[iti_after] = build_offline_matrix_from_mask(
            place_cells, subset_mask
        )

    records = []
    for template, rewarded in (
        ("all", None),
        ("rewarded", True),
        ("unrewarded", False),
    ):
        ensemble_matrix = build_run_ensembles(
            session, place_cells, rewarded, n_component_method=n_component_method
        )
        if ensemble_matrix.shape[1] == 0:
            print(f"  no significant components for the {template} template")
            continue

        for iti_after, offline_z in offline_matrices.items():
            mean_r, mean_rz, null_mean, null_std = score_offline_reactivation(
                offline_z, ensemble_matrix, n_shuffles=n_shuffles
            )
            records.extend(
                _reactivation_records(
                    session=session,
                    template=template,
                    subset_column="iti_after",
                    subset_value=iti_after,
                    ensemble_matrix=ensemble_matrix,
                    n_component_method=n_component_method,
                    pcs_mask=pcs_mask,
                    mean_r=mean_r,
                    mean_rz=mean_rz,
                    null_mean=null_mean,
                    null_std=null_std,
                    subset_seconds=frame_masks[iti_after].sum() / 30,
                    retained_seconds=retained_seconds,
                    n_trials_scored=len(per_trial),
                )
            )

    return pd.DataFrame(records)


def freeze_immobility_masks(
    session: Cached2pSession, n_frames: int
) -> Dict[str, np.ndarray]:
    """Full-session masks for immobility during the pre and post freeze epochs.

    The pre epoch precedes the task, so whatever the run ensembles express there cannot
    be a consequence of that day's running. That is the one contrast in this dataset
    that separates reactivation from static coupling: cells correlated during running
    are correlated offline for anatomical and neuropil reasons, and no amount of
    shuffling the weights distinguishes that from experience-driven reinstatement.
    post > pre with the same ensembles does.

    Expect the effect to be largest where there is a new map to build. On a familiar
    track the pre epoch legitimately already contains it, which is why Grosmark used
    belts that were novel on days 1 and 4 and restricted his headline analysis to
    newly formed place cells.
    """
    if session.wheel_freeze is None:
        return {}

    movement_pre, movement_post = get_movement_bool(wheel_freeze=session.wheel_freeze)
    epochs = {
        "pre": (
            session.wheel_freeze.pre_training_start_frame,
            session.wheel_freeze.pre_training_end_frame,
            movement_pre,
        ),
        "post": (
            session.wheel_freeze.post_training_start_frame,
            session.wheel_freeze.post_training_end_frame,
            movement_post,
        ),
    }

    masks = {}
    for name, (start, end, movement) in epochs.items():
        assert end - start == movement.size, (
            f"{session.mouse_name} {session.date}: {name} epoch is {end - start} frames "
            f"but its movement vector is {movement.size}. get_movement_bool pads to "
            f"27000, so an epoch of another length will not line up."
        )
        assert end <= n_frames, (
            f"{session.mouse_name} {session.date}: {name} epoch ends at frame {end} but "
            f"the imaging data has {n_frames} frames"
        )
        mask = np.zeros(n_frames, dtype=bool)
        mask[start:end] = ~movement
        masks[name] = mask

    return masks


def freeze_ensemble_reactivation(
    session: Cached2pSession,
    spks: np.ndarray,
    min_epoch_seconds: float = 60.0,
    n_shuffles: int = 500,
    use_place_cell_cache: bool = True,
    n_component_method: Literal["marcenko_pastur", "circular_shift"] = "circular_shift",
) -> pd.DataFrame:
    """Score the pre and post freeze epochs against the run ensembles.

    Mirrors iti_ensemble_reactivation exactly - same templates, same cell set, same
    measure - but the offline frames are the frozen-wheel blocks rather than the ITIs,
    so the paired contrast is post versus pre rather than one ITI subset versus another.

    None of main()'s per-timepoint machinery is needed for a mean-R contrast, so this
    does not touch the ensemble caches and is not affected by their being stale.
    """
    masks = freeze_immobility_masks(session, n_frames=spks.shape[1])
    if not masks:
        print(f"Skipping {session.mouse_name} {session.date}: no wheel freeze")
        return pd.DataFrame()

    pcs_mask, _, _ = get_place_cells(
        session=session,
        spks=spks,
        rewarded=None,
        config=grosmark_config,
        plot=False,
        use_cache=use_place_cell_cache,
    )
    if pcs_mask is None:
        return pd.DataFrame()

    place_cells = spks[pcs_mask, :]

    offline_matrices = {}
    for epoch, epoch_mask in masks.items():
        if epoch_mask.sum() / 30 < min_epoch_seconds:
            print(
                f"  skipping {epoch}: {epoch_mask.sum() / 30:.0f} s of immobility, "
                f"below {min_epoch_seconds:.0f} s"
            )
            continue
        offline_matrices[epoch] = build_offline_matrix_from_mask(
            place_cells, epoch_mask
        )

    if len(offline_matrices) < 2:
        print(f"  {session.mouse_name} {session.date}: need both epochs, skipping")
        return pd.DataFrame()

    retained_seconds = sum(m.sum() for m in masks.values()) / 30
    n_trials_scored = sum(trial_is_imaged(trial) for trial in session.trials)

    records = []
    for template, rewarded in (
        ("all", None),
        ("rewarded", True),
        ("unrewarded", False),
    ):
        ensemble_matrix = build_run_ensembles(
            session, place_cells, rewarded, n_component_method=n_component_method
        )
        if ensemble_matrix.shape[1] == 0:
            print(f"  no significant components for the {template} template")
            continue

        for epoch, offline_z in offline_matrices.items():
            mean_r, mean_rz, null_mean, null_std = score_offline_reactivation(
                offline_z, ensemble_matrix, n_shuffles=n_shuffles
            )
            records.extend(
                _reactivation_records(
                    session=session,
                    template=template,
                    subset_column="epoch",
                    subset_value=epoch,
                    ensemble_matrix=ensemble_matrix,
                    n_component_method=n_component_method,
                    pcs_mask=pcs_mask,
                    mean_r=mean_r,
                    mean_rz=mean_rz,
                    null_mean=null_mean,
                    null_std=null_std,
                    subset_seconds=masks[epoch].sum() / 30,
                    retained_seconds=retained_seconds,
                    n_trials_scored=n_trials_scored,
                )
            )

    return pd.DataFrame(records)


def _permutation_p(values: np.ndarray) -> float:
    """One-sided p from flipping the sign of each mouse's value, enumerated exactly.

    The unit is the mouse. With n mice the smallest achievable p is 1 / 2**n, so at
    n=5 nothing can beat 0.031 however large the effect - worth knowing before reading
    a p-value near that floor as weak.
    """
    if len(values) < 2:
        return float("nan")
    null = np.array(
        [
            np.mean(values * np.array(signs))
            for signs in product([-1, 1], repeat=len(values))
        ]
    )
    return float(np.mean(null >= values.mean()))


def report_freeze_reactivation(
    df: pd.DataFrame, template: str = "all", value: str = "excess_r"
) -> pd.Series:
    """Summarise the pre versus post freeze contrast and return the per-mouse deltas.

    excess_r (observed minus the shuffled null) is the default rather than mean_rz,
    because the z divides by a term that shrinks as place cells are added and so is not
    comparable across sessions or genotypes that differ in cell yield.
    """
    d = df[df.template == template]
    if d.empty:
        print(f"No rows for template={template}")
        return pd.Series(dtype=float)

    n_sessions = d.groupby(["mouse", "date"]).ngroups
    print(
        f"Freeze reactivation: {n_sessions} sessions, {d.mouse.nunique()} mice "
        f"(template={template}, value={value})\n"
    )

    print(f"  {'epoch':>6} {'sessions':>9} {'median':>9} {'mice > 0':>10}")
    for epoch in ("pre", "post"):
        e = d[d.epoch == epoch]
        if e.empty:
            continue
        per_mouse = (
            e.groupby(["mouse", "date"])[value].median().groupby("mouse").median()
        )
        print(
            f"  {epoch:>6} {e.groupby(['mouse','date']).ngroups:>9} "
            f"{per_mouse.median():>9.3f} "
            f"{f'{int((per_mouse > 0).sum())}/{len(per_mouse)}':>10}"
        )

    wide = d.pivot_table(
        index=["mouse", "date"], columns="epoch", values=value, aggfunc="median"
    ).dropna()
    if wide.empty or not {"pre", "post"}.issubset(wide.columns):
        print("\n  no session has both epochs")
        return pd.Series(dtype=float)

    per_session = wide["post"] - wide["pre"]
    per_mouse = per_session.groupby("mouse").median()
    v = per_mouse.to_numpy()

    print(f"\n  post - pre, per mouse ({len(per_session)} paired sessions):")
    for mouse, delta in per_mouse.items():
        n = (per_session.index.get_level_values("mouse") == mouse).sum()
        print(f"    {mouse:>7} {delta:>+8.3f}   ({n} sessions)")
    print(
        f"    mean {v.mean():>+.3f}   {int((v > 0).sum())}/{len(v)} positive   "
        f"p = {_permutation_p(v):.4f}   (floor {1 / 2 ** len(v):.4f})"
    )

    # The epochs differ in how much immobility they retain, which is the same covariate
    # that had to be watched for the ITI subsets
    secs = d.pivot_table(
        index=["mouse", "date"],
        columns="epoch",
        values="subset_seconds",
        aggfunc="first",
    )
    print(
        f"\n  immobility (s): pre {secs['pre'].median():.0f}  "
        f"post {secs['post'].median():.0f}"
    )
    per_sess_meta = d.groupby(["mouse", "date"])[
        ["n_components", "n_place_cells"]
    ].first()
    print(
        f"  median n_components {per_sess_meta.n_components.median():.0f}   "
        f"median n_place_cells {per_sess_meta.n_place_cells.median():.0f}"
    )

    if "session_type" in d.columns:
        stage = d.session_type.str.replace(r" day \d+", "", regex=True)
        d = d.assign(stage=stage)
        st = (
            d.pivot_table(
                index=["mouse", "date", "stage"],
                columns="epoch",
                values=value,
                aggfunc="median",
            )
            .dropna()
            .reset_index()
        )
        st["delta"] = st["post"] - st["pre"]
        print(f"\n  by stage (the effect should be largest where the map is new):")
        print(
            st.groupby("stage")
            .agg(sessions=("delta", "size"), median_delta=("delta", "median"))
            .round(3)
            .to_string()
            .replace("\n", "\n    ")
        )

    return per_mouse


def _reactivation_records(
    session: Cached2pSession,
    template: str,
    subset_column: str,
    subset_value: str,
    ensemble_matrix: np.ndarray,
    n_component_method: str,
    pcs_mask: np.ndarray,
    mean_r: np.ndarray,
    mean_rz: np.ndarray,
    null_mean: np.ndarray,
    null_std: np.ndarray,
    subset_seconds: float,
    retained_seconds: float,
    n_trials_scored: int,
) -> List[dict]:
    """One record per component for a single template x frame-subset combination.

    subset_column names whichever way the offline frames were split - "iti_after" for
    the ITI analysis, "epoch" for the freeze analysis - so both share a format without
    pretending the two splits mean the same thing.
    """
    return [
        {
            "mouse": session.mouse_name,
            "date": session.date,
            "session_type": session.session_type,
            "genotype": get_genotype(session.mouse_name),
            "template": template,
            subset_column: subset_value,
            "component": component,
            "n_components": ensemble_matrix.shape[1],
            "n_component_method": n_component_method,
            "n_place_cells": int(np.sum(pcs_mask)),
            "mean_r": mean_r[component],
            "mean_rz": mean_rz[component],
            # observed - null_mean: the effect size to use across groups
            "excess_r": mean_r[component] - null_mean[component],
            "null_mean": null_mean[component],
            "null_std": null_std[component],
            "subset_seconds": subset_seconds,
            "retained_seconds": retained_seconds,
            "n_trials_scored": n_trials_scored,
        }
        for component in range(ensemble_matrix.shape[1])
    ]


def compute_pcc_scores(
    reactivation: np.ndarray, ensemble_matrix: np.ndarray
) -> np.ndarray:
    """
    To assess the xth cell's contribution to ICA reactivation, a PCC score was defined as the mean across all components b and
    timepoints i of the reactivation score R computed from all PCs c minus the reactivation score Rcx computed after
    the exclusion xth cell from the activity and template matrices.
    """
    # TODO: should we keep offline_reactivation for computing R_full separately?
    # TODO: or should we try and change offline_reactivation to use the approach below?
    # this function is bypassing the offline_reactivation function to be faster while returning close results
    # it has been tested that the results are close to the non-vectorised approach

    n_cells = reactivation.shape[0]
    n_timepoints = reactivation.shape[1]
    n_components = ensemble_matrix.shape[1]

    offline_activity_matrix = get_offline_activity_matrix(reactivation=reactivation)
    Z = offline_activity_matrix  # (n_cells, n_timepoints)
    w = ensemble_matrix  # (n_cells, n_components)

    # for each component b compute w_b @ Z
    wZ = contract("kb,ki->bi", w, Z)  # (n_components, n_timepoints)
    assert wZ.shape == (n_components, n_timepoints)

    # for each cell k and component b: compute contribution
    # contribution_kb = 2 * w[k,b] * Z[k,:] * (wZ[b,:] - w[k,b] * Z[k,:])
    contributions = np.zeros((n_cells, n_components, n_timepoints))

    for k in range(n_cells):
        # w[k,:] is (n_components,), Z[k,:] is (n_timepoints,)
        # wZ is (n_components, n_timepoints)
        w_k = w[k, :, np.newaxis]  # (n_components, 1)
        assert w_k.shape == (n_components, 1)
        Z_k = Z[k, np.newaxis, :]  # (1, n_timepoints)
        assert Z_k.shape == (1, n_timepoints)

        # other cells' contribution for each component and timepoint
        other_contrib = wZ - w_k * Z_k  # (n_components, n_timepoints)
        assert other_contrib.shape == (n_components, n_timepoints)

        # cell k's contribution
        contributions[k] = 2 * w_k * Z_k * other_contrib

    pcc_scores = np.mean(contributions, axis=(1, 2))
    assert pcc_scores.shape == (n_cells,)

    return pcc_scores


def get_normalised_pcc_scores(
    reactivation: np.ndarray,
    preactivation: np.ndarray,
    ensemble_matrix: np.ndarray,
) -> np.ndarray:
    """
    To assess the xth cell's contribution to ICA reactivation, a PCC score was defined as the mean across all components b and
    timepoints i of the reactivation score R computed from all PCs c minus the reactivation score Rcx computed after
    the exclusion xth cell from the activity and template matrices.
    To account for putatively nonspecific changes in ICA reactivation strength from the pre to the post epochs, a normalized
    PCC score was taken per session as the pre to post change in within-epoch PCC rank.
    """
    post_pcc_scores = compute_pcc_scores(
        reactivation=reactivation, ensemble_matrix=ensemble_matrix
    )
    pre_pcc_scores = compute_pcc_scores(
        reactivation=preactivation, ensemble_matrix=ensemble_matrix
    )

    assert reactivation.shape[0] == preactivation.shape[0]

    # TODO: should there be a normalisation step?
    post_ranks = rankdata(post_pcc_scores)
    pre_ranks = rankdata(pre_pcc_scores)

    return post_ranks - pre_ranks


def sort_ensembles_by_reactivation_strength(
    reactivation_strength: np.ndarray, n_top: int = 2
) -> np.ndarray:
    """
    In Fig. 4j, panel II, it is not clear how they got to their 'run ensembles' A and B.
    I assumed, they select the two ensembles with the strongest reactivation.
    """
    # TODO: should we normalise? right now it is the strongest ensembles in total

    # TODO: should we only consider positive reactivation strength?
    # positive_only = np.maximum(reactivation_strength, 0)
    # total_strength = np.sum(positive_only, axis=1)

    total_strength = np.sum(reactivation_strength, axis=1)
    # total_strength = np.max(zscore(reactivation_strength, axis=1), axis=1)
    sorted_indices = np.argsort(total_strength)[::-1]
    return sorted_indices[:n_top]


def classify_and_sort_place_cells(
    ensemble_matrix: np.ndarray, top_ensembles: np.ndarray
) -> SortedPlaceCells:
    """
    'Panel (iii) shows the ICA component for each PC, with dashed lines separating those cells with large weights
    in ensemble A (top), ensemble B (middle) or neither (bottom; for the purposes of this illustration, large
    weights were those ≥1 s.d. above the mean for each template)'.
    """
    weights_a = ensemble_matrix[:, top_ensembles[0]]
    weights_b = ensemble_matrix[:, top_ensembles[1]]

    mean_a, std_a = np.mean(weights_a), np.std(weights_a)
    mean_b, std_b = np.mean(weights_b), np.std(weights_b)

    high_a = weights_a >= mean_a + std_a
    high_b = weights_b >= mean_b + std_b

    a_only = np.where(high_a & ~high_b)[0]
    b_only = np.where(high_b & ~high_a)[0]
    neither = np.where(~high_a & ~high_b)[0]

    a_only_sorted = a_only[np.argsort(-weights_a[a_only])]
    b_only_sorted = b_only[np.argsort(-weights_b[b_only])]
    neither_sorted = neither[
        np.argsort(-(np.maximum(weights_a[neither], weights_b[neither])))
    ]

    sorted_indices = np.concatenate([a_only_sorted, b_only_sorted, neither_sorted])
    return SortedPlaceCells(
        sorted_indices=sorted_indices,
        n_ensemble_a=len(a_only_sorted),
        n_ensemble_b=len(b_only_sorted),
    )


def plot_ensemble_reactivation_preactivation(
    reactivation_z: np.ndarray,
    preactivation_z: np.ndarray,
    top_ensembles: np.ndarray,
    smooth: bool = False,
    peak_z: float = 3.0,
) -> None:
    """
    Producing Fig. 4j, panel II. Plotting reactivation time courses for specified ensembles.

    Takes the shuffle-referenced z, so the null sits at zero and there is no separate
    shuffled trace to plot: peak_z is drawn as a reference line instead.
    """
    colours = ["red", "blue", "green", "yellow"]
    matrices = {
        "post": reactivation_z,
        "pre": preactivation_z,
    }

    processed_matrices = {}
    for name, matrix in matrices.items():
        if smooth:
            # processed = zscore(matrix, axis=1)
            # processed = gaussian_filter1d(processed, 30)
            processed = gaussian_filter1d(matrix, 30)
        else:
            # processed = zscore(matrix, axis=1)
            processed = matrix
        processed_matrices[name] = processed

    for name, matrix in processed_matrices.items():
        plt.figure(figsize=(14, 4))
        for i, idx in enumerate(top_ensembles):
            plt.plot(
                matrix[idx, :],
                color=colours[i],
                label=f"ensemble {i}",
            )
        plt.axhline(0, color="black", linestyle="dotted", label="shuffled null")
        plt.axhline(peak_z, color="grey", linestyle="--", label=f"z = {peak_z}")
        plt.xlabel("Time (frames)")
        plt.ylabel("Reactivation strength (z vs shuffled null)")
        plt.title(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{name}.svg", dpi=300)


def plot_cell_weights(
    ensemble_matrix: np.ndarray, top_ensembles: np.ndarray, sorted_pcs: SortedPlaceCells
) -> None:
    """
    Producing Fig. 4j, panel III. Plotting each cell's weight in the top ICA components/ensembles.
    """
    # TODO: is this correct? compare to Grosmark et al.
    colours = ["red", "blue", "green", "yellow"]
    sorted_cells, n_a, n_b = (
        sorted_pcs.sorted_indices,
        sorted_pcs.n_ensemble_a,
        sorted_pcs.n_ensemble_b,
    )
    plt.figure()
    for i, idx in enumerate(top_ensembles):
        plt.plot(
            ensemble_matrix[sorted_cells, idx],
            color=colours[i],
            label=f"ensemble {i}",
        )
    plt.axvline(
        x=n_a,
        color=colours[0],
        linestyle="dotted",
    )
    plt.axvline(
        x=n_a + n_b,
        color=colours[1],
        linestyle="dotted",
    )
    # calling it place cell ID as in Grosmark et al. as the order has been changed by sorting
    plt.xlabel("Place cell ID")
    plt.ylabel("Cell weight in template")
    plt.tight_layout()
    plt.savefig(f"plots/cell_weights.svg", dpi=300)
    # plt.show()


def plot_smoothed_offline_firing_rate_raster(
    reactivation: np.ndarray, sorted_pcs: SortedPlaceCells
) -> None:
    """
    Producing Fig. 4j, panel IV. Plotting the smoothed offline firing rate raster.
    """
    sorted_cells = sorted_pcs.sorted_indices
    offline_activity_matrix = get_offline_activity_matrix(reactivation=reactivation)
    plt.figure()
    plt.imshow(
        offline_activity_matrix[sorted_cells, :],
        cmap="gray_r",
        aspect="auto",
    )
    plt.ylabel("Place cell ID")
    plt.xlabel("Frames")
    plt.tight_layout()
    plt.savefig(f"plots/smoothed_offline_firing_rate_raster.svg", dpi=300)
    # plt.show()


def plot_pcc_scores(pcc_scores: np.ndarray) -> None:
    plt.figure()
    plt.plot(pcc_scores)
    plt.xlabel("Place cell ID")
    plt.ylabel("PCC score (normalised)")
    plt.tight_layout()
    plt.savefig(f"plots/pcc_scores.svg", dpi=300)
    # plt.show()


def plot_grosmark_panel(
    reactivation_strength: np.ndarray,
    ensemble_matrix: np.ndarray,
    top_ensembles: np.ndarray,
    sorted_pcs: SortedPlaceCells,
    reactivation: np.ndarray,
    smooth: bool = False,
) -> None:
    """
    Producing Fig. 4j
    """
    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 4], height_ratios=[3, 3])

    xmin = 0

    xmax = 27000

    # 1) reactivation strength (top)
    ax1 = fig.add_subplot(gs[0, 1])
    colours = ["red", "blue", "green", "yellow"]

    if smooth:
        processed_reactivation_strength = gaussian_filter1d(reactivation_strength, 30)
    else:
        processed_reactivation_strength = reactivation_strength

    for i, idx in enumerate(top_ensembles):
        ax1.plot(
            processed_reactivation_strength[idx, xmin:xmax],
            color=colours[i],
            label=f"ensemble {i}",
        )
    ax1.set_xlabel("Time (frames)")
    ax1.set_ylabel("Reactivation strength")
    ax1.set_title("Reactivation time course")

    # 2) cell weights (left)
    ax2 = fig.add_subplot(gs[1, 0])
    sorted_cells, n_a, n_b = (
        sorted_pcs.sorted_indices,
        sorted_pcs.n_ensemble_a,
        sorted_pcs.n_ensemble_b,
    )
    for i, idx in enumerate(top_ensembles):
        ax2.plot(
            ensemble_matrix[sorted_cells, idx],
            np.arange(len(sorted_cells)),
            color=colours[i],
            label=f"ensemble {i}",
        )
    ax2.axhline(y=n_a, color=colours[0], linestyle="dotted")
    ax2.axhline(y=n_a + n_b, color=colours[1], linestyle="dotted")
    ax2.set_ylabel("Place cell ID")
    ax2.set_xlabel("Cell weight in template")
    ax2.set_title("Cell weights")

    # 3) smoothed offline firing rate raster (bottom)
    ax3 = fig.add_subplot(
        gs[1, 1],
        sharex=ax1,
        sharey=ax2,
    )
    sorted_cells = sorted_pcs.sorted_indices
    # offline_activity_matrix = get_offline_activity_matrix(reactivation=reactivation)
    offline_activity_matrix = reactivation.copy()

    raster = offline_activity_matrix[sorted_cells, xmin:xmax]

    im = ax3.imshow(
        raster,
        cmap="gray_r",
        aspect="auto",
        interpolation="none",
    )
    # desired_ratio = raster.shape[1] / raster.shape[0]
    # ax3.set_aspect(desired_ratio / 5)  # Reduce the stretching

    ax3.set_ylabel("Place cell ID")
    ax3.set_xlabel("Frames")

    # ax3.set_title("Smoothed offline firing rate raster")

    plt.savefig("plots/grosmark_panel.svg", dpi=300)
    # plt.show()


def get_ssp_vectors(
    trials: List[TrialInfo],
    place_cells: np.ndarray,
    sigma: int = 30,
    mode: Literal["above", "below", "all"] = "above",
    speed_threshold: float = 5,
    n_consecutive_samples: int = 3 * 30,
    min_chunk_length: int | None = None,
) -> SSPVectorData:
    """Get sparsified binary spike estimate vector (Ssp) vector as in Grosmark et al.
    The actual binarisation and sparsification step is run in run_oasis.
    The place cell finding step is run in grosmark_analysis/get_place_cells

    This function gets the running bouts and smooths them. Based on these sections in the methods:
    'Online running epochs were defined as those in which the animal's smoothed velocity was above
        5cms-1 for at least 3 consecutive seconds.'

    'PC run running-bout spike estimate vectors, Ssp, were convolved with a 1-s Gaussian kernel
        corresponding to behavioral timescales.'

    Smoothing is applied ONCE across the whole continuous recording, and the running
    bouts are selected out of the result. The bouts are a selection, not a real
    discontinuity in the data, so there is no reason to filter each one in isolation.
    Doing that fed each bout to a kernel wider than itself: with sigma of 1 s and
    scipy's default reflect padding, a 4.5 s corridor traversal had 89% of its samples
    within 2 sigma of an edge.

    The damage is to the variance, and it is position-locked. Reflecting a bout back
    onto itself means the smoothed value near an edge averages fewer independent
    samples, so its variance is inflated. On uniform synthetic activity, per-chunk
    smoothing gave bout edges 2.05x the variance of the bout centre; continuous
    smoothing gives 1.09x, which is flat to within noise. Because bouts are
    trial-locked, that inflation lands at the same corridor position on every trial,
    and ICA works on the covariance matrix, so a trial-aligned modulation of how much
    each timepoint contributes to it is exactly the kind of structure it will return as
    an ensemble.

    min_chunk_length defaults to 4 * sigma, the kernel's support. With continuous
    smoothing a bout shorter than that has no interior: every sample in it draws
    substantially on the surrounding non-running activity that the kernel pulled in.
    Pass 0 to keep every bout that met the speed criterion.

    Note that at the default this discards bouts under 4 s, and a fast traversal of the
    180 cm corridor may not spend 4 s above 5 cm/s. That biases what is kept towards
    slower trials, so if running speed differs by genotype it becomes a confound - check
    the retained-frame diagnostic below against speed before trusting a group
    comparison. Setting min_chunk_length to n_consecutive_samples instead keeps the
    inclusion rule identical to the stated 3-second running-bout definition.
    """
    if min_chunk_length is None:
        min_chunk_length = 4 * sigma

    # Smooth the continuous recording once, then select. See the docstring.
    smoothed_place_cells = gaussian_filter1d(place_cells, sigma=sigma, axis=1)

    ssp_vectors = []
    position_vectors = []
    trial_start_indices = []
    chunk_start_indices = []
    current_idx = 0

    n_chunks_kept = 0
    n_chunks_dropped = 0
    n_frames_kept = 0
    n_frames_dropped = 0

    for trial in trials:
        trial_chunk_start_indices = []
        position = degrees_to_cm(
            np.array(trial.rotary_encoder_position),
            get_wheel_circumference_from_rig("2P"),
        )

        # actual position of the mouse at each imaging frame
        frame_position = np.array(
            [
                state.closest_frame_start
                for state in trial.states_info
                if state.name
                in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
            ]
        )
        assert len(position) == len(frame_position)

        speed = compute_speed_grosmark(position)

        # safely map indices in the ssp vector to indices in the position/speed vectors
        frame_to_pos_index = {int(f): idx for idx, f in enumerate(frame_position)}

        if mode == "above":
            idx_keep = above_threshold_for_n_consecutive_samples(
                speed, threshold=speed_threshold, n_samples=n_consecutive_samples
            )
        elif mode == "below":
            idx_keep = below_threshold_for_n_consecutive_samples(
                speed, threshold=speed_threshold, n_samples=n_consecutive_samples
            )
        elif mode == "all":
            # don't filter for speed
            idx_keep = np.ones_like(speed, dtype=bool)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Take the ITI out
        idx_keep = idx_keep & (position < 180)
        frames_keep = np.unique(frame_position[idx_keep])

        if frames_keep.size == 0:
            # skip trial if there aren't any valid frames
            continue

        trial_start_idx = None
        trial_has_chunks = False
        # Don't smooth across non-continuous chunks
        for chunk in split_continuous_chunks(frames_keep):
            if len(chunk) < min_chunk_length:
                n_chunks_dropped += 1
                n_frames_dropped += len(chunk)
                continue

            if np.any(np.array(chunk) < 0) or np.any(
                np.array(chunk) >= place_cells.shape[1]
            ):
                raise IndexError(
                    "Ssp chunk contains invalid frame indices for place_cells"
                )

            n_chunks_kept += 1
            n_frames_kept += len(chunk)

            # Already smoothed across the continuous recording, so just select
            segment = smoothed_place_cells[:, chunk]

            # add the smoothed segment of neural activity to the ssp_vectors
            ssp_vectors.append(segment)

            pos_idx = [frame_to_pos_index.get(int(f), None) for f in chunk]
            if any(x is None for x in pos_idx):
                raise IndexError("Missing frames")

            # search for the positions of the mouse at each frame of the chunk
            chunk_positions = position[np.searchsorted(frame_position, chunk)]
            position_vectors.append(chunk_positions)

            trial_chunk_start_indices.append(current_idx)
            # treat the beginning of the first valid chunk found as the trial start idx
            if not trial_has_chunks:
                trial_start_idx = current_idx
                trial_has_chunks = True

            current_idx += segment.shape[1]

        # only append the trial start idx if there are valid chunks within the trial
        if trial_has_chunks:
            trial_start_indices.append(trial_start_idx)
            # append the list of start indices of the chunks within the trial to the nested list
            chunk_start_indices.append((trial_chunk_start_indices))

    # Running bouts are assumed to be runs of consecutive frame indices. If the VR
    # update loop and the imaging clock drift apart the bouts fragment, and every
    # fragment under min_chunk_length is silently discarded, so report what survived.
    total_chunks = n_chunks_kept + n_chunks_dropped
    total_frames = n_frames_kept + n_frames_dropped
    if total_frames:
        print(
            f"Ssp bouts: kept {n_chunks_kept}/{total_chunks} chunks, "
            f"{n_frames_kept}/{total_frames} running frames "
            f"({n_frames_kept / total_frames:.1%}) at min_chunk_length="
            f"{min_chunk_length} ({min_chunk_length / 30:.1f} s)"
        )

    if len(ssp_vectors) == 0:
        print("No frames matched the criteria, returning empty ssp vector")
        return SSPVectorData(
            ssp_vectors=np.array([]),
            position_vectors=np.array([]),
            trial_start_indices=np.array([]),
            chunk_start_indices=np.array([]),
        )
    else:
        total_length = np.hstack(ssp_vectors).shape[1]
        assert all(
            0 <= s <= total_length for s in trial_start_indices
        ), "Invalid trial start indices"
        assert len(trial_start_indices) <= len(trials), "Too many trial start indices"

        assert len(ssp_vectors) == len(position_vectors)

        assert len(chunk_start_indices) == len(
            trial_start_indices
        ), "For each valid trial, there must be an array of chunk start indices"
        return SSPVectorData(
            ssp_vectors=np.hstack(ssp_vectors),
            position_vectors=np.hstack(position_vectors),
            trial_start_indices=np.array(trial_start_indices),
            chunk_start_indices=chunk_start_indices,
        )


def main(mouse: str, date: str, rewarded: bool | None, plot: bool = True) -> None:
    print(f"Processing mouse {mouse}, date {date}")

    verbose = True
    use_cache = True

    with open(CACHE_PATH / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())

    print(f"Working on {session.mouse_name}: {session.date} - {session.session_type}")

    if not session.wheel_freeze:
        print(f"Skipping {date} for mouse {mouse} as there was no wheel block")
        return

    assert (
        SERVER_PATH / "viral_caches" / "ensemble_caches"
    ).exists(), "Cache path does not exist, please create it"

    cache_file = (
        SERVER_PATH
        / "viral_caches"
        / "ensemble_caches"
        / f"{session.mouse_name}suite2p_{session.date}_ensemble_reactivation_{grosmark_config}_rewarded_{rewarded}.npz"
    )

    if use_cache and cache_file.exists():
        (
            pcs_mask,
            ensemble_matrix,
            reactivation_strength,
            reactivation_shuffle_mean,
            reactivation_shuffle_std,
            preactivation_strength,
            preactivation_shuffle_mean,
            preactivation_shuffle_std,
            reactivation,
            preactivation,
            pcc_scores,
        ) = load_data_from_cache(cache_file)
        if not plot:
            return
    else:
        print("No cached data found, processing data")

        _, spks, _ = load_imaging_data(mouse=mouse, date=date)

        t1 = time.time()
        pcs_mask, _, _ = get_place_cells(
            session=session,
            spks=spks,
            rewarded=rewarded,
            config=grosmark_config,
            plot=False,
        )

        print(f"Time to get place cells: {time.time() - t1}")
        place_cells = spks[pcs_mask, :]

        preactivation, _, reactivation = split_fluoresence_online_freeze(
            flu=place_cells, wheel_freeze=session.wheel_freeze
        )

        n_pre_all, n_post_all = preactivation.shape[1], reactivation.shape[1]
        preactivation, reactivation = restrict_to_immobility(
            offline_pre=preactivation,
            offline_post=reactivation,
            wheel_freeze=session.wheel_freeze,
        )
        print(
            f"Immobility frames kept: pre {preactivation.shape[1]}/{n_pre_all} "
            f"({preactivation.shape[1] / n_pre_all:.1%}), "
            f"post {reactivation.shape[1]}/{n_post_all} "
            f"({reactivation.shape[1] / n_post_all:.1%})"
        )

        trials = [
            trial
            for trial in session.trials
            if trial_is_imaged(trial)
            and ((rewarded is None) or trial.texture_rewarded == rewarded)
        ]

        ssp_result = get_ssp_vectors(
            trials=trials,
            place_cells=place_cells,
        )

        ssp_vectors = ssp_result.ssp_vectors

        ssp_vectors_shuffled = shuffle_rows(ssp_vectors)
        # TODO: are we returning the right thing here?
        ensemble_matrix = compute_ICA_components(ssp_vectors=ssp_vectors)
        num_shuffled_components = compute_ICA_components(
            ssp_vectors=ssp_vectors_shuffled
        ).shape[1]

        print("ICA done")
        # OFFLINE
        t2 = time.time()
        reactivation_strength = offline_reactivation(
            reactivation=reactivation, ensemble_matrix=ensemble_matrix
        )
        preactivation_strength = offline_reactivation(
            reactivation=preactivation, ensemble_matrix=ensemble_matrix
        )

        def compute_shuffled_strength(_: Any) -> Tuple[np.ndarray, np.ndarray]:
            ensemble_matrix_shuffled = permute_row_order(ensemble_matrix)
            return (
                offline_reactivation(
                    reactivation=reactivation,
                    ensemble_matrix=ensemble_matrix_shuffled,
                ),
                offline_reactivation(
                    reactivation=preactivation,
                    ensemble_matrix=ensemble_matrix_shuffled,
                ),
            )

        n_shuffles = 500

        # Accumulate the first two moments of the null rather than storing every
        # shuffle: we need a mean and s.d. per (component, timepoint) to z-score
        # against, and holding all 500 costs over a gigabyte on a normal session.
        reactivation_null = RunningMoments(reactivation_strength.shape)
        preactivation_null = RunningMoments(preactivation_strength.shape)

        do_concurrent = False
        if do_concurrent:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(compute_shuffled_strength, range(n_shuffles))
                for reac, preac in tqdm(results, total=n_shuffles):
                    reactivation_null.update(reac)
                    preactivation_null.update(preac)
        else:
            for _ in tqdm(range(n_shuffles)):
                reac, preac = compute_shuffled_strength(None)
                reactivation_null.update(reac)
                preactivation_null.update(preac)

        reactivation_shuffle_mean = reactivation_null.mean
        reactivation_shuffle_std = reactivation_null.std
        preactivation_shuffle_mean = preactivation_null.mean
        preactivation_shuffle_std = preactivation_null.std

        print(f"Time to get reactivation strength(s): {time.time() - t2}")
        t3 = time.time()
        pcc_scores = get_normalised_pcc_scores(
            reactivation=reactivation,
            preactivation=preactivation,
            ensemble_matrix=ensemble_matrix,
        )
        print(f"Time to get PCC scores: {time.time() - t3}")
        print("Saving cache")
        np.savez(
            cache_file,
            pcs_mask=pcs_mask,
            ensemble_matrix=ensemble_matrix,
            reactivation_strength=reactivation_strength,
            reactivation_shuffle_mean=reactivation_shuffle_mean,
            reactivation_shuffle_std=reactivation_shuffle_std,
            preactivation_strength=preactivation_strength,
            preactivation_shuffle_mean=preactivation_shuffle_mean,
            preactivation_shuffle_std=preactivation_shuffle_std,
            reactivation=reactivation,
            preactivation=preactivation,
            pcc_scores=pcc_scores,
        )
        if not plot:
            return
    if verbose:
        # print(f"# frames with running: {running_bouts.shape[1]}")
        # print(f"# place cells: {running_bouts.shape[0]}")
        print(f"# significant components (Marcenko-Pastur): {ensemble_matrix.shape[1]}")
        # TODO: remove eventually?
        print(
            f"# significant components (Marcenko-Pastur), shuffled data: {num_shuffled_components if not use_cache else 'not computed'}"
        )

    reactivation_z = reactivation_zscore(
        reactivation_strength=reactivation_strength,
        shuffle_mean=reactivation_shuffle_mean,
        shuffle_std=reactivation_shuffle_std,
    )
    preactivation_z = reactivation_zscore(
        reactivation_strength=preactivation_strength,
        shuffle_mean=preactivation_shuffle_mean,
        shuffle_std=preactivation_shuffle_std,
    )

    post_summary = summarise_reactivation(reactivation_z, offline_spks=reactivation)
    pre_summary = summarise_reactivation(preactivation_z, offline_spks=preactivation)

    print(
        f"Event rate (Hz of immobility): pre {np.mean(pre_summary.event_rate_hz):.4f} "
        f"post {np.mean(post_summary.event_rate_hz):.4f}\n"
        f"Mean peak z per event:         pre {np.nanmean(pre_summary.mean_peak_z):.3f} "
        f"post {np.nanmean(post_summary.mean_peak_z):.3f}\n"
        f"Immobility (s):                pre {pre_summary.immobility_seconds:.0f} "
        f"post {post_summary.immobility_seconds:.0f}\n"
        f"Excursions rejected for <5 participating cells: "
        f"pre {pre_summary.n_events_rejected.sum()}/"
        f"{pre_summary.n_events.sum() + pre_summary.n_events_rejected.sum()} "
        f"post {post_summary.n_events_rejected.sum()}/"
        f"{post_summary.n_events.sum() + post_summary.n_events_rejected.sum()}"
    )

    top_ensembles = sort_ensembles_by_reactivation_strength(
        reactivation_strength=reactivation_strength, n_top=2
    )
    sorted_pcs = classify_and_sort_place_cells(
        ensemble_matrix=ensemble_matrix, top_ensembles=top_ensembles
    )
    plot_pcc_scores(pcc_scores=pcc_scores)
    plot_ensemble_reactivation_preactivation(
        reactivation_z=reactivation_z,
        preactivation_z=preactivation_z,
        top_ensembles=top_ensembles,
        smooth=True,
    )
    plot_cell_weights(
        ensemble_matrix=ensemble_matrix,
        top_ensembles=top_ensembles,
        sorted_pcs=sorted_pcs,
    )
    plot_smoothed_offline_firing_rate_raster(
        reactivation=reactivation, sorted_pcs=sorted_pcs
    )
    plot_grosmark_panel(
        reactivation_strength=reactivation_strength,
        top_ensembles=top_ensembles,
        ensemble_matrix=ensemble_matrix,
        sorted_pcs=sorted_pcs,
        reactivation=reactivation,
        smooth=True,
    )
    1 / 0


def get_reactivation_strength_sum(
    reactivation_strength: np.ndarray,
    preactivation_strength: np.ndarray,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    total_reactivation = np.nanmean(reactivation_strength, axis=1)
    total_preactivation = np.nanmean(preactivation_strength, axis=1)

    if plot:
        plt.figure()
        plt.axhline(
            y=0,
            color="black",
            linestyle="dotted",
        )
        plt.plot(
            [0] * len(total_reactivation),
            total_reactivation - total_preactivation,
            ".",
            label="reactivation",
        )

        plt.ylabel("Reactivation strength (sum across time)")
        plt.title(
            f" Sum of values over threshold: Mean change = {np.mean(total_reactivation - total_preactivation):.2f} p ={wilcoxon(total_reactivation, total_preactivation)[1]:.2f}",
        )
    return total_reactivation, total_preactivation


def reactivation_triggered_average(
    data: np.ndarray, baseline: np.ndarray, length_event_samples: int = 30
) -> np.ndarray:
    """Get the average reactivation strength arond a suprathreshold event.
    Event is length_event_samples either side of the the time when data crosses baseline,
    """

    assert data.shape == baseline.shape

    result = []

    for idx in range(data.shape[0]):
        onset_times = threshold_detect_continuous(
            data[idx, :],
            baseline[idx, :],
        )
        assert len(onset_times) > 0, "No events found"

        component_response = []

        for i, onset in enumerate(onset_times):

            peak = np.argmax(data[idx, onset : onset + length_event_samples])

            if (
                peak + onset - length_event_samples < 0
                or peak + onset + length_event_samples > data.shape[1]
            ):
                continue

            component_response.append(
                data[
                    idx,
                    peak
                    + onset
                    - length_event_samples : peak
                    + onset
                    + length_event_samples,
                ]
            )

        result.append(np.nanmean(component_response, axis=0))

    return np.array(result)


def get_reactivation_triggered_averages(
    reactivation_strength: np.ndarray,
    reactivation_strength_baseline: np.ndarray,
    preactivation_strength: np.ndarray,
    preactivation_strength_baseline: np.ndarray,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    reactivation = reactivation_triggered_average(
        data=reactivation_strength,
        baseline=reactivation_strength_baseline,
    )
    preactivation = reactivation_triggered_average(
        data=preactivation_strength,
        baseline=preactivation_strength_baseline,
    )

    if plot:
        x_axis = np.arange(-30, 30) / 30

        shaded_line_plot(
            reactivation, x_axis=x_axis, color="blue", label="reactivation"
        )
        shaded_line_plot(
            preactivation, x_axis=x_axis, color="orange", label="preactivation"
        )

        plt.xlabel("Frames from event onset")
        plt.legend()

    return reactivation, preactivation


def get_reactivation_number_of_events(
    reactivation_strength: np.ndarray,
    reactivation_strength_baseline: np.ndarray,
    preactivation_strength: np.ndarray,
    preactivation_strength_baseline: np.ndarray,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    total_reactivation = np.array(
        [
            len(
                threshold_detect_continuous(
                    reactivation_strength[idx, :],
                    reactivation_strength_baseline[idx, :],
                )
            )
            for idx in range(reactivation_strength.shape[0])
        ]
    ) / (reactivation_strength.shape[1] / 30)

    total_preactivation = np.array(
        [
            len(
                threshold_detect_continuous(
                    preactivation_strength[idx, :],
                    preactivation_strength_baseline[idx, :],
                )
            )
            for idx in range(preactivation_strength.shape[0])
        ]
    ) / (preactivation_strength.shape[1] / 30)

    if plot:
        plt.figure()
        plt.axhline(
            y=0,
            color="black",
            linestyle="dotted",
        )
        plt.plot(
            [0] * len(total_reactivation),
            total_reactivation - total_preactivation,
            ".",
            label="reactivation",
        )

        plt.ylabel("Reactivation strength (sum across time)")
        plt.title(
            f"Number of events over threshold: Mean change = {np.mean(total_reactivation - total_preactivation):.2f} p ={wilcoxon(total_reactivation, total_preactivation)[1]:.2f}"
        )

    return total_reactivation, total_preactivation


def load_data_from_cache(cache_file: Path) -> tuple:
    print("Using cached data")
    cache = np.load(cache_file, allow_pickle=True)
    return (
        cache["pcs_mask"],
        cache["ensemble_matrix"],
        cache["reactivation_strength"],
        cache["reactivation_shuffle_mean"],
        cache["reactivation_shuffle_std"],
        cache["preactivation_strength"],
        cache["preactivation_shuffle_mean"],
        cache["preactivation_shuffle_std"],
        cache["reactivation"],
        cache["preactivation"],
        cache["pcc_scores"],
    )


def compare_run_results(W_py: np.ndarray, W_mat: np.ndarray) -> None:
    """There is no guarentee that two runs of ICA (particularly from different programming languages with different random seeds)
    will return the same:
        numerical values
        column order
        or even sign (i.e. the same component may be positive or negative)

    However their absolute sums should be similar. And the actual components (once sorted and the signs aligned)
    should span the same subspace. You can test this by looking at the angles between two components.
    They should be 0 (within floating point error)

    """

    assert np.sum(np.abs(W_py)) - np.sum(np.abs(W_mat)) < np.sum(np.abs(W_mat)) * 0.01

    def sort_and_align(W):
        # Sort columns by their L2 norm
        norms = np.sum(W**2, axis=1)
        order = np.argsort(norms)
        W_sorted = W[order, :]
        return W_sorted

    W_py = sort_and_align(W_py)
    W_mat = sort_and_align(W_mat)

    # Align signs
    for i in range(W_py.shape[1]):
        if np.dot(W_py[:, i], W_mat[:, i]) < 0:
            W_py[:, i] *= -1

    test_subspace(W_py, W_mat)
    # assert np.allclose(W_py, W_mat, atol=1e-3)
    print("All close")


def test_subspace(W1: np.ndarray, W2: np.ndarray) -> None:
    def orthonormalize(W):
        # QR decomposition for orthonormal basis
        Q, _ = np.linalg.qr(W)
        return Q

    Q1 = orthonormalize(W1)
    Q2 = orthonormalize(W2)

    # Compute principal angles (in radians)
    angles = subspace_angles(Q2, Q1)
    print("Max angle:", np.max(np.degrees(angles)))
    assert np.max(np.degrees(angles)) < 1e-9


def multiple_sessions(
    rewarded: bool | None = None,
) -> pd.DataFrame:
    """Summarise ICA ensemble reactivation across cached sessions.

    Returns one row per (session, component) with the paired pre and post measures,
    which is the shape the mixed model in dec.py wants: component nested in session
    nested in mouse, with the genotype contrast tested by exact permutation over mice.
    Do not collapse to one number per session first - with ~9 mice against 172
    sessions, the nesting is where the power is.
    """
    records = []

    cache_files = list(CACHE_PATH.glob("*.json"))

    mice = set(f.stem.split("_")[0] for f in cache_files)
    # So you don't have to build this for each sessions
    metadata_dict = {mouse: gsheet2df(SPREADSHEET_ID, mouse, 1) for mouse in mice}

    for cache_file in cache_files:
        file_parts = cache_file.stem.split("_")
        date = file_parts[1]
        mouse = file_parts[0]
        metadata = metadata_dict[mouse]
        stage = metadata.loc[metadata["Date"] == date, "Type"].values[0]

        ensemble_cache = (
            SERVER_PATH
            / "viral_caches"
            / "ensemble_caches"
            / f"{mouse}suite2p_{date}_ensemble_reactivation_{grosmark_config}_rewarded_{rewarded}.npz"
        )

        if not ensemble_cache.exists():
            print(f"Skipping {mouse} {date} as no ensemble cache found")
            continue

        (
            pcs_mask,
            ensemble_matrix,
            reactivation_strength,
            reactivation_shuffle_mean,
            reactivation_shuffle_std,
            preactivation_strength,
            preactivation_shuffle_mean,
            preactivation_shuffle_std,
            reactivation,
            preactivation,
            _,
        ) = load_data_from_cache(ensemble_cache)

        # Rescale to unit-norm weights. Exact and idempotent, so this works on caches
        # written before compute_ICA_components started normalising, with no re-run.
        reactivation_strength = rescale_strength_to_unit_norm(
            reactivation_strength, ensemble_matrix
        )
        preactivation_strength = rescale_strength_to_unit_norm(
            preactivation_strength, ensemble_matrix
        )
        reactivation_shuffle_mean = rescale_strength_to_unit_norm(
            reactivation_shuffle_mean, ensemble_matrix
        )
        reactivation_shuffle_std = rescale_strength_to_unit_norm(
            reactivation_shuffle_std, ensemble_matrix
        )
        preactivation_shuffle_mean = rescale_strength_to_unit_norm(
            preactivation_shuffle_mean, ensemble_matrix
        )
        preactivation_shuffle_std = rescale_strength_to_unit_norm(
            preactivation_shuffle_std, ensemble_matrix
        )

        # Mean strength over every offline frame: no threshold, no deduplication, no
        # participation criterion. This is what Grosmark's Extended Data Fig 6a plots,
        # and it is the measure least exposed to detection parameters.
        reactivation_z = reactivation_zscore(
            reactivation_strength=reactivation_strength,
            shuffle_mean=reactivation_shuffle_mean,
            shuffle_std=reactivation_shuffle_std,
        )
        preactivation_z = reactivation_zscore(
            reactivation_strength=preactivation_strength,
            shuffle_mean=preactivation_shuffle_mean,
            shuffle_std=preactivation_shuffle_std,
        )
        mean_r_post = reactivation_strength.mean(axis=1)
        mean_r_pre = preactivation_strength.mean(axis=1)
        mean_rz_post = reactivation_z.mean(axis=1)
        mean_rz_pre = preactivation_z.mean(axis=1)

        post = summarise_reactivation(
            reactivation_zscore(
                reactivation_strength=reactivation_strength,
                shuffle_mean=reactivation_shuffle_mean,
                shuffle_std=reactivation_shuffle_std,
            ),
            offline_spks=reactivation,
        )
        pre = summarise_reactivation(
            reactivation_zscore(
                reactivation_strength=preactivation_strength,
                shuffle_mean=preactivation_shuffle_mean,
                shuffle_std=preactivation_shuffle_std,
            ),
            offline_spks=preactivation,
        )

        # Every significant component contributes. Picking the top few by post-epoch
        # strength and then comparing post to pre would be circular.
        for component in range(ensemble_matrix.shape[1]):
            records.append(
                {
                    "mouse": mouse,
                    "genotype": get_genotype(mouse),
                    "date": date,
                    "stage": stage,
                    "component": component,
                    "n_components": ensemble_matrix.shape[1],
                    "n_place_cells": int(np.sum(pcs_mask)),
                    "mean_r_pre": mean_r_pre[component],
                    "mean_r_post": mean_r_post[component],
                    "mean_rz_pre": mean_rz_pre[component],
                    "mean_rz_post": mean_rz_post[component],
                    "weight_norm": float(np.linalg.norm(ensemble_matrix[:, component])),
                    "event_rate_hz_pre": pre.event_rate_hz[component],
                    "event_rate_hz_post": post.event_rate_hz[component],
                    "mean_peak_z_pre": pre.mean_peak_z[component],
                    "mean_peak_z_post": post.mean_peak_z[component],
                    "n_events_rejected_pre": pre.n_events_rejected[component],
                    "n_events_rejected_post": post.n_events_rejected[component],
                    "immobility_seconds_pre": pre.immobility_seconds,
                    "immobility_seconds_post": post.immobility_seconds,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        print("No sessions found")
        return df

    df["delta_event_rate_hz"] = df["event_rate_hz_post"] - df["event_rate_hz_pre"]
    df["delta_mean_peak_z"] = df["mean_peak_z_post"] - df["mean_peak_z_pre"]
    df["delta_mean_r"] = df["mean_r_post"] - df["mean_r_pre"]
    df["delta_mean_rz"] = df["mean_rz_post"] - df["mean_rz_pre"]
    return df


def plot_reactivation_summary(df: pd.DataFrame) -> None:
    """Paired pre versus post for the two measures, split by genotype.

    Also plots retained immobility, which is a covariate and a candidate confound:
    if the groups differ in how much they move while the wheel is frozen they are not
    contributing comparable amounts of offline data.
    """

    panels = [
        ("delta_mean_r", "Mean ICA ensemble reactivation\npost - pre"),
        ("delta_event_rate_hz", "Event rate (Hz of immobility)\npost - pre"),
        ("delta_mean_peak_z", "Mean peak z per event\npost - pre"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, (column, label) in zip(axes, panels):
        sns.boxplot(
            data=df, x="genotype", y=column, hue="genotype", showfliers=False, ax=ax
        )
        sns.stripplot(
            data=df, x="genotype", y=column, color="black", alpha=0.3, size=3, ax=ax
        )
        ax.axhline(0, color="black", linestyle="dotted")
        ax.set_ylabel(label)

        for genotype, group in df.groupby("genotype"):
            values = group[column].dropna()
            if len(values) > 1:
                p = wilcoxon(values, alternative="greater").pvalue
                print(f"{column} {genotype}: n={len(values)} p={p:.2e}")

    immobility = df.melt(
        id_vars=["genotype"],
        value_vars=["immobility_seconds_pre", "immobility_seconds_post"],
        var_name="epoch",
        value_name="immobility_seconds",
    )
    sns.boxplot(
        data=immobility,
        x="genotype",
        y="immobility_seconds",
        hue="epoch",
        showfliers=False,
        ax=axes[3],
    )
    axes[3].set_ylabel("Retained immobility (s)")

    plt.tight_layout()


def above_threshold_events(
    arr: np.ndarray, upper_threshold: float, lower_threshold: float
) -> List[Tuple[int, int]]:
    upper = threshold_detect(arr, upper_threshold)
    rising_lower, falling_lower = threshold_detect_edges(arr, lower_threshold)

    events = []
    for u in upper:
        lower_befores = rising_lower[rising_lower < u]
        if len(lower_befores) == 0:
            continue
        onset = lower_befores[-1]
        lower_afters = falling_lower[falling_lower > u]
        if len(lower_afters) == 0:
            continue
        offset = lower_afters[0]
        assert onset < u < offset
        events.append((onset, offset))

    return events


def get_three_sd_events(
    reactivation_strength: np.ndarray, preactivation_strength: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    From: www.nature.com/articles/s41467-023-43254-7#Sec10
    Reactivation events were identified when the reactivation strength exceeds
        three standard deviations above the mean.
    The onset and offset of these events were delimited by 25% of this threshold.

    The threshold is taken from the PRE epoch and applied unchanged to both epochs.
    Thresholding each epoch by its own mean and s.d. makes the measure
    self-normalising: if the post epoch genuinely reactivates more strongly its s.d.
    is larger, its threshold rises with it, and the event count barely moves. That
    cancels most of the effect the pre versus post contrast is meant to detect. The
    pre-freeze block is the natural reference because it precedes the task and so
    cannot contain task-driven reactivation.

    Thresholds stay per-component, since ICA components have arbitrary relative scale.
    """

    threshold = np.mean(preactivation_strength, axis=1) + 3 * np.std(
        preactivation_strength, axis=1
    )

    assert (
        reactivation_strength.shape[0] == preactivation_strength.shape[0]
    ), "Pre and post must be scored on the same set of components"

    reactivation_thresholds = [
        above_threshold_events(
            reactivation_strength[i, :],
            upper_threshold=threshold[i],
            lower_threshold=threshold[i] * 0.25,
        )
        for i in range(reactivation_strength.shape[0])
    ]

    preactivation_thresholds = [
        above_threshold_events(
            preactivation_strength[i, :],
            upper_threshold=threshold[i],
            lower_threshold=threshold[i] * 0.25,
        )
        for i in range(preactivation_strength.shape[0])
    ]

    preactivation_response = [
        get_onset_triggered_response(component, events)
        for component, events in zip(preactivation_strength, preactivation_thresholds)
    ]
    plt.plot(
        np.nanmean(preactivation_response, axis=0), color="blue", label="preactivation"
    )

    reactivation_response = [
        get_onset_triggered_response(component, events)
        for component, events in zip(reactivation_strength, reactivation_thresholds)
    ]
    plt.plot(
        np.nanmean(reactivation_response, axis=0), color="orange", label="reactivation"
    )
    plt.legend()

    return np.array([len(r) for r in reactivation_thresholds]), np.array(
        [len(p) for p in preactivation_thresholds]
    )


def get_onset_triggered_response(
    component: np.ndarray, events: List[Tuple[int, int]]
) -> np.ndarray:
    """Mean reactivation strength in a +/- 1 s window around each event onset.

    This must be a mean, not a sum. Summing makes the trace scale with the number of
    detected events, so an epoch with more events looks like it has larger events,
    which conflates reactivation rate with reactivation amplitude - exactly the two
    things this measure is supposed to separate.

    Returns all-NaN if no event has a full window, so that components with no events
    do not silently contribute zeros to the across-component average.
    """
    window = 30

    responses = []
    for onset, _ in events:
        if onset - window < 0 or onset + window > len(component):
            continue
        responses.append(component[onset - window : onset + window])

    if not responses:
        return np.full(window * 2, np.nan)

    return np.mean(responses, axis=0)


def all_mice_ensemble_results_plots(
    all_mice: List[EnsembleSessionResult],
) -> None:
    plt.figure()
    number_of_events_re = np.hstack([mouse.number_of_events[0] for mouse in all_mice])
    number_of_events_pre = np.hstack([mouse.number_of_events[1] for mouse in all_mice])

    sns.boxplot(number_of_events_re - number_of_events_pre, showfliers=False)
    p = wilcoxon(number_of_events_re, number_of_events_pre, alternative="greater")

    plt.title(
        f"total number events: Wilcoxon result: statistic: {p.statistic} p-value={p.pvalue:.2e},\nmean change={np.mean(number_of_events_re - number_of_events_pre):.3f}"
    )

    sns.stripplot(number_of_events_re - number_of_events_pre)

    plt.figure()
    sum_values_over_threshold_re = np.hstack(
        [mouse.sum_values_over_threshold[0] for mouse in all_mice]
    )
    sum_values_over_threshold_pre = np.hstack(
        [mouse.sum_values_over_threshold[1] for mouse in all_mice]
    )
    sns.boxplot(
        sum_values_over_threshold_re - sum_values_over_threshold_pre, showfliers=False
    )
    p = wilcoxon(
        sum_values_over_threshold_re,
        sum_values_over_threshold_pre,
        alternative="greater",
    )
    plt.title(
        f"Sum values: Wilcoxon result: statistic: {p.statistic} p-value={p.pvalue:.2e}\nmean change={np.mean(sum_values_over_threshold_re - sum_values_over_threshold_pre):.3f}"
    )

    sns.stripplot(sum_values_over_threshold_re - sum_values_over_threshold_pre)

    plt.figure()
    response_re = np.vstack(
        [mouse.reactivation_triggered_response[0] for mouse in all_mice]
    )
    response_pre = np.vstack(
        [mouse.reactivation_triggered_response[1] for mouse in all_mice]
    )
    shaded_line_plot(
        response_re - response_pre,
        x_axis=np.arange(-30, 30) / 30,
        color="blue",
        label="reactivation",
    )
    # shaded_line_plot(
    #     response_pre,
    #     x_axis=np.arange(-30, 30) / 30,
    #     color="orange",
    #     label="preactivation",
    # )
    plt.xlabel("Time (s) from event onset")
    1 / 0


def compare_run_results(W_py: np.ndarray, W_mat: np.ndarray) -> None:
    """There is no guarentee that two runs of ICA (particularly from different programming languages with different random seeds)
    will return the same:
        numerical values
        column order
        or even sign (i.e. the same component may be positive or negative)

    However their absolute sums should be similar. And the actual components (once sorted and the signs aligned)
    should span the same subspace. You can test this by looking at the angles between two components.
    They should be 0 (within floating point error)

    """

    assert np.sum(np.abs(W_py)) - np.sum(np.abs(W_mat)) < np.sum(np.abs(W_mat)) * 0.01

    def sort_and_align(W):
        # Sort columns by their L2 norm
        norms = np.sum(W**2, axis=1)
        order = np.argsort(norms)
        W_sorted = W[order, :]
        return W_sorted

    W_py = sort_and_align(W_py)
    W_mat = sort_and_align(W_mat)

    # Align signs
    for i in range(W_py.shape[1]):
        if np.dot(W_py[:, i], W_mat[:, i]) < 0:
            W_py[:, i] *= -1

    test_subspace(W_py, W_mat)
    # assert np.allclose(W_py, W_mat, atol=1e-3)
    print("All close")


def test_subspace(W1: np.ndarray, W2: np.ndarray) -> None:
    def orthonormalize(W):
        # QR decomposition for orthonormal basis
        Q, _ = np.linalg.qr(W)
        return Q

    Q1 = orthonormalize(W1)
    Q2 = orthonormalize(W2)

    # Compute principal angles (in radians)
    angles = subspace_angles(Q2, Q1)
    print("Max angle:", np.max(np.degrees(angles)))
    assert np.max(np.degrees(angles)) < 1e-9


def compare_to_matlab() -> None:

    # test_data = np.load(
    #     "/Volumes/hard_drive/VR-2p/2025-07-05/JB036/suite2p/plane0/oasis_spikes.npy"
    # )
    # test_data = gaussian_filter1d(test_data, sigma=30, axis=1)

    # # take a random-ish subset of the online data
    # test_data = test_data[:, 27000:100000]
    # np.save("test_ensemble_data.npy", test_data)

    matlab_result = np.load(
        "/Users/jamesrowland/Code/Cell-Assembly-Detection/assembly_templates.npy"
    )

    test_data = np.load("test_ensemble_data.npy")
    python_result = compute_ICA_components(test_data)
    compare_run_results(matlab_result, python_result)


def all_mouse_plots(
    all_mice: List[EnsembleSessionResult],
) -> None:
    number_of_events = np.hstack(
        [mouse.number_of_events[0] - mouse.number_of_events[1] for mouse in all_mice]
    )

    sum_values_over_threshold_re = np.hstack(
        [mouse.sum_values_over_threshold[0] for mouse in all_mice]
    )
    sum_values_over_threshold_pre = np.hstack(
        [mouse.sum_values_over_threshold[1] for mouse in all_mice]
    )
    sns.boxplot(
        sum_values_over_threshold_re - sum_values_over_threshold_pre, showfliers=False
    )

    plt.figure()
    response_re = np.vstack(
        [mouse.reactivation_triggered_response[0] for mouse in all_mice]
    )
    response_pre = np.vstack(
        [mouse.reactivation_triggered_response[1] for mouse in all_mice]
    )
    shaded_line_plot(
        response_re, x_axis=np.arange(-30, 30) / 30, color="blue", label="reactivation"
    )
    shaded_line_plot(
        response_pre,
        x_axis=np.arange(-30, 30) / 30,
        color="orange",
        label="preactivation",
    )
    plt.xlabel("Time (s) from event onset")
    1 / 0


def get_df_summary(genotype: str) -> pd.DataFrame:

    # main("J035", "2026-06-16", rewarded=None, plot=True)

    cache_files = list(CACHE_PATH.glob("*.json"))
    ensemble_cache_path = CACHE_PATH.parent / "ensemble_caches" / "new_method"

    assert ensemble_cache_path.exists(), f"{ensemble_cache_path} does not exist"

    all_df = []

    n = 0
    debug = True
    for cache_file in cache_files:
        mouse = cache_file.stem.split("_")[0]
        date = cache_file.stem.split("_")[1]

        if get_genotype(mouse) != genotype:
            continue

        if (ensemble_cache_path / f"{mouse}_{date}.csv").exists():
            print(f"Loading cached {mouse} {date}")
            all_df.append(
                pd.read_csv(
                    ensemble_cache_path / f"{mouse}_{date}.csv", index_col=False
                )
            )
            continue

        session = Cached2pSession.model_validate_json(
            (CACHE_PATH / f"{mouse}_{date}.json").read_text()
        )
        if not session.session_type.startswith("reversal learning"):
            continue

        print(f"Processing {mouse} {date}")

        if (LOCAL_DFF_PATH / f"{mouse}_{date}_dff.npy").exists():
            spks = np.load(LOCAL_DFF_PATH / f"{mouse}_{date}_spks.npy")
        else:
            try:
                _, spks, _ = load_imaging_data(mouse, date)
            except Exception as e:
                print(f"Error loading {mouse} {date}: {e}")
                continue

        if spks.shape[0] < 5:
            print(f"No cells :'( probably wrong pmt")
            continue

        try:
            df = freeze_ensemble_reactivation(
                session=session, spks=spks, n_component_method="circular_shift"
            )
            if len(df) > 0:
                print(f"Saving cached {mouse} {date}")
                df.to_csv(ensemble_cache_path / f"{mouse}_{date}.csv", index=False)
        except Exception as e:
            print(f"Error processing {mouse} {date}: {e}")
            if debug:
                raise
            continue
        all_df.append(df)

    df_summary = pd.concat(all_df)
    df_summary.to_csv(f"df_summary_just_reversals.csv", index=False)


def batch_runner(rewarded: bool | None = None) -> None:

    cache_files = list(CACHE_PATH.glob("*.json"))
    mice = set(f.stem.split("_")[0] for f in cache_files)
    mice = [mouse for mouse in mice if get_genotype(mouse) == "WT"]
    metadata_dict = {mouse: gsheet2df(SPREADSHEET_ID, mouse, 1) for mouse in mice}

    for cache_file in cache_files:
        file_parts = cache_file.stem.split("_")
        date = file_parts[1]
        mouse = file_parts[0]
        if mouse not in mice:
            continue
        metadata = metadata_dict[mouse]
        stage = metadata.loc[metadata["Date"] == date, "Type"].values[0]
        if not stage.lower().startswith("reversal") and not stage.lower().startswith(
            "learning"
        ):
            print(f"Skipping {stage} session for {mouse} {date}")
            continue

        main(mouse=mouse, date=date, rewarded=rewarded, plot=False)


if __name__ == "__main__":

    get_df_summary("WT")
    df = pd.read_csv("df_summary_just_reversals.csv")
    report_freeze_reactivation(df, template="all")
    1 / 0
    # 1 / 0

    # df = pd.read_csv("df_summary.csv")
    # a = df[(df.template == "all") & (df.iti_after != "all")]

    # pivoted = a.pivot_table(
    #     index=["mouse", "date"], columns="iti_after", values="mean_rz", aggfunc="median"
    # )
    # subbed = pivoted["rewarded"].to_numpy() - pivoted["unrewarded"].to_numpy()
    # subbed = subbed[~np.isnan(subbed)]
    # # print(wilcoxon(subbed, alternative="greater"))
    # per_mouse = (
    #     pd.Series(subbed, index=pivoted.dropna().index).groupby("mouse").median()
    # )
    # v = per_mouse.to_numpy()
    # null = np.array([np.mean(v * np.array(s)) for s in product([-1, 1], repeat=len(v))])
    # print(per_mouse, v.mean(), np.mean(null >= v.mean()))

    # 1 / 0
