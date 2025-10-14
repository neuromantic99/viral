from typing import Any, List, Tuple
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
import seaborn as sns
from tqdm import tqdm

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.constants import CACHE_PATH, SERVER_PATH, TIFF_UMBRELLA
from viral.models import (
    Cached2pSession,
    EnsembleSessionResult,
    GrosmarkConfig,
    SortedPlaceCells,
    TrialInfo,
)
from viral.rastermap_utils import (
    get_frame_position,
    get_speed_frame,
    align_validate_data,
    process_trials_data,
    filter_speed_position,
)
from viral.utils import (
    above_threshold_for_n_consecutive_samples,
    degrees_to_cm,
    get_wheel_circumference_from_rig,
    shaded_line_plot,
    shuffle_rows,
    split_continuous_chunks,
    threshold_detect,
    threshold_detect_continuous,
    trial_is_imaged,
)
from viral.imaging_utils import (
    compute_speed_grosmark,
    split_fluoresence_online_freeze,
)
from viral.grosmark_analysis import get_place_cells


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


def compute_ICA_components(ssp_vectors: np.ndarray) -> np.ndarray:
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
    # 'Compute the eigenvalues and right eigenvectors of a square array.' (NumPy documentation)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 'where σ2 is the variance of the elements of M (in our case σ2 = 1 due to z-score normalization)'
    assert np.isclose(np.var(ssp_vectors_z), 1)

    q = n_cols / n_rows

    # 'with q = Ncolumns/Nrows ≥ 1'
    assert q >= 1

    # 'λmax and λmin are the maximum and minimum bounds, respectively, and are calculated as:'
    lambda_max = (1 + np.sqrt(1 / q)) ** 2

    # 'Thus, if the rows of M are statistically independent, the probability of finding an eigenvalue outside these bounds is zero.
    #  In other words, the variance of the data in any axis cannot be larger than λmax when neurons are uncorrelated.
    #  Therefore, λmax can be used as a statistical threshold for detecting cell assembly activity'
    n_significant_components = np.sum(eigenvalues > lambda_max)

    if n_significant_components < 1:
        return np.zeros((ssp_vectors_z.shape[0], 0))

    return fast_ica_sklearn(ssp_vectors_z, n_significant_components)


def get_offline_activity_matrix(reactivation: np.ndarray) -> np.ndarray:
    """Offline reactivation was assessed from the 150-ms Gaussian kernel convolved offline activity matrix Z."""
    sigma = 150 / 1000 * 30  # 150 ms kernel
    return np.apply_along_axis(gaussian_filter1d, axis=1, arr=reactivation, sigma=sigma)


def offline_reactivation(
    reactivation: np.ndarray, ensemble_matrix: np.ndarray, do_shuffle: bool = False
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

    reactivation = zscore(reactivation, axis=1)
    # Remove nans from silent neurons
    reactivation = np.nan_to_num(reactivation)

    if do_shuffle:
        # """ICA components were shuffled by randomly permuting the weight matrix w across
        # PCs and recalculating the reactivation strength."""
        # Confusing, should we shuffle rows or columns (i.e. PCs or components)?
        # ensemble_matrix = ensemble_matrix[
        #     :, np.random.permutation(ensemble_matrix.shape[1])
        # ]
        ensemble_matrix = shuffle_rows(ensemble_matrix)

    offline_activity_matrix = get_offline_activity_matrix(reactivation=reactivation)

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
    reactivation_strength: np.ndarray,
    reactivation_strength_shuffled: np.ndarray,
    preactivation_strength: np.ndarray,
    preactivation_strength_shuffled: np.ndarray,
    top_ensembles: np.ndarray,
    smooth: bool = False,
) -> None:
    """
    Producing Fig. 4j, panel II. Plotting reactivation time courses for specified ensembles.
    """
    colours = ["red", "blue", "green", "yellow"]
    matrices = {
        "post": reactivation_strength,
        "post_shuffled": reactivation_strength_shuffled,
        "pre": preactivation_strength,
        "pre_shuffled": preactivation_strength_shuffled,
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
            # Sometimes the shuffled don't have enough components
            if matrix.shape[0] <= idx and "shuffled" in name:
                continue
            plt.plot(
                matrix[idx, :],
                color=colours[i],
                label=f"ensemble {i}",
            )
        # plt.ylim(-1.5, 7)
        plt.xlabel("Time (frames)")
        # plt.ylabel("Reactivation strength (zscored)")
        plt.ylabel("Reactivation strength")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(f"plots/{name}.svg", dpi=300)
        # plt.show()


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
) -> np.ndarray:
    """Get sparsified binary spike estimate vector (Ssp) vector as in Grosmark et al.
    The actual binarisation and sparsification step is run in run_oasis.
    The place cell finding step is run in grosmark_analysis/get_place_cells

    This function gets the running bouts and smooths them. Based on these sections in the methods:
    'Online running epochs were defined as those in which the animal's smoothed velocity was above
        5cms-1 for at least 3 consecutive seconds.'

    'PC run running-bout spike estimate vectors, Ssp, were convolved with a 1-s Gaussian kernel
        corresponding to behavioral timescales.'
    """
    sigma = 30
    ssp_vectors = []
    for trial in trials:
        position = degrees_to_cm(
            np.array(trial.rotary_encoder_position),
            get_wheel_circumference_from_rig("2P"),
        )

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

        speed_threshold = 5
        idx_keep = above_threshold_for_n_consecutive_samples(
            speed, threshold=speed_threshold, n_samples=3 * 30
        )
        # Take the ITI out
        idx_keep = idx_keep & (position < 180)
        frames_keep = np.unique(frame_position[idx_keep])

        # Don't smooth across non-continuous chunks
        for chunk in split_continuous_chunks(frames_keep):
            if len(chunk) < 2 * 30:  # Arbitrary removal of short chunks
                continue
            ssp_vectors.append(
                gaussian_filter1d(
                    input=place_cells[:, chunk],
                    sigma=sigma,
                    axis=1,
                )
            )

    return np.hstack(ssp_vectors)


def main(mouse: str, date: str, plot: bool = True) -> None:
    print(f"Processing mouse {mouse}, date {date}")

    verbose = True
    use_cache = True

    assert (
        TIFF_UMBRELLA
        / date
        / mouse
        / "suite2p"
        / "plane0"
        / "full_grosmark_oasis_preprocessed.npy"
    ).exists(), f"Correct oasis not run for {mouse} on {date}"

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
        / f"{session.mouse_name}suite2p_{session.date}_ensemble_reactivation.npz"
    )

    if use_cache and cache_file.exists():
        (
            pcs_mask,
            ensemble_matrix,
            reactivation_strength,
            reactivation_strength_shuffled,
            preactivation_strength,
            preactivation_strength_shuffled,
            reactivation,
            preactivation,
            pcc_scores,
        ) = load_data_from_cache(cache_file)
        if not plot:
            return
    else:
        print("No cached data found, processing data")
        config = GrosmarkConfig(
            bin_size=5,
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

        t1 = time.time()
        pcs_mask, _ = get_place_cells(
            session=session, spks=spks, rewarded=None, config=config, plot=True
        )

        print(f"Time to get place cells: {time.time() - t1}")
        place_cells = spks[pcs_mask, :]

        preactivation, _, reactivation = split_fluoresence_online_freeze(
            flu=place_cells, wheel_freeze=session.wheel_freeze
        )

        trials = [trial for trial in session.trials if trial_is_imaged(trial)]

        ssp_vectors = get_ssp_vectors(
            trials=trials,
            place_cells=place_cells,
        )

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
            ensemble_matrix_shuffled = shuffle_rows(ensemble_matrix)
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
        reactivation_strength_shuffled = []
        preactivation_strength_shuffled = []

        do_concurrent = False
        if do_concurrent:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(
                    tqdm(
                        executor.map(compute_shuffled_strength, range(n_shuffles)),
                        total=n_shuffles,
                    )
                )
                for reac, preac in results:
                    reactivation_strength_shuffled.append(reac)
                    preactivation_strength_shuffled.append(preac)

        else:
            for _ in tqdm(range(n_shuffles)):
                reac, preac = compute_shuffled_strength(None)
                reactivation_strength_shuffled.append(reac)
                preactivation_strength_shuffled.append(preac)

        reactivation_strength_shuffled = np.percentile(
            np.array(reactivation_strength_shuffled), 95, axis=0
        )
        preactivation_strength_shuffled = np.percentile(
            np.array(preactivation_strength_shuffled), 95, axis=0
        )

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
            reactivation_strength_shuffled=reactivation_strength_shuffled,
            preactivation_strength=preactivation_strength,
            preactivation_strength_shuffled=preactivation_strength_shuffled,
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

    top_ensembles = sort_ensembles_by_reactivation_strength(
        reactivation_strength=reactivation_strength, n_top=2
    )
    sorted_pcs = classify_and_sort_place_cells(
        ensemble_matrix=ensemble_matrix, top_ensembles=top_ensembles
    )
    plot_pcc_scores(pcc_scores=pcc_scores)
    plot_ensemble_reactivation_preactivation(
        reactivation_strength=reactivation_strength,
        reactivation_strength_shuffled=reactivation_strength_shuffled,
        preactivation_strength=preactivation_strength,
        preactivation_strength_shuffled=preactivation_strength_shuffled,
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


def get_reactivation_strength_sum(
    reactivation_strength: np.ndarray,
    preactivation_strength: np.ndarray,
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    assert reactivation_strength.shape == preactivation_strength.shape

    total_reactivation = np.nansum(reactivation_strength, axis=1)
    total_preactivation = np.nansum(preactivation_strength, axis=1)

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
    )

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
    )

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
        cache["reactivation_strength_shuffled"],
        cache["preactivation_strength"],
        cache["preactivation_strength_shuffled"],
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


def multiple_sessions() -> None:
    """Run ensemble reactivation on multiple cached sessions"""

    cache_files = list(
        (SERVER_PATH / "viral_caches" / "ensemble_caches").glob(
            "*_ensemble_reactivation.npz"
        )
    )
    assert cache_files, "No cache files found"

    all_mice: List[EnsembleSessionResult] = []

    baseline_multiplier = 1
    for cache_file in cache_files:

        mouse, date = cache_file.stem.split("_")[:2]
        mouse = mouse.strip("suite2p")  # dunno why this is in the path lol
        data = np.load(cache_file, allow_pickle=True)
        (
            reactivation_strength,
            reactivation_strength_shuffled,
            preactivation_strength,
            preactivation_strength_shuffled,
            ensemble_matrix,
        ) = (
            data["reactivation_strength"],
            data["reactivation_strength_shuffled"],
            data["preactivation_strength"],
            data["preactivation_strength_shuffled"],
            data["ensemble_matrix"],
        )

        if ensemble_matrix.shape[1] < 5:
            print(
                f"Skipping {mouse} {date} as there are only {ensemble_matrix.shape[1]} components"
            )
            continue

        significant_reactivation = (
            reactivation_strength > reactivation_strength_shuffled * baseline_multiplier
        )
        significant_preactivation = (
            preactivation_strength
            > preactivation_strength_shuffled * baseline_multiplier
        )

        only_significant_reactivation = reactivation_strength.copy()
        only_significant_preactivation = preactivation_strength.copy()

        only_significant_reactivation[~significant_reactivation] = np.nan
        only_significant_preactivation[~significant_preactivation] = np.nan

        reactivation_number_of_events = get_reactivation_number_of_events(
            reactivation_strength=reactivation_strength,
            preactivation_strength=preactivation_strength,
            reactivation_strength_baseline=reactivation_strength_shuffled
            * baseline_multiplier,
            preactivation_strength_baseline=preactivation_strength_shuffled
            * baseline_multiplier,
            plot=False,
        )

        reactivation_strength_sum_over_threshold = get_reactivation_strength_sum(
            reactivation_strength=only_significant_reactivation,
            preactivation_strength=only_significant_preactivation,
        )

        reactivation_triggered_response = get_reactivation_triggered_averages(
            reactivation_strength=reactivation_strength,
            reactivation_strength_baseline=reactivation_strength_shuffled
            * baseline_multiplier,
            preactivation_strength=preactivation_strength,
            preactivation_strength_baseline=preactivation_strength_shuffled
            * baseline_multiplier,
        )

        all_mice.append(
            EnsembleSessionResult(
                reactivation_triggered_response=reactivation_triggered_response,
                number_of_events=reactivation_number_of_events,
                sum_values_over_threshold=reactivation_strength_sum_over_threshold,
            )
        )
    all_mice_ensemble_results_plots(all_mice=all_mice)


def all_mice_ensemble_results_plots(
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


def multiple_sessions() -> None:

    cache_files = list(
        (SERVER_PATH / "viral_caches" / "ensemble_caches").glob(
            "*_ensemble_reactivation.npz"
        )
    )
    assert cache_files, "No cache files found"

    all_mice: List[EnsembleSessionResult] = []

    baseline_multiplier = 1
    for cache_file in cache_files:

        mouse, date = cache_file.stem.split("_")[:2]
        mouse = mouse.strip("suite2p")  # dunno why this is in the path lol
        if mouse not in {"JB030", "JB031"}:
            continue
        data = np.load(cache_file, allow_pickle=True)
        (
            reactivation_strength,
            reactivation_strength_shuffled,
            preactivation_strength,
            preactivation_strength_shuffled,
            ensemble_matrix,
        ) = (
            data["reactivation_strength"],
            data["reactivation_strength_shuffled"],
            data["preactivation_strength"],
            data["preactivation_strength_shuffled"],
            data["ensemble_matrix"],
        )

        if ensemble_matrix.shape[1] < 5:
            print(
                f"Skipping {mouse} {date} as there are only {ensemble_matrix.shape[1]} components"
            )
            continue

        significant_reactivation = (
            reactivation_strength > reactivation_strength_shuffled * baseline_multiplier
        )
        significant_preactivation = (
            preactivation_strength
            > preactivation_strength_shuffled * baseline_multiplier
        )

        only_significant_reactivation = reactivation_strength.copy()
        only_significant_preactivation = preactivation_strength.copy()

        only_significant_reactivation[~significant_reactivation] = np.nan
        only_significant_preactivation[~significant_preactivation] = np.nan

        reactivation_number_of_events = get_reactivation_number_of_events(
            reactivation_strength=reactivation_strength,
            preactivation_strength=preactivation_strength,
            reactivation_strength_baseline=reactivation_strength_shuffled
            * baseline_multiplier,
            preactivation_strength_baseline=preactivation_strength_shuffled
            * baseline_multiplier,
            plot=False,
        )

        reactivation_strength_sum_over_threshold = get_reactivation_strength_sum(
            reactivation_strength=only_significant_reactivation,
            preactivation_strength=only_significant_preactivation,
        )

        reactivation_triggered_response = get_reactivation_triggered_averages(
            reactivation_strength=reactivation_strength,
            reactivation_strength_baseline=reactivation_strength_shuffled
            * baseline_multiplier,
            preactivation_strength=preactivation_strength,
            preactivation_strength_baseline=preactivation_strength_shuffled
            * baseline_multiplier,
        )

        all_mice.append(
            EnsembleSessionResult(
                reactivation_triggered_response=reactivation_triggered_response,
                number_of_events=reactivation_number_of_events,
                sum_values_over_threshold=reactivation_strength_sum_over_threshold,
            )
        )
    all_mouse_plots(all_mice=all_mice)


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


if __name__ == "__main__":
    # This is the file that's saved by the full grosmark oasis preprocessing as a flag
    valid_sessions = list(TIFF_UMBRELLA.rglob("full_grosmark_oasis_preprocessed.npy"))

    for session_idx in range(len(valid_sessions)):
        session_path = valid_sessions[session_idx]
        mouse = session_path.parts[-4]
        date = session_path.parts[-5]

        # Bad imaging, replace eventually with column
        # in spreadsheet, or filter by number of place cells
        if mouse in {"JB033", "JB032"}:
            continue

        main(mouse, date, plot=False)
