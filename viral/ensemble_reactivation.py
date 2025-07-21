from typing import List
import numpy as np
import sys
import os
import time
from pathlib import Path
from matplotlib import pyplot as plt
from deprecated import deprecated
from scipy.linalg import fractional_matrix_power, subspace_angles
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore, rankdata
from opt_einsum import contract
import seaborn as sns
from tqdm import tqdm

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.constants import CACHE_PATH, TIFF_UMBRELLA
from viral.models import Cached2pSession, GrosmarkConfig, SortedPlaceCells, TrialInfo
from viral.rastermap_utils import (
    get_frame_position,
    get_speed_frame,
    load_data,
    align_validate_data,
    process_trials_data,
    filter_speed_position,
)
from viral.utils import (
    degrees_to_cm,
    get_wheel_circumference_from_rig,
    shuffle_rows,
    trial_is_imaged,
)
from viral.imaging_utils import get_ITI_start_frame, get_frozen_wheel_flu
from viral.grosmark_analysis import binarise_spikes, get_place_cells


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


def get_ssp_vectors(
    place_cells_running: np.ndarray,
) -> np.ndarray:
    """Briefly, PC run running-bout spike
    estimate vectors, Ssp, were convolved with a 1-s Gaussian kernel corresponding to
    behavioral timescales."""
    sigma = 30  # 1 second * 30 Hz sampling rate
    return np.apply_along_axis(
        gaussian_filter1d, axis=1, arr=place_cells_running, sigma=sigma
    )


@deprecated("Use fast_ica_sklearn instead, tested as its the same")
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

    # Transpose to match your return shape: (n_cells, n_components)
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
        raise ValueError("There are no significant components!")

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
    # TODO:
    # is Z_b a typo in the Grosmark paper? It would be ZiT * Pb * Zi

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
    colours = ["r", "b"]
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
    colours = ["r", "b"]
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
    colours = ["r", "b"]
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
    1 / 0
    # plt.show()


def main() -> None:
    mouse = "JB036"
    date = "2025-07-05"

    verbose = True
    use_cache = False

    with open(CACHE_PATH / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())

    print(f"Working on {session.mouse_name}: {session.date} - {session.session_type}")

    if not session.wheel_freeze:
        print(f"Skipping {date} for mouse {mouse} as there was no wheel block")
        return

    cache_file = (
        HERE.parent
        / f"{session.mouse_name}suite2p_{session.date}_ensemble_reactivation.npz"
    )

    if use_cache and cache_file.exists():
        (
            pcs_mask,
            ensemble_matrix,
            ensemble_matrix_shuffled_data,
            reactivation_strength,
            reactivation_strength_shuffled,
            preactivation_strength,
            preactivation_strength_shuffled,
            reactivation,
            preactivation,
            running_bouts,
            pcc_scores,
        ) = load_data_from_cache(cache_file)
    else:
        print("No cached data found, processing data")
        # TODO: think about speed_bin_size -> set to 30 i.e. 1s (30 fps)
        config = GrosmarkConfig(
            bin_size=5,
            start=0,
            end=170,
        )
        spks_raw, _, _, _ = load_data(
            session,
            # AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
            Path("/Volumes/hard_drive/VR-2p/2025-03-25/JB031/suite2p/plane0"),
            # TIFF_UMBRELLA / session.date / session.mouse_name / "suite2p" / "plane0",
            "spks",
        )

        # """Based on the observed differences in calcium activity waveforms between the online and
        # offline epochs (Supplementary Fig. 2), a threshold of 1.5 m.a.d. was used for online running epochs,
        # while a lower threshold of 1.25 m.a.d. were used for offline immobility epochs."""

        online_spks = binarise_spikes(
            spks_raw[
                :,
                session.wheel_freeze.pre_training_end_frame : session.wheel_freeze.post_training_start_frame,
            ],
            mad_threshold=1.5,
        )
        offline_spks_pre = binarise_spikes(
            spks_raw[:, : session.wheel_freeze.pre_training_end_frame],
            mad_threshold=1.25,
        )

        offline_spks_post = binarise_spikes(
            spks_raw[:, session.wheel_freeze.post_training_start_frame :],
            mad_threshold=1.25,
        )
        spks = np.hstack([offline_spks_pre, online_spks, offline_spks_post])

        assert spks_raw.shape == spks.shape
        t1 = time.time()
        pcs_mask, _ = get_place_cells(
            session=session, spks=spks, rewarded=None, config=config, plot=True
        )
        print(f"Time to get place cells: {time.time() - t1}")
        place_cells = spks[pcs_mask, :]
        reactivation = offline_spks_post[pcs_mask]
        preactivation = offline_spks_pre[pcs_mask]

        ## Taken out while we figure out how best to do this
        # running_bouts = get_running_bouts(
        #     place_cells=place_cells,
        #     speed=speed,
        #     frames_positions=positions,
        #     aligned_trial_frames=aligned_trial_frames,
        #     config=config,
        # )

        trials = [trial for trial in session.trials if trial_is_imaged(trial)]

        sigma = 30
        # Doing the gaussian filter on a trial by trial basis to prevent edge effects
        # Pretty stupid doing this in a oneliner
        ssp_vectors = np.hstack(
            [
                np.apply_along_axis(
                    gaussian_filter1d,
                    axis=1,
                    arr=place_cells[
                        :, trial.trial_start_closest_frame : get_ITI_start_frame(trial)
                    ],
                    sigma=sigma,
                )
                for trial in trials
            ]
        )

        ssp_vectors_shuffled = shuffle_rows(ssp_vectors)
        # TODO: are we returning the right thing here?
        ensemble_matrix = compute_ICA_components(ssp_vectors=ssp_vectors)
        ensemble_matrix_shuffled_data = compute_ICA_components(
            ssp_vectors=ssp_vectors_shuffled
        )
        print("ICA done")
        # OFFLINE
        t2 = time.time()
        reactivation_strength = offline_reactivation(
            reactivation=reactivation, ensemble_matrix=ensemble_matrix
        )
        preactivation_strength = offline_reactivation(
            reactivation=preactivation, ensemble_matrix=ensemble_matrix
        )

        total_reactivation = np.sum(reactivation_strength, axis=1)
        total_preactivation = np.sum(preactivation_strength, axis=1)

        # Plot the component changes

        # TODO: get rid of the "do_shuffle" argument
        preactivation_shuffled = shuffle_rows(preactivation)
        reactivation_shuffled = shuffle_rows(reactivation)
        reactivation_strength_shuffled = offline_reactivation(
            reactivation=reactivation_shuffled,
            ensemble_matrix=ensemble_matrix_shuffled_data,
        )
        preactivation_strength_shuffled = offline_reactivation(
            reactivation=preactivation_shuffled,
            ensemble_matrix=ensemble_matrix_shuffled_data,
        )
        print(f"Time to get reactivation strength(s): {time.time() - t2}")
        t3 = time.time()
        pcc_scores = get_normalised_pcc_scores(
            reactivation=reactivation,
            preactivation=preactivation,
            ensemble_matrix=ensemble_matrix,
        )
        print(f"Time to get PCC scores: {time.time() - t3}")

        # TODO: should the top ensembles be the same for reactivation and preactivation?
        # I mean we would look at the reactivation strength of the same ensembles?

        print("Saving cache")
        np.savez(
            cache_file,
            pcs_mask=pcs_mask,
            ensemble_matrix=ensemble_matrix,
            ensemble_matrix_shuffled_data=ensemble_matrix_shuffled_data,
            reactivation_strength=reactivation_strength,
            reactivation_strength_shuffled=reactivation_strength_shuffled,
            preactivation_strength=preactivation_strength,
            preactivation_strength_shuffled=preactivation_strength_shuffled,
            reactivation=reactivation,
            preactivation=preactivation,
            # running_bouts=,
            pcc_scores=pcc_scores,
        )
    if verbose:
        # print(f"# frames with running: {running_bouts.shape[1]}")
        # print(f"# place cells: {running_bouts.shape[0]}")
        print(f"# significant components (Marcenko-Pastur): {ensemble_matrix.shape[1]}")
        # TODO: remove eventually?
        print(
            f"# significant components (Marcenko-Pastur), shuffled data: {ensemble_matrix_shuffled_data.shape[1]}"
        )

    top_ensembles = sort_ensembles_by_reactivation_strength(
        reactivation_strength=reactivation_strength
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
        smooth=False,
    )

    total_reactivation = np.sum(reactivation_strength, axis=1)
    total_preactivation = np.sum(preactivation_strength, axis=1)
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
    plt.title(f"Mean change = {np.mean(total_reactivation - total_preactivation):.2f}")
    plt.show()


def load_data_from_cache(cache_file: Path) -> tuple:
    print("Using cached data")
    cache = np.load(cache_file, allow_pickle=True)
    return (
        cache["pcs_mask"],
        cache["ensemble_matrix"],
        cache["ensemble_matrix_shuffled_data"],
        cache["reactivation_strength"],
        cache["reactivation_strength_shuffled"],
        cache["preactivation_strength"],
        cache["preactivation_strength_shuffled"],
        cache["reactivation"],
        cache["preactivation"],
        None,
        # cache["running_bouts"],
        cache["pcc_scores"],
    )


def plot_speed_and_position_logic(trials: List[TrialInfo]) -> None:
    """Checking logic of get_speed_frame and get_frame_position. Delete this later."""
    for trial in trials:
        trial_frames = np.arange(
            trial.trial_start_closest_frame, trial.trial_end_closest_frame + 1, 1
        )
        fig, ax1 = plt.subplots(figsize=(10, 5))
        position = get_frame_position(
            trial, trial_frames, get_wheel_circumference_from_rig("2P")
        )
        possy = degrees_to_cm(
            np.array(trial.rotary_encoder_position),
            get_wheel_circumference_from_rig("2P"),
        )
        (p1,) = ax1.plot(
            possy,
            color="black",
            label="rotary encoder position",
        )

        (p2,) = ax1.plot(position[:, 1], color="blue", label="frame position")

        speed = get_speed_frame(position, 30)[:, 1]
        ax2 = ax1.twinx()
        (p3,) = ax2.plot(speed, color="orange", label="speed")
        lines = [p1, p2, p3]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="upper left")


def compare_run_results(W_py: np.ndarray, W_mat: np.ndarray) -> None:
    """There is no guarentee that two runs of ICA (particularly from different programming languages with different random seeds)
    will return the same:
        numerical values
        column order
        or even sign (i.e. the same component may be positive or negative)

    However their absolute sums should be similar. And the actual components (once sorted and the signs aligned)
    should span the same subspace. You can test this by looking at the angles between two components. They should be 0 (within floating point error)

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


if __name__ == "__main__":
    compare_to_matlab()
