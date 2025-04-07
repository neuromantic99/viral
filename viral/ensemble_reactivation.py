import numpy as np
import sys
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
from numpy.linalg import eigvalsh
from sklearn.decomposition import FastICA

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.rastermap_utils import filter_speed_position, filter_out_ITI
from viral.utils import remove_diagonal


def get_running_bouts(
    place_cells: np.ndarray,
    speed: np.ndarray,
    frames_positions: np.ndarray,
    ITI_starts_ends: np.ndarray,
) -> np.ndarray:
    # TODO: do we want to keep the ITI out?
    valid_frames = filter_speed_position(
        speed=speed,
        frames_positions=frames_positions,
        speed_threshold=1,  # cm/s
        position_threshold=None,
    ) & filter_out_ITI(ITI_starts_ends=ITI_starts_ends, positions=frames_positions)
    return place_cells[:, valid_frames]


def get_ssp_vectors(
    place_cells: np.ndarray,
    speed: np.ndarray,
    frames_positions: np.ndarray,
    ITI_starts_ends: np.ndarray,
) -> np.ndarray:
    """Briefly, PC run running-bout spike
    estimate vectors, Ssp, were convolved with a 1-s Gaussian kernel corresponding to
    behavioral timescales."""
    # TODO: change the args
    place_cells_running = get_running_bouts(
        place_cells=place_cells,
        speed=speed,
        frames_positions=frames_positions,
        ITI_starts_ends=ITI_starts_ends,
    )
    sigma = 30  # 1 second * 30 Hz sampling rate
    return np.apply_along_axis(
        gaussian_filter1d, axis=1, arr=place_cells_running, sigma=sigma
    )


def detect_significant_components(ica_comp: np.ndarray) -> np.ndarray:
    # TODO: add args and return
    """
    Use Marcenko-Pastur distribution to detect significant compontents.

    'Only the subset of components found to be significant under the Marcenko–Pasteur
    distribution were considered ICA ensembles and included in the ICA ensemble
    matrix¸ w, with rows corresponding to PCs and columns corresponding to
    significant ICA components.' (Grosmark et al., 2021)

    Used description in '2.2. Determination of the number of cell assemblies', Lopes-dos-Santos, Ribeiro and Tort, 2013.
    """
    # TODO: check that all of these operations are correct
    n_components, n_timepoints = ica_comp.shape

    ica_comp_z = zscore(ica_comp, axis=1, ddof=1)
    # TODO: sigma2 = 1!!!
    corr_matrix = np.corrcoef(ica_comp_z)
    eigenvalues = eigvalsh(corr_matrix)

    # compute threshold using Marcenko-Pastur distribution (lambda max)
    q = n_components / n_timepoints
    lambda_max = (1 + 1 / np.sqrt(q)) ** 2
    return np.where(eigenvalues > lambda_max)[0]


def compute_ICA_components(ssp_vectors: np.ndarray) -> np.ndarray:
    """Subsequently, the ICA components of these smoothed run
    activity estimates were calculated across time using the fastICA algorithm. Only
    the subset of components found to be significant under the Marcenko–Pasteur
    distribution were considered ICA ensembles and included in the ICA ensemble
    matrix¸ w, with rows corresponding to PCs and columns corresponding to
    significant ICA components."""
    ica_transfomer = FastICA()
    ensemble_matrix = ica_transfomer.fit_transform(ssp_vectors)
    significant_components = detect_significant_components(ica_transfomer.components_.T)
    return ensemble_matrix[:, significant_components]


def square_projection_matrix(arr: np.ndarray) -> np.ndarray:
    """For each component, b, of ICA ensemble matrix w, a
    square projection matrix, P, was computed from wb as follows:
    Pb = wb * wbT
    Where T denotes the transpose operator. Subsequently, the diagonal of the
    projection matrix P was set to zero to exclude each cell’s individual firing rate
    variance."""
    square_projection = np.matmul(arr, arr.T)
    return remove_diagonal(square_projection)
