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

from viral.constants import TIFF_UMBRELLA
from viral.models import Cached2pSession, GrosmarkConfig
from viral.rastermap_utils import (
    load_data,
    align_validate_data,
    process_trials_data,
    filter_speed_position,
    filter_out_ITI,
)
from viral.utils import get_wheel_circumference_from_rig, remove_diagonal
from viral.grosmark_analysis import binarise_spikes, grosmark_place_field


def process_behaviour(
    # TODO: add args and return
    session: Cached2pSession,
    wheel_circumference: float,
    speed_bin_size: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s2p_path = TIFF_UMBRELLA / session.date / session.mouse_name / "suite2p" / "plane0"
    print(f"Working on {session.mouse_name}: {session.date} - {session.session_type}")
    signal, xpos, ypos, trials = load_data(session, s2p_path, "spks")
    aligned_trial_frames, neural_data_trials = align_validate_data(signal, trials)
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
    ITI_starts_ends = np.array(
        [[trial.iti_start_frame, trial.iti_end_frame] for trial in imaged_trials_infos]
    )
    neural_data = np.concatenate(
        [trial.neural_data for trial in imaged_trials_infos], axis=1
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
    return positions, speed, ITI_starts_ends, aligned_trial_frames


def get_running_bouts(
    place_cells: np.ndarray,
    speed: np.ndarray,
    frames_positions: np.ndarray,
    ITI_starts_ends: np.ndarray,
    aligned_trial_frames: np.ndarray,
) -> np.ndarray:
    # TODO: do we want to keep the ITI out?
    # TODO: do we want to have start end and stuff same as in grosmark config??
    # TODO: what are missing frames here? not imaged trials?
    trial_activities = list()
    for start, end, _ in aligned_trial_frames:
        trial_activity = place_cells[:, start : end + 1]
        trial_activities.append(trial_activity)
    place_cells_behaviour = np.concatenate(trial_activities, axis=1)
    behaviour_mask = filter_speed_position(
        speed=speed,
        frames_positions=frames_positions,
        speed_threshold=1,  # cm/s
        position_threshold=None,
    ) & filter_out_ITI(ITI_starts_ends=ITI_starts_ends, positions=frames_positions)
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


def main():
    # Load stuff from rastermap utils
    # get place cells
    # do rest
    mouse = "JB031"
    date = "2025-03-07"

    with open(HERE.parent / "data" / "cached_2p" / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())
    positions, speed, ITI_starts_ends, aligned_trial_frames = process_behaviour(
        session,
        wheel_circumference=get_wheel_circumference_from_rig(
            "2P",
        ),
    )
    config = config = GrosmarkConfig(
        bin_size=2,
        start=30,
        end=160,
    )
    spks, _, _, _ = load_data(
        session,
        TIFF_UMBRELLA / session.date / session.mouse_name / "suite2p" / "plane0",
        "spks",
    )
    pcs_mask = grosmark_place_field(
        session=session, spks=spks, rewarded=None, config=config
    )
    place_cells = spks[pcs_mask, :]
    running_bouts = get_running_bouts(
        place_cells=place_cells,
        speed=speed,
        frames_positions=positions,
        ITI_starts_ends=ITI_starts_ends,
        aligned_trial_frames=aligned_trial_frames,
    )
    ssp_vectors = get_ssp_vectors(place_cells_running=running_bouts)
    ensemble_matrix = compute_ICA_components(ssp_vectors=ssp_vectors)
    square_projection_matrix = square_projection_matrix(arr=ensemble_matrix)


if __name__ == "__main__":
    main()
