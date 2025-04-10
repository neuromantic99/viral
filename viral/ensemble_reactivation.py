import numpy as np
import sys
from pathlib import Path
from matplotlib import pyplot as plt
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
from viral.imaging_utils import (
    activity_trial_position,
    trial_is_imaged,
    get_reactivation,
)
from viral.grosmark_analysis import (
    binarise_spikes,
    grosmark_place_field,
    has_five_consecutive_trues,
    filter_additional_check,
)


def get_place_cell_mask(
    session: Cached2pSession,
    spks: np.ndarray,
    rewarded: bool | None,
    config: GrosmarkConfig,
) -> np.ndarray:
    """Returns a boolean mask of place cells among all cells in spks.
    Compare to grosmark_place_field."""

    n_cells_total = spks.shape[0]
    sigma_cm = 7.5
    sigma_bins = sigma_cm / config.bin_size
    n_shuffles = 2000

    all_trials = np.array(
        [
            activity_trial_position(
                trial=trial,
                flu=spks,
                wheel_circumference=get_wheel_circumference_from_rig("2P"),
                bin_size=config.bin_size,
                start=config.start,
                max_position=config.end,
                verbose=False,
                do_shuffle=False,
                smoothing_sigma=sigma_bins,
            )
            for trial in session.trials
            if trial_is_imaged(trial)
            and (rewarded is None or trial.texture_rewarded == rewarded)
        ]
    )

    smoothed_matrix = np.mean(all_trials, axis=0)

    shuffled_matrices = np.array(
        [
            np.mean(
                np.array(
                    [
                        activity_trial_position(
                            trial=trial,
                            flu=spks,
                            wheel_circumference=get_wheel_circumference_from_rig("2P"),
                            bin_size=config.bin_size,
                            start=config.start,
                            max_position=config.end,
                            verbose=False,
                            do_shuffle=True,
                            smoothing_sigma=sigma_bins,
                        )
                        for trial in session.trials
                        if trial_is_imaged(trial)
                        and (rewarded is None or trial.texture_rewarded == rewarded)
                    ]
                ),
                axis=0,
            )
            for _ in range(n_shuffles)
        ]
    )

    # Threshold (99th percentile of shuffled matrices)
    place_threshold = np.percentile(shuffled_matrices, 99, axis=0)

    # filter firing above threshold in at least 5 consecutive bins
    initial_pcs = has_five_consecutive_trues(smoothed_matrix > place_threshold)

    # apply lap-based filter
    final_pcs = filter_additional_check(
        all_trials=all_trials[:, initial_pcs, :],
        place_threshold=place_threshold[initial_pcs, :],
        smoothed_matrix=smoothed_matrix[initial_pcs, :],
    )

    # final mask in original cell template
    place_cell_mask = np.zeros(n_cells_total, dtype=bool)
    place_cell_mask[np.flatnonzero(initial_pcs)] = final_pcs
    return place_cell_mask


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
    ica_transfomer.fit_transform(ssp_vectors)
    significant_components = detect_significant_components(ica_transfomer.components_.T)
    component_matrix = ica_transfomer.components_.T
    return component_matrix[:, significant_components]


def square_projection_matrix(arr: np.ndarray) -> np.ndarray:
    """For each component, b, of ICA ensemble matrix w, a
    square projection matrix, P, was computed from wb as follows:
    Pb = wb * wbT
    Where T denotes the transpose operator. Subsequently, the diagonal of the
    projection matrix P was set to zero to exclude each cell’s individual firing rate
    variance."""
    square_projection = np.matmul(arr, arr.T)
    return remove_diagonal(square_projection)


def get_offline_activity_matrix(reactivation: np.ndarray) -> np.ndarray:
    """Offline reactivation was assessed from the 150-ms Gaussian kernel convolved offline activity matrix Z."""
    sigma = 150 / 1000 * 30  # 150 ms kernel
    return np.apply_along_axis(gaussian_filter1d, axis=1, arr=reactivation, sigma=sigma)


def offline_reactivation(
    reactivation: np.ndarray, ensemble_matrix: np.ndarray
) -> np.ndarray:
    """Offline reactivation was assessed from the 150-ms Gaussian kernel convolved offline activity matrix Z.
    For the ith time point (frame) in Z, the reactivation strength Rb,i
    of the bth ICA component was calculated as the square of the projection length of Zi on Pb as follows:
    Rbi = ZiT * Pb * Zb
    """
    offline_activity_matrix = get_offline_activity_matrix(reactivation=reactivation)
    # TODO: is this correctly getting the number of components (columns)??
    n_components = ensemble_matrix.shape[1]
    n_timepoints = reactivation.shape[1]
    reactivation_strength = np.zeros((n_components, n_timepoints))
    for t in range(n_timepoints):
        offline_activity_matrix_t = offline_activity_matrix[:, t]
        for c in range(n_components):
            projection_length = np.outer(ensemble_matrix[:, c], ensemble_matrix[:, c])
            reactivation_strength[c, t] = np.matmul(
                np.matmul(offline_activity_matrix_t.T, projection_length),
                offline_activity_matrix_t,
            )
    return reactivation_strength


def pcc_scores(reactivation: np.ndarray, ensemble_matrix: np.ndarray) -> np.ndarray:
    # TODO: add args and return
    """
    To assess the xth cell’s contribution to ICA reactivation, a PCC score was defined as the mean across all components b and
    timepoints i of the reactivation score R computed from all PCs c minus the reactivation score Rcx computed after
    the exclusion xth cell from the activity and template matrices.
    """
    n_cells = reactivation.shape[0]

    pcc_scores = list()
    for cell_idx in range(n_cells):
        # reactivation score R computer from all PCs
        R_full = offline_reactivation(
            reactivation=reactivation, ensemble_matrix=ensemble_matrix
        )

        # reactivation score Rc\x after excluding xth cell
        reactivation_excluded = np.delete(reactivation, cell_idx, axis=0)
        ensemble_matrix_excluded = np.delete(ensemble_matrix, cell_idx, axis=0)
        R_excluded = offline_reactivation(
            reactivation=reactivation_excluded, ensemble_matrix=ensemble_matrix_excluded
        )

        delta_R = R_full - R_excluded
        # mean across all components and timepoints
        pcc_scores.append(np.mean(delta_R))
    return np.array([pcc_scores])


def sort_ensembles_by_reactivation_strength(
    reactivation_strength: np.ndarray, n_top: int = 2
) -> np.ndarray:
    """
    In Fig. 4j, panel II, it is not clear how they got to their 'run ensembles' A and B.
    I assumed, they select the two ensembles with the strongest reactivation.
    """
    # TODO: should we normalise?
    total_strength = np.sum(reactivation_strength, axis=1)
    sorted_indices = np.argsort(total_strength)[::-1]
    return sorted_indices[:n_top]


def plot_ensemble_reactivation(
    reactivation_strength: np.ndarray, top_ensembles: np.ndarray
) -> None:
    """
    Producing Fig. 4j, panel II. Plotting reactivation time courses for specified ensembles.
    """
    colours = ["r", "b"]
    # TODO: is reactivation strength already z-scored?
    # TODO: do we need to specify colours?
    plt.figure()
    for i, idx in enumerate(top_ensembles):
        plt.plot(reactivation_strength[idx, :], color=colours[i], label=f"ensemble {i}")
    plt.xlabel("Time (frames)")
    plt.ylabel("Reactivation strength")
    plt.tight_layout()


def plot_cell_weights(ensemble_matrix: np.ndarray, top_ensembles: np.ndarray) -> None:
    """
    Producing Fig. 4j, panel III. Plotting each cell's weight in the top ICA components/ensembles.
    """
    # TODO: classify as having a preference
    plt.figure()
    for i, idx in enumerate(top_ensembles):
        plt.plot(ensemble_matrix[:, idx], marker="o", label=f"ensemble {i}")
    plt.xlabel("Place cell no.")
    plt.ylabel("Cell weight in template")
    plt.tight_layout()


def plot_smoothed_offline_firing_rate_raster(reactivation: np.ndarray) -> None:
    # TODO: classify as having a preference
    """
    Producing Fig. 4j, panel IV. Plotting the smoothed offline firing rate raster.
    """
    offline_activity_matrix = get_offline_activity_matrix(reactivation=reactivation)
    plt.figure()
    plt.plot(offline_activity_matrix)
    plt.tight_layout()


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
    spks = binarise_spikes(spks)
    print("Getting place cells")
    pcs_mask = get_place_cell_mask(
        session=session, spks=spks, rewarded=None, config=config
    )
    place_cells = spks[pcs_mask, :]
    reactivation = get_reactivation(flu=spks, wheel_freeze=session.wheel_freeze)[
        pcs_mask
    ]
    print("Got place cells")
    running_bouts = get_running_bouts(
        place_cells=place_cells,
        speed=speed,
        frames_positions=positions,
        ITI_starts_ends=ITI_starts_ends,
        aligned_trial_frames=aligned_trial_frames,
    )
    print("Got runninbg bouts")
    ssp_vectors = get_ssp_vectors(place_cells_running=running_bouts)
    print("Got ssp vectors")
    ensemble_matrix = compute_ICA_components(ssp_vectors=ssp_vectors)
    print("Got ensemble matrix")
    square_projection_matrix = square_projection_matrix(arr=ensemble_matrix)
    print("Got square projection matrix")
    reactivation_strength = offline_reactivation(
        reactivation=reactivation, ensemble_matrix=ensemble_matrix
    )
    pcc_scores = pcc_scores(reactivation=reactivation, ensemble_matrix=ensemble_matrix)
    # """ICA components were shuffled by randomly permuting the weight matrix w across
    # PCs and recalculating the reactivation strength."""
    ensemble_matrix_shuffled = np.copy(ensemble_matrix)
    ensemble_matrix_shuffled = np.random.shuffle(ensemble_matrix_shuffled)
    reactivation_strength_shuffled = offline_reactivation(
        reactivation=reactivation, ensemble_matrix=ensemble_matrix_shuffled
    )
    top_ensembles = sort_ensembles_by_reactivation_strength(
        reactivation_strength=reactivation_strength
    )
    plot_ensemble_reactivation(top_ensembles=top_ensembles)
    plot_cell_weights(ensemble_matrix=ensemble_matrix, top_ensembles=top_ensembles)
    plot_smoothed_offline_firing_rate_raster(reactivation=reactivation)


if __name__ == "__main__":
    main()
