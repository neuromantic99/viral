import numpy as np
import sys
import time
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore, rankdata
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
from viral.utils import (
    get_wheel_circumference_from_rig,
    remove_diagonal,
    shuffle_rows,
    shuffle,
)
from viral.imaging_utils import (
    activity_trial_position,
    trial_is_imaged,
    get_preactivation_reactivation,
)
from viral.grosmark_analysis import (
    binarise_spikes,
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
    matrix w, with rows corresponding to PCs and columns corresponding to
    significant ICA components.' (Grosmark et al., 2021)

    Used description in '2.2. Determination of the number of cell assemblies', Lopes-dos-Santos, Ribeiro and Tort, 2013.
    """
    n_components, n_timepoints = ica_comp.shape

    ica_comp_z = zscore(ica_comp, axis=1, ddof=1)
    assert np.allclose(np.var(ica_comp_z, axis=1, ddof=1), 1.0, rtol=1e-5)
    np.allclose(np.var(ica_comp_z, axis=1, ddof=1), 1.0)
    corr_matrix = np.corrcoef(ica_comp_z)
    eigenvalues = eigvalsh(corr_matrix)

    # compute threshold using Marcenko-Pastur distribution (lambda max)
    q = n_timepoints / n_components  # n_cols / n_rows
    lambda_max = (1 + 1 / np.sqrt(q)) ** 2
    return np.where(eigenvalues > lambda_max)[0]


def compute_ICA_components(ssp_vectors: np.ndarray) -> np.ndarray:
    """Subsequently, the ICA components of these smoothed run
    activity estimates were calculated across time using the fastICA algorithm. Only
    the subset of components found to be significant under the Marcenko-Pasteur
    distribution were considered ICA ensembles and included in the ICA ensemble
    matrix w, with rows corresponding to PCs and columns corresponding to
    significant ICA components."""
    ica_transfomer = FastICA()
    # expected input: (n_samples, n_features)
    # independent components (frames, components)
    independent_components = ica_transfomer.fit_transform(ssp_vectors.T)
    # mixing matrix (n_features, n_components)
    mixing_matrix = ica_transfomer.mixing_
    significant_components = detect_significant_components(ica_transfomer.components_)
    return mixing_matrix[:, significant_components]


def get_offline_activity_matrix(reactivation: np.ndarray) -> np.ndarray:
    """Offline reactivation was assessed from the 150-ms Gaussian kernel convolved offline activity matrix Z."""
    sigma = 150 / 1000 * 30  # 150 ms kernel
    return np.apply_along_axis(gaussian_filter1d, axis=1, arr=reactivation, sigma=sigma)


def offline_reactivation(
    reactivation: np.ndarray, ensemble_matrix: np.ndarray
) -> np.ndarray:
    """
    For each component, b, of ICA ensemble matrix w, a
    square projection matrix, P, was computed from wb as follows:
    Pb = wb * wbT
    Where T denotes the transpose operator. Subsequently, the diagonal of the
    projection matrix P was set to zero to exclude each cell'2s individual firing rate
    variance.
    Offline reactivation was assessed from the 150-ms Gaussian kernel convolved offline activity matrix Z.
    For the ith time point (frame) in Z, the reactivation strength Rb,i
    of the bth ICA component was calculated as the square of the projection length of Zi on Pb as follows:
    Rbi = ZiT * Pb * Zb
    """
    offline_activity_matrix = get_offline_activity_matrix(reactivation=reactivation)
    n_components = ensemble_matrix.shape[1]
    n_timepoints = reactivation.shape[1]
    reactivation_strength = np.zeros((n_components, n_timepoints))
    for b in range(n_components):
        wb = ensemble_matrix[:, b]  # shape: (place cells,)
        pb = np.outer(wb, wb)
        np.fill_diagonal(pb, 0)  # shape: (place cells, place cells)
        # pb = remove_diagonal(pb)
        for i in range(n_timepoints):
            zi = offline_activity_matrix[:, i]  # shape: (place cells,)
            # is Z_b a typo in the Grosmark paper? It would be ZiT * Pb * Zi
            projection = zi.T @ pb @ zi
            reactivation_strength[b, i] = projection
    return reactivation_strength


def compute_pcc_scores(
    reactivation: np.ndarray, ensemble_matrix: np.ndarray
) -> np.ndarray:
    """
    To assess the xth cell's contribution to ICA reactivation, a PCC score was defined as the mean across all components b and
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
    return np.array(pcc_scores)


def get_normalised_pcc_scores(
    reactivation: np.ndarray, preactivation: np.ndarray, ensemble_matrix: np.ndarray
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
    # n_cells = reactivation.shape[0]

    post_ranks = rankdata(post_pcc_scores)
    pre_ranks = rankdata(pre_pcc_scores)

    # TODO: No normalisation? 'a normalized PCC score was taken per session as the pre to post change in within-epoch PCC rank.'
    # pre_ranks_norm = (pre_ranks - 1) / (n_cells - 1)
    # post_ranks_norm = (post_ranks - 1) / (n_cells - 1)

    return post_ranks - pre_ranks


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


def classify_and_sort_place_cells(
    ensemble_matrix: np.ndarray, top_ensembles: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """
    Panel (iii) shows the ICA component for each PC, with dashed lines separating those cells with large weights
    in ensemble A (top), ensemble B (middle) or neither (bottom; for the purposes of this illustration, large
    weights were those ≥1 s.d. above the mean for each template).
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
    return sorted_indices, len(a_only_sorted), len(b_only_sorted)


def plot_ensemble_reactivation(
    reactivation_strength: np.ndarray,
    top_ensembles: np.ndarray,
    shuffled: bool,
    smooth: bool = False,
) -> None:
    # TODO: remove the shuffled tag, this is dumb
    """
    Producing Fig. 4j, panel II. Plotting reactivation time courses for specified ensembles.
    """
    colours = ["r", "b"]
    reactivation_strength_z_scored = zscore(reactivation_strength, axis=1)
    if smooth:
        reactivation_strength_z_scored = gaussian_filter1d(
            reactivation_strength_z_scored, 30
        )
    plt.figure(figsize=(14, 4))
    for i, idx in enumerate(top_ensembles):
        plt.plot(
            reactivation_strength_z_scored[idx, :],
            color=colours[i],
            label=f"ensemble {i}",
        )
    plt.ylim(-1.5, 7)
    plt.xlabel("Time (frames)")
    plt.ylabel("Reactivation strength (zscored)")
    plt.tight_layout()
    # TODO: change back!
    file_name = (
        f"plots/{mouse}_{date}_ensemble_reactivation_shuffled.svg"
        if shuffled
        else f"plots/{mouse}_{date}_ensemble_reactivation.svg"
    )
    plt.savefig(file_name, dpi=300)


def plot_cell_weights(ensemble_matrix: np.ndarray, top_ensembles: np.ndarray) -> None:
    """
    Producing Fig. 4j, panel III. Plotting each cell's weight in the top ICA components/ensembles.
    """
    colours = ["r", "b"]
    sorted_cells, n_a, n_b = classify_and_sort_place_cells(
        ensemble_matrix, top_ensembles
    )
    plt.figure()
    for i, idx in enumerate(top_ensembles):
        plt.plot(
            ensemble_matrix[sorted_cells, idx],
            # marker="o",
            # linestyle="",
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
    # we changed the place cell no through sorting, should it be place cell ID as in Grosmark et al.?
    plt.xlabel("Place cell ID")
    plt.ylabel("Cell weight in template")
    plt.tight_layout()
    plt.savefig(f"plots/{mouse}_{date}_cell_weights.svg", dpi=300)


def plot_smoothed_offline_firing_rate_raster(reactivation: np.ndarray) -> None:
    """
    Producing Fig. 4j, panel IV. Plotting the smoothed offline firing rate raster.
    """
    # TODO: make this a dataclass or some kind of argument
    sorted_cells, n_a, n_b = classify_and_sort_place_cells(
        ensemble_matrix, top_ensembles
    )
    offline_activity_matrix = get_offline_activity_matrix(reactivation=reactivation)
    # TODO: axis labels!
    plt.imshow(
        offline_activity_matrix[sorted_cells, :],
        cmap="gray_r",
        aspect="auto",
        # vmin=0,
        # vmax=np.percentile(offline_activity_matrix, 99),
    )
    plt.ylabel("Place cell ID")
    plt.xlabel("Frames")
    plt.tight_layout()
    plt.savefig(
        f"plots/{mouse}_{date}_smoothed_offline_firing_rate_raster.svg", dpi=300
    )


# def main():
if __name__ == "__main__":
    mouse = "JB031"
    date = "2025-03-28"

    verbose = True
    use_cache = True

    with open(HERE.parent / "data" / "cached_2p" / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())
    if not session.wheel_freeze:
        print(f"Skipping {date} for mouse {mouse} as there was no wheel block")

    cache_file = HERE / f"{session.mouse_name}_{session.date}_place_cells.npy"
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
    spks_raw, _, _, _ = load_data(
        session,
        TIFF_UMBRELLA / session.date / session.mouse_name / "suite2p" / "plane0",
        "spks",
    )
    # Based on the observed differences in calcium activity waveforms between the online and
    # offline epochs (Supplementary Fig. 2), a threshold of 1.5 m.a.d. was used for online running epochs,
    # while a lower threshold of 1.25 m.a.d. were used for offline immobility epochs.
    online_spks = binarise_spikes(
        spks_raw[
            :,
            session.wheel_freeze.pre_training_end_frame : session.wheel_freeze.post_training_start_frame,
        ],
        mad_threshold=1.5,
    )
    offline_spks_pre, offline_spks_post = get_preactivation_reactivation(
        flu=spks_raw, wheel_freeze=session.wheel_freeze
    )
    # TODO: should pre and post be binarised as one??
    offline_spks_pre = binarise_spikes(
        offline_spks_pre,
        mad_threshold=1.25,
    )
    offline_spks_post = binarise_spikes(offline_spks_post, mad_threshold=1.25)
    spks = np.hstack([offline_spks_pre, online_spks, offline_spks_post])
    # TODO: shape is off by one frame (pre is one short), do we care?
    # assert spks_raw.shape == spks.shape
    if not use_cache:
        t1 = time.time()
        # TODO: why is place cells now soooo low (23 vs 101 before??)
        # TODO: probably slicing issue??? Put back the pre-period!!!
        pcs_mask = get_place_cell_mask(
            session=session, spks=spks, rewarded=None, config=config
        )
        print(f"Time to get place cells: {time.time() - t1}")

    else:
        print("Using cached pcs mask")
        pcs_mask = np.load(cache_file)
    place_cells = spks[pcs_mask, :]
    reactivation = offline_spks_post[pcs_mask]
    preactivation = offline_spks_pre[pcs_mask]
    # ONLINE
    running_bouts = get_running_bouts(
        place_cells=place_cells,
        speed=speed,
        frames_positions=positions,
        ITI_starts_ends=ITI_starts_ends,
        aligned_trial_frames=aligned_trial_frames,
    )
    print("Got running bouts")
    ssp_vectors = get_ssp_vectors(place_cells_running=running_bouts)
    print("Got ssp vectors")
    ensemble_matrix = compute_ICA_components(ssp_vectors=ssp_vectors)
    print("Got ensemble matrix")
    if verbose:
        print(f"# place cells: {running_bouts.shape[0]}")
        print(f"# significant components (Marcenko-Pastur): {ensemble_matrix.shape[1]}")
    # OFFLINE
    reactivation_strength = offline_reactivation(
        reactivation=reactivation, ensemble_matrix=ensemble_matrix
    )
    preactivation_strength = offline_reactivation(
        reactivation=preactivation, ensemble_matrix=ensemble_matrix
    )
    pcc_scores = get_normalised_pcc_scores(
        reactivation=reactivation,
        preactivation=preactivation,
        ensemble_matrix=ensemble_matrix,
    )
    # TODO: Move or remove eventually
    plt.figure()
    plt.plot(pcc_scores)
    plt.savefig("plots/pcc_scores.svg", dpi=300)
    plt.close()

    # """ICA components were shuffled by randomly permuting the weight matrix w across
    # PCs and recalculating the reactivation strength."""
    # TODO: Where do they use the shuffled ICA components???
    ensemble_matrix_shuffled = np.copy(ensemble_matrix)
    # ensemble_matrix_shuffled = shuffle_rows(ensemble_matrix)
    ensemble_matrix_shuffled = shuffle(ensemble_matrix)
    # np.random.shuffle(ensemble_matrix_shuffled)
    reactivation_strength_shuffled = offline_reactivation(
        reactivation=reactivation, ensemble_matrix=ensemble_matrix_shuffled
    )
    top_ensembles = sort_ensembles_by_reactivation_strength(
        reactivation_strength=reactivation_strength
    )
    sorted_place_cells = classify_and_sort_place_cells(
        ensemble_matrix=ensemble_matrix, top_ensembles=top_ensembles
    )
    # TODO: Change back
    plot_ensemble_reactivation(
        reactivation_strength=reactivation_strength,
        top_ensembles=top_ensembles,
        shuffled=False,
        smooth=True,
    )
    plot_ensemble_reactivation(
        reactivation_strength=reactivation_strength_shuffled,
        top_ensembles=top_ensembles,
        shuffled=True,
        smooth=True,
    )
    plot_cell_weights(ensemble_matrix=ensemble_matrix, top_ensembles=top_ensembles)
    plot_smoothed_offline_firing_rate_raster(reactivation=reactivation)


# if __name__ == "__main__":
#     main()
