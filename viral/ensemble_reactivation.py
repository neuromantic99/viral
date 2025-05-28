import numpy as np
import sys
import os
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
from viral.models import Cached2pSession, GrosmarkConfig, SortedPlaceCells
from viral.rastermap_utils import (
    load_data,
    align_validate_data,
    process_trials_data,
    filter_speed_position,
)
from viral.utils import get_wheel_circumference_from_rig, shuffle_rows
from viral.imaging_utils import (
    get_preactivation_reactivation,
)
from viral.grosmark_analysis import binarise_spikes, get_place_cells


def process_behaviour(
    session: Cached2pSession,
    wheel_circumference: float,
    speed_bin_size: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s2p_path = TIFF_UMBRELLA / session.date / session.mouse_name / "suite2p" / "plane0"
    signal, _, _, trials = load_data(session, s2p_path, "spks")
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
    return positions, speed, ITI_starts_ends, aligned_trial_frames


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
        speed_threshold=5,  # cm/s
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


def detect_significant_components(ica_comp: np.ndarray) -> np.ndarray:
    """
    Use Marcenko-Pastur distribution to detect significant compontents.

    'Only the subset of components found to be significant under the Marcenko-Pasteur
    distribution were considered ICA ensembles and included in the ICA ensemble
    matrix w, with rows corresponding to PCs and columns corresponding to
    significant ICA components.' (Grosmark et al., 2021)

    Used description in '2.2. Determination of the number of cell assemblies', Lopes-dos-Santos, Ribeiro and Tort, 2013.
    """
    n_components, n_timepoints = ica_comp.shape

    ica_comp_z = zscore(ica_comp, axis=1, ddof=1)
    assert np.allclose(np.var(ica_comp_z, axis=1, ddof=1), 1.0, rtol=1e-5)
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
    # TODO: keep random state as ICA is non-deterministic to keep results reproducible?
    ica_transfomer = FastICA(random_state=999)
    # expected input: (n_samples, n_features)
    # independent components (frames, components)
    ica_transfomer.fit_transform(ssp_vectors.T)
    # mixing matrix (n_features, n_components)
    mixing_matrix = ica_transfomer.mixing_
    # TODO: there are two ways of retrieving the components
    # 1. ica_transfomer.components_ (n_components, n_features)
    # 2. ica_transfomer.fit_transform(ssp_vectors.T) (n_samples, n_components)
    # I think number 1 is correct for our use case? it would be components, frames
    significant_components = detect_significant_components(ica_transfomer.components_)
    return mixing_matrix[:, significant_components]


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
    projection matrix P was set to zero to exclude each cell'2s individual firing rate
    variance.
    Offline reactivation was assessed from the 150-ms Gaussian kernel convolved offline activity matrix Z.
    For the ith time point (frame) in Z, the reactivation strength Rb,i
    of the bth ICA component was calculated as the square of the projection length of Zi on Pb as follows:
    Rbi = ZiT * Pb * Zb
    """
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
    n_components = ensemble_matrix.shape[1]
    n_timepoints = reactivation.shape[1]
    reactivation_strength = np.zeros((n_components, n_timepoints))
    for b in range(n_components):
        wb = ensemble_matrix[:, b]  # shape: (place cells,)
        pb = np.outer(wb, wb)  # shape: (place cells, place cells)
        np.fill_diagonal(pb, 0)  # shape: (place cells, place cells); diagonal set to 0
        for i in range(n_timepoints):
            zi = offline_activity_matrix[:, i]  # shape: (place cells,)
            # Rbi = ZiT * Pb * Zi
            reactivation_strength[b, i] = zi.T @ pb @ zi
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

    # reactivation score R computer from all PCs
    R_full = offline_reactivation(
        reactivation=reactivation, ensemble_matrix=ensemble_matrix
    )

    pcc_scores = list()
    for cell_idx in range(n_cells):
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


def main() -> None:
    mouse = "JB027"
    date = "2025-02-26"

    verbose = True
    use_cache = True

    with open(HERE.parent / "data" / "cached_2p" / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())

    print(f"Working on {session.mouse_name}: {session.date} - {session.session_type}")

    if not session.wheel_freeze:
        print(f"Skipping {date} for mouse {mouse} as there was no wheel block")
        exit()

    cache_file = (
        HERE.parent / f"{session.mouse_name}_{session.date}_ensemble_reactivation.npz"
    )

    if use_cache and os.path.exists(cache_file):
        print("Using cached data")
        pcs_mask = np.load(cache_file)["pcs_mask"]
        ensemble_matrix = np.load(cache_file)["ensemble_matrix"]
        reactivation_strength = np.load(cache_file)["reactivation_strength"]
        reactivation_strength_shuffled = np.load(cache_file)[
            "reactivation_strength_shuffled"
        ]
        preactivation_strength = np.load(cache_file)["preactivation_strength"]
        preactivation_strength_shuffled = np.load(cache_file)[
            "preactivation_strength_shuffled"
        ]
        reactivation = np.load(cache_file)["reactivation"]
        preactivation = np.load(cache_file)["preactivation"]
        running_bouts = np.load(cache_file)["running_bouts"]
        pcc_scores = np.load(cache_file)["pcc_scores"]
    else:
        print("No cached data found, processing data")
        # TODO: think about speed_bin_size -> set to 30 i.e. 1s (30 fps)
        positions, speed, ITI_starts_ends, aligned_trial_frames = process_behaviour(
            session,
            wheel_circumference=get_wheel_circumference_from_rig(
                "2P",
            ),
            speed_bin_size=30,
        )
        config = GrosmarkConfig(
            bin_size=2,
            start=30,
            end=160,
        )
        spks_raw, _, _, _ = load_data(
            session,
            TIFF_UMBRELLA / session.date / session.mouse_name / "suite2p" / "plane0",
            "spks",
        )
        # TODO: should this be one function that is called by grosmark_analysis.py and ensemble_reactivation.py?
        # """Based on the observed differences in calcium activity waveforms between the online and
        # offline epochs (Supplementary Fig. 2), a threshold of 1.5 m.a.d. was used for online running epochs,
        # while a lower threshold of 1.25 m.a.d. were used for offline immobility epochs.""
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
        # TODO: should pre and post be binarised as one?
        offline_spks_pre = binarise_spikes(
            offline_spks_pre,
            mad_threshold=1.25,
        )
        offline_spks_post = binarise_spikes(offline_spks_post, mad_threshold=1.25)
        spks = np.hstack([offline_spks_pre, online_spks, offline_spks_post])
        # TODO: shape is off by one frame (pre is one short), do we care?
        # assert spks_raw.shape == spks.shape
        t1 = time.time()
        pcs_mask, _ = get_place_cells(
            session=session, spks=spks, rewarded=None, config=config, plot=False
        )
        print(f"Time to get place cells: {time.time() - t1}")
        place_cells = spks[pcs_mask, :]
        reactivation = offline_spks_post[pcs_mask]
        preactivation = offline_spks_pre[pcs_mask]
        # ONLINE
        running_bouts = get_running_bouts(
            place_cells=place_cells,
            speed=speed,
            frames_positions=positions,
            aligned_trial_frames=aligned_trial_frames,
            config=config,
        )
        ssp_vectors = get_ssp_vectors(place_cells_running=running_bouts)
        ensemble_matrix = compute_ICA_components(ssp_vectors=ssp_vectors)
        # OFFLINE
        reactivation_strength = offline_reactivation(
            reactivation=reactivation, ensemble_matrix=ensemble_matrix
        )
        preactivation_strength = offline_reactivation(
            reactivation=preactivation, ensemble_matrix=ensemble_matrix
        )
        reactivation_strength_shuffled = offline_reactivation(
            reactivation=reactivation, ensemble_matrix=ensemble_matrix, do_shuffle=True
        )
        preactivation_strength_shuffled = offline_reactivation(
            reactivation=preactivation, ensemble_matrix=ensemble_matrix, do_shuffle=True
        )
        t2 = time.time()
        pcc_scores = get_normalised_pcc_scores(
            reactivation=reactivation,
            preactivation=preactivation,
            ensemble_matrix=ensemble_matrix,
        )
        print(f"Time to get pcc scores: {time.time() - t2}")
    if verbose:
        print(f"# frames with running: {running_bouts.shape[1]}")
        print(f"# place cells: {running_bouts.shape[0]}")
        print(f"# significant components (Marcenko-Pastur): {ensemble_matrix.shape[1]}")

    plot_pcc_scores(pcc_scores=pcc_scores)
    # TODO: should the top ensembles be the same for reactivation and preactivation?
    # I mean we would look at the reactivation strength of the same ensembles?
    top_ensembles = sort_ensembles_by_reactivation_strength(
        reactivation_strength=reactivation_strength
    )
    sorted_pcs = classify_and_sort_place_cells(
        ensemble_matrix=ensemble_matrix, top_ensembles=top_ensembles
    )
    if use_cache and not os.path.exists(cache_file):
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
            running_bouts=running_bouts,
            pcc_scores=pcc_scores,
        )
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


if __name__ == "__main__":
    main()
