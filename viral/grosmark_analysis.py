import itertools
from pathlib import Path
import sys
from matplotlib import pyplot as plt
from scipy.stats import median_abs_deviation, zscore
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
import numpy as np

# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


from viral.constants import HERE
from viral.imaging_utils import (
    get_ITI_matrix,
    get_dff,
    trial_is_imaged,
    activity_trial_position,
    get_resting_chunks,
)

from viral.models import Cached2pSession, GrosmarkConfig

from viral.utils import (
    cross_correlation_pandas,
    find_five_consecutive_trues_center,
    get_wheel_circumference_from_rig,
    has_five_consecutive_trues,
    remove_consecutive_ones,
    remove_diagonal,
    shuffle_rows,
    sort_matrix_peak,
)


def grosmark_place_field(
    session: Cached2pSession,
    spks: np.ndarray,
    rewarded: bool | None,
    config: GrosmarkConfig,
) -> None:
    """The position of the animal during online running epochs on the 2-m-long run belts was binned into 100,
      2-cm spatial bins. For each cell, the as within spatial-bin firing rate was calculated across all bins
    based on its sparsified spike estimate vector, Ssp. This firing rate by position vector was subsequently
    smoothed with a 7.5-cm Gaussian kernel leading to the smoothed firing rate by position vector.
    In addition, for each cell, 2,000 shuffled smoothed firing rate by position vectors were computed for each
    cell following the per-lap randomized circular permutation of estimated activity vector, Ssp. For each cell,
    putative PFs were defined as those in which the observed smoothed firing rate by position vectors exceeded the
    99th percentile of their shuffled smoothed firing rate by position vectors as assessed on a per spatial-bin basis for
    at least five consecutive spatial bins.

    As an additional control, only those putative PFs in which the cell had a greater within-PF than outside-of-PF firing rates
    in at least 3 or 15% of laps (whichever was greater for each session) were considered bona fide PFs and kept for further analysis.

    """

    n_cells_total = spks.shape[0]

    sigma_cm = 7.5  # Desired smoothing in cm
    sigma_bins = sigma_cm / config.bin_size  # Convert to bin units

    n_shuffles = 2000

    # spks = shuffle_rows(spks)

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

    smoothed_matrix = np.mean(all_trials, 0)

    # Probably delete cache logic once we're all sorted
    use_cache = True
    cache_file = (
        HERE
        / f"{session.mouse_name}_{session.date}_rewarded_{rewarded}_shuffled_matrices.npy"
    )
    if use_cache and cache_file.exists():
        shuffled_matrices = np.load(cache_file)
    else:
        # Create array of shape (n_shuffles, n_cells, n_bins)
        # where each (n_cells x bins) matrix is trial averaged but shuffled on a per-trial basis (as in Grosmark)
        # You can then apply percentiles along the first dimension to find "real" place cells
        shuffled_matrices = np.array(
            [
                np.mean(
                    np.array(
                        [
                            activity_trial_position(
                                trial=trial,
                                flu=spks,
                                wheel_circumference=get_wheel_circumference_from_rig(
                                    "2P"
                                ),
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
                    0,
                )
                for _ in range(n_shuffles)
            ]
        )
        np.save(cache_file, shuffled_matrices)

    place_threshold = np.percentile(shuffled_matrices, 99, axis=0)

    shuffled_place_cells = np.array(
        [
            has_five_consecutive_trues(shuffled_matrices[idx, :, :] > place_threshold)
            for idx in range(shuffled_matrices.shape[0])
        ]
    )

    pcs = has_five_consecutive_trues(smoothed_matrix > place_threshold)

    spks = spks[pcs, :]
    smoothed_matrix = smoothed_matrix[pcs, :]

    print(f"percent place cells before extra check {np.sum(pcs) / n_cells_total}")

    pcs = filter_additional_check(
        all_trials=all_trials[:, pcs, :],
        place_threshold=place_threshold[pcs, :],
        smoothed_matrix=smoothed_matrix,
    )

    # Don't love this double indexing
    spks = spks[pcs, :]
    smoothed_matrix = smoothed_matrix[pcs, :]

    plot_place_cells(
        smoothed_matrix=smoothed_matrix,
        shuffled_matrices=shuffled_matrices,
        shuffled_place_cells=shuffled_place_cells,
    )

    print(f"percent place cells after extra check {np.sum(pcs) / n_cells_total}")

    print(
        f"percent place cells shuffed {np.mean(np.sum(shuffled_place_cells, axis=1) / n_cells_total)}"
    )

    peak_indices = np.argmax(smoothed_matrix, axis=1)
    sorted_order = np.argsort(peak_indices)

    plot_circular_distance_matrix(smoothed_matrix, sorted_order)

    offline_correlations(
        session,
        spks[sorted_order, :],
        rewarded=rewarded,
    )

    plt.show()


def offline_correlations(
    session: Cached2pSession, spks: np.ndarray, rewarded: bool | None
) -> None:
    """Correlated offline activity with running sequences. There used to be a lot of alternative definitions of offline activity
    that can be found in the commit history (e.g. d2e7852f54282a52722767e52cca1ab71e56851b) if you need them

    From Grosmark:
    For pair-wise reactivation analysis, the run PF peak distance between pairs of PCs was compared to their offline firing-rate
    Pearsons correlation coefficients in either the pre or post epochs. For calculating offline firing rate correlations,
    the sparsified binary spike estimate vectors Ssp were restricted to periods of immobility during either the pre or post epoch
    and convolved with a 150-ms Gaussian kernel.

    """

    offline = get_ITI_matrix(
        trials=[
            trial
            for trial in session.trials
            if trial_is_imaged(trial)
            and (rewarded is None or trial.texture_rewarded == rewarded)
        ],
        flu=spks,
        bin_size=None,
    )

    shuffled_corrs = get_offline_correlation_matrix(offline, do_shuffle=True, plot=True)
    real_corrs = get_offline_correlation_matrix(offline, do_shuffle=False, plot=True)

    correlations_vs_peak_distance(real_corrs, config=config)


def get_offline_correlation_matrix(
    offline: np.ndarray, do_shuffle: bool = False, plot: bool = False
) -> np.ndarray:
    n_trials = offline.shape[0]
    all_corrs = []
    for trial in range(n_trials):
        trial_matrix = offline[trial, :, :]
        if do_shuffle:
            trial_matrix = shuffle_rows(trial_matrix)
        # 150-ms kernel convolution
        ITI_trial = gaussian_filter1d(trial_matrix, sigma=4.5, axis=1)
        all_corrs.append(cross_correlation_pandas(ITI_trial.T))

    corrs = np.nanmean(np.array(all_corrs), 0)

    if plot:
        plt.figure()
        plt.title("shuffled" if do_shuffle else "real")
        plt.imshow(
            gaussian_filter1d(remove_diagonal(corrs), sigma=2.5),
            vmin=0,
            vmax=0.1,
            cmap="bwr",
        )

    return corrs


def correlations_vs_peak_distance(corrs: np.ndarray, config: GrosmarkConfig) -> None:
    n_cells = corrs.shape[0]
    # This assumes cells equally span the field. Not true but ok for now
    # DOOOOOOOOOO THISSSSSSSSSSSSSSSSSSSSSSs
    all_cell_peak_positions = np.linspace(config.start, config.end, n_cells)
    peak_distances = []
    cell_corrs = []

    for i, j in itertools.combinations(range(n_cells), r=2):
        cell1_peak = all_cell_peak_positions[i]
        cell2_peak = all_cell_peak_positions[j]
        peak_distances.append(abs(cell1_peak - cell2_peak))
        cell_corrs.append(corrs[i, j])

    peak_distances = np.array(peak_distances)
    cell_corrs = np.array(cell_corrs)

    x = []
    y = []

    for bin_start in np.arange(0, 80):
        in_bin = np.logical_and(
            peak_distances > bin_start, peak_distances < bin_start + 20
        )
        x.append(bin_start)
        y.append(np.mean(cell_corrs[in_bin]))

    plt.figure()
    plt.plot(x, y)

    plt.xlabel("Distance between peaks")
    plt.ylabel("Pearson correlation")
    plt.show()


def plot_circular_distance_matrix(
    smoothed_matrix: np.ndarray, sorted_order: np.ndarray
) -> None:

    plt.figure()
    plt.imshow(
        circular_distance_matrix(smoothed_matrix[sorted_order, :]), cmap="RdYlBu"
    )
    plt.colorbar()
    plt.ylabel("Cell number")
    plt.xlabel("Cell number")


def plot_place_cells(
    smoothed_matrix: np.ndarray,
    shuffled_matrices: np.ndarray,
    shuffled_place_cells: np.ndarray,
) -> None:
    plt.figure()
    plt.imshow(
        zscore(sort_matrix_peak(smoothed_matrix), axis=1),
        aspect="auto",
        cmap="bwr",
        vmin=-1,
        vmax=2,
    )

    plt.title("Real")
    plt.xlabel("Corridor position (cm)")
    plt.ylabel("Cell number")

    plt.colorbar()

    plt.figure()

    plt.imshow(
        zscore(
            sort_matrix_peak(shuffled_matrices[0, shuffled_place_cells[0, :], :]),
            axis=1,
        ),
        aspect="auto",
        cmap="bwr",
        vmin=-1,
        vmax=2,
    )

    plt.title("shuffled")
    plt.colorbar()


def filter_additional_check(
    all_trials: np.ndarray, place_threshold: np.ndarray, smoothed_matrix: np.ndarray
) -> np.ndarray:
    """Runs the following check from the Grosmark paper:
    As an additional control, only those putative PFs in which the cell had a greater within-PF than outside-of-PF firing rates
    in at least 3 or 15% of laps (whichever was greater for each session) were considered bona fide PFs and kept for further analysis.

    Currently have made the threshold more conservative (40%) as 15% does not filter any cells out, but review.
    """

    centers = find_five_consecutive_trues_center(smoothed_matrix > place_threshold)

    n_trials, n_cells, n_bins = all_trials.shape

    valid_pcs = np.array([False] * n_cells)
    for cell in range(n_cells):
        center = centers[cell]
        assert center + 3 <= n_bins
        assert center - 2 >= 0

        cell_place_field = np.array([False] * n_bins)
        cell_place_field[center - 2 : center + 3] = True
        cell_out_of_place_field = np.logical_not(cell_place_field)

        cell_place_activity = all_trials[:, cell, cell_place_field]
        cell_not_place_activity = all_trials[:, cell, cell_out_of_place_field]
        count = 0
        for trial in range(n_trials):
            if np.mean(cell_place_activity[trial, :]) > np.mean(
                cell_not_place_activity[trial, :]
            ):
                count += 1

        if count / n_trials > 0.15:
            valid_pcs[cell] = True

    return np.array(valid_pcs)


def circular_distance_matrix(activity_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise circular distance matrix of place field (PF) peaks.

    Parameters:
    - activity_matrix: np.array of shape (n_cells, n_positions)
      Each row corresponds to the neural activity of a single cell across positions.

    Returns:
    - circular_dist_matrix: np.array of shape (n_cells, n_cells)
      The pairwise circular peak distance matrix.

    I haven't unit tested this but the results look good.
    """

    n_positions = activity_matrix.shape[1]  # Number of spatial bins

    # Step 1: Find peak firing positions for each cell
    peak_positions = np.argmax(activity_matrix, axis=1)

    # Step 2: Compute pairwise circular distances
    pairwise_distances = cdist(
        peak_positions[:, None], peak_positions[:, None], metric="cityblock"
    )

    # Step 3: Apply circular distance correction
    circular_dist_matrix = np.minimum(
        pairwise_distances, n_positions - pairwise_distances
    )

    return circular_dist_matrix


def binarise_spikes(spks: np.ndarray) -> np.ndarray:
    """Implements the calcium imaging preprocessing stepts here:
    https://www.nature.com/articles/s41593-021-00920-7#Sec12

    Though the first steps done in our oasis fork.

    Currently we are not doing wavelet denoising as I've found this makes the fit much worse.
    We have added zhang baseline step. As without this, if our baseline drifts, higher baseline
    periods are considered to have more spikes.

    We are also not normalising by the residual between denoised and actual. It's not clear
    how they do this. What factor are they reducing the residual by? The residual is some
    massive number.

    They threshold based on the MAD. But is it just the MAD or is the MAD deviation from the median?
    I also had to take the MAD of only non-zero periods. As the raw MAD of all cells is 0. This may
    not be true in the hippocampus which is why they may not do this. We're also not currently
    altering the threshold depending or running or not. TOOD: DO THIS


    """

    non_zero_spikes = np.copy(spks)
    non_zero_spikes[non_zero_spikes == 0] = np.nan

    mad = median_abs_deviation(non_zero_spikes, axis=1, nan_policy="omit")

    # Maybe
    # threshold = mad * 1.5

    # Or maybe
    threshold = np.nanmedian(non_zero_spikes, axis=1) + mad * 1.5
    mask = spks - threshold[:, np.newaxis] > 0
    spks[~mask] = 0
    spks[mask] = 1
    return remove_consecutive_ones(spks)


if __name__ == "__main__":

    mouse = "JB027"
    date = "2025-02-26"
    # date = "2024-12-10"

    with open(HERE.parent / "data" / "cached_2p" / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())

    print(f"Total number of trials: {len(session.trials)}")
    print(
        f"number of trials imaged {len([trial for trial in session.trials if trial_is_imaged(trial)])}"
    )

    dff, spks, denoised = get_dff(mouse, date)

    print("Got dff")

    assert (
        max(
            trial.states_info[-1].closest_frame_start
            for trial in session.trials
            if trial.states_info[-1].closest_frame_start is not None
        )
        < dff.shape[1]
    ), "Tiff is too short"

    spks = binarise_spikes(spks)

    is_unsupervised = session.session_type.lower().startswith("unsupervised learning")

    config = GrosmarkConfig(
        bin_size=2,
        start=20,
        end=160,
    )

    grosmark_place_field(
        session, spks, rewarded=None if is_unsupervised else True, config=config
    )
