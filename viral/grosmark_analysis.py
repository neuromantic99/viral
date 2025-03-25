import itertools
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
import numpy as np
from scipy.stats import zscore

from viral.constants import HERE
from viral.imaging_utils import get_ITI_matrix, trial_is_imaged, get_resting_chunks

from viral.models import Cached2pSession

from viral.utils import (
    cross_correlation_pandas,
    find_five_consecutive_trues_center,
    get_wheel_circumference_from_rig,
    has_five_consecutive_trues,
    sort_matrix_peak,
)

from viral.imaging_utils import activity_trial_position

bin_size = 2

start = 20
end = 160


def grosmark_place_field(session: Cached2pSession, spks: np.ndarray) -> np.ndarray:
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
    sigma_bins = sigma_cm / bin_size  # Convert to bin units

    n_shuffles = 2000

    # spks = shuffle_rows(spks)

    all_trials = np.array(
        [
            activity_trial_position(
                trial=trial,
                flu=spks,
                wheel_circumference=get_wheel_circumference_from_rig("2P"),
                bin_size=bin_size,
                start=start,
                max_position=end,
                verbose=False,
                do_shuffle=False,
                smoothing_sigma=sigma_bins,
            )
            for trial in session.trials
            if not trial.texture_rewarded and trial_is_imaged(trial)
        ]
    )

    smoothed_matrix = np.mean(all_trials, 0)
    # shuffled_matrices = np.load(HERE / "shuffled_matrices.npy")
    shuffled_matrices = np.array(
        [
            np.mean(
                np.array(
                    [
                        activity_trial_position(
                            trial=trial,
                            flu=spks,
                            wheel_circumference=get_wheel_circumference_from_rig("2P"),
                            bin_size=bin_size,
                            start=start,
                            max_position=end,
                            verbose=False,
                            do_shuffle=True,
                            smoothing_sigma=sigma_bins,
                        )
                        for trial in session.trials
                        if not trial.texture_rewarded and trial_is_imaged(trial)
                    ]
                ),
                0,
            )
            for _ in range(n_shuffles)
        ]
    )
    np.save(HERE / "shuffled_matrices.npy", shuffled_matrices)
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

    print(f"percent place cells after extra check {np.sum(pcs) / n_cells_total}")

    print(
        f"percent place cells shuffed {np.mean(np.sum(shuffled_place_cells, axis=1) / n_cells_total)}"
    )

    # TODO: no additional check for shuffled, probably not necessary

    print(f"percent place cells not shuffled {np.sum(pcs) / n_cells_total}")

    peak_indices = np.argmax(smoothed_matrix, axis=1)
    sorted_order = np.argsort(peak_indices)

    # plot_circular_distance_matrix(smoothed_matrix, sorted_order)

    offline_correlations(
        session,
        spks[sorted_order, :],
    )

    return np.ndarray([])


def offline_correlations(
    session: Cached2pSession,
    spks: np.ndarray,
) -> None:
    """Correlated offline activity with running. Currently a load of different options for
    how you define offline activity"""

    offline = get_ITI_matrix(
        trials=[
            trial
            for trial in session.trials
            if trial_is_imaged(trial) and not trial.texture_rewarded
        ],
        dff=spks,
        bin_size=1,
        average=False,
    )

    n_trials, n_cells, n_frames = offline.shape

    # HORIZONAL STACK ITI
    # offline = (
    #     offline.reshape(-1, n_cells, n_frames).transpose(1, 0, 2).reshape(n_cells, -1)
    # )
    # offline = gaussian_filter1d(offline, sigma=4.5, axis=1)

    # plt.figure()
    # plt.title("real")
    # corrs = cross_correlation_pandas(offline.T)
    # corrs = remove_diagonal(corrs)
    # corrs = gaussian_filter1d(corrs, sigma=2.5)
    # plt.imshow(corrs, vmin=0, vmax=0.2)

    # plt.figure()

    # np.random.shuffle(offline)
    # plt.title("Shuffled")
    # corrs = cross_correlation_pandas(offline.T)
    # corrs = remove_diagonal(corrs)
    # corrs = gaussian_filter1d(corrs, sigma=2.5)
    # plt.imshow(corrs, vmin=0, vmax=0.2)

    # total_spikes_cells = np.sum(offline, (0, 2))
    # offline = offline[:, total_spikes_cells > 100, :]

    #####################################

    # DONT STACK ITI
    plt.figure()
    all_corrs = []
    for trial in range(n_trials):
        # 150-ms kernel convolution
        shuffled_trial = np.copy(offline[trial, :, :])
        np.random.shuffle(shuffled_trial)
        ITI_trial = gaussian_filter1d(shuffled_trial, sigma=4.5, axis=1)
        all_corrs.append(cross_correlation_pandas(ITI_trial.T))

    corrs = np.nanmean(np.array(all_corrs), 0)
    # corrs = remove_diagonal(corrs)
    # corrs = gaussian_filter1d(corrs, sigma=2.5)
    plt.title("shuffled")
    plt.imshow(corrs, vmin=0, vmax=0.1, cmap="bwr")

    plt.figure()
    all_corrs = []
    for trial in range(n_trials):
        # 150-ms kernel convolution
        ITI_trial = gaussian_filter1d(offline[trial, :, :], sigma=4.5, axis=1)
        all_corrs.append(cross_correlation_pandas(ITI_trial.T))

    corrs = np.nanmean(np.array(all_corrs), 0)
    # corrs = remove_diagonal(corrs)
    # corrs = gaussian_filter1d(corrs, sigma=2.5)
    plt.title("REAL")
    plt.imshow(corrs, vmin=0, vmax=0.1, cmap="bwr")

    # This assumes cells equally span the field. Not true but ok for now
    all_cell_peak_positions = np.linspace(start, end, n_cells)
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
    plt.show()

    ############ RESTING CHUNKS #########################
    # offline = get_resting_chunks(
    #     trials=[
    #         session.trials[idx]
    #         for idx in range(len(session.trials))
    #         if trial_is_imaged(session.trials[idx])
    #         and session.trials[idx - 1].texture_rewarded
    #     ],
    #     dff=spks,
    #     chunk_size_frames=5 * 30,
    #     speed_threshold=1,
    # )

    # offline = gaussian_filter1d(offline, sigma=4.5, axis=1)

    # all_coors = []
    # for offline_trial in offline:
    #     # offline = gaussian_filter1d(offline, sigma=4.5, axis=1)
    #     corr_trial = cross_correlation_pandas(offline_trial.T)
    #     corr_trial = remove_diagonal(corr_trial)
    #     all_coors.append(corr_trial)


def plot_circular_distance_matrix(
    smoothed_matrix: np.ndarray, sorted_order: np.ndarray
) -> None:

    plt.imshow(
        circular_distance_matrix(smoothed_matrix[sorted_order, :]), cmap="RdYlBu"
    )
    plt.colorbar()
    plt.show()


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

    plt.title("real")

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
    plt.show()


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

        if count / n_trials > 0.4:
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
