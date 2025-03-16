from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy.stats import zscore

from viral.imaging_utils import trial_is_imaged
from viral.models import Cached2pSession
from viral.utils import (
    find_five_consecutive_trues_center,
    get_wheel_circumference_from_rig,
    has_five_consecutive_trues,
    sort_matrix_peak,
)
from viral.imaging_utils import activity_trial_position


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

    bin_size = 2
    start = 30
    end = 170

    sigma_cm = 3  # Desired smoothing in cm
    sigma_bins = sigma_cm / bin_size  # Convert to bin units

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
            )
            for trial in session.trials
            if not trial.texture_rewarded and trial_is_imaged(trial)
        ]
    )

    activity_matrix = np.mean(all_trials, 0)

    # Apply Gaussian smoothing along each row (axis=1)
    # TODO: Should we smooth each trial independently?
    smoothed_matrix = gaussian_filter1d(activity_matrix, sigma=sigma_bins, axis=1)
    shuffled_matrices = np.array(
        [
            gaussian_filter1d(
                np.mean(
                    np.array(
                        [
                            activity_trial_position(
                                trial=trial,
                                flu=spks,
                                wheel_circumference=get_wheel_circumference_from_rig(
                                    "2P"
                                ),
                                bin_size=bin_size,
                                start=start,
                                max_position=end,
                                verbose=False,
                                do_shuffle=True,
                            )
                            for trial in session.trials
                            if not trial.texture_rewarded and trial_is_imaged(trial)
                        ]
                    ),
                    0,
                ),
                sigma=sigma_bins,
                axis=1,
            )
            for _ in range(2000)
        ]
    )

    place_threshold = np.percentile(shuffled_matrices, 99, axis=0)

    shuffled_place_cells = np.array(
        [
            has_five_consecutive_trues(shuffled_matrices[idx, :, :] > place_threshold)
            for idx in range(shuffled_matrices.shape[0])
        ]
    )

    pcs = has_five_consecutive_trues(smoothed_matrix > place_threshold)
    smoothed_matrix = smoothed_matrix[pcs, :]

    pcs = filter_additional_check(
        all_trials=all_trials[:, pcs, :],
        place_threshold=place_threshold[pcs, :],
        smoothed_matrix=smoothed_matrix,
    )

    print(
        f"percent place cells shuffed {np.mean(np.sum(shuffled_place_cells, axis=1) / shuffled_place_cells.shape[1])}"
    )

    # TODO: no additional check for shuffled, probably not necessary
    print(f"percent place cells not shuffled {np.sum(pcs) / len(pcs)}")

    plt.figure()
    plt.imshow(
        zscore(sort_matrix_peak(smoothed_matrix[pcs, :]), axis=1),
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
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
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plt.title("shuffled")
    plt.colorbar()
    plt.show()

    return np.ndarray([])


def filter_additional_check(
    all_trials: np.ndarray, place_threshold: np.ndarray, smoothed_matrix: np.ndarray
) -> np.ndarray:
    # TODO: Do we need to smooth this?

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

        # in_field.append(all_trials[:,

    return np.array(valid_pcs)
