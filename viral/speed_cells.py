# Move eventually, but do it here for now
import logging
from pathlib import Path
import random
import sys
import os
from typing import List
from scipy.stats import zscore, pearsonr
import seaborn as sns
import math


# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


sns.set_theme(context="talk", style="ticks")

from viral.utils import (
    degrees_to_cm,
    get_wheel_circumference_from_rig,
    shuffle,
    trial_is_imaged,
    pad_to_max_length,
    pad_to_max_length_bins,
)

from viral.models import Cached2pSession, TrialInfo

from viral.two_photon import normalize, get_dff, get_spks

# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from matplotlib import pyplot as plt
import numpy as np


from viral.constants import SERVER_PATH, TIFF_UMBRELLA

from viral.rastermap_utils import (
    get_frame_position,
    get_speed_frame,
    get_signal_for_trials,
    get_dff,
    align_trial_frames,
)


def activity_trial_speed(
    trial: TrialInfo,
    signal: np.ndarray,
    trial_frames: np.ndarray,
    wheel_circumference: float,
    aggregation_fcn: str = "mean",
    bin_size: int = 5,
    return_raw: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Bin frames with their respective activity according to speed and return the activity for each cell.

    Args:
        trial (TrialInfo):              A TrialInfo object representing the trial to be analysed.
        signal (np.ndarray):            A Numpy array with the signal (e.g. dff or spks) for the trial.
        trial_frames (np.ndarray):      A Numpy array representing the frames in the trial.
        wheel_circumference (float):    The circumference of the wheel on the rig used in centimetres.
        aggregation_fcn (str):          The aggregation function to be used (mean or sum). Defaults to "mean".
        bin_size (int):                 Bin size for speed binning. Defaults to 5.
        return_raw (bool):              Whether to return the raw activity. Defaults to False.
        verbose (bool):                 Whether to print additional information. Defaults to False.

    Returns:
        - If return_raw=True: A 3D Numpy array of shape(cells, speed_bins, frames) where:
            - The first dimension corresponds to cells.
            - The second dimension corresponds to speed bins.
            - The third dimension corresponds to frames within each speed bin (padded with NaNs).
        - If `return_raw=False`: A 2D array of shape (cells, speed_bins) where:
            - Rows correspond to cells.
            - Columns correspond to speed bins.
            - Values represent the aggregated activity (mean or sum) of each cell in each speed bin.
            - NaN values appear in bins with no associated frames.

    """

    frames_positions = get_frame_position(
        trial, trial_frames, wheel_circumference
    )  # each row: frame_idx, position (cm)
    frames_speed = get_speed_frame(
        frames_positions
    )  # each row: frame_idx, speed (cm/frame)

    speed = (frames_speed[:, 1] * 30).astype(
        int
    )  # convert cm/frame to cm/second (@30 fps)
    max_speed = np.max(speed)

    frames = np.arange(0, len(trial_frames), 1)  # Continuous frames to index the dff

    signal_speed = list()

    for bin_start in range(0, max_speed, bin_size):
        # search for indices of frame that have the bin speed
        frame_idx_bin = np.unique(
            frames[np.logical_and(speed >= bin_start, speed < bin_start + bin_size)]
            # found an edge case through testing: if max_speed == bin_stop
            # the speed and associated frames did not get counted
            # hence, include max_speed in last bin
            if bin_start + bin_size < max_speed
            else frames[speed >= bin_start]
        )

        signal_bin = signal[:, frame_idx_bin]

        if verbose:
            print(f"bin_start: {bin_start}")
            print(f"bin_end: {bin_start + bin_size}")
            print(f"n_frames in bin: {len(frame_idx_bin)}")

        if return_raw:
            signal_speed.append(signal_bin)
        else:
            # Can and will return NaN for speed bins without frames!
            if aggregation_fcn == "mean":
                signal_speed.append(np.mean(signal_bin, axis=1))
            elif aggregation_fcn == "sum":
                signal_speed.append(np.sum(signal_bin, axis=1))
            else:
                raise AttributeError("aggregation_fcn must be either 'mean' or 'sum'")

    if return_raw:
        # max_cols = max(arr.shape[1] for arr in signal_speed)
        # padded_arrays = [
        #     np.pad(
        #         arr,
        #         ((0, 0), (0, max_cols - arr.shape[1])),
        #         constant_values=np.nan,
        #     )
        #     for arr in signal_speed
        # ]
        return pad_to_max_length_bins(signal_speed).transpose(1, 0, 2)
        # return np.array(padded_arrays).transpose(1, 0, 2)
    else:
        return np.array(signal_speed).T


# TODO: for sanity testing, remove eventually
def create_random_activity_data(cells, speed_bins) -> np.ndarray:
    data = np.random.rand(cells, speed_bins)
    for i in range(cells):
        data[i] += np.exp(-0.5 * ((np.arange(speed_bins) - i) ** 2) / 2)
    data /= data.sum(axis=1, keepdims=True)
    peak_indices = np.argmax(data, axis=1)
    sorted_order = np.argsort(peak_indices)
    return data[sorted_order, :]


def get_speed_activity(
    trials: List[TrialInfo],
    aligned_trial_frames: np.ndarray,
    dff: np.ndarray,
    wheel_circumference: float,
    bin_size: int = 5,
    train_test_split: bool = True,
) -> np.ndarray:
    """Sort cells by their speed activity.

    Args:
        trials (List[TrialInfo]):       A list of TrialInfo objects to be analysed.
        trial_frames (np.ndarray):      Numpy array representing the frames in the trials above.
        dff (np.ndarray):               The dff array for the session to be analysed.
        wheel_circumference (float):    The circumference of the wheel on the rig used in centimetres.
        bin_size (int):                 Bin size for speed binning. Defaults to 5.
        train_test_split (bool):        Whether to perform a train-test-split. Defaults to True

    Returns:
        np.ndarray:                     A Numpy array of shape (cells, binned_speeds) where:
            - Rows correspond to cells.
            - Columns correspond to speed bins.
            - Values represent normalized activity of given cell in given speed bin.
            - The cells at the beginning of the array have their peaks at the first speed bin and so on.

    """
    dff_trials = get_signal_for_trials(dff, aligned_trial_frames)

    if train_test_split:
        test_matrices = []
        for _ in range(10):
            train_idx = random.sample(range(len(trials)), len(trials) // 2)
            test_idx = [idx for idx in range(len(trials)) if idx not in train_idx]

            # Find the order in which to sort neurons in a random 50% of the trials
            train_matrix = np.nanmean(
                pad_to_max_length_bins(
                    [
                        activity_trial_speed(
                            trials[idx],
                            dff_trials[idx],
                            np.arange(
                                aligned_trial_frames[idx][0],
                                aligned_trial_frames[idx][1] + 1,
                                1,
                            ),
                            wheel_circumference,
                            bin_size=bin_size,
                        )
                        for idx in train_idx
                    ]
                ),
                axis=0,
            )
            peak_indices = np.argmax(train_matrix, axis=1)
            sorted_order = np.argsort(peak_indices)

            # TODO: Review whether this is the best normalisation function
            test_matrix = normalize(
                np.nanmean(
                    pad_to_max_length_bins(
                        [
                            activity_trial_speed(
                                trials[idx],
                                dff_trials[idx],
                                np.arange(
                                    aligned_trial_frames[idx][0],
                                    aligned_trial_frames[idx][1] + 1,
                                    1,
                                ),
                                wheel_circumference,
                                bin_size=bin_size,
                            )
                            for idx in test_idx
                        ]
                    ),
                    axis=0,
                ),
                axis=1,
            )
            test_matrices.append(test_matrix[sorted_order, :])

        padded_test_matrices = pad_to_max_length_bins(
            test_matrices
        )  # the number of speed bins may very in between trials
        return np.nanmean(padded_test_matrices, 0)
    else:
        metrices_raw = [
            activity_trial_speed(
                trials[idx],
                dff_trials[idx],
                np.arange(
                    aligned_trial_frames[idx][0],
                    aligned_trial_frames[idx][1] + 1,
                    1,
                ),
                wheel_circumference,
                bin_size=bin_size,
            )
            for idx in range(len(trials))
        ]
        metrices_padded = normalize(
            np.nanmean(pad_to_max_length_bins(metrices_raw), axis=0), axis=1
        )
        peak_indices = np.argmax(metrices_padded, axis=1)
        sorted_order = np.argsort(peak_indices)
        return metrices_padded[sorted_order, :]


# TODO: Test get_speed_activity


def speed_cells(
    session: Cached2pSession,
    dff: np.ndarray,
    wheel_circumference: float,
    bin_size: int = 5,
) -> None:
    # TODO: Is this even a good way of plotting it?
    # Might be skewed because the frames in each speed bin certainly are not even
    # E.g. look at unrewarded trials, after learning most frames and most signals will be in there, right?
    aligned_trial_frames = align_trial_frames(
        [trial for trial in session.trials if trial_is_imaged(trial)], False
    )

    plt.figure()
    plt.title("Rewarded trials")
    # TODO: add train-test split back in!
    # Testing with artificial data
    # speed_activity_rewarded = create_random_activity_data(cells=100, speed_bins=10)
    speed_activity_rewarded = get_speed_activity(
        [
            trial
            for trial in session.trials
            if trial_is_imaged(trial) and trial.texture_rewarded
        ],
        aligned_trial_frames[aligned_trial_frames[:, -1] == 1],
        dff,
        wheel_circumference,
        bin_size,
        True,
    )
    speed_activity_rewarded = speed_activity_rewarded[
        :, ~np.isnan(speed_activity_rewarded).all(axis=0)
    ]
    # TODO: Careful: removing entire columns might be dangerous!
    plt.imshow(
        speed_activity_rewarded,
        aspect="auto",
        cmap="viridis",
    )

    clb = plt.colorbar()
    clb.ax.set_title("Normalised\nactivity", fontsize=12)
    plt.ylabel("cell number")
    plt.xlabel("Speed")
    xticks_rewarded = np.arange(
        bin_size, (speed_activity_rewarded.shape[1] + 1) * bin_size, bin_size
    )
    plt.xticks(np.arange(len(xticks_rewarded)), xticks_rewarded)
    plt.tight_layout()
    plt.savefig(
        HERE.parent
        / "plots"
        / "speed_cells"
        / f"speed-cells-rewarded-{session.mouse_name}-{session.date}"
    )
    plt.close()

    #  TODO: The xticks for unrewarded are off goddamn it
    plt.figure()
    plt.title("Unrewarded trials")
    # TODO: add train-test split back in!
    speed_activity_unrewarded = get_speed_activity(
        [
            trial
            for trial in session.trials
            if trial_is_imaged(trial) and not trial.texture_rewarded
        ],
        aligned_trial_frames[aligned_trial_frames[:, -1] == 0],
        dff,
        wheel_circumference,
        bin_size,
        True,
    )
    speed_activity_unrewarded = speed_activity_unrewarded[
        :, ~np.isnan(speed_activity_unrewarded).all(axis=0)
    ]
    # TODO: Careful: removing entire columns might be dangerous!
    plt.imshow(
        speed_activity_unrewarded,
        aspect="auto",
        cmap="viridis",
    )
    clb = plt.colorbar()
    clb.ax.set_title("Normalised\nactivity", fontsize=12)
    plt.ylabel("cell number")
    plt.xlabel("Speed")
    xticks_unrewarded = np.arange(
        bin_size, (speed_activity_unrewarded.shape[1] + 1) * bin_size, bin_size
    )
    plt.xticks(np.arange(len(xticks_unrewarded)), xticks_unrewarded)
    plt.tight_layout()
    plt.savefig(
        HERE.parent
        / "plots"
        / "speed_cells"
        / f"speed-cells-unrewarded-{session.mouse_name}-{session.date}"
    )


# TODO: Testing!
def get_speed_firing_rate(
    trials: List[TrialInfo],
    aligned_trial_frames: np.ndarray,
    spks: np.ndarray,
    wheel_circumference: float,
    bin_size: int = 5,
) -> np.ndarray:
    spks_trials = get_signal_for_trials(spks, aligned_trial_frames)
    speed_firing_rates_list = list()
    for idx, trial in enumerate(trials):
        trial_start_end_frames = aligned_trial_frames[idx, [0, 1]]
        speed_activity = activity_trial_speed(
            trial,
            spks_trials[idx],
            np.arange(trial_start_end_frames[0], trial_start_end_frames[1]),
            wheel_circumference,
            bin_size=bin_size,
            return_raw=True,
        )
        speed_firing_rate = np.nansum(speed_activity, axis=2) / np.sum(
            ~np.isnan(speed_activity), axis=2
        )
        speed_firing_rates_list.append(speed_firing_rate)
    speed_firing_rates = pad_to_max_length_bins(speed_firing_rates_list)
    return np.nanmean(speed_firing_rates, axis=0)


def plot_speed_firing_rate(session: Cached2pSession) -> None:
    # TODO: Add bin size, better path handling
    if not os.path.exists(
        Path(HERE.parent)
        / "data"
        / f"{session.mouse_name}-{session.date}-speed_firing_rate.npy"
    ):
        spks = get_spks(session.mouse_name, session.date)
        trials = [trial for trial in session.trials if trial_is_imaged(trial)]
        aligned_trial_frames = align_trial_frames(trials)
        speed_firing_rate = get_speed_firing_rate(
            trials, aligned_trial_frames, spks, 34.7, 5
        )
        np.save(
            Path(HERE.parent)
            / "data"
            / f"{session.mouse_name}-{session.date}-speed_firing_rate.npy",
            speed_firing_rate,
        )
    else:
        speed_firing_rate = np.load(
            Path(HERE.parent)
            / "data"
            / f"{session.mouse_name}-{session.date}-speed_firing_rate.npy"
        )

    speed_bins = np.arange(speed_firing_rate.shape[1])
    pearson_values = list()
    p_values = list()

    for cell_idx in range(speed_firing_rate.shape[0]):
        firing_rates = speed_firing_rate[cell_idx, :]
        valid_indices = ~np.isnan(firing_rates)
        valid_speeds = speed_bins[valid_indices]
        valid_firing_rates = firing_rates[valid_indices]

        pearson, p_value = pearsonr(valid_speeds, valid_firing_rates)
        pearson_values.append(pearson)
        p_values.append(p_value)

    sorted_order = np.argsort(np.abs(pearson_values))[::-1]
    speed_firing_rate = speed_firing_rate[sorted_order, :]
    pearson_arr = np.array(pearson_values)[sorted_order]
    p_arr = np.array(p_values)[sorted_order]

    n_max_cells = 10
    n_cols = 5
    n_rows = math.ceil(n_max_cells / n_cols)

    max_cells = speed_firing_rate[:n_max_cells, :]
    fig, ax = plt.subplots(
        ncols=n_cols, nrows=n_rows, figsize=(n_cols * 10, n_rows * 5)
    )
    ax = np.ravel(ax)
    for i in range(n_max_cells):
        firing_rates = max_cells[i, :]
        ax[i].scatter(speed_bins, firing_rates, s=10)
        ax[i].text(
            0.50,
            0.95,
            f"r = {pearson_arr[i]:.2f}\np = {p_arr[i]:.2f}",
            transform=ax[i].transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"),
        )
        ax[i].set_xticks(speed_bins)
        ax[i].set_xticklabels([f"{x * 5}" for x in speed_bins], fontsize=8)
        ax[i].set_xlabel("Speed (cm/s)")
        ax[i].set_ylabel("Firing Rate")
        ax[i].set_title(f"Cell {i + 1}")
    plt.tight_layout()
    # plt.show()
    plt.savefig(
        Path(HERE.parent)
        / "plots"
        / "speed_firing_rate"
        / f"{session.mouse_name}-{session.date}-speed_firing_rate.png",
        dpi=300,
    )


if __name__ == "__main__":

    mouse = "JB011"
    date = "2024-10-31"

    with open(HERE.parent / "data" / "cached_2p" / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())
    print(f"Total number of trials: {len(session.trials)}")
    print(
        f"number of trials imaged {len([trial for trial in session.trials if trial_is_imaged(trial)])}"
    )

    # dff = get_dff(mouse, date)

    # assert (
    #     max(
    #         trial.states_info[-1].closest_frame_start
    #         for trial in session.trials
    #         if trial.states_info[-1].closest_frame_start is not None
    #     )
    #     < dff.shape[1]
    # ), "Tiff is too short"

    if "unsupervised" in session.session_type.lower():
        # TODO: speed cells for unsupervised sessions
        print()
        # speed_cells(session, dff, 34.7)
    else:
        # speed_cells(session, dff, 34.7)
        plot_speed_firing_rate(session)
