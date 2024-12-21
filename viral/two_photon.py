import logging
from pathlib import Path
import random
import sys
from typing import List
from scipy.stats import zscore
import seaborn as sns


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
    get_speed_positions,
)

from viral.models import Cached2pSession, TrialInfo


# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from matplotlib import pyplot as plt
import numpy as np


from viral.constants import SERVER_PATH, TIFF_UMBRELLA


def compute_dff(f: np.ndarray) -> np.ndarray:
    flu_mean = np.expand_dims(np.mean(f, 1), 1)
    return (f - flu_mean) / flu_mean


def subtract_neuropil(f_raw: np.ndarray, f_neu: np.ndarray) -> np.ndarray:
    return f_raw - f_neu * 0.7


def get_dff(mouse: str, date: str) -> np.ndarray:
    s2p_path = TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0"
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)
    spks = np.load(s2p_path / "spks.npy")[iscell, :]
    f_raw = np.load(s2p_path / "F.npy")[iscell, :]
    f_neu = np.load(s2p_path / "Fneu.npy")[iscell, :]
    # return spks
    return compute_dff(subtract_neuropil(f_raw, f_neu))


def activity_trial_position(
    trial: TrialInfo, dff: np.ndarray, wheel_circumference: float, verbose: bool = False
) -> np.ndarray | None:

    position = degrees_to_cm(
        np.array(trial.rotary_encoder_position), wheel_circumference
    )

    frame_position = np.array(
        [
            state.closest_frame_start
            for state in trial.states_info
            if state.name in ["trigger_panda", "trigger_panda_post_reward"]
        ]
    )

    assert len(position) == len(frame_position)

    # Bin frames by position
    bin_size = 1
    start = 10
    max_position = 170

    dff_position = []

    for bin_start in range(start, max_position, bin_size):
        frame_idx_bin = frame_position[
            np.logical_and(position >= bin_start, position < bin_start + bin_size)
        ]
        dff_bin = dff[:, frame_idx_bin]

        if verbose:
            print(f"bin_start: {bin_start}")
            print(f"bin_end: {bin_start + bin_size}")
            print(f"n_frames in bin: {len(frame_idx_bin)}")
        dff_position.append(np.mean(dff_bin, axis=1))

    print("\n")
    return np.array(dff_position).T


def sort_matrix_peak(matrix: np.ndarray) -> np.ndarray:
    peak_indices = np.argmax(matrix, axis=1)
    sorted_order = np.argsort(peak_indices)
    return matrix[sorted_order]


# def normalize(data):
#     return (data - np.min(data)) / (np.max(data) - np.min(data))


def normalize(array: np.ndarray, axis: int) -> np.ndarray:
    # Calculate the min and max along the specified axis
    min_val = np.min(array, axis=axis, keepdims=True)
    max_val = np.max(array, axis=axis, keepdims=True)
    return (array - min_val) / (max_val - min_val)


def get_position_activity(
    trials: List[TrialInfo], dff: np.ndarray, wheel_circumference: float
) -> np.ndarray:

    test_matrices = []
    # 33, 40
    # 76, 85
    # 120, 132

    for _ in range(10):
        train_idx = random.sample(range(len(trials)), len(trials) // 2)
        test_idx = [idx for idx in range(len(trials)) if idx not in train_idx]

        # Find the order in which to sort neurons in a random 50% of the trials
        train_matrix = np.nanmean(
            np.array(
                [
                    activity_trial_position(trials[idx], dff, wheel_circumference)
                    for idx in train_idx
                ]
            ),
            axis=0,
        )
        train_matrix[:, 33 // 1 : 40 // 1] = 0
        train_matrix[:, 76 // 1 : 85 // 1] = 0
        train_matrix[:, 120 // 1 : 132 // 1] = 0

        peak_indices = np.argmax(train_matrix, axis=1)
        sorted_order = np.argsort(peak_indices)

        test_matrix = normalize(
            np.nanmean(
                np.array(
                    [
                        activity_trial_position(trials[idx], dff, wheel_circumference)
                        for idx in test_idx
                    ]
                ),
                axis=0,
            ),
            axis=1,
        )

        test_matrices.append(test_matrix[sorted_order, :])

    return np.mean(np.array(test_matrices), 0)


def place_cells_unsupervised(session: Cached2pSession, dff: np.ndarray) -> None:

    plt.figure()
    plt.title("Unsupervised Session")
    data = get_position_activity(
        [trial for trial in session.trials if trial_is_imaged(trial)],
        dff,
        get_wheel_circumference_from_rig("2P"),
    )
    plt.imshow(data, aspect="auto", cmap="viridis")
    clb = plt.colorbar()
    clb.ax.set_title("Normalised\nactivity", fontsize=12)

    plt.ylabel("cell number")
    plt.xlabel("corrdior position")
    ticks = np.array([50, 100, 150])
    plt.xticks(ticks - 10, ticks)

    plt.title("Unsupervised Session")

    plt.tight_layout()

    plt.savefig(
        HERE.parent
        / "plots"
        / "place_cells"
        / f"place-cells-unsupervised-{session.mouse_name}-{session.date}"
    )


def place_cells(session: Cached2pSession, dff: np.ndarray) -> None:

    plt.figure()
    plt.title("Rewarded trials")
    plt.imshow(
        get_position_activity(
            [
                trial
                for trial in session.trials
                if trial_is_imaged(trial) and trial.texture_rewarded
            ],
            dff,
            get_wheel_circumference_from_rig("2P"),
        ),
        aspect="auto",
        cmap="viridis",
    )

    clb = plt.colorbar()
    clb.ax.set_title("Normalised\nactivity", fontsize=12)
    plt.ylabel("cell number")
    plt.xlabel("corrdior position")
    ticks = np.array([50, 100, 150])
    plt.xticks(ticks - 10, ticks)
    plt.tight_layout()
    plt.savefig(
        HERE.parent
        / "plots"
        / "place_cells"
        / f"place-cells-rewarded-{session.mouse_name}-{session.date}"
    )

    plt.figure()
    plt.title("Unrewarded trials")
    plt.imshow(
        get_position_activity(
            [
                trial
                for trial in session.trials
                if trial_is_imaged(trial) and not trial.texture_rewarded
            ],
            dff,
            get_wheel_circumference_from_rig("2P"),
        ),
        aspect="auto",
        cmap="viridis",
    )

    clb = plt.colorbar()
    clb.ax.set_title("Normalised\nactivity", fontsize=12)
    plt.ylabel("cell number")
    plt.xlabel("corrdior position")
    ticks = np.array([50, 100, 150])
    plt.xticks(ticks - 10, ticks)
    plt.tight_layout()
    plt.savefig(
        HERE.parent
        / "plots"
        / "place_cells"
        / f"place-cells-unrewarded-{session.mouse_name}-{session.date}"
    )


def activity_trial_speed(
    trial: TrialInfo, dff: np.ndarray, wheel_circumference: float, verbose: bool = False
) -> np.ndarray | None:
    position = degrees_to_cm(
        np.array(trial.rotary_encoder_position), wheel_circumference
    )

    frame_position = np.array(
        [
            state.closest_frame_start
            for state in trial.states_info
            if state.name in ["trigger_panda", "trigger_panda_post_reward"]
        ]
    )

    assert len(position) == len(frame_position)

    # TODO: Are these the right parameters
    first_position = 10
    last_position = 170
    step_size = 30
    sampling_rate = 30
    speed = get_speed_positions(
        position=position,
        first_position=first_position,
        last_position=last_position,
        step_size=step_size,
        sampling_rate=sampling_rate,
    )

    # Bin frames by speed
    bin_size = 1
    start = 0
    max_speed = (
        50  # arbitrary value, double-check but should be fine, TODO: make it dynamic?
    )

    dff_speed = list()
    for bin_start in range(start, max_speed, bin_size):
        frame_idx_bin = frame_position[
            np.logical_and(speed >= bin_start, speed < bin_start + bin_size)
        ]
        dff_bin = dff[:, frame_idx_bin]

        if verbose:
            print(f"bin_start: {bin_start}")
            print(f"bin_end: {bin_start + bin_size}")
            print(f"n_frames in bin: {len(frame_idx_bin)}")
        dff_speed.append(np.mean(dff_bin, axis=1))

    print("\n")
    return np.array(dff_speed).T


def get_speed_activity(
    trials: List[TrialInfo], dff: np.ndarray, wheel_circumference: float
) -> np.ndarray:
    test_matrices = list()

    for _ in range(10):
        train_idx = random.sample(range(len(trials)), len(trials) // 2)
        test_idx = [idx for idx in range(len(trials)) if idx not in train_idx]

        # Find the order in which to sort neurons in a random 50% of the trials
        train_matrix = np.nanmean(
            np.array(
                [
                    activity_trial_speed(trials[idx], dff, wheel_circumference)
                    for idx in train_idx
                ]
            ),
            axis=0,
        )
        train_matrix[:, 33 // 1 : 40 // 1] = 0
        train_matrix[:, 76 // 1 : 85 // 1] = 0
        train_matrix[:, 120 // 1 : 132 // 1] = 0

        peak_indices = np.argmax(train_matrix, axis=1)
        sorted_order = np.argsort(peak_indices)

        test_matrix = normalize(
            np.nanmean(
                np.array(
                    [
                        activity_trial_speed(trials[idx], dff, wheel_circumference)
                        for idx in test_idx
                    ]
                ),
                axis=0,
            ),
            axis=1,
        )

        test_matrices.append(test_matrix[sorted_order, :])

    return np.mean(np.array(test_matrices), 0)


def speed_cells_unsupervised(session: Cached2pSession, dff: np.ndarray) -> None:

    plt.figure()
    plt.title("Unsupervised Session")
    data = get_speed_activity(
        [trial for trial in session.trials if trial_is_imaged(trial)],
        dff,
        get_wheel_circumference_from_rig("2P"),
    )
    plt.imshow(data, aspect="auto", cmap="viridis")
    clb = plt.colorbar()
    clb.ax.set_title("Normalised\nactivity", fontsize=12)

    plt.ylabel("cell number")
    plt.xlabel("speed")
    ticks = np.array([1, 50])  # arbitrary speed x-ticks
    plt.xticks(ticks)

    plt.title("Unsupervised Session")

    plt.tight_layout()

    plt.savefig(
        HERE.parent
        / "plots"
        / "speed_cells"
        / f"speed-cells-unsupervised-{session.mouse_name}-{session.date}"
    )


if __name__ == "__main__":

    mouse = "JB018"
    date = "2024-12-06"

    with open(HERE.parent / "data" / "cached_2p" / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())
    print(f"Total number of trials: {len(session.trials)}")
    print(
        f"number of trials imaged {len([trial for trial in session.trials if trial_is_imaged(trial)])}"
    )

    dff = get_dff(mouse, date)

    assert (
        max(
            trial.states_info[-1].closest_frame_start
            for trial in session.trials
            if trial.states_info[-1].closest_frame_start is not None
        )
        < dff.shape[1]
    ), "Tiff is too short"

    if "unsupervised" in session.session_type.lower():
        place_cells_unsupervised(session, dff)
    else:
        place_cells(session, dff)

    # all_trial_activity = []
    # for trial in session.trials:
    #     if not trial_is_imaged(trial):
    #         continue

    #     trial_activity = activity_trial_position(
    #         trial, dff, get_wheel_circumference_from_rig("2P")
    #     )

    #     if trial_activity is None:
    #         continue

    #     all_trial_activity.append(trial_activity)

    # averaged = np.mean(np.array(all_trial_activity), axis=0)

    # for cell in range(averaged.shape[0]):
    #     plt.plot(averaged[cell, :] + cell * 0.5)
    # plt.show()
