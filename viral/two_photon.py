import logging
from pathlib import Path
import random
import sys
from typing import List
from scipy.stats import zscore
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
from viral.utils import trial_is_imaged
# SERVER_PATH = Path("/Volumes/MarcBusche/Josef/")
# TIFF_UMBRELLA = SERVER_PATH / "2P"


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


def speed_by_frame(trial: TrialInfo, wheel_circumference: float, verbose: bool = False) -> np.ndarray | None:
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

    # TODO: Think about step_size! Has to evenly divide the range, however we had it at 30 before (0-150; 150-180; ...)
    first_position = 0      # had to set it to max so that rastermap has dataset w/o NaNs
    last_position = 200     # had to set it to max so that rastermap has dataset w/o NaNs
    step_size = 1
    sampling_rate = 30
    speed_positions = get_speed_positions(
        position=position,
        first_position=first_position,
        last_position=last_position,
        step_size=step_size,
        sampling_rate=sampling_rate,
    )

    # Bin speed by frame
    frame_speeds = np.zeros_like(frame_position, dtype=float)
    for idx, sp in enumerate(speed_positions):
        if idx < len(frame_position) and frame_position[idx] < len(frame_speeds):
            frame_speeds[frame_position[idx]] = sp.speed
    if verbose:
        print(f"Calculated speeds for {len(frame_speeds)} frames.")
        print(f"Sample speeds: {frame_speeds[:10]}")
    return frame_speeds

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
    trial: TrialInfo, dff: np.ndarray, wheel_circumference: float, bin_size: int = 5, max_speed: int = 60, verbose: bool = False
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

    # TODO: Think about step_size! Has to evenly divide the range, however we had it at 30 before (0-150; 150-180; ...)
    first_position = 10
    last_position = 170
    step_size = 10
    sampling_rate = 30
    speed_positions = get_speed_positions(
        position=position,
        first_position=first_position,
        last_position=last_position,
        step_size=step_size,
        sampling_rate=sampling_rate,
    )

    # Bin frames by speed
    # bin_size = 5
    start = 0
    # max_speed = 60

    dff_speed = list()
    for bin_start in range(start, int(max_speed), bin_size):
        if bin_start == 0:
            frame_idx_bin = [
                frame_position[idx] for idx, sp in enumerate(speed_positions) if sp.speed == 0
            ]
            # TODO: Is the handling of speed = 0 correct?
            if len(frame_idx_bin) > 0:
                dff_bin = dff[:, frame_idx_bin]
                dff_speed.append(np.mean(dff_bin, axis=1))
            else:
                dff_speed.append(np.zeros(dff.shape[0]))  # handle no data for speed = 0
            continue
        frame_idx_bin = [
            frame_position[idx]
            for idx, sp in enumerate(speed_positions)
            if bin_start <= sp.speed < bin_start + bin_size
        ]
        if len(frame_idx_bin) == 0:
            dff_speed.append(np.zeros(dff.shape[0])) # speed = 0 gets included
            continue
        dff_bin = dff[:, frame_idx_bin]

        if verbose:
            print(f"bin_start: {bin_start}")
            print(f"bin_end: {bin_start + bin_size}")
            print(f"n_frames in bin: {len(frame_idx_bin)}")
        dff_speed.append(np.mean(dff_bin, axis=1))

    # print("\n")
    return np.array(dff_speed).T


def get_speed_activity(
    trials: List[TrialInfo], dff: np.ndarray, wheel_circumference: float, bin_size: int = 5, max_speed: int = 60
) -> np.ndarray:
    test_matrices = list()

    for _ in range(10):
        train_idx = random.sample(range(len(trials)), len(trials) // 2)
        test_idx = [idx for idx in range(len(trials)) if idx not in train_idx]

        # Find the order in which to sort neurons in a random 50% of the trials
        train_matrix = np.nanmean(
            np.array(
                [
                    activity_trial_speed(trials[idx], dff, wheel_circumference, bin_size, max_speed)
                    for idx in train_idx
                ]
            ),
            axis=0,
        )
        train_matrix[:, 33 // 1 : 40 // 1] = 0
        train_matrix[:, 76 // 1 : 85 // 1] = 0
        train_matrix[:, 120 // 1 : 132 // 1] = 0

        # Sort the matrix by greatest overall activity
        overall_activity = np.sum(train_matrix, axis=1)
        sorted_order = np.argsort(overall_activity)

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


def get_assert_dff(session: Cached2pSession) -> np.ndarray:
    dff = get_dff(session.mouse_name, session.date)

    assert (
        max(
            trial.states_info[-1].closest_frame_start
            for trial in session.trials
            if trial.states_info[-1].closest_frame_start is not None
        )
        < dff.shape[1]
    ), "Tiff is too short"

    return dff

def get_cached_sessions_by_type(mouse: str, supervised: bool = True) -> list[tuple[Cached2pSession, np.ndarray]]:
    cached_sessions = list()
    for file in (HERE.parent / "data" / "cached_2p").glob(f"{mouse}_*.json"):
        with open(file, "r") as f:
            session = Cached2pSession.model_validate_json(f.read())
            if supervised:
                if "learning day" in session.session_type.lower() and "unsupervised" not in session.session_type.lower():
                    dff = get_assert_dff(session)
                    cached_sessions.append((session, dff))
            else:
                if "unsupervised learning day" in session.session_type.lower():
                    dff = get_assert_dff(session)
                    cached_sessions.append((session, dff))
    cached_sessions.sort(key=lambda x: x[0].date)
    return cached_sessions

def speed_cells_unsupervised(sessions: list[tuple[Cached2pSession, np.ndarray]]) -> None:
    n_sessions = len(sessions)
    fig, axes = plt.subplots(nrows=n_sessions, ncols=1, figsize=(5, n_sessions * 5), constrained_layout=True)
    fig.suptitle("Unsupervised Sessions", fontsize=20)
    bin_size = 5
    max_speed = 60
    if n_sessions == 1:
        axes = [axes]
    for i, (session, dff) in enumerate(sessions):
        ax = axes[i]
        data = get_speed_activity(
            [trial for trial in session.trials if trial_is_imaged(trial)],
            dff,
            get_wheel_circumference_from_rig("2P"),
            bin_size,
            max_speed
        )
        num_speeds = data.shape[1]
        speeds = np.arange(0, num_speeds) * bin_size
        im = ax.imshow(data, aspect="auto", cmap="viridis")
        ax.set_title(f"{session.session_type} - {session.date}", fontsize=10)
        ax.set_ylabel("cell number")
        ax.set_xlabel("speed")
        ax.set_xticks(np.arange(num_speeds))
        ax.set_xticklabels([f"{speed}" for speed in speeds])
        clb = fig.colorbar(im, ax=ax)
        clb.ax.set_title("Normalised\nactivity", fontsize=12)

    plt.savefig(
        HERE.parent
        / "plots"
        / "speed_cells"
        / f"speed-cells-unsupervised-{mouse}", dpi=300
    )


def speed_cells(sessions: list[tuple[Cached2pSession, np.ndarray]]) -> None:
    n_sessions = len(sessions)
    fig, axes = plt.subplots(nrows=n_sessions, ncols=2, figsize=(12, n_sessions * 5), constrained_layout=True)
    fig.suptitle("Supervised Sessions", fontsize=18)
    bin_size = 5
    max_speed = 60
    if n_sessions == 1:
        axes = np.array([axes])
    for i, (session, dff) in enumerate(sessions):
        rewarded_ax = axes[i, 0]
        unrewarded_ax = axes[i, 1]

        rewarded_data = get_speed_activity(
            [trial for trial in session.trials if trial_is_imaged(trial) and trial.texture_rewarded],
            dff,
            get_wheel_circumference_from_rig("2P"),
            bin_size,
            max_speed
        )
    
        unrewarded_data = get_speed_activity(
            [trial for trial in session.trials if trial_is_imaged(trial) and not trial.texture_rewarded],
            dff,
            get_wheel_circumference_from_rig("2P"),
            bin_size,
            max_speed
        )

        num_speeds_rewarded = rewarded_data.shape[1]
        num_speeds_unrewarded = unrewarded_data.shape[1]
        rewarded_speeds = np.arange(0, num_speeds_rewarded) * bin_size
        unrewarded_speeds = np.arange(0, num_speeds_unrewarded) * bin_size

        im_rewarded = rewarded_ax.imshow(rewarded_data, aspect="auto", cmap="viridis")
        rewarded_ax.set_title(f"{session.session_type} - {session.date} (Rewarded)", fontsize=10)
        rewarded_ax.set_ylabel("cell number")
        rewarded_ax.set_xlabel("speed")
        rewarded_ax.set_xticks(np.arange(num_speeds_rewarded))
        rewarded_ax.set_xticklabels([f"{speed}" for speed in rewarded_speeds])
        clb_rewarded = fig.colorbar(im_rewarded, ax=rewarded_ax)
        clb_rewarded.ax.set_title("Normalised\nactivity", fontsize=12)

        im_unrewarded = unrewarded_ax.imshow(unrewarded_data, aspect="auto", cmap="viridis")
        unrewarded_ax.set_title(f"{session.session_type} - {session.date} (Unrewarded)", fontsize=10)
        unrewarded_ax.set_ylabel("cell number")
        unrewarded_ax.set_xlabel("speed")
        unrewarded_ax.set_xticks(np.arange(num_speeds_unrewarded))
        unrewarded_ax.set_xticklabels([f"{speed}" for speed in unrewarded_speeds])
        clb_unrewarded = fig.colorbar(im_unrewarded, ax=unrewarded_ax)
        clb_unrewarded.ax.set_title("Normalised\nactivity", fontsize=12)

    plt.savefig(
        HERE.parent
        / "plots"
        / "speed_cells"
        / f"speed-cells-supervised-{mouse}", dpi=300
    )

if __name__ == "__main__":

    mice = ["JB016", "JB018", "JB019", "JB020", "JB021", "JB022", "JB023", "JB026", "JB027"] # "JB011", "JB014", "JB015"
    # # mice = ["JB016", "JB018"]
    for mouse in mice:
    # mouse = "JB021"
    # dates = ["2024-12-12"]
    
    # for date in dates:
    #     with open(HERE.parent / "data" / "cached_2p" / f"{mouse}_{date}.json", "r") as f:
    #         session = Cached2pSession.model_validate_json(f.read())
    #     print(f"Total number of trials: {len(session.trials)}")
    #     print(
    #         f"number of trials imaged {len([trial for trial in session.trials if trial_is_imaged(trial)])}"
    #     )

    #     dff = get_dff(mouse, date)

    #     assert (
    #         max(
    #             trial.states_info[-1].closest_frame_start
    #             for trial in session.trials
    #             if trial.states_info[-1].closest_frame_start is not None
    #         )
    #         < dff.shape[1]
    #     ), "Tiff is too short"

    #     # for trial in session.trials:
    #     #     speed_by_frame(trial=trial, wheel_circumference=get_wheel_circumference_from_rig("2P"), verbose=True)

    #     if "unsupervised" in session.session_type.lower():
    #         place_cells_unsupervised(session, dff)
    #         speed_cells_unsupervised([(session, dff)])
    #     else:
    #         place_cells(session, dff)
    #         speed_cells([(session, dff)])

        unsupervised_sessions = get_cached_sessions_by_type(mouse, supervised=False)
        if unsupervised_sessions:
            speed_cells_unsupervised(unsupervised_sessions)

        supervised_sessions = get_cached_sessions_by_type(mouse, supervised=True)
        if supervised_sessions:
            speed_cells(supervised_sessions)

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
