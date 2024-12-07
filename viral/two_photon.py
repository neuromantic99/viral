import logging
from pathlib import Path
import sys
from typing import List
from scipy.stats import zscore


# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


from viral.utils import (
    degrees_to_cm,
    get_wheel_circumference_from_rig,
    shuffle,
    trial_is_imaged,
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
    trial: TrialInfo,
    dff: np.ndarray,
    wheel_circumference: float,
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
    bin_size = 2
    max_position = 170
    start = 10

    dff_position = []

    for bin_start in range(start, max_position, bin_size):
        frame_idx_bin = frame_position[
            np.logical_and(position >= bin_start, position < bin_start + bin_size)
        ]
        dff_bin = dff[:, frame_idx_bin]
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
    return sort_matrix_peak(
        normalize(
            np.nanmean(
                np.array(
                    [
                        activity_trial_position(trial, dff, wheel_circumference)
                        for trial in trials
                    ]
                ),
                axis=0,
            ),
            axis=1,
        )
    )


def place_cells_unsupervised(session: Cached2pSession, dff: np.ndarray) -> None:

    plt.figure()
    plt.title("Unsupervised Session")
    data = get_position_activity(
        [trial for trial in session.trials if trial_is_imaged(trial)],
        dff,
        get_wheel_circumference_from_rig("2P"),
    )
    plt.imshow(data, aspect="auto")

    # From activity trial position
    start = 10
    stop = 170
    # bin_size = (stop - start) / data.shape[1]
    # tick_step = 20

    plt.xticks(
        np.linspace(0, data.shape[1], 10),
        np.linspace(start, stop, 10).astype(int),
    )
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
    )
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
    )
    plt.savefig(
        HERE.parent
        / "plots"
        / "place_cells"
        / f"place-cells-unrewarded-{session.mouse_name}-{session.date}"
    )


if __name__ == "__main__":

    mouse = "JB018"
    date = "2024-11-18"

    with open(HERE.parent / "data" / "cached_2p" / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())
    print(f"number of trials {len(session.trials)}")
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
