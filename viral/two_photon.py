import logging
from pathlib import Path
import random
import sys
from typing import List
from scipy.stats import zscore
import seaborn as sns
from rastermap import Rastermap


# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


sns.set_theme(context="talk", style="ticks")

from viral.imaging_utils import trial_is_imaged
from viral.rastermap_utils import get_ITI_start_frame
from viral.utils import (
    array_bin_mean,
    degrees_to_cm,
    get_wheel_circumference_from_rig,
    normalize,
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
    print(f"Suite 2p path is {s2p_path}")
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)
    spks = np.load(s2p_path / "spks.npy")[iscell, :]
    # f_raw = np.load(s2p_path / "F.npy")[iscell, :]
    # f_neu = np.load(s2p_path / "Fneu.npy")[iscell, :]
    return spks
    # return compute_dff(subtract_neuropil(f_raw, f_neu))


def activity_trial_position(
    trial: TrialInfo,
    dff: np.ndarray,
    wheel_circumference: float,
    bin_size: int = 1,
    start: int = 10,
    max_position: int = 170,
    verbose: bool = False,
) -> np.ndarray:

    position = degrees_to_cm(
        np.array(trial.rotary_encoder_position), wheel_circumference
    )

    frame_position = np.array(
        [
            state.closest_frame_start
            for state in trial.states_info
            if state.name
            in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
        ]
    )

    assert len(position) == len(frame_position)

    dff_position = []

    for bin_start in range(start, max_position, bin_size):
        frame_idx_bin = np.unique(
            frame_position[
                np.logical_and(position >= bin_start, position < bin_start + bin_size)
            ]
        )
        dff_bin = dff[:, frame_idx_bin]

        if verbose:
            print(f"bin_start: {bin_start}")
            print(f"bin_end: {bin_start + bin_size}")
            print(f"n_frames in bin: {len(frame_idx_bin)}")
        dff_position.append(np.mean(dff_bin, axis=1))

    return np.array(dff_position).T


def remove_landmarks_from_train_matrix(train_matrix: np.ndarray) -> np.ndarray:
    """We seem to get a lot of neurons with a peak at the landmark. This removes the landmark from the
    train matrix so cells do not get sorted by their landmark peak
    TODO: This will not work if bin_size != 1, originally had an integer division which may deal with this
    """

    train_matrix[:, 33:40] = 0
    train_matrix[:, 76:85] = 0
    train_matrix[:, 120:132] = 0
    return train_matrix


def get_position_activity(
    trials: List[TrialInfo],
    dff: np.ndarray,
    wheel_circumference: float,
    bin_size: int,
    start: int,
    max_position: int,
    remove_landmarks: bool,
    ITI_bin_size: int,
) -> np.ndarray:

    test_matrices = []
    for _ in range(10):
        train_idx = random.sample(range(len(trials)), len(trials) // 2)
        test_idx = [idx for idx in range(len(trials)) if idx not in train_idx]

        # Find the order in which to sort neurons in a random 50% of the trials
        train_matrix = np.nanmean(
            np.array(
                [
                    activity_trial_position(
                        trials[idx],
                        dff,
                        wheel_circumference,
                        bin_size=bin_size,
                        start=start,
                        max_position=max_position,
                    )
                    for idx in train_idx
                ]
            ),
            axis=0,
        )

        if remove_landmarks:
            train_matrix = remove_landmarks_from_train_matrix(train_matrix)

        peak_indices = np.argmax(train_matrix, axis=1)
        sorted_order = np.argsort(peak_indices)

        # model = Rastermap(n_PCs=200, n_clusters=5, locality=1, time_lag_window=0).fit(
        #     train_matrix
        # )
        # sorted_order = model.isort

        # TODO: Review whether this is the best normalisation function
        test_matrix = normalize(
            np.nanmean(
                np.array(
                    [
                        activity_trial_position(
                            trials[idx],
                            dff,
                            wheel_circumference,
                            bin_size=bin_size,
                            start=start,
                            max_position=max_position,
                        )
                        for idx in test_idx
                    ]
                ),
                axis=0,
            ),
            axis=1,
        )

        test_matrices.append(test_matrix[sorted_order, :])

    test_matrices_averaged = np.mean(np.array(test_matrices), 0)

    ITI = get_ITI_matrix(trials, dff, bin_size=ITI_bin_size)
    ITI = ITI[np.argsort(np.argmax(test_matrices_averaged, axis=1)), :]

    return np.hstack((test_matrices_averaged, ITI))


def running_during_ITI(trial: TrialInfo) -> bool:

    trigger_states = np.array(
        [
            state.name
            for state in trial.states_info
            if state.name
            in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
        ]
    )

    ITI_positions = degrees_to_cm(
        np.array(trial.rotary_encoder_position)[trigger_states == "trigger_panda_ITI"],
        get_wheel_circumference_from_rig("2P"),
    )
    return max(ITI_positions) - min(ITI_positions) > 20


def get_ITI_matrix(
    trials: List[TrialInfo],
    dff: np.ndarray,
    bin_size: int,
) -> np.ndarray:
    """
    In theory will be 600 frames in an ITI (as is always 20 seconds)
    In practise it's 599 or 598 (or could be something like 597 or 600, fix
    the assertion if so).
    Or could be less if you stop the trial in the ITI.
    So we'll take the first 598 frames of any trial that has 599 or 598
    """

    matrices = []

    for trial in trials:
        assert trial.trial_end_closest_frame is not None

        if running_during_ITI(trial):
            continue
        chunk = dff[:, get_ITI_start_frame(trial) : int(trial.trial_end_closest_frame)]
        n_frames = chunk.shape[1]
        if n_frames < 550:
            # Imaging stopped in the middle
            continue
        elif n_frames in {598, 599}:
            matrices.append(array_bin_mean(chunk[:, :598], bin_size=bin_size))
        else:
            raise ValueError(f"Chunk with {n_frames} frames not understood")

    print(f"Number of still ITI trials: {len(matrices)}")
    # return normalize(np.mean(np.array(matrices), 0), axis=1)
    print(f"ITI shape {matrices[0].shape}")

    # for idx in range(len(matrices)):
    #     matrices[idx] = np.hstack([matrices[idx], np.ones((matrices[idx].shape[0], 1))])
    # return np.hstack([normalize(matrix, axis=1) for matrix in matrices])

    return np.mean(np.array([normalize(matrix, axis=1) for matrix in matrices]), axis=0)


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

    start = 30
    max_position = 180
    bin_size = 5
    ITI_bin_size = 10

    vmin = 0
    vmax = 1

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
            start=start,
            bin_size=bin_size,
            max_position=max_position,
            remove_landmarks=False,
            ITI_bin_size=ITI_bin_size,
        ),
        aspect="auto",
        cmap="viridis",
        vmax=vmax,
        vmin=vmin,
    )

    clb = plt.colorbar()
    clb.ax.set_title("Normalised\nactivity", fontsize=12)
    plt.ylabel("cell number")
    plt.xlabel("corrdior position")
    ticks = np.array([50, 100, 150])

    plt.xticks((ticks - start) / bin_size, ticks)
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
            start=start,
            bin_size=bin_size,
            max_position=max_position,
            remove_landmarks=False,
            ITI_bin_size=ITI_bin_size,
        ),
        aspect="auto",
        cmap="viridis",
        vmax=vmax,
        vmin=vmin,
    )

    clb = plt.colorbar()
    clb.ax.set_title("Normalised\nactivity", fontsize=12)
    plt.ylabel("cell number")
    plt.xlabel("corrdior position")
    plt.xticks((ticks - start) / bin_size, ticks)
    plt.tight_layout()
    plt.savefig(
        HERE.parent
        / "plots"
        / "place_cells"
        / f"place-cells-unrewarded-{session.mouse_name}-{session.date}"
    )
    plt.show()


if __name__ == "__main__":

    mouse = "JB027"
    date = "2025-02-26"

    with open(HERE.parent / "data" / "cached_2p" / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())

    print(f"Total number of trials: {len(session.trials)}")
    print(
        f"number of trials imaged {len([trial for trial in session.trials if trial_is_imaged(trial)])}"
    )

    dff = get_dff(mouse, date)
    print("Got dff")

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
