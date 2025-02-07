import numpy as np
import sys
import random
from pathlib import Path
from scipy.interpolate import interp1d
from typing import List

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.utils import (
    TrialInfo,
    trial_is_imaged,
    degrees_to_cm,
    get_wheel_circumference_from_rig,
)
from viral.models import Cached2pSession
from viral.constants import TIFF_UMBRELLA
from viral.two_photon import (
    compute_dff,
    subtract_neuropil,
    normalize,
)


def get_spks_pos(s2p_path: str) -> tuple[np.ndarray]:
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)
    spks = np.load(s2p_path / "spks.npy")[iscell, :]
    stat = np.load(s2p_path / "stat.npy", allow_pickle=True)[iscell]
    pos = np.array([[int(coord) for coord in cell["med"]] for cell in stat])
    xpos = pos[:, 0]
    ypos = pos[:, 1]
    return spks, xpos, ypos


def get_dff(s2p_path: str) -> np.ndarray:
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)
    f_raw = np.load(s2p_path / "F.npy")[iscell, :]
    f_neu = np.load(s2p_path / "Fneu.npy")[iscell, :]
    return compute_dff(subtract_neuropil(f_raw, f_neu))


def align_trial_frames(trials: list[TrialInfo], ITI: bool = False) -> np.ndarray:
    """Extract trial start and end frames, reward condition"""
    trial_frames = np.array(
        [
            [trial.trial_start_closest_frame, trial.trial_end_closest_frame]
            for trial in trials
        ]
    )
    if ITI == False:
        ITI_start_frames = np.array([get_ITI_start_frame(trial) for trial in trials])
        trial_frames[:, 1] = ITI_start_frames
    assert (
        len(trials) == trial_frames.shape[0]
    ), "Number of trials does not match number of trial times"
    sorted_trial_frames = trial_frames[np.argsort(trial_frames[:, 0])]
    for i in range(len(sorted_trial_frames) - 1):
        assert (
            sorted_trial_frames[i, 1] < sorted_trial_frames[i + 1, 0]
        ), "Overlapping frames for trials"
    rewarded = np.array([trial.texture_rewarded for trial in trials])
    return np.column_stack((sorted_trial_frames, rewarded))


def get_ITI_start_frame(trial: TrialInfo) -> float:
    for state in trial.states_info:
        if state.name == "ITI":
            ITI_start_frame = state.closest_frame_start
    return ITI_start_frame


def get_signal_for_trials(
    signal: np.ndarray, trial_frames: np.ndarray
) -> list[np.ndarray]:
    "Return spks or dff for trials"
    signal_trials = list()
    for trial in trial_frames:
        trial_start, trial_end, _ = trial.astype(int)
        signal_trials.append(signal[:, trial_start : trial_end + 1])
    return signal_trials


def get_frame_position(
    trial: TrialInfo,
    trial_frames: np.ndarray,
    wheel_circumference: float,
    ITI: bool = False,
) -> np.ndarray:
    if ITI == True:
        frames_start_end = np.array(
            [
                [state.closest_frame_start, state.closest_frame_end]
                for state in trial.states_info
                if state.name
                in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
            ]
        )
    else:
        frames_start_end = np.array(
            [
                [state.closest_frame_start, state.closest_frame_end]
                for state in trial.states_info
                if state.name in ["trigger_panda", "trigger_panda_post_reward"]
            ]
        )
    degrees = np.array(trial.rotary_encoder_position)
    positions = degrees_to_cm(degrees, wheel_circumference)
    assert len(frames_start_end) == len(positions)
    frames_positions_combined = list()
    for i in range(len(frames_start_end)):
        frames_positions_combined.append([frames_start_end[i, 0], positions[i]])
        frames_positions_combined.append([frames_start_end[i, 1], positions[i]])
    frames_positions_combined = np.array(frames_positions_combined)
    unique_frames_positions = np.unique(frames_positions_combined, axis=0)
    # Interpolation is needed as the position is not stored when the reward comes on, hence position for several frames might be missing
    interp_func = interp1d(
        unique_frames_positions[:, 0],
        unique_frames_positions[:, 1],
        kind="linear",
        fill_value="extrapolate",
    )
    interpolated_positions = interp_func(trial_frames)
    assert len(trial_frames) == len(
        interpolated_positions
    ), "# of trial frames is unequal to # of interpolated positions"
    return np.column_stack((trial_frames, interpolated_positions))


def get_speed_frame(frame_position: np.ndarray) -> np.ndarray:
    # TODO: Is this ok like this?
    frames_diff = np.diff(frame_position[:, 0])
    positions_diff = np.diff(frame_position[:, 1])
    speeds = positions_diff / frames_diff
    # little hack, have to account for one more frame
    last_speed = speeds[-1] if len(speeds) > 0 else 0
    # Add last frame with speed of the last computed frame
    new_frame = np.array([[frame_position[-1, 0], last_speed]])
    return np.vstack([np.column_stack((frame_position[:-1, 0], speeds)), new_frame])


def get_lick_index(trial: TrialInfo) -> np.ndarray | None:
    licks_start = np.array(
        [event.closest_frame for event in trial.events_info if event.name == "Port1In"]
    )
    licks_end = np.array(
        [event.closest_frame for event in trial.events_info if event.name == "Port1Out"]
    )
    # Handle trial w/o licks
    if len(licks_start) == 0 or len(licks_end) == 0:
        return None
    # Cutting of pairs with e.g. missing end frame
    # TODO: is this a good approach?
    valid_pairs = list()
    start_idx, end_idx = 0, 0
    while start_idx < len(licks_start) and end_idx < len(licks_end):
        if licks_start[start_idx] <= licks_end[end_idx]:
            valid_pairs.append((licks_start[start_idx], licks_end[end_idx]))
            start_idx += 1
            end_idx += 1
        else:
            end_idx += 1
    start_end_frames = np.array(valid_pairs, dtype=int)
    lick_segments = list()
    for start, end in start_end_frames.astype(int):
        if end > start:
            segment_frames = np.arange(start, end + 1, 1)
            lick_segments.append(segment_frames)
    if len(lick_segments) != 0:
        licks = np.concatenate(lick_segments)
        return np.unique(licks, axis=0)
    else:
        return None


def get_reward_index(trial: TrialInfo) -> np.ndarray | None:
    # Caution: When changing number of rewards, the state names have to be changed
    start_frames = np.array(
        [
            state.closest_frame_start
            for state in trial.states_info
            if state.name in ["reward_on1"]
        ]
    )
    end_frames = np.array(
        [
            state.closest_frame_start
            for state in trial.states_info
            if state.name in ["reward_off3"]
        ]
    )
    # Handle trial w/o rewards
    if len(start_frames) == 0 or len(end_frames) == 0:
        return None
    # Cutting of pairs with e.g. missing end time
    # TODO: is this a good approach?
    valid_pairs = list()
    start_idx, end_idx = 0, 0
    while start_idx < len(start_frames) and end_idx < len(end_frames):
        if start_frames[start_idx] <= end_frames[end_idx]:
            valid_pairs.append((start_frames[start_idx], end_frames[end_idx]))
            start_idx += 1
            end_idx += 1
        else:
            end_idx += 1
    start_end_frames = np.array(valid_pairs, dtype=int)
    reward_segments = list()
    for start, end in start_end_frames.astype(int):
        if end > start:
            segment_frames = np.arange(start, end + 1, 1)
            reward_segments.append(segment_frames)
    if len(reward_segments) != 0:
        rewards = np.concatenate(reward_segments) if reward_segments else None
        return np.unique(rewards, axis=0)
    else:
        return None


def remap_to_continuous_indices(
    original_indices: np.ndarray, frame_mapping: dict
) -> np.ndarray:
    return np.array(
        [frame_mapping[idx] for idx in original_indices if idx in frame_mapping]
    )


def process_session(session: Cached2pSession, wheel_circumference: float) -> None:
    s2p_path = TIFF_UMBRELLA / session.date / session.mouse_name / "suite2p" / "plane0"
    spks, xpos, ypos = get_spks_pos(s2p_path)
    trials = [trial for trial in session.trials if trial_is_imaged(trial)]
    if not trials:
        print("No trials imaged")
        exit()
    aligned_trial_frames = align_trial_frames(trials)
    assert len(trials) == len(
        aligned_trial_frames
    ), "Number of trials and aligned_trial_frames do not match"
    spks_trials = get_signal_for_trials(signal=spks, trial_frames=aligned_trial_frames)
    dff = get_dff(s2p_path)
    # dff_trials = get_signal_for_trials(signal=dff, trial_frames=aligned_trial_frames)
    assert len(trials) == len(
        spks_trials
    ), "Number of trials and number of spks for trials do not match"
    lcks_combined = np.array([], dtype=int)
    rwrds_combined = np.array([], dtype=int)
    pstns_combined = np.empty((0, 2))
    spd_combined = np.empty((0, 2))
    crrdr_wdths_combined = aligned_trial_frames[:, 1] - aligned_trial_frames[:, 0]
    for trial, (start, end, rewarded) in zip(trials, aligned_trial_frames):
        trial_frames = np.arange(start, end + 1, 1)
        activity_trial_speed(
            trial, dff, trial_frames, wheel_circumference=wheel_circumference
        )
        licks = get_lick_index(trial)
        rewards = get_reward_index(trial)
        frames_positions = get_frame_position(trial, trial_frames, wheel_circumference)
        speed = get_speed_frame(frames_positions)
        if licks is not None:
            lcks_combined = np.concatenate((lcks_combined, licks))
        if rewards is not None:
            rwrds_combined = np.concatenate((rwrds_combined, rewards))
        pstns_combined = np.vstack((pstns_combined, frames_positions))
        spd_combined = np.vstack((spd_combined, speed))
    spks_combined = np.hstack(spks_trials)
    # TODO: sanity check for the frame mapping
    continuous_frames = np.arange(pstns_combined.shape[0])
    discontinuous_frames = pstns_combined[:, 0]
    frame_mapping = {
        orig: cont for orig, cont in zip(discontinuous_frames, continuous_frames)
    }
    lcks_combined = remap_to_continuous_indices(lcks_combined, frame_mapping)
    rwrds_combined = remap_to_continuous_indices(rwrds_combined, frame_mapping)
    crrdr_strts_combined = np.column_stack(
        (
            remap_to_continuous_indices(aligned_trial_frames[:, 0], frame_mapping),
            aligned_trial_frames[:, -1],
        )
    )
    # corridor_starts
    # corridor_widths
    # corridor_imgs
    # VRpos
    # reward_inds
    # sound_inds
    # lick_inds
    # run
    neur_path = (
        HERE.parent
        / "data"
        / "cached_for_rastermap"
        / f"{session.mouse_name}_{session.date}_corridor_neur.npz"
    )
    np.savez(neur_path, spks=spks_combined, xpos=xpos, ypos=ypos)
    behavior_path = (
        HERE.parent
        / "data"
        / "cached_for_rastermap"
        / f"{session.mouse_name}_{session.date}_corridor_behavior.npz"
    )
    np.savez(
        behavior_path,
        corridor_starts=crrdr_strts_combined,
        corridor_widths=crrdr_wdths_combined,
        VRpos=pstns_combined[:, 1],
        reward_inds=rwrds_combined,
        lick_inds=lcks_combined,
        run=spd_combined[:, 1],
        corridor_imgs=np.zeros((2, 1200, 100)),  # placeholder
        sound_inds=np.zeros(1),  # placeholder
    )
    print(f"Done")


# Move eventually, but do it here for now
def activity_trial_speed(
    trial: TrialInfo,
    dff: np.ndarray,
    trial_frames: np.ndarray,
    wheel_circumference: float,
    remove_landmarks: bool = True,
) -> np.ndarray:
    """Return frames x position x speed"""

    frames_positions = get_frame_position(trial, trial_frames, wheel_circumference)
    speed = get_speed_frame(frames_positions)
    frames_positions_speed_activity = np.column_stack(
        [
            frames_positions,
            speed[
                :,
                1,
            ],
            get_signal_for_trials(dff, trial_frames),
        ]
    )

    1 / 0


def get_speed_activity(
    trials: List[TrialInfo],
    aligned_trial_frames: np.ndarray,
    dff: np.ndarray,
    wheel_circumference: float,
    bin_size: int = 1,
    start: int = 10,
    max_position: int = 170,
    remove_landmarks: bool = True,
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

    return np.mean(np.array(test_matrices), 0)


if __name__ == "__main__":
    wheel_circumference = get_wheel_circumference_from_rig("2P")
    for mouse_name in ["JB011"]:
        print(f"Off we go for {mouse_name}...")
        for file in (HERE.parent / "data" / "cached_2p").glob(f"{mouse_name}_*.json"):
            print(file)
            with open(file, "r") as f:
                session = Cached2pSession.model_validate_json(f.read())
            f.close()
            process_session(session, wheel_circumference)
