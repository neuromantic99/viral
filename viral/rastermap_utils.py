import numpy as np
import sys
import random
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional

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
from viral.two_photon import compute_dff, subtract_neuropil, normalize, get_dff


def get_spks_pos(s2p_path: Path) -> tuple[np.ndarray]:
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)
    spks = np.load(s2p_path / "spks.npy")[iscell, :]
    stat = np.load(s2p_path / "stat.npy", allow_pickle=True)[iscell]
    pos = np.array([[int(coord) for coord in cell["med"]] for cell in stat])
    xpos = pos[:, 1]
    ypos = -1 * pos[:, 0]  # y-axis is inverted
    return spks, xpos, ypos


def align_trial_frames(trials: List[TrialInfo], ITI: bool = True) -> np.ndarray:
    """Align trial frames and return array with trial frames and reward condition.

    Args:
        trials (List[TrialInfo]):       A list of TrialInfo objects.
        ITI (bool, optional):           Include the frames in the inter-trial interval (ITI). Defaults to False.
                                        If ITI, is False, then trial end is defined as the start of the ITI.

    Returns:
        np.ndarray:                     A Numpy array of shape (n_trials, 3). Each row representing a trial with
            - start_frame (int)
            - end_frame (int)
            - rewarded (bool)
    """
    trial_frames = np.array(
        [
            [trial.trial_start_closest_frame, trial.trial_end_closest_frame]
            for trial in trials
        ]
    )
    if not ITI:
        ITI_start_frames = np.array([get_ITI_start_frame(trial) for trial in trials])
        trial_frames[:, 1] = ITI_start_frames - 1
    assert (
        len(trials) == trial_frames.shape[0]
    ), "Number of trials does not match number of trial times"
    for i in range(len(trial_frames) - 1):
        assert (
            trial_frames[i, 1] < trial_frames[i + 1, 0]
        ), "Overlapping frames for trials"
    rewarded = np.array([trial.texture_rewarded for trial in trials])
    return np.column_stack((trial_frames, rewarded))


def get_ITI_start_frame(trial: TrialInfo) -> int:
    for state in trial.states_info:
        if state.name == "ITI":
            return state.closest_frame_start
        elif state.name == "trigger_ITI":
            return state.closest_frame_start
    raise ValueError("ITI state not found")


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
    ITI: bool = True,
) -> np.ndarray:
    """Get the position for every frame and return frames and positions as array.

    Args:
        trial (TrialInfo):              A TrialInfo object representing the trial to be analysed.
        trial_frames (np.ndarray):      Numpy array representing the frames in the trial.
        wheel_circumference (float):    The circumference of the wheel on the rig used in centimetres.
        ITI (bool, optional):           Whether to include the frames in the inter-trial interval (ITI). Defaults to True.

    Returns:
        np.ndarray:                     A Numpy array of shape (frames, 2). Each row representing a frame with:
            frame_idx (int)
            position (float)
    """
    if ITI:
        # There is an old version of our IBL rig code which does not store any positions during the ITI
        ITI_states = {state.name for state in trial.states_info if "ITI" in state.name}
        assert len(ITI_states) != 0, "No ITI state found"
        ITI_patch = True if "ITI" in ITI_states else False
        frames_start_end = np.array(
            [
                [state.closest_frame_start, state.closest_frame_end]
                for state in trial.states_info
                if state.name in ["trigger_panda", "trigger_panda_post_reward"]
                or (state.name == "trigger_panda_ITI" and not ITI_patch)
            ]
        )
    else:
        ITI_patch = False
        frames_start_end = np.array(
            [
                [state.closest_frame_start, state.closest_frame_end]
                for state in trial.states_info
                if state.name in ["trigger_panda", "trigger_panda_post_reward"]
                or (state.name == "trigger_panda_ITI" and not ITI_patch)
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

    if not ITI:
        ITI_start_frame = get_ITI_start_frame(trial)
        ITI_end_frame = trial.trial_end_closest_frame
        mask = ~(
            (frames_positions_combined[:, 0] >= ITI_start_frame)
            & (frames_positions_combined[:, 0] <= ITI_end_frame)
        )

        # Apply the mask to remove ITI frames
        frames_positions_combined = frames_positions_combined[mask]

    # Backfilling with the first recorded position as there is no position being recorded for the first frame, hence, the interpolation would error
    if trial_frames[0] < unique_frames_positions[0, 0]:
        first_position = np.array([[trial_frames[0], unique_frames_positions[0, 1]]])
        unique_frames_positions = np.vstack([first_position, unique_frames_positions])
    # Backfilling with last recorded position as there might not be a position recorded for the last frame, hence, the interpolation would error
    if trial_frames[-1] > np.max(unique_frames_positions[-1, 0]):
        last_position = np.array([trial_frames[-1], unique_frames_positions[-1, 1]])
        unique_frames_positions = np.vstack([unique_frames_positions, last_position])
    # Interpolation is needed as the position is not stored when the reward comes on, hence position for several frames might be missing
    interp_func = interp1d(
        unique_frames_positions[:, 0],
        unique_frames_positions[:, 1],
        kind="linear",
        fill_value="interpolate",
    )
    interpolated_positions = interp_func(trial_frames)

    if ITI_patch:
        # If patching is needed, fill up the ITI frames with positions zero
        ITI_start_frame = get_ITI_start_frame(trial)
        ITI_end_frame = trial.trial_end_closest_frame
        ITI_frame_indices = np.where(
            (trial_frames >= ITI_start_frame) & (trial_frames <= ITI_end_frame)
        )[0]
        interpolated_positions[ITI_frame_indices] = 0

    assert (interpolated_positions >= -15).all(), "one or more position(s) are negative"
    # Wanted to account for the mouse rolling backwardwards/forwards resulting in slightly negative values
    # 15 is obviously a bit arbitrary
    assert len(trial_frames) == len(
        interpolated_positions
    ), "# of trial frames is unequal to # of interpolated positions"
    return np.column_stack((trial_frames, interpolated_positions))


def get_speed_frame(frame_position: np.ndarray, bin_size: int = 5) -> np.ndarray:
    """Get the speed for every frame and return frames and speed as array.

    Args:
        frame_position (np.ndarray):    A Numpy array of shape (frames, 2). Each row representing a frame with frame index and corresponding speed.
        bin_size (int):                 How many frames to bin over. Defaults to 5.

    Returns:
        np.ndarray:                     A Numpy array of shape (frames, 2). Each row representing a frame with:
            frame_idx (int)
            speed (float)
    """
    n_frames = frame_position.shape[0]
    n_bins = n_frames // bin_size
    speeds = np.zeros(frame_position.shape[0])
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size
        start_frame, start_pos = frame_position[start_idx]
        end_frame, end_pos = frame_position[end_idx - 1]
        time_diff = end_frame - start_frame
        pos_diff = end_pos - start_pos
        speed = pos_diff / time_diff if time_diff != 0 else 0
        speeds[start_idx:end_idx] = speed

    remaining = n_frames % bin_size
    if remaining > 0:
        start_idx = n_bins * bin_size
        start_frame, start_pos = frame_position[start_idx]
        end_frame, end_pos = frame_position[-1]

        time_diff = end_frame - start_frame
        pos_diff = end_pos - start_pos
        speed = pos_diff / time_diff if time_diff != 0 else 0

        speeds[start_idx:] = speed

    # Replace negative speeds with 0 *** TODO: is this a good approach? Should we not rather fix it in the positions?
    speeds[speeds < 0] = 0
    speeds[speeds > 5] = np.mean(speeds)
    result = np.column_stack((frame_position[:, 0], speeds))
    return result


def get_lick_index(trial: TrialInfo) -> np.ndarray | None:
    """Return frame indices of frames when licks occured."""
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
    """Return frame indices of frames when rewards came on."""
    # Caution: When changing number of rewards, the state names have to be changed
    if not trial.texture_rewarded:
        return None
    else:
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
        allowed_rewards = {"reward_on1", "reward_on2", "reward_on3"}
        # Ensuring that all allowed reward_on states are to be found in the states info
        for reward in allowed_rewards:
            assert any(
                state.name == reward for state in trial.states_info
            ), f"'{reward}' is not in states_info"

        # Ensuring no other "reward_onX" states exist (e.g. 'reward_on4')
        for state in trial.states_info:
            if state.name.startswith("reward_on") and state.name not in allowed_rewards:
                raise AssertionError(f"Unexpected reward state found: {state.name}")
        # Cutting of pairs with e.g. missing end time
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


def create_frame_mapping(positions_combined: np.ndarray) -> dict:
    """Create a frame_mapping dictionary with original and continuous frame indices"""
    # Takes postions_combined array and takes the first column with the frame indices
    # (The positions array is the one that should have all frames in an imaged trial)
    continuous_frames = np.arange(positions_combined.shape[0])
    discontinuous_frames = positions_combined[:, 0]
    return {orig: cont for orig, cont in zip(discontinuous_frames, continuous_frames)}


def remap_to_continuous_indices(
    original_indices: np.ndarray, frame_mapping: dict
) -> np.ndarray:
    """Re-assign frame indices to be continuous, according to the frame_mapping dictionary."""
    return np.array(
        [frame_mapping[idx] for idx in original_indices if idx in frame_mapping]
    )


def load_data(session: Cached2pSession, s2p_path: Path) -> tuple:
    """Return spks, xpos, ypos and trials loaded from cache."""
    spks, xpos, ypos = get_spks_pos(s2p_path)
    trials = [trial for trial in session.trials if trial_is_imaged(trial)]
    if not trials:
        print("No trials imaged")
        exit()
    return spks, xpos, ypos, trials


def align_validate_data(spks: np.ndarray, trials: List[TrialInfo]) -> tuple:
    """Alignes frames and spikes to trial data."""
    aligned_trial_frames = align_trial_frames(trials)
    assert len(trials) == len(
        aligned_trial_frames
    ), "Number of trials and aligned_trial_frames do not match"
    spikes_trials = get_signal_for_trials(
        signal=spks, trial_frames=aligned_trial_frames
    )
    assert len(trials) == len(
        spikes_trials
    ), "Number of trials and number of spks for trials do not match"

    return aligned_trial_frames, spikes_trials


def get_corridor_width(
    trial: TrialInfo, valid_frame_indices: np.ndarray, include_ITI: bool = True
) -> int:
    """Return the corridor width in frames for a trial, with option to exclude ITI frames.

    Args:
        trial (TrialInfo):      A TrialInfo object representing the trial to be analysed.
        include_ITI (bool):     Whether to include ITI frames in the width calculation. Defaults to False.

    Returns:
        tuple[int, int]: A tuple containing:
            - start_frame (int): The starting frame of the corridor
            - width (int): The width of the corridor in frames
    """
    if include_ITI:
        start_frame = trial.trial_start_closest_frame
        end_frame = trial.trial_end_closest_frame
    else:
        start_frame = trial.trial_start_closest_frame
        ITI_frame = get_ITI_start_frame(trial)
        end_frame = ITI_frame - 1

    valid_trial_frames = valid_frame_indices[
        (valid_frame_indices >= start_frame) & (valid_frame_indices <= end_frame)
    ]
    return len(valid_trial_frames)


def process_behavioural_data(
    trials: List[TrialInfo],
    aligned_trial_frames: np.ndarray,
    wheel_circumference: float,
    speed_bin_size: int = 5,
) -> tuple:
    """Process behavioural data and return licks, rewards, positions and speed as arrays."""
    licks_combined = np.array([], dtype=int)
    rewards_combined = np.array([], dtype=int)
    positions_combined = np.empty((0, 2))
    speed_combined = np.empty((0, 2))

    for trial, (start, end, rewarded) in zip(trials, aligned_trial_frames):
        trial_frames = np.arange(start, end + 1, 1)
        licks = get_lick_index(trial)
        rewards = get_reward_index(trial)
        frames_positions = get_frame_position(trial, trial_frames, wheel_circumference)
        speed = get_speed_frame(frames_positions, speed_bin_size)

        if licks is not None:
            licks_combined = np.concatenate((licks_combined, licks))
        if rewards is not None:
            rewards_combined = np.concatenate((rewards_combined, rewards))
        positions_combined = np.vstack((positions_combined, frames_positions))
        speed_combined = np.vstack((speed_combined, speed))
    return licks_combined, rewards_combined, positions_combined, speed_combined


def filter_speed_position(
    speed: np.ndarray,
    frames_positions: np.ndarray,
    speed_threshold: Optional[float],
    position_threshold: Optional[float],
    filter_speed: bool = True,
    filter_position: bool = True,
) -> np.ndarray:
    """Return a mask of valid frames"""

    valid_mask = np.ones(len(speed), dtype=bool)

    # specified positions that should always be included
    if filter_position:
        position_mask = frames_positions[:, 1] >= position_threshold
        valid_mask &= position_mask  # bitwise AND
    # filter out frames with sub-threshold speeds
    # converting cm/s to cm/frames
    if filter_speed:
        speed_mask = (
            speed[:, 1] >= (speed_threshold / 30)
        ) | position_mask  # bitwise OR
        valid_mask &= speed_mask

    return valid_mask


def get_trial_valid_starts(
    aligned_trial_frames: np.ndarray, valid_frame_indices: np.ndarray
) -> np.ndarray:
    """Get the first valid frame for each trial.

    Args:
        aligned_trial_frames (np.ndarray):  A Numpy array of trial start/end frames and reward conditions.
        valid_frame_indices (np.ndarray):   A Numpy array of frame indices that passed filtering.

    Returns:
        np.ndarray:                         A Numpy array containing the first valid frame index for each trial and reward condition.
    """
    trial_valid_starts = list()
    for trial_frames in aligned_trial_frames:
        start, end, rewarded = trial_frames
        # find first valid frame that falls within this trial's bounds
        valid_trial_frames = valid_frame_indices[
            (valid_frame_indices >= start) & (valid_frame_indices <= end)
        ]
        # TODO: will this if loop cause issues in the future?
        if len(valid_trial_frames) > 0:
            trial_valid_starts.append([valid_trial_frames[0], rewarded])
    return np.array(trial_valid_starts)


# TODO: test the ITI_split
def save_neural_data(
    session: Cached2pSession,
    spikes_combined: np.ndarray,
    spikes_combined_noITI: Optional[np.ndarray],
    xpos: np.ndarray,
    ypos: np.ndarray,
    ITI_split: bool = True,  # Whether to split the spks into files with and without ITI
) -> None:
    base_path = HERE.parent / "data" / "cached_for_rastermap"
    neur_path_ITI = base_path / f"{session.mouse_name}_{session.date}_corridor_neur.npz"
    np.savez(neur_path_ITI, spks=spikes_combined, xpos=xpos, ypos=ypos)
    if ITI_split:
        neur_path_noITI = (
            base_path / f"{session.mouse_name}_{session.date}_corridor_neur_noITI.npz"
        )
        np.savez(neur_path_noITI, spks=spikes_combined_noITI, xpos=xpos, ypos=ypos)


def save_behavioural_data(
    session: Cached2pSession,
    corridor_starts_combined: np.ndarray,
    corridor_widths_combined: np.ndarray,
    positions_combined: np.ndarray,
    rewards_combined: np.ndarray,
    licks_combined: np.ndarray,
    speed_combined: np.ndarray,
) -> None:
    base_path = HERE.parent / "data" / "cached_for_rastermap"
    behavior_path = (
        base_path / f"{session.mouse_name}_{session.date}_corridor_behavior.npz"
    )
    # MinMax scale running speed and VR position
    scaler = MinMaxScaler(feature_range=(0, 1))
    np.savez(
        behavior_path,
        corridor_starts=corridor_starts_combined,
        corridor_widths=corridor_widths_combined,
        VRpos=(scaler.fit_transform(positions_combined[:, 1].reshape(-1, 1))).flatten(),
        reward_inds=rewards_combined,
        lick_inds=licks_combined,
        run=(scaler.fit_transform(speed_combined[:, 1].reshape(-1, 1))).flatten(),
        # run=speed_combined[:, 1],
    )


def process_session(
    session: Cached2pSession,
    wheel_circumference: float,
    ITI_split: bool = True,
    speed_bin_size: int = 5,
) -> None:
    s2p_path = TIFF_UMBRELLA / session.date / session.mouse_name / "suite2p" / "plane0"
    print(f"Working on {session.mouse_name}: {session.date} - {session.session_type}")
    spks, xpos, ypos, trials = load_data(session, s2p_path)
    aligned_trial_frames, spikes_trials = align_validate_data(spks, trials)
    licks_combined, rewards_combined, positions_combined, speed_combined = (
        process_behavioural_data(
            trials, aligned_trial_frames, wheel_circumference, speed_bin_size
        )
    )

    spikes_combined = np.hstack(spikes_trials)

    # filtering out frames
    valid_mask = filter_speed_position(
        speed=speed_combined,
        frames_positions=positions_combined,
        speed_threshold=0.1,
        position_threshold=180,
        filter_speed=True,
        filter_position=True,
    )
    valid_frame_indices = np.where(valid_mask)[0]

    positions_combined = positions_combined[valid_frame_indices]
    speed_combined = speed_combined[valid_frame_indices]
    spikes_combined = spikes_combined[:, valid_frame_indices]
    corridor_starts_combined = get_trial_valid_starts(
        aligned_trial_frames, valid_frame_indices
    )
    corridor_widths_combined = np.array(
        [
            get_corridor_width(
                trial, valid_frame_indices=valid_frame_indices, include_ITI=False
            )
            for trial in trials
        ]
    )
    assert len(corridor_widths_combined) == len(
        aligned_trial_frames
    ), "Number of corridor widths and aligned_trial_frames do not match"

    # remapping frame indices
    frame_mapping = create_frame_mapping(positions_combined)
    licks_combined = remap_to_continuous_indices(licks_combined, frame_mapping)
    rewards_combined = remap_to_continuous_indices(rewards_combined, frame_mapping)
    corridor_starts_combined[:, 0] = remap_to_continuous_indices(
        corridor_starts_combined[:, 0], frame_mapping
    )

    if ITI_split:
        aligned_trial_frames_noITI = align_trial_frames(session.trials, ITI=False)
        spikes_trials_noITI = get_signal_for_trials(spks, aligned_trial_frames_noITI)
        spikes_combined_noITI = np.hstack(spikes_trials_noITI)

        positions_combined_noITI = np.empty((0, 2))
        speed_combined_noITI = np.empty((0, 2))
        for trial, (start, end, rewarded) in zip(trials, aligned_trial_frames):
            trial_frames = np.arange(start, end + 1, 1)
            frames_positions = get_frame_position(
                trial, trial_frames, wheel_circumference, ITI=False
            )
            speed = get_speed_frame(frames_positions, speed_bin_size)
            positions_combined_noITI = np.vstack(
                (positions_combined_noITI, frames_positions)
            )
            speed_combined_noITI = np.vstack((speed_combined_noITI, speed))

        # TODO: in theory, could not positions and speeds be slighly off here as the other frames are not considered in here?
        valid_mask_noITI = filter_speed_position(
            speed=speed_combined_noITI,
            frames_positions=positions_combined_noITI,
            speed_threshold=0.1,
            position_threshold=180,
            filter_speed=True,
            filter_position=True,
        )

        valid_frame_indices_noITI = np.where(valid_mask_noITI)[0]

        spikes_combined_noITI = spikes_combined_noITI[:, valid_frame_indices_noITI]

        save_neural_data(
            session=session,
            spikes_combined=spikes_combined,
            spikes_combined_noITI=spikes_combined_noITI,
            xpos=xpos,
            ypos=ypos,
            ITI_split=True,
        )
    else:
        save_neural_data(
            session=session,
            spikes_combined=spikes_combined,
            xpos=xpos,
            ypos=ypos,
            ITI_split=False,
        )
    save_behavioural_data(
        session,
        corridor_starts_combined,
        corridor_widths_combined,
        positions_combined,
        rewards_combined,
        licks_combined,
        speed_combined,
    )
    print(f"Successfully cached 2P session for rastermap!")


if __name__ == "__main__":
    wheel_circumference = get_wheel_circumference_from_rig("2P")
    for mouse_name in ["JB027"]:
        print(f"Off we go for {mouse_name}...")
        for file in (HERE.parent / "data" / "cached_2p").glob(f"{mouse_name}_*.json"):
            print(file)
            with open(file, "r") as f:
                session = Cached2pSession.model_validate_json(f.read())
            f.close()
            process_session(session, wheel_circumference)
