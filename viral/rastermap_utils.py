import numpy as np
import sys
from pathlib import Path
from scipy.interpolate import interp1d
from typing import List, Optional

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.imaging_utils import get_ITI_start_frame, load_imaging_data, trial_is_imaged
from viral.utils import (
    trial_is_imaged,
    degrees_to_cm,
    get_wheel_circumference_from_rig,
)
from viral.models import Cached2pSession, TrialInfo, ImagedTrialInfo
from viral.constants import TIFF_UMBRELLA


def get_spks_pos(s2p_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)
    spks = np.load(s2p_path / "spks.npy")[iscell, :]
    stat = np.load(s2p_path / "stat.npy", allow_pickle=True)[iscell]
    pos = np.array([[int(coord) for coord in cell["med"]] for cell in stat])
    xpos = pos[:, 1]
    ypos = -1 * pos[:, 0]  # y-axis is inverted
    return spks, xpos, ypos


def get_dff_pos(
    session: Cached2pSession, s2p_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dff = load_imaging_data(mouse=session.mouse_name, date=session.date)
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)
    stat = np.load(s2p_path / "stat.npy", allow_pickle=True)[iscell]
    pos = np.array([[int(coord) for coord in cell["med"]] for cell in stat])
    xpos = pos[:, 1]
    ypos = -1 * pos[:, 0]  # y-axis is inverted
    return dff, xpos, ypos


def align_trial_frames(trials: List[TrialInfo], include_ITI: bool = True) -> np.ndarray:
    """Align trial frames and return array with trial frames and reward condition.

    Args:
        trials (List[TrialInfo]):       A list of TrialInfo objects.
        include_ITI (bool):   Include the frames in the inter-trial interval (ITI). Defaults to True.
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
    if not include_ITI:
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
    return np.column_stack((trial_frames, rewarded)).astype(int)


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

    degrees = np.array(trial.rotary_encoder_position)
    positions = degrees_to_cm(degrees, wheel_circumference)
    assert len(frames_start_end) == len(positions)

    frames_positions_combined = list()
    for i in range(len(frames_start_end)):
        frames_positions_combined.append([frames_start_end[i, 0], positions[i]])
        frames_positions_combined.append([frames_start_end[i, 1], positions[i]])
    frames_positions_combined = np.array(frames_positions_combined)
    unique_frames_positions = np.unique(frames_positions_combined, axis=0)

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
    if bin_size <= 1:
        raise ValueError("You cannot select a bin size <= 1 for speed!")
    n_frames = frame_position.shape[0]
    # Including upper bound
    bins = np.arange(0, n_frames, bin_size)
    speeds = np.zeros(frame_position.shape[0])
    for i in range(len(bins)):
        start_idx = bins[i]
        if i == len(bins) - 1:
            # Handle smaller last bin
            end_idx = n_frames - 1
        else:
            end_idx = bins[i + 1] - 1
        frame_start_idx, position_frame_start = frame_position[start_idx, :]
        frame_end_idx, position_frame_end = frame_position[end_idx, :]
        frame_diff = frame_end_idx - frame_start_idx
        position_diff = position_frame_end - position_frame_start
        speed = 0 if frame_diff == 0 else (position_diff / frame_diff)
        if i == len(bins) - 1:
            speeds[start_idx:] = speed
        else:
            speeds[start_idx : bins[i + 1]] = speed
        # convert from cm / frame to cm / second
    speeds = speeds * 30
    speeds[speeds < 0] = 0
    speeds[speeds > 100] = np.mean(speeds)
    return np.column_stack([frame_position[:, 0], speeds])


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
    continuous_frames = np.arange(positions_combined.shape[0], dtype=int)
    discontinuous_frames = positions_combined.astype(int)
    return {orig: cont for orig, cont in zip(discontinuous_frames, continuous_frames)}


def remap_to_continuous_indices(
    original_indices: np.ndarray, frame_mapping: dict
) -> np.ndarray:
    """Re-assign frame indices to be continuous, according to the frame_mapping dictionary."""
    return np.array(
        [frame_mapping[idx] for idx in original_indices if idx in frame_mapping]
    )


def load_data(session: Cached2pSession, s2p_path: Path, signal_type: str) -> tuple:
    """Return spks/dff, xpos, ypos and trials loaded from cache."""
    trials = [trial for trial in session.trials if trial_is_imaged(trial)]
    if not trials:
        print("No trials imaged")
        exit()
    if signal_type == "spks":
        spks, xpos, ypos = get_spks_pos(s2p_path)
        return spks, xpos, ypos, trials
    elif signal_type == "dff":
        dff, xpos, ypos = get_dff_pos(session, s2p_path)
        return dff, xpos, ypos, trials
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
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


def process_trials_data(
    trials: List[TrialInfo],
    aligned_trial_frames: np.ndarray,
    spikes_trials: np.ndarray,
    wheel_circumference: float,
    speed_bin_size: int = 10,
) -> List[ImagedTrialInfo]:
    """Process behavioural data and return licks, rewards, positions and speed as arrays."""
    imaged_trials_infos = list()

    for trial, (start, end, rewarded), spikes_trial in zip(
        trials, aligned_trial_frames, spikes_trials, strict=True
    ):
        trial_frames = np.arange(start, end + 1, 1)
        ITI_start_frame = get_ITI_start_frame(trial)
        licks = get_lick_index(trial)
        rewards = get_reward_index(trial)
        frames_positions = get_frame_position(trial, trial_frames, wheel_circumference)
        assert frames_positions.shape[0] == len(trial_frames)
        speed = get_speed_frame(frames_positions, speed_bin_size)
        assert speed.shape[0] == len(trial_frames)
        valid_mask = filter_speed_position(
            speed=speed,
            frames_positions=frames_positions,
        )
        if not np.any(valid_mask):
            print("No valid frames in the trial")
        valid_trial_frames = trial_frames[valid_mask]
        valid_trial_start = valid_trial_frames[0]
        if not np.any(licks):
            licks = np.array([])
        if not np.any(rewards):
            rewards = np.array([])
        imaged_trials_infos.append(
            ImagedTrialInfo(
                trial_start_frame=valid_trial_start,
                trial_end_frame=end,
                rewarded=rewarded,
                trial_frames=trial_frames,
                iti_start_frame=ITI_start_frame,
                iti_end_frame=end,
                frames_positions=frames_positions[valid_mask, :],
                frames_speed=speed[valid_mask, :],
                corridor_width=(ITI_start_frame - start),
                lick_idx=licks,
                reward_idx=rewards,
                signal=spikes_trial[:, valid_mask],
            )
        )
    return imaged_trials_infos


def filter_speed_position(
    speed: np.ndarray,
    frames_positions: np.ndarray,
    speed_threshold: Optional[float] = None,
    position_threshold: Optional[tuple[float, float]] = None,
    use_or: bool = True,
) -> np.ndarray:
    """Return a mask of valid frames"""

    if speed_threshold is None and position_threshold is None:
        return np.ones(len(speed), dtype=bool)
    valid_mask = np.zeros(len(speed), dtype=bool)

    # filter out frames with sub-threshold speeds
    # converting cm/s to cm/frames
    if speed_threshold is not None:
        speed_mask = speed[:, 1] >= (speed_threshold)
        if use_or:
            valid_mask |= speed_mask
        else:
            valid_mask = speed_mask
    # specified positions that should always be included
    if position_threshold is not None:
        position_mask = frames_positions[:, 1] >= position_threshold[0]
        position_mask &= frames_positions[:, 1] <= position_threshold[1]
        if use_or:
            valid_mask |= position_mask
        else:
            valid_mask &= position_mask

    return valid_mask


def filter_out_ITI(ITI_starts_ends: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Return a mask for filtering out frames that fall into the ITI"""
    valid_frame_mask = np.ones_like(positions[:, 0], dtype=bool)
    combined_ITI_mask = np.zeros_like(valid_frame_mask, dtype=bool)
    for start, end in zip(ITI_starts_ends[:, 0], ITI_starts_ends[:, 1], strict=True):
        ITI_mask = (positions[:, 0].astype(int) >= start.astype(int)) & (
            positions[:, 0].astype(int) <= end.astype(int)
        )
        combined_ITI_mask |= ITI_mask
    valid_frame_mask[combined_ITI_mask] = False
    assert valid_frame_mask.shape[0] == positions.shape[0]
    return valid_frame_mask


def save_neural_data(
    session: Cached2pSession,
    spikes_combined: np.ndarray,
    xpos: np.ndarray,
    ypos: np.ndarray,
    valid_frame_mask: Optional[np.ndarray] = None,
    ITI_split: bool = True,
) -> None:
    base_path = HERE.parent / "data" / "cached_for_rastermap"
    neur_path_ITI = base_path / f"{session.mouse_name}_{session.date}_corridor_neur.npz"
    np.savez(neur_path_ITI, spks=spikes_combined, xpos=xpos, ypos=ypos)
    if ITI_split:
        neur_path_noITI = (
            base_path / f"{session.mouse_name}_{session.date}_corridor_neur_noITI.npz"
        )
        np.savez(
            neur_path_noITI,
            spks=spikes_combined[:, valid_frame_mask],
            xpos=xpos,
            ypos=ypos,
        )


def save_behavioural_data(
    session: Cached2pSession,
    corridor_starts_combined: np.ndarray,
    corridor_widths_combined: np.ndarray,
    positions_combined: np.ndarray,
    rewards_combined: np.ndarray,
    licks_combined: np.ndarray,
    speed_combined: np.ndarray,
    valid_frame_mask: Optional[np.ndarray] = None,
    ITI_split: bool = True,
) -> None:
    base_path = HERE.parent / "data" / "cached_for_rastermap"
    behavior_path = (
        base_path / f"{session.mouse_name}_{session.date}_corridor_behavior.npz"
    )
    assert positions_combined.shape[0] == speed_combined.shape[0]
    frame_mapping = create_frame_mapping(positions_combined[:, 0])

    licks = remap_to_continuous_indices(licks_combined, frame_mapping)
    rewards = remap_to_continuous_indices(rewards_combined, frame_mapping)
    corridor_starts = np.copy(corridor_starts_combined)
    corridor_starts[:, 0] = remap_to_continuous_indices(
        corridor_starts[:, 0], frame_mapping
    )
    np.savez(
        behavior_path,
        corridor_starts=corridor_starts,
        corridor_widths=corridor_widths_combined,
        VRpos=positions_combined[:, 1],
        reward_inds=rewards,
        lick_inds=licks,
        run=speed_combined[:, 1],
    )
    if ITI_split:
        assert (
            positions_combined.shape[0]
            == speed_combined.shape[0]
            == valid_frame_mask.shape[0]
        )
        behavior_path_noITI = (
            base_path
            / f"{session.mouse_name}_{session.date}_corridor_behavior_noITI.npz"
        )
        positions_noITI = positions_combined[valid_frame_mask]
        speed_noITI = speed_combined[valid_frame_mask]

        licks_noITI = licks_combined[np.isin(licks_combined, positions_noITI[:, 0])]
        rewards_noITI = rewards_combined[
            np.isin(rewards_combined, positions_noITI[:, 0])
        ]

        frame_mapping = create_frame_mapping(positions_noITI[:, 0])

        licks_noITI = remap_to_continuous_indices(licks_noITI, frame_mapping)
        rewards_noITI = remap_to_continuous_indices(rewards_noITI, frame_mapping)
        corridor_starts_combined_noITI = np.copy(corridor_starts_combined)
        corridor_starts_combined_noITI[:, 0] = remap_to_continuous_indices(
            corridor_starts_combined_noITI[:, 0], frame_mapping
        )
        np.savez(
            behavior_path_noITI,
            corridor_starts=corridor_starts_combined,
            corridor_widths=corridor_widths_combined,
            VRpos=positions_noITI[:, 1],
            reward_inds=rewards_noITI,
            lick_inds=licks_noITI,
            run=speed_noITI[:, 1],
        )


def process_session(
    session: Cached2pSession,
    wheel_circumference: float,
    ITI_split: bool = True,
    speed_bin_size: int = 10,
) -> None:
    s2p_path = TIFF_UMBRELLA / session.date / session.mouse_name / "suite2p" / "plane0"
    print(f"Working on {session.mouse_name}: {session.date} - {session.session_type}")
    # TODO: it is actually dff and not spks but I will still call it that
    spks, xpos, ypos, trials = load_data(session, s2p_path, "spks")
    aligned_trial_frames, spikes_trials = align_validate_data(spks, trials)
    imaged_trials_infos = process_trials_data(
        trials, aligned_trial_frames, spikes_trials, wheel_circumference, speed_bin_size
    )

    corridor_starts = np.array(
        [[trial.trial_start_frame, trial.rewarded] for trial in imaged_trials_infos]
    )
    corridor_widths = np.array([trial.corridor_width for trial in imaged_trials_infos])
    ITI_starts_ends = np.array(
        [[trial.iti_start_frame, trial.iti_end_frame] for trial in imaged_trials_infos]
    )
    positions = np.vstack([trial.frames_positions for trial in imaged_trials_infos])
    speed = np.vstack([trial.frames_speed for trial in imaged_trials_infos])
    licks = np.hstack(
        [trial.lick_idx for trial in imaged_trials_infos if trial.lick_idx is not None]
    )
    rewards = np.hstack(
        [
            trial.reward_idx
            for trial in imaged_trials_infos
            if trial.reward_idx is not None
        ]
    )
    signals = np.concatenate([trial.signal for trial in imaged_trials_infos], axis=1)
    assert len(corridor_widths) == len(
        aligned_trial_frames
    ), "Number of corridor widths and aligned_trial_frames do not match"

    assert positions.shape[0] == speed.shape[0] == signals.shape[1]
    # Checking that neither positions nor speed have values that exceed the trial boundaries
    assert np.min(positions[:, 0]) == aligned_trial_frames[0, 0]
    assert np.max(positions[:, 0]) == aligned_trial_frames[-1, 1]
    assert np.min(speed[:, 0]) == aligned_trial_frames[0, 0]
    assert np.max(speed[:, 0]) == aligned_trial_frames[-1, 1]
    valid_frames = set()
    for start, end, _ in aligned_trial_frames:
        valid_frames.update(np.arange(start, end + 1))
    assert np.all(np.isin(positions[:, 0], list(valid_frames)))
    assert np.all(np.isin(speed[:, 0], list(valid_frames)))

    if ITI_split:
        valid_frame_mask = filter_out_ITI(ITI_starts_ends, positions)

        save_neural_data(
            session=session,
            spikes_combined=signals,
            xpos=xpos,
            ypos=ypos,
            valid_frame_mask=valid_frame_mask,
            ITI_split=True,
        )
        save_behavioural_data(
            session=session,
            corridor_starts_combined=corridor_starts,
            corridor_widths_combined=corridor_widths,
            positions_combined=positions,
            rewards_combined=rewards,
            licks_combined=licks,
            speed_combined=speed,
            valid_frame_mask=valid_frame_mask,
            ITI_split=True,
        )
    else:
        save_neural_data(
            session=session,
            spikes_combined=signals,
            xpos=xpos,
            ypos=ypos,
            ITI_split=False,
        )
        save_behavioural_data(
            session=session,
            corridor_starts_combined=corridor_starts,
            corridor_widths_combined=corridor_widths,
            positions_combined=positions,
            rewards_combined=rewards,
            licks_combined=licks,
            speed_combined=speed,
            ITI_split=False,
        )
    print("Successfully cached 2P session for rastermap!")


if __name__ == "__main__":
    wheel_circumference = get_wheel_circumference_from_rig("2P")
    for mouse_name in ["JB027"]:
        print(f"Off we go for {mouse_name}...")
        for file in (HERE.parent / "data" / "cached_2p").glob(f"{mouse_name}_*.json"):
            print(file)
            with open(file, "r") as f:
                session = Cached2pSession.model_validate_json(f.read())
            f.close()
            process_session(session, wheel_circumference, True, 15)
