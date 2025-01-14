import numpy as np
import logging
import json
import time
import sys
from pathlib import Path
from typing import List
from nptdms import TdmsFile
from ScanImageTiffReader import ScanImageTiffReader
from matplotlib import pyplot as plt

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.utils import TrialInfo, trial_is_imaged, degrees_to_cm, get_speed_positions, licks_to_position
from viral.cache_2p_sessions import count_spacers, add_imaging_info_to_trials
from viral.models import Cached2pSession
from viral.constants import SPREADSHEET_ID, BEHAVIOUR_DATA_PATH, SYNC_FILE_PATH, TIFF_UMBRELLA

# logger = logging.getLogger("2p_cacher")

def align_trial_times_tbehav(trials: list[TrialInfo], tbehav: np.ndarray) -> np.ndarray:
    """Extract trial start and end times from trials and align them to tbehav"""
    trial_times = np.array([[trial.trial_start_time_daq, trial.trial_end_time_daq] for trial in trials])
    assert len(trials) == trial_times.shape[0], "Number of trials does not match number of trial times"
    aligned_trial_times = np.empty_like(trial_times, dtype=float)
    for i, (start, end) in enumerate(trial_times):
        # closest indices in tbehav
        start_idx = np.searchsorted(tbehav, start, side="left")
        end_idx = np.searchsorted(tbehav, end, side="left")
        
        # ensure indices are within bounds
        start_idx = np.clip(start_idx, 0, len(tbehav) - 1)
        end_idx = np.clip(end_idx, 0, len(tbehav) - 1)
        
        # align trial times
        aligned_trial_times[i, 0] = tbehav[start_idx]
        aligned_trial_times[i, 1] = tbehav[end_idx]
    
    return aligned_trial_times


def get_time_trial(tbehav: np.ndarray, trial_times: np.ndarray) -> np.ndarray:
    # Caution, currently, the trial index is index of imaged trials, but does not include the non-imaged trials!
    # trial_matrix = np.zeros_like(tbehav, dtype=int)
    # for trial_index, (start, end) in enumerate(trial_times):
    #     trial_mask = (tbehav >= start) & (tbehav <= end)
    #     trial_matrix[trial_mask] = trial_index +1
    # return np.column_stack((tbehav, trial_matrix))
    trial_segments = list()
    for index, (start, end) in enumerate(trial_times.astype(int)):
        if end > start:
            segment_times = np.arange(start, end, 1)
            segment_index = np.full_like(segment_times, index + 1)
            trial_segments.append(np.column_stack((segment_times, segment_index)))
    return np.concatenate(trial_segments, axis=0) if trial_segments else np.empty((0, 2), dtype=int)
    


def get_time_position(trial: TrialInfo, wheel_circumference: float) -> np.ndarray:
    times = np.array(
        [
            state.start_time_daq
            for state in trial.states_info
            if state.name in ["trigger_panda", "trigger_panda_post_reward"]
        ]
    )
    degrees = np.array(trial.rotary_encoder_position)
    positions = degrees_to_cm(degrees, wheel_circumference)
    assert len(times) == len(positions)
    return np.column_stack((times, positions))


def get_speed_time(trial: TrialInfo, wheel_circumference: float, trial_times: np.ndarray) -> np.ndarray:
    time_position = get_time_position(trial, wheel_circumference).astype(int)
    # Cannot use the "get_speed_positions" function because we would never end up with a speed of 0! (speed as function of position)
    # speed_positions = get_speed_positions(position=time_position[:, 1], first_position=first_position, last_position=last_position, step_size=step_size, sampling_rate=sampling_rate)
    # positions_start, positions_stop, speeds = [speed.position_start for speed in speed_positions], [speed.position_stop for speed in speed_positions], [speed.speed for speed in speed_positions]
    # Need to calculate the speed as a function of time
    # TODO: Is this ok like this?
    times_diff = np.diff(time_position[:, 0])
    positions_diff = np.diff(time_position[:, 1])
    speeds = (positions_diff / times_diff)
    return np.column_stack((time_position[:-1, 0], speeds))
    

def get_lick_time(trial: TrialInfo) -> np.ndarray:
    start_times = np.array(
        [
            event.start_time_daq
            for event in trial.events_info
            if event.name == "Port1In"
        ]
    )
    end_times = np.array(
        [
            event.start_time_daq
            for event in trial.events_info
            if event.name == "Port1Out"
        ]
    )
    # Handle trial w/o licks
    if len(start_times) == 0 or len(end_times) == 0:
        return np.empty((0, 2), dtype=int)
    # Cutting of pairs with e.g. missing end time
    # TODO: is this a good approach?
    valid_pairs = list()
    start_idx, end_idx = 0, 0
    while start_idx < len(start_times) and end_idx < len(end_times):
        if start_times[start_idx] < end_times[end_idx]:
            valid_pairs.append((start_times[start_idx], end_times[end_idx]))
            start_idx += 1
            end_idx += 1
        else:
            end_idx += 1
    start_end_times = np.array(valid_pairs, dtype=int)
    lick_segments = list()
    for start, end in start_end_times.astype(int):
        if end > start:
            segment_times = np.arange(start, end, 1)
            segment_licks = np.ones_like(segment_times, dtype=int)
            lick_segments.append(np.column_stack((segment_times, segment_licks)))
    return np.concatenate(lick_segments) if lick_segments else np.empty((0, 2), dtype=int)

def get_reward_time(trial: TrialInfo) -> np.ndarray:
    # Caution: When changing number of rewards, the state names have to be changed
    start_times = np.array(
        [
            state.start_time_daq
            for state in trial.states_info
            if state.name in ["reward_on1"]
        ]
    )
    end_times = np.array(
        [
            state.start_time_daq
            for state in trial.states_info
            if state.name in ["reward_off3"]
        ]
    )
    # Handle trial w/o rewards
    if len(start_times) == 0 or len(end_times) == 0:
        return np.empty((0, 2), dtype=int)
    # Cutting of pairs with e.g. missing end time
    # TODO: is this a good approach?
    valid_pairs = list()
    start_idx, end_idx = 0, 0
    while start_idx < len(start_times) and end_idx < len(end_times):
        if start_times[start_idx] < end_times[end_idx]:
            valid_pairs.append((start_times[start_idx], end_times[end_idx]))
            start_idx += 1
            end_idx += 1
        else:
            end_idx += 1
    start_end_times = np.array(valid_pairs, dtype=int)
    reward_segments = list()
    for start, end in start_end_times.astype(int):
        if end > start:
            segment_times = np.arange(start, end, 1)
            segment_rewards = np.ones_like(segment_times, dtype=int)
            reward_segments.append(np.column_stack((segment_times, segment_rewards)))
    return np.concatenate(reward_segments) if reward_segments else np.empty((0, 2), dtype=int)
    


def trim_tbehav(tbehav: np.ndarray, aligned_trial_times: np.ndarray) -> np.ndarray:
    """Trim tbehav to the range of aligned_trial_times"""
    # Not sure about this, could probably make up the tbehav without having the behaviour_clock cached
    return tbehav[(tbehav >= aligned_trial_times[0, 0]) & (tbehav <= aligned_trial_times[-1, 1])]

def stack_behaviour_data(tbehav: np.ndarray, arrays: list[np.ndarray]) -> np.ndarray:
    # TODO: downsample before? the function is currently too slow
    aligned_arrays = list()
    for array in arrays:
        aligned_array = np.zeros((len(tbehav), array.shape[1]))
        for i in range(array.shape[0]):
            time_idx = np.where(tbehav == array[i, 0])[0]
            if len(time_idx) > 0:
                    time_idx = time_idx[0]
                    aligned_array[time_idx, :] = array[i, 1:]
            else:
               raise ValueError(f"Warning: Time {array[i, 0]} not found in tbehav.")
    return np.stack(aligned_arrays, axis=-1)


def save_data() -> None:
    # Save spks.py
    # Save behav.py
    # Save tbehav.py
    # Save tneural.py
    # Save behav_names
    np.savez()

if __name__ == "__main__":
    file = "/home/josef/code/viral/data/cached_2p/JB018_2024-12-05.json"
    with open(file, "r") as f:
        session = Cached2pSession.model_validate_json(f.read())
    tbehav = np.load("/home/josef/code/viral/data/cached_2p/JB018_2024-12-05_behaviour_clock.npy")
    tneural = np.load("/home/josef/code/viral/data/cached_2p/JB018_2024-12-05_valid_frame_times.npy")
    
    # Not ideal, could be saved while caching the sessions
    if 30 < len(tbehav) / 1000 / 60 < 100:
        sampling_rate = 1000
    elif 30 < len(tbehav) / 10000 / 60 < 100:
        sampling_rate = 10000
    else:
        raise ValueError("Could not determine sampling rate")
    trials = [trial for trial in session.trials if trial_is_imaged(trial)]
    if not trials:
        print("No trials imaged")
        exit()
    aligned_trial_times = align_trial_times_tbehav(trials, tbehav)
    
    tbehav = trim_tbehav(tbehav, aligned_trial_times)
    # downsample to bpod sampling rate?!!!

    assert len(trials) == len(aligned_trial_times), "Number of trials and aligned_trial_times do not match"

    for trial, (start_time, end_time) in zip(trials, aligned_trial_times):
        trial_times = np.arange(start=start_time, stop=end_time, step=1)
        speed_time = get_speed_time(trial, 34.7, trial_times)
        print(speed_time)
        print(np.min(speed_time[1]))
        print(np.max(speed_time[1]))
        plt.figure(figsize=(20,7), dpi=300)
        plt.scatter(x=speed_time[:, 0], y=speed_time[:, 1])
        plt.savefig(f"/home/josef/code/viral/plots/speed_test.png")
        lick_time = get_lick_time(trial)
        plt.figure(figsize=(20,7), dpi=300)
        plt.scatter(x=lick_time[:, 0], y=lick_time[:, 1])
        plt.savefig(f"/home/josef/code/viral/plots/lick_test.png")
        reward_time = get_reward_time(trial)
        plt.figure(figsize=(20,7), dpi=300)
        plt.scatter(x=reward_time[:, 0], y=reward_time[:, 1])
        plt.savefig(f"/home/josef/code/viral/plots/reward_test.png")
        print(stack_behaviour_data(tbehav=tbehav, arrays=[speed_time, lick_time, reward_time]))

    