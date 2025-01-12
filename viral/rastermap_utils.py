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
    print("positions")
    print(positions.astype(int))
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
    
    
def get_lick_time(trial: TrialInfo, wheel_circumference: float, tbehav: np.ndarray) -> np.ndarray:
    licks_position = licks_to_position(trial, wheel_circumference)
    # licks_binary = np.isin(time_position[:, 1], licks_position).astype(int)
    licks_binary_tbehav = np.isin(tbehav, licks_position).astype(int)
    # licks_binary_tbehav = np.zeros_like(tbehav, dtype=int)
    # for i, t in enumerate(tbehav):
    #     if t in time_position[:, 0]:
    #         licks_binary_tbehav[i] = licks_binary[np.where(time_position[:, 0] == t)[0][0]]
    return np.column_stack((tbehav, licks_binary_tbehav))

def get_reward_time(trial: TrialInfo, tbehav: np.ndarray) -> np.ndarray:
    reward_on_time = [
        state.start_time_daq
        for state in trial.states_info
        if state.name in ["reward_on1"]
    ]
    reward_binary_tbehav = np.isin(tbehav, reward_on_time).astype(int)
    return np.column_stack((tbehav, reward_binary_tbehav))
    

def get_trial_start_time(trial: TrialInfo) -> float:
    return next(
        (state.start_time_daq for state in trial.states_info if state.name == "spacer_high_00"),
        None
    )

# def get_trial_end_time(trial: TrialInfo) -> float:
#     # TODO: Is this a close enough approximation? Could be slightly off
#     return next(
#         (state.end_time_daq for state in trial.states_info if state.name == "ITI"),
#         None
#     )

def align_trial_times_tbehav(trials: list[TrialInfo], tbehav: np.ndarray) -> np.ndarray:
    """Extract trial start and end times from trials and align them to tbehav"""
    trial_times = np.array([[trial.trial_start_time_daq, trial.trial_end_time_daq] for trial in trials])
    print("trial times")
    print(trial_times)
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

def trim_tbehav(tbehav: np.ndarray, aligned_trial_times: np.ndarray) -> np.ndarray:
    """Trim tbehav to the range of aligned_trial_times"""
    return tbehav[(tbehav >= aligned_trial_times[0, 0]) & (tbehav <= aligned_trial_times[-1, 1])]

def save_data() -> None:
    # Save spks.py
    # Save behav.py
    # Save tbehav.py
    # Save tneural.py
    # Save behav_names
    np.savez()

if __name__ == "__main__":
    # file = "/home/josef/code/viral/data/cached_2p/JB011_2024-10-28.json"
    # file = "/home/josef/code/viral/data/cached_2p/JB011_2024-10-30.json"
    # file = "/home/josef/code/viral/data/cached_2p/JB011_2024-10-31.json"
    file = "/home/josef/code/viral/data/cached_2p/JB018_2024-12-05.json"
    with open(file, "r") as f:
        session = Cached2pSession.model_validate_json(f.read())
    # tbehav = np.load("/home/josef/code/viral/data/cached_2p/JB011_2024-10-30_behaviour_times.npy")
    # tbehav = np.load("/home/josef/code/viral/data/cached_2p/JB011_2024-10-30_behaviour_clock.npy")
    # tbehav = np.load("/home/josef/code/viral/data/cached_2p/JB011_2024-10-31_behaviour_clock.npy")
    tbehav = np.load("/home/josef/code/viral/data/cached_2p/JB018_2024-12-05_behaviour_clock.npy")
    # tneural = np.load("/home/josef/code/viral/data/cached_2p/JB011_2024-10-30_valid_frame_times.npy")
    # tneural = np.load("/home/josef/code/viral/data/cached_2p/JB011_2024-10-31_valid_frame_times.npy")
    tneural = np.load("/home/josef/code/viral/data/cached_2p/JB018_2024-12-05_valid_frame_times.npy")
    # behaviour_chunk_lens = np.load("/home/josef/code/viral/data/cached_2p/JB011_2024-10-30_behaviour_chunk_lens.npy")
    # behaviour_chunk_lens = np.load("/home/josef/code/viral/data/cached_2p/JB011_2024-10-31_behaviour_chunk_lens.npy")
    # behaviour chunk lens have trials in there which have not been imaged, hurray
    
    # Not ideal, should be saved while caching the sessions
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
    print("behaviour_clock")
    print(tbehav.astype(int))
    aligned_trial_times = align_trial_times_tbehav(trials, tbehav)
    print("aligned trial times")
    print(aligned_trial_times)
    print("spacer high times")
    print([get_trial_start_time(trial) for trial in trials])
    
    tbehav = trim_tbehav(tbehav, aligned_trial_times)
    
    assert len(trials) == len(aligned_trial_times), "Number of trials and aligned_trial_times do not match"

    for trial, (start_time, end_time) in zip(trials, aligned_trial_times):
        trial_times = np.arange(start=start_time, stop=end_time, step=1)
        speed_time = get_speed_time(trial, 34.7, trial_times)
        print(speed_time)
        print(np.min(speed_time[1]))
        print(np.max(speed_time[1]))
        plt.figure()
        plt.plot(speed_time)
        plt.savefig(f"/home/josef/code/viral/plots/speed_test.png")

    
    # 1/0
        

        # if timepoints.size == 0:
        #     raise ValueError(f"No timepoints found for trial '{trial.pc_timestamp}'")
        
        # speed_time = get_speed_time(trial, 34.7, tbehav)
        # plt.figure()
        # plt.plot(speed_time[:, 0], speed_time[:, 1])
        # plt.savefig(f"/home/josef/code/viral/plots/speed_test.png")
        # lick_time = get_lick_time(trial, 34.7, tbehav)
        # print(lick_time.shape)
        # reward_on = get_reward_time(trial, tbehav)
        # print(reward_on.shape)
        # print(reward_on)
        # trial_start = get_trial_start_time(trial, tbehav)
        # print(trial_start.shape)
        # print(trial_start)
        # 1/0
    
    # for i, trial in enumerate(trials):
    #     tbehav_trial = trial.behaviour_times
    #     tneural_trial = trial.valid_frame_times
    #     np.concatenate(tbehav, tbehav_trial)
    #     np.concatenate(tneural, tneural_trial)
    