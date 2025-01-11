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
    print(f"Len times {len(times)}")
    degrees = np.array(trial.rotary_encoder_position)
    positions = degrees_to_cm(degrees, wheel_circumference)
    assert len(times) == len(positions)
    return np.column_stack((times, positions))


def get_speed_time(trial: TrialInfo, wheel_circumference: float, tbehav: np.ndarray) -> np.ndarray:
    # Settings for speed calculation
    sampling_rate = 30
    first_position = 0
    last_position = 200
    step_size = 5
    time_position = get_time_position(trial, wheel_circumference)
    speed_positions = get_speed_positions(position=time_position[:, 1], first_position=first_position, last_position=last_position, step_size=step_size, sampling_rate=sampling_rate)
    positions_start, positions_stop, speeds = [speed.position_start for speed in speed_positions], [speed.position_stop for speed in speed_positions], [speed.speed for speed in speed_positions]
    interpolated_speeds = np.interp(time_position[:, 1], np.concatenate([positions_start, positions_stop]), np.concatenate([speeds, speeds]))
    speeds_tbehav = np.zeros_like(tbehav)
    for i, t in enumerate(tbehav):
        if t in time_position[:, 0]:
            speeds_tbehav[i] = interpolated_speeds[np.where(time_position[:, 0] == t)[0][0]]
    return np.column_stack((tbehav, speeds_tbehav))

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
    # TODO: Is this a close enough approximation? Could be slightly off
    return next(
        (state.start_time_daq for state in trial.states_info if state.name == "trial_start"),
        None
    )
def get_trial_end_time(trial: TrialInfo) -> float:
    # TODO: Is this a close enough approximation? Could be slightly off
    return next(
        (state.end_time_daq for state in trial.states_info if state.name == "ITI"),
        None
    )

def align_trial_times_tbehav(trials: list[TrialInfo], tbehav: np.ndarray) -> np.ndarray:
    """Extract trial start and end times from trials and align them to tbehav"""
    trial_start_times = list()
    trial_end_times = list()
    for trial in trials:
        trial_start_time = get_trial_start_time(trial)
        trial_end_time = get_trial_end_time(trial)
        assert trial_start_time is not None and trial_end_time is not None, "trial_start_time or trial_end_time is 'None'"
        assert trial_start_time < trial_end_time, "trial_end_time is sooner than trial_start_time"
        trial_start_times.append(trial_start_time)
        trial_end_times.append(trial_end_time)

    # tbehav must cover the range of trial times
    assert tbehav[0] <= np.min(trial_start_times) and tbehav[-1] >= np.max(trial_end_times), "tbehav does not cover the range of trial times!"

    trial_start_times = np.array(trial_start_times)
    trial_end_times = np.array(trial_end_times)

    # Check that no more than one trial starts at the same time
    unique_start_times, start_counts = np.unique(trial_start_times, return_counts=True)
    unique_end_times, end_counts = np.unique(trial_end_times, return_counts=True)
    assert np.all(start_counts == 1), f"Duplicate start times found: {trial_start_times[start_counts > 1]}"
    assert np.all(end_counts == 1), f"Duplicate end times found: {trial_end_times[end_counts > 1]}"

    for start_time, end_time in zip(trial_start_times, trial_end_times):
        assert start_time in tbehav, f"start_time '{start_time}' is not in tbehav"
        assert end_time in tbehav, f"end_time '{end_time}' is not in tbehav"

    aligned_trial_times = np.column_stack([trial_start_times, trial_end_times])
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
    file = "/home/josef/code/viral/data/cached_2p/JB011_2024-10-30.json"
    with open(file, "r") as f:
        session = Cached2pSession.model_validate_json(f.read())
    # tbehav = np.load("/home/josef/code/viral/data/cached_2p/JB011_2024-10-30_behaviour_times.npy")
    tbehav = np.load("/home/josef/code/viral/data/cached_2p/JB011_2024-10-30_behaviour_clock.npy")
    tneural = np.load("/home/josef/code/viral/data/cached_2p/JB011_2024-10-30_valid_frame_times.npy")
    trials = [trial for trial in session.trials if trial_is_imaged(trial)]
    if not trials:
        print("No trials imaged")
        exit()
    aligned_trial_times = align_trial_times_tbehav(trials, tbehav)
    tbehav = trim_tbehav(tbehav, aligned_trial_times)
    
    assert len(trials) == len(aligned_trial_times), "Number of trials and aligned_trial_times do not match"

    for trial, (start_time, end_time) in zip(trials, aligned_trial_times):
        speed_time = get_speed_time(trial, 34.7, tbehav)
    # plt.figure()
    # plt.plot(aligned_trial_times)
    # plt.savefig(f"/home/josef/code/viral/plots/trial_times_test.png")

    
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
    