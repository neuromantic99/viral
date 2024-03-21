from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from models import TrialInfo
from utils import pad_lists_to_array, rolling_count, shaded_line_plot
import pandas as pd
import seaborn as sns

# DATA_PATH = Path("/Volumes/MarcBusche/James/Behaviour")
DATA_PATH = Path("/Users/jamesrowland/Documents/behaviour")
MOUSE = "J002"
DATE = "2024-03-20"
SESSION_NUMBER = "016"
SESSION_PATH = DATA_PATH / MOUSE / DATE / SESSION_NUMBER


WHEEL_CIRCUMFERENCE = 48  # cm
ENCODER_TICKS_PER_TURN = 360  # check me
SAMPLING_RATE = 10


def load_data(session_path: Path):
    trial_files = list(session_path.glob("trial*.json"))
    trials: List[TrialInfo] = []
    for trial_file in trial_files:
        with open(trial_file) as f:
            trials.append(TrialInfo.model_validate_json(f.read()))
    return trials


def plot_lick_raster(trials: List[TrialInfo]) -> None:
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)

    all_trials = []
    for idx, trial in enumerate(trials):
        plt.scatter(
            trial.lick_start,
            np.zeros(len(trial.lick_start)) + idx,
            marker=".",
            c="black",
        )
        all_trials.extend(trial.lick_start)

    plt.xlabel("Time (s)")
    plt.ylabel("Trial number")

    plt.subplot(2, 1, 2)

    window = 1
    rolling = rolling_count(np.array(all_trials), window)

    plt.plot(np.linspace(0, max(all_trials), len(rolling)), rolling)
    plt.ylabel("Number of licks")
    plt.xlabel("Time (s)")
    plt.show()


def plot_position_whole_session(trials: List[TrialInfo]) -> None:
    all_positions = np.array([], dtype="float")
    previous_rotary_end = 0
    for trial in trials:
        position = np.array(trial.rotary_encoder_position)
        all_positions = np.concatenate((all_positions, position + previous_rotary_end))
        previous_rotary_end += position[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(len(all_positions)) / SAMPLING_RATE,
        (all_positions / ENCODER_TICKS_PER_TURN) * WHEEL_CIRCUMFERENCE / 100,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.show()


def compute_speed(
    position: np.ndarray, first: int, last: int, bin_size: int, sampling_rate: int
):
    """Speed in each bin (set by "step"). Assuming each integer is a second. Return unit is position units / second"""

    speed = []
    for start, stop in zip(
        range(first, last - bin_size, bin_size), range(first + bin_size, last, bin_size)
    ):
        n = np.sum(np.logical_and(position > start, position < stop))
        if n == 0:
            speed.append(0)
        else:
            speed.append(bin_size / (n / sampling_rate))
    return speed


def plot_speed(trials: List[TrialInfo]) -> None:
    plt.figure(figsize=(10, 6))
    rewarded = []
    not_rewarded = []
    for trial in trials:

        position = (
            np.array(trial.rotary_encoder_position) / ENCODER_TICKS_PER_TURN
        ) * WHEEL_CIRCUMFERENCE

        speed = compute_speed(position, 0, 200, 20, SAMPLING_RATE)

        if trial.texture_rewarded:
            rewarded.append(speed)
        else:
            not_rewarded.append(speed)

    x_axis = np.linspace(0, 200, len(rewarded[0]))
    shaded_line_plot(np.array(rewarded), x_axis, "green", "Rewarded")
    shaded_line_plot(np.array(not_rewarded), x_axis, "red", "Unrewarded")
    plt.legend()
    plt.xlabel("Distance (cm)")
    plt.ylabel("Speed (cm / s)")
    plt.title(MOUSE)
    plt.show()


if __name__ == "__main__":
    trials = load_data(SESSION_PATH)
    plot_speed(trials)
