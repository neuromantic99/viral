from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from models import TrialInfo

DATA_PATH = Path("/Volumes/MarcBusche/James/Behaviour")
MOUSE = "J003"
DATE = "2024-03-17"
SESSION_NUMBER = "001"
SESSION_PATH = DATA_PATH / MOUSE / DATE / SESSION_NUMBER


def load_data(session_path: Path):
    trial_files = list(session_path.glob("trial*.json"))
    trials: List[TrialInfo] = []
    for trial_file in trial_files:
        with open(trial_file) as f:
            trials.append(TrialInfo.model_validate_json(f.read()))
    return trials


def plot_lick_raster(trials: List[TrialInfo]) -> None:
    plt.figure(figsize=(10, 6))
    for idx, trial in enumerate(trials):
        plt.scatter(
            trial.lick_start,
            np.zeros(len(trial.lick_start)) + idx,
            marker=".",
            c="black",
        )
    plt.xlabel("Time (s)")
    plt.ylabel("Trial number")
    plt.show()


def plot_position(trials: List[TrialInfo]) -> None:
    all_positions: List[float] = []
    previous_rotary_end = 0
    for trial in trials:
        position = np.array(trial.rotary_encoder_position)
        all_positions.extend(position + previous_rotary_end)
        previous_rotary_end += position[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(all_positions)
    plt.title("Rotary Encoder Position")
    plt.xlabel("Sample")
    plt.ylabel("Position")
    plt.show()


if __name__ == "__main__":
    trials = load_data(SESSION_PATH)
    plot_lick_raster(trials)
