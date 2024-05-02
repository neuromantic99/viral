from typing import List
from matplotlib import pyplot as plt
import numpy as np

from constants import ENCODER_TICKS_PER_TURN, WHEEL_CIRCUMFERENCE
from models import TrialInfo


def shaded_line_plot(arr: np.ndarray, x_axis, color: str, label: str):

    mean = np.mean(arr, 0)
    sem = np.std(arr, 0) / np.sqrt(arr.shape[1])
    plt.plot(x_axis, mean, color=color, label=label)
    plt.fill_between(
        x_axis,
        np.subtract(
            mean,
            sem,
        ),
        np.add(
            mean,
            sem,
        ),
        alpha=0.2,
        color=color,
    )


def licks_to_position(trial: TrialInfo) -> np.ndarray:
    """Tested with hardware does not give false anticipatory licks. Write software tests still."""

    position = np.array(trial.rotary_encoder_position)
    lick_start = np.array(trial.lick_start)

    # The rotary encoder position is stored each time the trigger_panda state is exited. So this
    # is the time at which each position element was recorded.
    time_position = np.array(
        [
            state.end_time
            for state in trial.states_info
            if state.name in ["trigger_panda", "trigger_panda_post_reward"]
        ]
    )
    assert (
        position.shape == time_position.shape
    ), "Maybe, we might have off-by-ones here"
    # You can get NaNs if a state is not entered in a trial. Replace with -inf to stop
    # it being detected as the argmin
    time_position[np.isnan(time_position)] = -np.inf

    # The index of the position closest to each lick
    min_diff_indices = np.argmin(
        np.abs(lick_start[:, None] - time_position[None, :]), axis=1
    )
    return (position[min_diff_indices] / ENCODER_TICKS_PER_TURN) * WHEEL_CIRCUMFERENCE


###################### ChatGPT's idea of how to plot speed vs position ##################
def compute_speed_chat(positions, dt):
    positions = np.array(positions)
    displacements = np.diff(positions, axis=0)
    return np.abs(displacements) / dt


def plot_speed_vs_position(positions, speeds):
    # For plotting purposes, we can consider the position at each speed measurement to be the average of the two positions used for its calculation
    position_averages = (positions[:-1] + positions[1:]) / 2
    plt.figure(figsize=(8, 5))
    plt.plot(position_averages, speeds, marker="o", linestyle="-", color="blue")
    plt.title("Speed as a Function of Position")
    plt.xlabel("Position")
    plt.ylabel("Speed (units per second)")
    plt.grid(True)
    plt.show()


############ My speed computation function, test this relative to the above ########################33
def compute_speed(
    position: np.ndarray, first: int, last: int, bin_size: int, sampling_rate: int
) -> np.ndarray:
    """Speed in each bin (set by "step"). Assuming each integer is a second. Return unit is position units / second.

    This also needs testing before using for anything serious
    """

    speed = []
    for start, stop in zip(
        range(first, last - bin_size, bin_size), range(first + bin_size, last, bin_size)
    ):
        n = np.sum(np.logical_and(position > start, position < stop))
        if (
            n == 0
        ):  # Really this is inf, but more likely to arise from the mouse not running to this position
            speed.append(0)
        else:
            speed.append(bin_size / (n / sampling_rate))
    return speed


def degrees_to_cm(degrees: float | np.ndarray) -> float | np.ndarray:
    return (degrees / ENCODER_TICKS_PER_TURN) * WHEEL_CIRCUMFERENCE
