from typing import Any, List, TypeVar
from matplotlib import pyplot as plt
import numpy as np

from constants import ENCODER_TICKS_PER_TURN, WHEEL_CIRCUMFERENCE
from models import SpeedPosition, TrialInfo


def shaded_line_plot(arr: np.ndarray[float], x_axis, color: str, label: str):

    mean = np.mean(arr, 0)
    sem = np.std(arr, 0) / np.sqrt(arr.shape[1])
    plt.plot(x_axis, mean, color=color, label=label, marker="o")
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


def licks_to_position(trial: TrialInfo) -> np.ndarray[float]:
    """Tested with hardware does not give false anticipatory licks. Write software tests still."""

    position = np.array(trial.rotary_encoder_position).astype(float)
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


def get_speed_positions(
    position: np.ndarray,
    first_position: int,
    last_position: int,
    step_size: int,
    sampling_rate: int,
) -> List[SpeedPosition]:
    """Compute speed as function of position

    position: The rotary encoder position at each sample
    first_position: The first position to consider
    last_position: The last position to consider inclusive
    step_size: The size of the position bins (needs to evenly divide the range)
    sampling_rate: The sampling rate of the rotary encoder in Hz

    Return unit is position units / second.

    Currently only works for evenly spaced integer positions

    TODO: Add tests for this function
    """

    assert (
        last_position - first_position
    ) % step_size == 0, "step_size should evenly divide the range"

    speed_position: List[SpeedPosition] = []
    for start, stop in zip(
        range(first_position, last_position - step_size + 1, step_size),
        range(first_position + step_size, last_position + 1, step_size),
        strict=True,
    ):
        # TODO: This will often be zero after the reward is triggered. Deal with this
        n = np.sum(np.logical_and(position >= start, position < stop))
        if n == 0 and start < 180:
            raise ValueError("Likely the rotary encoder has jumped in a weird way.")
            # n = np.nan

        speed_position.append(
            SpeedPosition(
                position_start=start,
                position_stop=stop,
                speed=step_size / (n / sampling_rate),
            )
        )
    return speed_position


T = TypeVar("T", float, np.ndarray)


def degrees_to_cm(degrees: T) -> T:
    return (degrees / ENCODER_TICKS_PER_TURN) * WHEEL_CIRCUMFERENCE
