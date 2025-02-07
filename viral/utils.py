from datetime import datetime
import math
from pathlib import Path
from typing import List, Tuple, TypeVar
import warnings
from matplotlib import pyplot as plt
import numpy as np

from .constants import ENCODER_TICKS_PER_TURN
from .models import SpeedPosition, TrialInfo


def shaded_line_plot(
    arr: np.ndarray,
    x_axis: np.ndarray | List[float],
    color: str,
    label: str,
) -> None:

    mean = np.mean(arr, 0)
    sem = np.std(arr, 0) / np.sqrt(arr.shape[1])
    plt.plot(x_axis, mean, color=color, label=label, marker="", zorder=1)
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


def licks_to_position(trial: TrialInfo, wheel_circumference: float) -> np.ndarray:
    """Tested with hardware does not give false anticipatory licks. Write software tests still."""

    position = np.array(trial.rotary_encoder_position).astype(float)
    lick_start = np.array(trial.lick_start)

    # The rotary encoder position is stored each time the trigger_panda state is exited. So this
    # is the time at which each position element was recorded.
    time_position = np.array(
        [
            state.end_time
            for state in trial.states_info
            if state.name
            in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
        ]
    )

    match abs(position.shape[0] - time_position.shape[0]):
        case 0:
            pass
        case 1:
            warnings.warn(
                "There is an off-by-one here. That happens occassionally for a reason i don't understand but, probably doesn't matter."
            )
        case _:
            raise ValueError("posiiton and time_position should have the same length.")

    # You can get NaNs if a state is not entered in a trial. Replace with -inf to stop
    # it being detected as the argmin
    time_position[np.isnan(time_position)] = -np.inf

    # The index of the position closest to each lick
    min_diff_indices = np.argmin(
        np.abs(lick_start[:, None] - time_position[None, :]), axis=1
    )
    return (position[min_diff_indices] / ENCODER_TICKS_PER_TURN) * wheel_circumference


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
        # # TODO: This will often be zero after the reward is triggered. Deal with this
        n = np.sum(np.logical_and(position >= start, position < stop))
        if n == 0 and start < 180:
            raise ValueError("Likely the rotary encoder has jumped in a weird way.")

        speed_position.append(
            SpeedPosition(
                position_start=start,
                position_stop=stop,
                speed=step_size / (n / sampling_rate),
            )
        )
    return speed_position


T = TypeVar("T", float, np.ndarray)


def degrees_to_cm(degrees: T, wheel_circumference: float) -> T:
    return (degrees / ENCODER_TICKS_PER_TURN) * wheel_circumference


def threshold_detect(signal: np.ndarray, threshold: float) -> np.ndarray:
    """lloyd russell"""
    thresh_signal = signal > threshold
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False
    times = np.where(thresh_signal)
    return times[0]


def pade_approx_norminv(p: float) -> float:
    q = (
        math.sqrt(2 * math.pi) * (p - 1 / 2)
        - (157 / 231) * math.sqrt(2) * math.pi ** (3 / 2) * (p - 1 / 2) ** 3
    )
    r = (
        1
        - (78 / 77) * math.pi * (p - 1 / 2) ** 2
        + (241 * math.pi**2 / 2310) * (p - 1 / 2) ** 4
    )
    return q / r


def d_prime(hit_rate: float, false_alarm_rate: float) -> float:
    return pade_approx_norminv(hit_rate) - pade_approx_norminv(false_alarm_rate)


def threshold_detect_edges(
    signal: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    rising_edges = (signal[:-1] <= threshold) & (signal[1:] > threshold)
    falling_edges = (signal[:-1] > threshold) & (signal[1:] <= threshold)
    rising_indices = (
        np.where(rising_edges)[0] + 1
    )  # Shift by 1 to get the index where the crossing occurs
    falling_indices = (
        np.where(falling_edges)[0] + 1
    )  # Shift by 1 to get the index where the crossing occurs
    return rising_indices, falling_indices


def get_tiff_paths_in_directory(directory: Path) -> List[Path]:
    return list(directory.glob("*.tif"))


def extract_TTL_chunks(
    frame_clock: np.ndarray, sampling_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    frame_times = threshold_detect(frame_clock, 4)
    diffed = np.diff(frame_times)
    chunk_starts = np.where(diffed > sampling_rate)[0] + 1
    # The first chunk starts on the first frame detected
    chunk_starts = np.insert(chunk_starts, 0, 0)
    # Add the final frame to allow the diff to work on the last chunk
    chunk_starts = np.append(chunk_starts, len(frame_times))
    return frame_times, np.diff(chunk_starts)


def get_wheel_circumference_from_rig(rig: str) -> float:
    if rig == "2P":
        return 34.7
        # return 11.05 * math.pi
    elif rig in {"Box", "Box2.0", "Box2.5"}:
        return 53.4
    else:
        raise ValueError(f"Unknown rig: {rig}")


def time_list_to_datetime(time_list: List[float]) -> datetime:
    assert len(time_list) == 6, "time_list should have 6 elements"
    whole_seconds = int(time_list[5])
    fractional_seconds = time_list[5] - whole_seconds
    return datetime(
        int(time_list[0]),
        int(time_list[1]),
        int(time_list[2]),
        int(time_list[3]),
        int(time_list[4]),
        whole_seconds,
        int(fractional_seconds * 1e6),
    )


def find_chunk(chunk_lens: List[int], index: int) -> int:
    """Given a list of chunk lengths and an index, find the chunk that contains the index"""
    cumulative_length = 0
    for i, length in enumerate(chunk_lens):
        cumulative_length += length
        if index < cumulative_length:
            return i
    return -1  # If index is out of bounds


def trial_is_imaged(trial: TrialInfo) -> bool:
    trigger_panda_states = [
        state
        for state in trial.states_info
        if state.name in {"trigger_panda", "trigger_panda_post_reward"}
    ]
    start_times_bpod = [state.start_time for state in trigger_panda_states]
    length_trial_bpod = start_times_bpod[-1] - start_times_bpod[0]

    start_time_frames = [state.closest_frame_start for state in trigger_panda_states]

    assert start_time_frames[-1] is not None
    assert start_time_frames[0] is not None

    length_trial_frames = (start_time_frames[-1] - start_time_frames[0]) / 30

    # A little but of a error in this calculation is allowed here as the frame rate is not exactly 30.
    # Possibly you will get a false positive if the trial is stopped exactly at the end but unlikely
    return length_trial_bpod - 0.5 <= length_trial_frames <= length_trial_bpod + 0.5


def average_different_lengths(data: List[np.ndarray]) -> np.ndarray:
    max_length = max(len(d) for d in data)

    for idx, d in enumerate(data):
        if len(d) < max_length:
            data[idx] = np.append(d, np.repeat(np.nan, max_length - len(d)))

    return np.nanmean(data, axis=0)


def get_genotype(mouse_name: str) -> str:
    if mouse_name in {"JB014", "JB015", "JB018", "JB020", "JB022", "JB022"}:
        return "Oligo-BACE1-KO"
    elif mouse_name in {
        "JB011",
        "JB012",
        "JB013",
        "JB016",
        "JB017",
        "JB019",
        "JB021",
        "JB023",
    }:
        return "NLGF"

    elif mouse_name in {"JB025", "JB024", "JB026", "JB027"}:
        return "WT"
    else:
        raise ValueError(f"Unknown genotype for mouse: {mouse_name}")


def get_sex(mouse_name: str) -> str:
    if mouse_name in {
        "JB013",
        "JB014",
        "JB016",
        "JB017",
        "JB018",
        "JB024",
        "JB025",
        "JB026",
        "JB027",
    }:
        return "male"
    if mouse_name in {
        "JB011",
        "JB012",
        "JB015",
        "JB019",
        "JB020",
        "JB021",
        "JB022",
        "JB023",
    }:
        return "female"
    else:
        raise ValueError(f"Unknown sex for mouse: {mouse_name}")


def get_setup(mouse_name: str) -> str:
    if mouse_name in {
        "JB011",
        "JB014",
        "JB015",
        "JB016",
        "JB018",
        "JB019",
        "JB020",
        "JB021",
        "JB022",
        "JB023",
        "JB026",
        "JB027",
    }:
        return "2P"
    if mouse_name in {
        "JB012",
        "JB013",
        "JB017",
        "JB024",
        "JB025",
    }:
        return "box"
    else:
        raise ValueError(f"Unknown sex for mouse: {mouse_name}")


def shuffle(x: np.ndarray) -> np.ndarray:
    """shuffles along all dimensions of an array"""
    shape = x.shape
    x = np.ravel(x)
    np.random.shuffle(x)
    return x.reshape(shape)


def get_sampling_rate(frame_clock: np.ndarray) -> int:
    """Bit of a hack as the sampling rate is not stored in the tdms file I think. I've used
    two different sampling rates: 1,000 and 10,000. The sessions should be between 30 and 100 minutes.
    """
    if 30 < len(frame_clock) / 1000 / 60 < 100:
        return 1000
    elif 30 < len(frame_clock) / 10000 / 60 < 100:
        return 10000
    raise ValueError("Could not determine sampling rate")
