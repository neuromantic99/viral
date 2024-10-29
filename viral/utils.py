import math
from pathlib import Path
from typing import List, Tuple, TypeVar
import warnings
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter

from .constants import ENCODER_TICKS_PER_TURN, WHEEL_CIRCUMFERENCE
from .models import SpeedPosition, TrialInfo


from nptdms import TdmsFile
from ScanImageTiffReader import ScanImageTiffReader


def shaded_line_plot(
    arr: np.ndarray[float],
    x_axis: np.ndarray[float] | List[float],
    color: str,
    label: str,
) -> None:

    mean = np.mean(arr, 0)
    sem = np.std(arr, 0) / np.sqrt(arr.shape[1])
    plt.plot(x_axis, mean, color=color, label=label, marker="")
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
    return (position[min_diff_indices] / ENCODER_TICKS_PER_TURN) * WHEEL_CIRCUMFERENCE


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


def count_spacers(trial: TrialInfo) -> int:
    return len([state for state in trial.states_info if "spacer_high" in state.name])


def process_sync_file(
    tdms_path: Path, tiff_directory: Path, trials: List[TrialInfo]
) -> None:

    # tiffs = sorted(get_tiff_paths_in_directory(tiff_directory))
    # stack_lengths = [ScanImageTiffReader(str(tiff)).shape()[0] for tiff in tiffs]
    stack_lengths = [10963, 23779, 19146, 8960, 340, 15313, 13176]

    tdms_file = TdmsFile.read(tdms_path)
    group = tdms_file["Analog"]
    frame_clock = group["AI0"][:]
    behaviour_clock = group["AI1"][:]

    # Bit of a hack as the sampling rate is not stored in the tdms file I think. I've used
    # two different sampling rates: 1,000 and 10,000. The sessions should be between 30 and 100 minutes.
    if 30 < len(frame_clock) / 1000 / 60 < 100:
        sampling_rate = 1000
    elif 30 < len(frame_clock) / 10000 / 60 < 100:
        sampling_rate = 10000
    else:
        raise ValueError("Could not determine sampling rate")

    print(f"Sampling rate: {sampling_rate}")

    behaviour_times, behaviour_chunk_lens = extract_TTL_chunks(
        behaviour_clock, sampling_rate
    )
    num_spacers_per_trial = np.array([count_spacers(trial) for trial in trials])

    # Behaviour crashed half way through a trial, so manual fix
    if "JB011" in str(tdms_path) and "2024-10-22" in str(tdms_path):
        behaviour_chunk_lens = np.delete(behaviour_chunk_lens, 52)

    assert np.array_equal(
        behaviour_chunk_lens, num_spacers_per_trial
    ), "Spacers recorded in txt file do not match sync"

    frame_times, chunk_lens = extract_TTL_chunks(frame_clock, sampling_rate)
    sanity_check_imaging_frames(frame_times, sampling_rate)

    plt.plot(frame_clock)
    plt.plot(frame_times, np.ones(len(frame_times)), ".", color="green")
    print(chunk_lens - stack_lengths)

    21 / 0


def extract_TTL_chunks(
    frame_clock: np.ndarray, sampling_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    frame_times = threshold_detect(frame_clock, 1)
    diffed = np.diff(frame_times)
    chunk_starts = np.where(diffed > sampling_rate)[0] + 1
    # The first chunk starts on the first frame
    chunk_starts = np.insert(chunk_starts, 0, 0)
    chunk_starts = np.append(chunk_starts, len(frame_times))

    return frame_times, np.diff(chunk_starts)


def sanity_check_imaging_frames(frame_times: np.ndarray, sampling_rate: float) -> None:
    """Make sure there are no frame clocks that don't make sense:
    Less than one frame apart , or more than one frame apart but less than a second apart
    """
    diffed = np.diff(frame_times)
    bad_times = np.where(
        np.logical_or(
            diffed < 29 * sampling_rate / 1000,
            np.logical_and(diffed > 40 * sampling_rate / 1000, diffed < sampling_rate),
        )
    )[0]
    assert len(bad_times) == 0
    ## Useful plot to debug
    # if len(bad_times) != 0:
    #     plt.plot(frame_clock)
    #     plt.plot(frame_times, np.ones(len(frame_times)), ".", color="green")
    #     plt.plot(frame_times[bad_times], np.ones(len(bad_times)), ".", color="red")
    #     plt.show()
    #     raise ValueError("Bad inter-frame-interval. Inspect the clock using the plot")
