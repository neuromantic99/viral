from datetime import datetime
import math
from pathlib import Path
from typing import List, Tuple, TypeVar, Any
import warnings
from zoneinfo import ZoneInfo
from matplotlib import pyplot as plt
import numpy as np
from enum import Enum
import pandas as pd

from viral.constants import ENCODER_TICKS_PER_TURN
from viral.models import (
    Cached2pSession,
    SpeedPosition,
    TrialInfo,
    MouseSummary,
)


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


def pad_to_max_length(sequences: Any, fill_value=np.nan) -> np.ndarray:
    """Return numpy array with the length of the longest sequence, padded with NaN values"""
    max_len = max(len(seq) for seq in sequences)
    return np.array(
        [
            np.pad(seq, (0, max_len - len(seq)), constant_values=fill_value)
            for seq in sequences
        ]
    )


def get_wheel_circumference_from_rig(rig: str) -> float:
    if rig in {"2P", "2P_1.5"}:
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
        if state.name
        in {"trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"}
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
    if mouse_name in {"JB014", "JB015", "JB018", "JB020", "JB022"}:
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

    elif mouse_name in {
        "JB024",
        "JB025",
        "JB026",
        "JB027",
        "JB030",
        "JB031",
        "JB032",
        "JB033",
    }:
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
        "JB030",
        "JB031",
        "JB032",
        "JB033",
    }:
        return "female"
    else:
        raise ValueError(f"Unknown sex for mouse: {mouse_name}")


class SetupType(Enum):
    TWO_PHOTON = "2P"
    BOX = "box"


def get_setup(setup_name: str) -> str:
    if "2P" in setup_name.upper().strip():
        return SetupType.TWO_PHOTON.value
    elif "box" in setup_name.lower().strip():
        return SetupType.BOX.value
    else:
        raise ValueError(f"Unknown setup '{setup_name}' for mouse!")


def get_setup_for_session_type(mouse: MouseSummary, session_type: str) -> str:
    return mouse.setup[session_type]


class RewardedTexture(Enum):
    PEBBLE = "pebble.jpg"
    BLACK_AND_WHITE_CIRCLES = "blackAndWhiteCircles.png"


def get_rewarded_texture(texture_name: str) -> str:
    if "pebble" in texture_name.lower().strip():
        return RewardedTexture.PEBBLE.value
    elif "blackandwhitecircles" in texture_name.lower().strip():
        return RewardedTexture.BLACK_AND_WHITE_CIRCLES.value
    else:
        raise ValueError(f"Unknown rewarded texture '{texture_name}'.")


def get_rewarded_texture_for_session_type(
    mouse: MouseSummary, session_type: str
) -> str:
    return mouse.rewarded_texture[session_type]


class SessionType(Enum):
    REVERSAl = "reversal"
    RECALL_REVERSAL = "recall_reversal"
    RECALL = "recall"
    LEARNING = "learning"


def get_session_type(session_name: str) -> str:
    session_name = session_name.lower().strip()
    if "reversal" in session_name:
        return (
            SessionType.RECALL_REVERSAL.value
            if "recall" in session_name
            else SessionType.REVERSAl.value
        )
    elif "recall" in session_name:
        return SessionType.RECALL.value
    elif "learning" in session_name:
        return SessionType.LEARNING.value
    else:
        raise ValueError(f"Invalid session type: {session_name}")


def shuffle(x: np.ndarray) -> np.ndarray:
    """shuffles along all dimensions of an array"""
    shape = x.shape
    x = np.ravel(x)
    np.random.shuffle(x)
    return x.reshape(shape)


def sort_matrix_peak(matrix: np.ndarray) -> np.ndarray:
    peak_indices = np.argmax(matrix, axis=1)
    sorted_order = np.argsort(peak_indices)
    return matrix[sorted_order]


def array_bin_mean(arr: np.ndarray, bin_size: int = 2, axis: int = 1) -> np.ndarray:
    """Bins elements along a given axis  with a specified bin size, computing the mean in the bin"""
    shape = arr.shape[axis]
    indices = np.arange(0, shape, bin_size)
    binned_sum = np.add.reduceat(arr, indices, axis=axis)

    # Count elements in each bin (handling the last bin if it's smaller)
    counts = (
        np.diff(indices, append=shape)[:, None]
        if axis == 0
        else np.diff(indices, append=shape)
    )
    return binned_sum / counts


def remove_consecutive_ones(matrix: np.ndarray) -> np.ndarray:

    def driver(row: np.ndarray) -> np.ndarray:
        # Create a mask to identify the first occurrence of 1 in consecutive sequences
        mask = np.diff(row, prepend=0) == 1
        # Apply the mask to keep only the first 1 in consecutive sequences
        return row * mask

    return np.apply_along_axis(driver, 1, matrix)


def shuffle_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Shuffles the elements within each row of the given matrix independently.

    Parameters:
    matrix (numpy.ndarray): A 2D NumPy array where each row's elements are shuffled.

    Returns:
    numpy.ndarray: A new matrix with shuffled rows.
    """
    shuffled_matrix = (
        matrix.copy()
    )  # Make a copy to avoid modifying the original matrix
    for row in shuffled_matrix:
        np.random.shuffle(row)  # Shuffle elements within the row
    return shuffled_matrix


def has_five_consecutive_trues(matrix: np.ndarray) -> np.ndarray:
    matrix = np.array(matrix, dtype=bool)  # Ensure it's a boolean NumPy array
    kernel = np.ones(5, dtype=int)  # Kernel to check consecutive 5 Trues
    # Perform a 1D convolution along each row
    conv_results = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), axis=1, arr=matrix
    )
    # Check if any value in the result equals 5 (meaning 5 consecutive Trues)
    return np.any(conv_results == 5, axis=1)


def find_five_consecutive_trues_center(matrix: np.ndarray) -> np.ndarray:
    def find_center(row: np.ndarray) -> int:
        conv_result = np.convolve(row, np.ones(5, dtype=int), mode="valid") == 5
        if np.any(conv_result):
            start = np.argmax(conv_result).astype(
                int
            )  # First occurrence of 5 consecutive Trues
            return start + 2  # Center index
        raise ValueError(
            "You should only pass PCs run through has_five_consective_trues to this function"
        )

    matrix = np.asarray(matrix, dtype=bool)
    return np.apply_along_axis(find_center, axis=1, arr=matrix)


def remove_diagonal(A: np.ndarray) -> np.ndarray:
    return A[~np.eye(A.shape[0], dtype=bool)].reshape(A.shape[0], -1)


def cross_correlation_pandas(matrix: np.ndarray) -> np.ndarray:

    df = pd.DataFrame(matrix)
    corr = df.corr(method="pearson")
    return corr.to_numpy()


def session_is_unsupervised(session: Cached2pSession) -> bool:
    return session.session_type.lower().startswith("unsupervised learning")


def uk_to_utc(dt: datetime) -> datetime:
    """Converts a datetime object in UK time to UTC time and strips the timezone info for further calculations.
    dt: datetime object in UK time
    """
    return (
        dt.replace(tzinfo=ZoneInfo("Europe/London"))
        .astimezone(ZoneInfo("UTC"))
        .replace(tzinfo=None)
    )
