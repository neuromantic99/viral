import gc
import math
import warnings
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Tuple, TypeVar
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.linalg import issymmetric
from scipy.ndimage import gaussian_filter1d
from statsmodels.formula.api import mixedlm

from viral.constants import ENCODER_TICKS_PER_TURN
from viral.models import (
    Cached2pSession,
    MouseSummary,
    SpeedPosition,
    TrialInfo,
)


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(arr, np.ones(window), "valid") / window


def shaded_line_plot(
    arr: np.ndarray,
    x_axis: np.ndarray | list[float],
    color: str,
    label: str,
    do_moving_average: bool = False,
    axis: plt.Axes | None = None,
) -> None:
    plotter = axis if axis is not None else plt

    if do_moving_average:
        mean = gaussian_filter1d(np.nanmean(arr, 0), sigma=1)
        sem = gaussian_filter1d(np.nanstd(arr, 0) / np.sqrt(arr.shape[1]), sigma=1)
    else:
        mean = np.nanmean(arr, 0)
        sem = np.nanstd(arr, 0) / np.sqrt(arr.shape[1])

    plotter.plot(x_axis, mean, color=color, label=label, marker="", zorder=1)
    plotter.fill_between(
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
) -> list[SpeedPosition]:
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

    speed_position: list[SpeedPosition] = []
    for start, stop in zip(
        range(first_position, last_position - step_size + 1, step_size),
        range(first_position + step_size, last_position + 1, step_size),
        strict=True,
    ):
        # TODO: This will often be zero after the reward is triggered. Deal with this
        n = np.sum(np.logical_and(position >= start, position < stop))
        if n == 0 and start < 180:
            warnings.warn("Likely the rotary encoder has jumped in a weird way.")

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


def threshold_detect_continuous(
    signal: np.ndarray, threshold: np.ndarray
) -> np.ndarray:
    """
    Detect threshold crossings where signal > threshold (elementwise).
    Suppresses consecutive detections to only return first index of each crossing.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    threshold : np.ndarray
        Threshold array of same shape as signal.

    Returns
    -------
    np.ndarray
        Indices where signal crosses threshold.
    """
    if signal.shape != threshold.shape:
        raise ValueError("signal and threshold must have the same shape")

    # Compare elementwise
    thresh_signal = signal > threshold

    # Keep only rising edge detections
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False

    # Return indices
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


def get_tiff_paths_in_directory(directory: Path) -> list[Path]:
    return list(directory.glob("*.tif*"))


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


def time_list_to_datetime(time_list: list[float]) -> datetime:
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


def find_chunk(chunk_lens: list[int] | np.ndarray, index: int) -> int:
    """Given a list of chunk lengths and an index, find the chunk that contains the index"""
    cumulative_length = 0
    for i, length in enumerate(chunk_lens):
        cumulative_length += length
        if index < cumulative_length:
            return i
    return -1  # If index is out of bounds


def average_different_lengths(data: list[np.ndarray]) -> np.ndarray:
    max_length = max(len(d) for d in data)

    for idx, d in enumerate(data):
        if len(d) < max_length:
            data[idx] = np.append(d, np.repeat(np.nan, max_length - len(d)))

    return np.nanmean(data, axis=0)


def get_genotype(
    mouse_name: str,
) -> Literal["Oligo-BACE1-KO", "NLGF", "WT", "Neuronal-BACE1-KO"]:
    if mouse_name in {"JB014", "JB015", "JB018", "JB020", "JB022"}:
        return "Oligo-BACE1-KO"
    if mouse_name in {"JB034", "JB035", "J036", "J037", "J038"}:
        return "Neuronal-BACE1-KO"
    if mouse_name in {
        "JB011",
        "JB012",
        "JB013",
        "JB016",
        "JB017",
        "JB019",
        "JB021",
        "JB023",
        "JB036",
        "J030",
        "J031",
        "J032",
    }:
        return "NLGF"

    if mouse_name in {
        "JB024",
        "JB025",
        "JB026",
        "JB027",
        "JB030",
        "JB031",
        "JB032",
        "JB033",
        "J034",
        "J035",
    }:
        return "WT"

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
        "JB034",
        "JB036",
        "J030",
        "J031",
        "J034",
        "J036",
        "J037",
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
        "JB035",
        "J032",
        "J035",
        "J038",
    }:
        return "female"
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
    UNSUPERVISED = "unsupervised"


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
    elif "unsupervised" in session_name:
        return SessionType.UNSUPERVISED.value
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
    peak_indices = np.nanargmax(matrix, axis=1)
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


def has_n_consecutive_trues(matrix: np.ndarray, n: int = 5) -> np.ndarray:
    matrix = np.array(matrix, dtype=bool)  # Ensure it's a boolean NumPy array
    kernel = np.ones(n, dtype=int)  # Kernel to check consecutive 5 Trues
    # Perform a 1D convolution along each row
    conv_results = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), axis=1, arr=matrix
    )
    # Check if any value in the result equals n (meaning n consecutive Trues)
    return np.any(conv_results == n, axis=1)


def find_n_consecutive_trues_center(matrix: np.ndarray, n: int = 5) -> np.ndarray:
    def find_center(row: np.ndarray) -> int:
        conv_result = np.convolve(row, np.ones(n, dtype=int), mode="valid") == n
        if np.any(conv_result):
            start = np.argmax(conv_result).astype(
                int
            )  # First occurrence of 5 consecutive Trues
            return start + (n // 2)  # Center index
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


def below_threshold_for_n_consecutive_samples(
    arr: np.ndarray,
    threshold: float,
    n_samples: int,
) -> np.ndarray:
    """
    Returns a boolean mask where True indicates the array element is within a bout of being
    below threshold for n_samples length (all elements in any qualifying window are True).

    Returns:
        np.ndarray: Boolean mask, same length as arr.
    """
    below = arr < threshold
    # Rolling sum to find windows of n_samples below threshold
    run_lengths = np.convolve(
        below.astype(int), np.ones(n_samples, dtype=int), mode="valid"
    )
    # Find start indices of valid runs
    valid_starts = np.where(run_lengths >= n_samples)[0]
    mask = np.zeros_like(arr, dtype=bool)
    for start in valid_starts:
        mask[start : start + n_samples] = True
    return mask


def above_threshold_for_n_consecutive_samples(
    arr: np.ndarray,
    threshold: float,
    n_samples: int,
) -> np.ndarray:
    """
    Returns a boolean mask where True indicates the array element is within a bout of being
    above threshold for n_samples length (all elements in any qualifying window are True).

    Returns:
        np.ndarray: Boolean mask, same length as arr.
    """
    above = arr > threshold
    # Rolling sum to find windows of n_samples above threshold
    run_lengths = np.convolve(
        above.astype(int), np.ones(n_samples, dtype=int), mode="valid"
    )
    # Find start indices of valid runs
    valid_starts = np.where(run_lengths >= n_samples)[0]
    mask = np.zeros_like(arr, dtype=bool)
    for start in valid_starts:
        mask[start : start + n_samples] = True
    return mask


def split_continuous_chunks(arr: np.ndarray) -> list[np.ndarray]:
    """Split an array into continuous chunks"""
    split_indices = np.where(np.diff(arr) != 1)[0] + 1
    return np.split(arr, split_indices)


def check_trial_file_sorting(trial_files: list[Path]) -> None:
    """Check that trials are sorted by trial number"""
    for trial, next_trial in zip(trial_files[:-1], trial_files[1:], strict=True):
        this_number = int(trial.stem.split("trial")[-1])
        next_number = int(next_trial.stem.split("trial")[-1])
        assert next_number == this_number + 1


def basic_normalise(data: np.ndarray) -> np.ndarray:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def imshow(matrix: np.ndarray, vmax: float | None = None) -> None:
    """Wrapper with the settings we use everytime"""
    plt.imshow(matrix, aspect="auto", interpolation="none", vmax=vmax)
    plt.colorbar()


def exp_model(t: np.ndarray, A: float, tau: float, C: float) -> np.ndarray:
    return A * np.exp(-t / tau) + C


def corr_vs_distance(A: np.ndarray) -> np.ndarray:
    assert A.shape[0] == A.shape[1]
    assert issymmetric(A, atol=0.01), "Input matrix must be symmetric"
    n = A.shape[0]
    return np.array([np.diag(A, k).mean() for k in range(n)])


def round_up_to_base(x: float, base: int) -> int:
    return base * math.ceil(x / base)


def compute_linear_slope(
    x: np.ndarray, y: np.ndarray, plot: bool = False, title: str = ""
) -> float:
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    if plot:
        plt.figure()
        plt.plot(x, y, label="data")
        plt.plot(x, slope * x + intercept, label="fit")
        plt.title(f"{title} slope: {slope:.4f}, p: {p_value:.4f}")
        plt.legend()
    return slope


def boxplot(result: Dict[str, Any]) -> None:
    sns.boxplot(result, showfliers=False)
    sns.stripplot(result, edgecolor="black", linewidth=1)
    plt.tight_layout()


def upper_triangle_no_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Return the upper triangle of a square matrix, excluding the diagonal."""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def mixed_effects(
    df: pd.DataFrame,
    dependent_var: str,
    independent_var: str,
    group_name: str,
) -> pd.Series:
    df[independent_var] = df[independent_var].astype("category")

    md = mixedlm(
        f"{dependent_var} ~ C({independent_var})",
        df,
        groups=df[group_name],
    )
    mdf = md.fit(reml=False)
    assert mdf.converged, "MixedLM did not converge for resting baseline firing rates"
    return mdf.pvalues


def interpolate_nans_vector(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 1, "Input array must be one-dimensional"
    # arr = np.asarray(arr, dtype=float)
    nans = np.isnan(arr)

    if not nans.any():
        return arr  # nothing to do

    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


def save_figure(path: Path) -> None:
    plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(path, bbox_inches="tight", transparent=True)


def compute_motion_energy(video: np.ndarray) -> np.ndarray:
    """
    Efficient per-frame motion energy for video shaped (frames, height, width).

    Returns one value per frame transition.
    """
    if video.ndim != 3:
        raise ValueError("Expected video with shape (frames, height, width)")

    # Use float32 to avoid uint8 wraparound and reduce memory vs float64.
    v = video.astype(np.float32, copy=False)

    # Allocate exactly one difference array: shape (frames - 1, height, width)
    diff = np.empty_like(v[:-1], dtype=np.float32)

    # diff[t] = v[t+1] - v[t]
    np.subtract(v[1:], v[:-1], out=diff)

    # Square in-place to avoid another large temporary.
    np.multiply(diff, diff, out=diff)

    return diff.sum(axis=(1, 2))


def motion_energy_in_chunks(mp4_path: Path, chunk_size: int) -> np.ndarray:
    """
    Processes a video file in chunks and compute motion energy for the entire video

    Parameters:
        mp4_path (Path): Path to the .mp4 file.
        chunk_size (int): Number of frames to process per chunk.

    Returns:
        numpy.ndarray: The diffed vector for the entire video.
    """

    cap = cv2.VideoCapture(str(mp4_path))
    motion_energy = []

    previous_chunk_last_frame = None  # To store the last frame of the previous chunk

    while True:
        gc.collect()
        frames = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(
                frame[:, :, 0]
            )  # Video is grayscale but three channels are loaded

        if not frames:  # Break the loop if no frames were read
            break

        frames_array = np.array(frames)

        # If there was a previous chunk, include the last frame for continuity in diff computation
        if previous_chunk_last_frame is not None:
            frames_array = np.vstack(
                [previous_chunk_last_frame[np.newaxis, ...], frames_array]
            )

        # Compute the diffed vector
        motion_energy_chunk = compute_motion_energy(frames_array)
        motion_energy.extend(motion_energy_chunk)

        # Store the last frame of this chunk for continuity in the next iteration
        previous_chunk_last_frame = frames_array[-1]

    cap.release()

    return np.array(motion_energy)


def is_ordered_subset(a: np.ndarray, b: np.ndarray) -> Tuple[bool, int | None]:
    """Is a contained within b in order"""
    n = len(a)

    if n > len(b):
        return False, None

    start = np.where([np.array_equal(a, b[i : i + n]) for i in range(len(b) - n + 1)])[
        0
    ]
    if len(start) == 1:
        return True, start[0]
    if len(start) > 1:
        raise ValueError("a is contained in b more than once")
    return False, None


def detect_events(
    vector: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> list[Tuple[int, int]]:
    in_event = False
    upper_exceeded = False
    start_event = 0

    events = []

    for idx, value in enumerate(vector):
        if value > lower_threshold and not in_event:
            start_event = idx
            in_event = True

        # If you bounce on the lower threshold
        if value < lower_threshold and in_event and not upper_exceeded:
            in_event = False

        if value > upper_threshold:
            upper_exceeded = True

        if value < lower_threshold and in_event and upper_exceeded:
            events.append((start_event, idx))
            in_event = False
            upper_exceeded = False

    return events


def subset_frames_mp4(mp4_path: Path, frames: Iterable[int], outfile: Path) -> None:
    """Subset the frames in 'frames' from an MP4 video and save to a new file.

    Decodes the video in a single sequential forward pass rather than seeking
    to each frame individually - cv2.CAP_PROP_POS_FRAMES has to decode forward
    from the nearest preceding keyframe on every call, so seeking per frame
    ends up re-decoding large stretches of the video once per requested frame.
    """
    cap = cv2.VideoCapture(mp4_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        outfile,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    target_frames = sorted(set(frames))
    target_idx = 0
    frame_idx = 0
    while target_idx < len(target_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == target_frames[target_idx]:
            out.write(frame)
            target_idx += 1
        frame_idx += 1

    cap.release()
    out.release()
