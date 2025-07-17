"""Contains functions that act on fluorescence data, either spks or dff, either alone or with behavioural data"""

from scipy.ndimage import gaussian_filter1d
from typing import List, Tuple
import numpy as np
from viral.models import TrialInfo
from viral.utils import (
    array_bin_mean,
    degrees_to_cm,
    get_wheel_circumference_from_rig,
    shuffle_rows,
    threshold_detect,
)

from viral.constants import TIFF_UMBRELLA


def compute_dff(f: np.ndarray) -> np.ndarray:
    flu_mean = np.expand_dims(np.mean(f, 1), 1)
    return (f - flu_mean) / flu_mean


def subtract_neuropil(f_raw: np.ndarray, f_neu: np.ndarray) -> np.ndarray:
    return f_raw - f_neu * 0.7


def load_imaging_data(
    mouse: str, date: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s2p_path = TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0"
    print(f"Suite 2p path is {s2p_path}")
    if not s2p_path.exists():
        raise FileNotFoundError("This session likely was not suite2p'ed yet")
    if not (s2p_path / "oasis_spikes.npy").exists():
        from viral.run_oasis import main as run_oasis

        run_oasis(mouse=mouse, date=date, grosmark=False)
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)

    spks = np.load(s2p_path / "oasis_spikes.npy")[iscell, :]
    denoised = np.load(s2p_path / "oasis_denoised.npy")[iscell, :]

    f_raw = np.load(s2p_path / "F.npy")[iscell, :]
    f_neu = np.load(s2p_path / "Fneu.npy")[iscell, :]

    dff = compute_dff(subtract_neuropil(f_raw, f_neu))
    return dff, spks, denoised


def get_ITI_start_frame(trial: TrialInfo) -> int:
    for state in trial.states_info:
        if state.name in {"ITI", "trigger_ITI"}:
            assert (
                state.closest_frame_start is not None
            ), "Imaging data not added to trial"
            return state.closest_frame_start
    raise ValueError("ITI state not found")


def get_sampling_rate(frame_clock: np.ndarray) -> int:
    """Bit of a hack as the sampling rate is not stored in the tdms file I think. I've used
    two different sampling rates: 1,000 and 10,000. The sessions should be between 30 and 100 minutes.
    """
    if 30 < len(frame_clock) / 1000 / 60 < 120:
        return 1000
    elif 30 < len(frame_clock) / 10000 / 60 < 120:
        return 10000
    raise ValueError("Could not determine sampling rate")


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

    if any(
        np.isnan(state.start_time) or np.isnan(state.end_time)
        for state in trial.states_info
    ):
        return False

    if any(np.isnan(event.start_time) for event in trial.events_info):
        return False

    length_trial_frames = (start_time_frames[-1] - start_time_frames[0]) / 30

    # A little but of a error in this calculation is allowed here as the frame rate is not exactly 30.
    # Possibly you will get a false positive if the trial is stopped exactly at the end but unlikely
    return length_trial_bpod - 0.5 <= length_trial_frames <= length_trial_bpod + 0.5


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


def activity_trial_position(
    trial: TrialInfo,
    flu: np.ndarray,
    wheel_circumference: float,
    smoothing_sigma: float | None,
    bin_size: int = 1,
    start: int = 10,
    max_position: int = 170,
    verbose: bool = False,
    do_shuffle: bool = False,
) -> np.ndarray:
    """Returns the dff activity of the trial binned by position in matrix of shape (n_cells, n_bins)
    trial: TrialInfo
    flu: dff or spks fluorescence data
    wheel_circumference: in cm
    smoothing_sigma: if None, no smoothing is applied otherwise smooth with a gaussian filter of width smoothing_sigma (cm)
    bin_size: in cm
    start: in cm
    max_position: in cm
    verbose: if True, print the binning information
    do_shuffle: if True, shuffle the rows of the dff matrix
    """

    # TODO: remove non running epochs?

    position = degrees_to_cm(
        np.array(trial.rotary_encoder_position), wheel_circumference
    )

    frame_position = np.array(
        [
            state.closest_frame_start
            for state in trial.states_info
            if state.name
            in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
        ]
    )

    assert len(position) == len(frame_position)

    dff_position_list = []

    for bin_start in range(start, max_position, bin_size):
        frame_idx_bin = np.unique(
            frame_position[
                np.logical_and(position >= bin_start, position < bin_start + bin_size)
            ]
        )
        dff_bin = flu[:, frame_idx_bin]

        if verbose:
            print(f"bin_start: {bin_start}")
            print(f"bin_end: {bin_start + bin_size}")
            print(f"n_frames in bin: {len(frame_idx_bin)}")
        dff_position_list.append(np.mean(dff_bin, axis=1))

    dff_position = np.array(dff_position_list).T

    if smoothing_sigma is not None:
        dff_position = gaussian_filter1d(dff_position, sigma=smoothing_sigma, axis=1)

    if do_shuffle:
        return shuffle_rows(dff_position)

    return dff_position


def get_resting_chunks(
    trials: List[TrialInfo],
    dff: np.ndarray,
    chunk_size_frames: int,
    speed_threshold: float,
) -> np.ndarray:

    all_chunks = []
    for trial in trials:

        lick_frames = []

        # TODO: This will error if the trial ends during a lick event
        for onset, offset in zip(
            [
                event.closest_frame
                for event in trial.events_info
                if event.name == "Port1In"
            ],
            [
                event.closest_frame
                for event in trial.events_info
                if event.name == "Port1Out"
            ],
            strict=True,
        ):
            assert onset is not None
            assert offset is not None
            lick_frames.extend(range(onset, offset + 1))

        position_frames = np.array(
            [
                state.closest_frame_start
                for state in trial.states_info
                if state.name
                in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
            ]
        )

        positions = degrees_to_cm(
            np.array(trial.rotary_encoder_position),
            get_wheel_circumference_from_rig("2P"),
        )

        n_chunks = (position_frames[-1] - position_frames[0]) // chunk_size_frames

        for chunk in range(n_chunks):
            start = chunk * chunk_size_frames
            end = start + chunk_size_frames

            frame_start = position_frames[0] + start
            frame_end = frame_start + chunk_size_frames

            distance_travelled = max(positions[start:end]) - min(positions[start:end])
            speed = distance_travelled / (chunk_size_frames / 30)

            if (
                speed < speed_threshold
                and len(
                    set(lick_frames).intersection(set(range(frame_start, frame_end)))
                )
                == 0
            ):

                all_chunks.append(
                    dff[:, frame_start:frame_end],
                )

    return np.array(all_chunks)


def running_during_ITI(trial: TrialInfo) -> bool:

    trigger_states = np.array(
        [
            state.name
            for state in trial.states_info
            if state.name
            in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
        ]
    )

    ITI_positions = degrees_to_cm(
        np.array(trial.rotary_encoder_position)[trigger_states == "trigger_panda_ITI"],
        get_wheel_circumference_from_rig("2P"),
    )
    return max(ITI_positions) - min(ITI_positions) > 20


def get_ITI_matrix(
    trials: List[TrialInfo],
    flu: np.ndarray,
    bin_size: int | None,
) -> np.ndarray:
    """
    In theory will be 600 frames in an ITI (as is always 20 seconds)
    In practise it's 599 or 598 (or could be something like 597 or 600, fix
    the assertion if so).
    Or could be less if you stop the trial in the ITI.
    So we'll take the first 598 frames of any trial that has 599 or 598
    """

    matrices = []

    for trial in trials:
        assert trial.trial_end_closest_frame is not None

        # This would be good, but in practise it rarely occurs
        # if running_during_ITI(trial):
        #     continue

        chunk = flu[:, get_ITI_start_frame(trial) : int(trial.trial_end_closest_frame)]

        n_frames = chunk.shape[1]

        if n_frames < 550:
            # Imaging stopped in the middle
            continue
        if n_frames in {598, 599, 600}:
            if bin_size is None:
                matrices.append(chunk[:, :598])
            else:
                matrices.append(array_bin_mean(chunk[:, :598], bin_size=bin_size))
        else:
            raise ValueError(f"Chunk with {n_frames} frames not understood")

    return np.array(matrices)


def get_imaging_crashed(mouse_name: str, date: str) -> bool:
    """Manually define if sessions have crashed imaging, based on the metadata"""
    return (mouse_name, date) in [
        ("JB011", "2024-10-22"),
        ("JB011", "2024-10-25"),
        ("JB034", "2025-07-04"),
    ]
