import json
import re
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nptdms import TdmsFile
from ScanImageTiffReader import ScanImageTiffReader

# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

import logging

from viral.constants import (
    BEHAVIOUR_DATA_PATH,
    CACHE_PATH,
    SPREADSHEET_ID,
    SYNC_FILE_PATH,
    TEMP_CACHE_PATH,
    TIFF_UMBRELLA,
)
from viral.correct_2p_sessions import apply_session_correction
from viral.gsheets_importer import gsheet2df
from viral.imaging_utils import (
    extract_TTL_chunks,
    get_daq_crashed,
    get_imaging_crashed,
    get_sampling_rate,
    trial_is_imaged,
)
from viral.models import Cached2pSession, SessionImagingInfo, TrialInfo, WheelFreeze
from viral.multiple_sessions import parse_session_number
from viral.single_session import HERE, load_data
from viral.utils import (
    SessionType,
    find_chunk,
    get_session_type,
    get_tiff_paths_in_directory,
    is_ordered_subset,
    time_list_to_datetime,
    uk_to_utc,
)
from viral.wheel_freeze_movement import (
    get_wheel_freeze_movement,
    load_freeze_trials,
)

logging.basicConfig(
    filename="2p_cacher_log.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)

logging.info("Starting 2p cacher")

logger = logging.getLogger("2p_cacher")


def sanity_check_imaging_frames(
    frame_times: np.ndarray, sampling_rate: float, frame_clock: np.ndarray
) -> None:
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
    ## Useful plot to debug
    if len(bad_times) != 0:
        plt.plot(frame_clock)
        plt.plot(frame_times, np.ones(len(frame_times)), ".", color="green")
        plt.plot(frame_times[bad_times], np.ones(len(bad_times)), ".", color="red")
        plt.show()
        raise ValueError("Bad inter-frame-interval. Inspect the clock using the plot")

    assert len(bad_times) == 0


def count_spacers(trial: TrialInfo) -> int:
    return len([state for state in trial.states_info if "spacer_high" in state.name])


def add_daq_times_to_trial(
    trial: TrialInfo,
    trial_idx: int,
    valid_frame_times: np.ndarray,
    behaviour_times: np.ndarray,
    behaviour_chunk_lens: np.ndarray,
    daq_sampling_rate: int,
    wheel_freeze: WheelFreeze | None = None,
    offset_after_pre_epoch: int = 0,
    task_sync_start: int | None = None,
    is_freeze_session: bool = False,
) -> None:

    # If we recorded a task for the pre-freeze, remove these syncs from the daq so
    # only task syncs are used for the trial
    task_sync_start = task_sync_start if task_sync_start is not None else 0
    behaviour_times = behaviour_times[np.sum(behaviour_chunk_lens[:task_sync_start]) :]
    behaviour_chunk_lens = behaviour_chunk_lens[task_sync_start:]

    trial_spacer_daq_times = behaviour_times[
        np.sum(behaviour_chunk_lens[:trial_idx]) : np.sum(
            behaviour_chunk_lens[: trial_idx + 1]
        )
    ]
    assert len(trial_spacer_daq_times) == count_spacers(trial)

    trial_spacer_bpod_times = np.array(
        [state.start_time for state in trial.states_info if "spacer_high" in state.name]
    )

    # Should be the first state, but verify
    assert trial_spacer_bpod_times[0] == 0
    # Check that the clocks are equal to the millisecond
    np.testing.assert_almost_equal(
        (trial_spacer_daq_times - trial_spacer_daq_times[0]) / daq_sampling_rate,
        trial_spacer_bpod_times,
        decimal=3,
    )

    bpod_to_daq = (
        lambda bpod_time: bpod_time * daq_sampling_rate + trial_spacer_daq_times[0]
    )

    # Another probably redundant sanity check
    np.testing.assert_almost_equal(
        np.array([bpod_to_daq(bpod_time) for bpod_time in trial_spacer_bpod_times])
        / daq_sampling_rate,
        trial_spacer_daq_times / daq_sampling_rate,
        decimal=3,
    )

    # Bit of monkey patching but oh well
    # TODO: think of permanent fix for behaviour time stamps being NaN
    for state in trial.states_info:
        state.start_time_daq = bpod_to_daq(state.start_time).astype(float)
        state.end_time_daq = bpod_to_daq(state.end_time).astype(float)
        if np.isnan(state.start_time):
            state.closest_frame_start = np.nan
        else:
            state.closest_frame_start = (
                int(np.argmin(np.abs(valid_frame_times - state.start_time_daq)))
                + offset_after_pre_epoch
            )
        if np.isnan(state.end_time):
            state.closest_frame_end = np.nan
        else:
            state.closest_frame_end = (
                int(np.argmin(np.abs(valid_frame_times - state.end_time_daq)))
                + offset_after_pre_epoch
            )

        if state.name == "spacer_high_00":
            trial.trial_start_closest_frame = state.closest_frame_start

    if trial_is_imaged(trial):
        for state in trial.states_info:
            assert_against_wheel_freeze_indices(
                wheel_freeze, is_freeze_session, state.closest_frame_start
            )

            assert_against_wheel_freeze_indices(
                wheel_freeze, is_freeze_session, state.closest_frame_end
            )

    last_state = trial.states_info[-1]
    trial.trial_end_closest_frame = last_state.closest_frame_end

    for event in trial.events_info:
        event.start_time_daq = float(bpod_to_daq(event.start_time))
        if np.isnan(event.start_time):
            event.closest_frame = np.nan
        else:
            event.closest_frame = (
                int(np.argmin(np.abs(valid_frame_times - event.start_time_daq)))
                + offset_after_pre_epoch
            )

        if trial_is_imaged(trial):
            assert_against_wheel_freeze_indices(
                wheel_freeze, is_freeze_session, event.closest_frame
            )


def assert_against_wheel_freeze_indices(
    wheel_freeze: WheelFreeze | None, is_freeze_session: bool, frame_time: float
) -> None:
    if wheel_freeze is None:
        return

    if np.isnan(frame_time):
        return

    if is_freeze_session:
        assert (
            frame_time >= wheel_freeze.pre_training_start_frame
            and frame_time <= wheel_freeze.pre_training_end_frame
        ) or (
            frame_time >= wheel_freeze.post_training_start_frame
            and frame_time <= wheel_freeze.post_training_end_frame
        ), "Wheel freeze trials assigned to main task period"
        return

    assert (
        frame_time >= wheel_freeze.pre_training_end_frame
        and frame_time <= wheel_freeze.post_training_start_frame
    ), "Main task trials assigned to wheel freeze period"


def extract_frozen_wheel_chunks(
    stack_lengths_tiffs: np.ndarray,
    valid_frame_times: np.ndarray,
    behaviour_times: np.ndarray,
    sampling_rate: int,
    frame_rate: int = 30,
    check_first_chunk: bool = True,
    recorded_movement_during_freezes: bool = False,
) -> tuple[tuple[int, int], tuple[int, int]] | tuple[None, tuple[int, int]]:
    """Extract start and end frame index for pre-training and post-training imaging chunks.
    Args:
        stack_lengths_tiffs (np.ndarray):           A NumPy array of length tiffs, with the number of frames in each tiff.
        valid_frame_times (np.ndarray):             A NumPy array with times for valid frames.
        behaviour_times (np.ndarray):               A NumPy array with all times when a Bpod spacer signal occured.
        sampling_rate (int):                        The sampling rate of the DAQ system in Hz.
        frame_rate (int):                           The frame rate of the 2P in frames per second.
        check_first_chunk (bool):                   Whether to check the first imaging chunk for behaviour pulses.
                                                    Defaults to True.
                                                    Set to False for sessions where the DAQ was started after imaging the pre-session epoch.

    Returns:
        tuple[tuple[int, int], tuple[int, int]]:    First chunk and last chunk, with their respective start and end frame.
    """

    if check_first_chunk:
        # first chunk (before behavioural chunks)
        first_chunk_len = stack_lengths_tiffs[0]
        first_chunk = (0, first_chunk_len)  # start and end frame
        first_chunk_times = valid_frame_times[first_chunk[0] : first_chunk[1]]
    else:
        first_chunk = None

    # last chunk (after all behavioural chunks)
    last_chunk_len = stack_lengths_tiffs[-1]
    prev_frames_total = sum(stack_lengths_tiffs[:-1])
    last_chunk = (
        prev_frames_total,
        prev_frames_total + last_chunk_len,
    )  # start and end frame
    last_chunk_times = valid_frame_times[last_chunk[0] : last_chunk[1]]

    # We want to make sure the chunks are >= 15 mins but < 20 mins
    # 15 mins = 27,000 frames
    # 20 mins = 36,000 frames
    if check_first_chunk:
        assert (
            27000 <= first_chunk_len < 36000
        ), "First chunk length does not match expected length"
        assert (
            15 * 60 * sampling_rate
            <= (
                first_chunk_times[-1]
                - first_chunk_times[0]
                + sampling_rate
                / frame_rate  # accounting for the duration of the last frame
            )
            <= 20 * 60 * sampling_rate
        ), "First chunk length does not match expected length"

    assert (
        27000 <= last_chunk_len < 36000
    ), "Last chunk length does not match expected length"
    assert (
        15 * 60 * sampling_rate
        <= (
            last_chunk_times[-1] - last_chunk_times[0] + sampling_rate / frame_rate
        )  # accounting for the duration of the last frame
        <= 20 * 60 * sampling_rate
    ), "Last chunk length does not match expected length"

    # Need to disable this check with the most recent verion as
    # we are now recording movement on the bpod during the wheel freezes
    if check_first_chunk and not recorded_movement_during_freezes:
        assert not np.any(
            (behaviour_times >= first_chunk_times[0])
            & (behaviour_times <= first_chunk_times[-1])
        ), "Behavioural pulses detected in pre-training period!"

    if not recorded_movement_during_freezes:
        assert not np.any(
            (behaviour_times >= last_chunk_times[0])
            & (behaviour_times <= last_chunk_times[-1])
        ), "Behavioural pulses detected in post-training period!"

    return first_chunk, last_chunk


def manual_wheel_freezes(
    mouse_name: str, date: str, session_sync: SessionImagingInfo
) -> WheelFreeze | None:
    if (mouse_name, date) == ("JB035", "2025-07-04"):
        return WheelFreeze(
            pre_training_start_frame=0,
            pre_training_end_frame=19011 + 8000,
            post_training_start_frame=sum(session_sync.stack_lengths_tiffs[:-1]),
            post_training_end_frame=sum(session_sync.stack_lengths_tiffs),
        )

    return None


def get_wheel_freeze(
    session_sync: SessionImagingInfo, recorded_movement_during_freezes: bool
) -> WheelFreeze:
    """Get wheel freeze object."""
    # TODO: if this occurs more often, find a more elegant fix
    # manually set wheel freeze objects for crashed recordings
    manual_wheel_freeze = {
        "JB034_2025-07-04": WheelFreeze(
            pre_training_start_frame=0,
            pre_training_end_frame=27000,
            post_training_start_frame=sum([27000, 30240, 69451, 10372]),
            post_training_end_frame=sum([27000, 30240, 69451, 10372, 14200, 13000]),
        )
    }
    if f"{session_sync.mouse_name}_{session_sync.date}" in manual_wheel_freeze.keys():
        print("Using a manually set WheelFreeze object")
        return manual_wheel_freeze[f"{session_sync.mouse_name}_{session_sync.date}"]
    frozen_wheel_chunks = extract_frozen_wheel_chunks(
        stack_lengths_tiffs=session_sync.stack_lengths_tiffs,
        valid_frame_times=session_sync.valid_frame_times,
        behaviour_times=session_sync.behaviour_times,
        sampling_rate=session_sync.sampling_rate,
        check_first_chunk=session_sync.offset_after_pre_epoch == 0,
        recorded_movement_during_freezes=recorded_movement_during_freezes,
    )
    if session_sync.offset_after_pre_epoch > 0:
        print("Using offset after pre-epoch")
        return WheelFreeze(
            pre_training_start_frame=0,
            pre_training_end_frame=session_sync.offset_after_pre_epoch,
            post_training_start_frame=frozen_wheel_chunks[1][0]
            + session_sync.offset_after_pre_epoch,
            post_training_end_frame=frozen_wheel_chunks[1][1]
            + session_sync.offset_after_pre_epoch,
        )
    assert frozen_wheel_chunks[0] is not None
    return WheelFreeze(
        pre_training_start_frame=frozen_wheel_chunks[0][0],
        pre_training_end_frame=frozen_wheel_chunks[0][1],
        post_training_start_frame=frozen_wheel_chunks[1][0],
        post_training_end_frame=frozen_wheel_chunks[1][1],
    )


def add_imaging_info_to_trials(
    trials: List[TrialInfo],
    session_sync: SessionImagingInfo,
    wheel_freeze: WheelFreeze | None = None,
    daq_crashed: bool = False,
    is_freeze_session: bool = False,
) -> List[TrialInfo]:
    """Adds imaging info to trials."""
    logger.info("Adding imaging info to trials")

    for idx, trial in enumerate(trials):
        # Works in place, maybe not ideal
        add_daq_times_to_trial(
            trial,
            idx,
            session_sync.valid_frame_times,
            session_sync.behaviour_times,
            session_sync.behaviour_chunk_lens,
            session_sync.sampling_rate,
            wheel_freeze,
            session_sync.offset_after_pre_epoch,
            session_sync.task_sync_start,
            is_freeze_session=is_freeze_session,
        )

    for idx, trial in enumerate(trials):
        check_timestamps(
            epochs=session_sync.epochs,
            trial=trial,
            all_tiff_timestamps=session_sync.all_tiff_timestamps,
            chunk_lens=session_sync.chunk_lengths_daq,
            valid_frame_times=session_sync.valid_frame_times,
            sampling_rate=session_sync.sampling_rate,
            daq_start_time=session_sync.daq_start_time,
            wheel_blocked=bool(wheel_freeze),
            offset_after_pre_epoch=session_sync.offset_after_pre_epoch,
            loosen_assertion=daq_crashed,
        )

    return trials


def get_session_sync(
    tdms_path: Path,
    mouse_name: str,
    date: str,
    tiff_directory: Path,
    trials: List[TrialInfo],
    imaging_crashed: bool = False,
) -> SessionImagingInfo:
    """Handles all the necessary logics for syncing imaging and behaviour."""
    t1 = time.time()

    tiff_paths = sorted(get_tiff_paths_in_directory(tiff_directory))

    stack_lengths_tiffs, epochs, all_tiff_timestamps = get_tiff_metadata(
        tiff_paths=tiff_paths, use_cache=True
    )
    print("Got tiff metadata")

    tdms_file = TdmsFile.read(tdms_path)
    group = tdms_file["Analog"]
    frame_clock = group["AI0"][:]
    behaviour_clock = group["AI1"][:]
    daq_start_time = pd.Timestamp(
        group.__dict__["properties"]["StartTime"]
    ).to_pydatetime()

    print(f"Time to load data: {time.time() - t1}")

    sampling_rate = get_sampling_rate(frame_clock)

    print(f"Sampling rate: {sampling_rate}")

    print("Extracting TTL chunks from behaviour")
    behaviour_times, behaviour_chunk_lens = extract_TTL_chunks(
        behaviour_clock, sampling_rate
    )
    num_spacers_per_trial = np.array([count_spacers(trial) for trial in trials])

    # Behaviour crashed half way through a trial, so manual fix
    if "JB011" in str(tdms_path) and "2024-10-22" in str(tdms_path):
        behaviour_chunk_lens = np.delete(behaviour_chunk_lens, 52)

    check, start = is_ordered_subset(num_spacers_per_trial, behaviour_chunk_lens)
    assert check, "Spacers recorded in txt file do not match sync"

    print("Extracting TTL chunks from frame clock")
    frame_times_daq, chunk_lengths_daq = extract_TTL_chunks(frame_clock, sampling_rate)

    correction = apply_session_correction(
        mouse_name=mouse_name,
        date=date,
        epochs=epochs,
        all_tiff_timestamps=all_tiff_timestamps,
        stack_lengths_tiffs=stack_lengths_tiffs,
        chunk_lengths_daq=chunk_lengths_daq,
        frame_times_daq=frame_times_daq,
    )
    epochs = correction.epochs
    all_tiff_timestamps = correction.all_tiff_timestamps
    stack_lengths_tiffs = correction.stack_lengths_tiffs
    chunk_lengths_daq = correction.chunk_lengths_daq
    frame_times_daq = correction.frame_times_daq
    # a bit of a hack
    # when DAQ was started before the pre-session epoch, this will be 0
    # when DAQ was started after the pre-session epoch, this will be #frames in the pre-session epoch
    # (this is necessary so that the valid frames and check tiff timestamps logic works)
    offset_after_pre_epoch = correction.offset_after_pre_epoch

    sanity_check_imaging_frames(frame_times_daq, sampling_rate, frame_clock)

    print("Getting valid frame times")
    valid_frame_times = get_valid_frame_times(
        stack_lengths_tiffs=stack_lengths_tiffs,
        frame_times_daq=frame_times_daq,
        chunk_lengths_daq=chunk_lengths_daq,
        loosen_assertions=imaging_crashed,
    )

    # crazy hack to bypass the frame times validation for post session wheel freeze when DAQ crashed
    if mouse_name == "JB034" and date == "2025-07-04":
        print("JB034_2025-07-04 fix for valid_frame_times")
        valid_frame_times = np.append(
            valid_frame_times, np.ones(shape=sum([14200, 13000]))
        )

    print("Checking tiff timestamps against suite2p output")
    check_against_suite2p_output(mouse_name, date, valid_frame_times)

    # not the most beautiful solution, but works and relieves add_imaging_info_to_trials
    return SessionImagingInfo(
        mouse_name=mouse_name,
        date=date,
        stack_lengths_tiffs=stack_lengths_tiffs,
        epochs=epochs,
        all_tiff_timestamps=all_tiff_timestamps,
        chunk_lengths_daq=chunk_lengths_daq,
        daq_start_time=daq_start_time,
        valid_frame_times=valid_frame_times,
        behaviour_chunk_lens=behaviour_chunk_lens,
        behaviour_times=behaviour_times,
        sampling_rate=sampling_rate,
        offset_after_pre_epoch=offset_after_pre_epoch,
        task_sync_start=start,
    )


def get_valid_frame_times(
    stack_lengths_tiffs: np.ndarray,
    frame_times_daq: np.ndarray,
    chunk_lengths_daq: np.ndarray,
    loosen_assertions: bool = False,
) -> np.ndarray:
    """
    Consistently, the number of triggers recorded is two more than the number of frames recorded (for the imaged behaviour chunks).
    This only occurs when the imaging is manually stopped before a grab is complete (confirmed by counting triggers
    from a completed grab).
    The reason for first extra frame is obvious (we stop the imaging mid-way through a frame so it is not saved).
    The second happens for unclear reasons but must be at the end as there are no extra frames in the middle and the first
    frame is reliably correct
    Possible we may see a recording with one extra frame if the imaging is stopped on flyback. The error below will catch this

    We also now have a one recording that was not aborted (i.e. ran to 100,000 frames. The assertion below deals with this.

    loosen_assertions: fudge flag to deal with crashed sessions. If e.g. the grab crashes you'll see lots of frames (currently 13 but could be more)
                       that are in the daq but not in the tiff. Only include this flag if the notes say the session was crashed.

    """

    valid_frame_times = np.array([])
    offset = 0
    for stack_len_tiff, chunk_len_daq in zip(
        stack_lengths_tiffs, chunk_lengths_daq, strict=True
    ):
        assert (
            chunk_len_daq - stack_len_tiff
            in {
                0,
                2,
                3,
            }
            or loosen_assertions
            and chunk_len_daq - stack_len_tiff <= 13
        ), f"""The difference between daq chunk length and tiff length is not 0 or 2. Rather it is {chunk_len_daq - stack_len_tiff}./n
        This will occur, especially on crashed recordings. Think about a fix."""

        valid_frame_times = np.append(
            valid_frame_times, frame_times_daq[offset : offset + stack_len_tiff]
        )
        offset += chunk_len_daq

    assert len(valid_frame_times) == sum(stack_lengths_tiffs) and (
        (
            0
            <= len(frame_times_daq) - len(valid_frame_times)
            <= 3 * len(chunk_lengths_daq)
        )
        or loosen_assertions
    )

    return valid_frame_times


def get_tiff_metadata(
    tiff_paths: List[Path], use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mouse_name = tiff_paths[0].parent.name
    date = tiff_paths[0].parent.parent.name

    if use_cache:
        if (TEMP_CACHE_PATH / f"{mouse_name}_{date}_stack_lengths.npy").exists():
            print("Using cached tiff metadata")
            stack_lengths = np.load(
                TEMP_CACHE_PATH / f"{mouse_name}_{date}_stack_lengths.npy"
            )
            epochs = np.load(TEMP_CACHE_PATH / f"{mouse_name}_{date}_epochs.npy")
            all_tiff_timestamps = np.load(
                TEMP_CACHE_PATH / f"{mouse_name}_{date}_all_tiff_timestamps.npy"
            )
            return stack_lengths, epochs, all_tiff_timestamps

    print("Could not find cached tiff metadata. Reading tiffs (takes a long time)")
    tiffs = []
    for tiff in tiff_paths:
        try:
            tiffs.append(ScanImageTiffReader(str(tiff)))
        except Exception as e:
            raise Exception(f"Error reading tiff file: {tiff}") from e

    stack_lengths = [tiff.shape()[0] for tiff in tiffs]
    epochs = []
    all_tiff_timestamps = []
    for tiff in tiffs:
        _, epoch, tiff_timestamps = extract_metadata(tiff)
        epochs.append(epoch)
        all_tiff_timestamps.extend(tiff_timestamps)
        check_no_dropped_frames(tiff_timestamps)

    for variable, name in zip(
        [stack_lengths, all_tiff_timestamps, epochs],
        ["stack_lengths", "all_tiff_timestamps", "epochs"],
    ):
        np.save(
            TEMP_CACHE_PATH / f"{mouse_name}_{date}_{name}.npy",
            variable,
        )

    return stack_lengths, epochs, all_tiff_timestamps


def extract_metadata(tiff: ScanImageTiffReader) -> Tuple[int, List[float], List[float]]:
    stack_length = tiff.shape()[0]
    tiff_timestamps = [
        float(
            re.search(
                r"frameTimestamps_sec\s*=\s*(-?\d+\.\d+)",
                tiff.description(idx),
            )[1]
        )
        for idx in range(tiff.shape()[0])
    ]
    if any([t is None for t in tiff_timestamps]):
        raise ValueError("Could not extract all timestamps from tiff description")
    # epoch (same across frames, grab from first description)
    epoch_match = re.search(r"epoch\s*=\s*\[([^\]]+)\]", tiff.description(0))
    if epoch_match is None:
        raise ValueError("Could not extract epoch from tiff description")
    epoch = list(map(float, epoch_match[1].split()))
    return stack_length, epoch, tiff_timestamps


def check_no_dropped_frames(tiff_timestamps: List[float]) -> None:
    diffed = np.diff(tiff_timestamps)
    # Check no dropped frames in the middle
    assert (
        round(np.max(diffed), 3) == round(np.min(diffed), 3) == 0.033
    ), f"Dropped frames in the middle based on tiff timestamps. Min diffed = {np.min(diffed)}, max diffed = {np.max(diffed)}"


def check_timestamps(
    epochs: List[List[float]],
    trial: TrialInfo,
    all_tiff_timestamps: np.ndarray,
    chunk_lens: np.ndarray,
    valid_frame_times: np.ndarray,
    sampling_rate: int,
    daq_start_time: datetime,
    wheel_blocked: bool = False,
    offset_after_pre_epoch: int = 0,
    loosen_assertion: bool = False,
) -> None:
    """Compares the timestamps in the tiff to the timestamps in the Daq (the time of the trigger, offset to the timestamp that the daq started)
    Currently works trial by trial which isn't really necessary.
    There is a bit of a drift here, frames at the start have a closer match in times to frames at the end. This is probably because the sampling rate
    is not exactly 10,000. But if they are off by less than 10ms that's fine.
    This will not process ITI frames in the original version of task.py but will after we added the position storage
    """

    if not trial_is_imaged(trial):
        return

    # loosen the assertion for all tiff frames being accounted for in the valid frame times (DAQ), for when the DAQ crashed
    if not loosen_assertion:
        assert len(all_tiff_timestamps) == len(valid_frame_times)

    first_frame_trial = trial.trial_start_closest_frame - offset_after_pre_epoch
    last_frame_trial = trial.trial_end_closest_frame - offset_after_pre_epoch
    assert first_frame_trial is not None
    assert last_frame_trial is not None

    assert first_frame_trial <= len(valid_frame_times) and last_frame_trial <= len(
        valid_frame_times
    )

    epoch_trial = find_chunk(chunk_lens, first_frame_trial)
    # Epoch is in uk time, convert to UTC to match the daq
    chunk_start = uk_to_utc(time_list_to_datetime(epochs[epoch_trial]))

    for frame in range(first_frame_trial, last_frame_trial):
        # The time in the tiff. Not sure if this is the end or the start of the tiff
        frame_datetime = chunk_start + timedelta(seconds=all_tiff_timestamps[frame])
        frame_daq_time = daq_start_time + timedelta(
            seconds=valid_frame_times[frame] / sampling_rate
        )

        offset = (frame_datetime - frame_daq_time).total_seconds()

        # Allow for some drift up to 20 ms at the start and 50ms at the end
        # The clocks do drift slightly but if we're within 50ms at the end
        # We have almost definitely asigned to the correct frames
        increase_offset_allowance_time = 30
        if frame / 60 / 30 < increase_offset_allowance_time:
            assert abs(offset) <= 0.02, "Tiff timestamp does not match daq timestamp"
        else:
            # Probably ideally this would be a bit lower, but drifting by one frame
            # is probably ok. And it's just the timestamp, the actual frame match should be ok
            assert abs(offset) <= 0.05, "Tiff timestamp does not match daq timestamp"


def process_session(
    trials: List[TrialInfo],
    tiff_directory: Path,
    tdms_path: Path,
    mouse_name: str,
    date: str,
    session_type: str,
    wheel_blocked: bool,
    row: pd.Series,
) -> None:
    print(f"Off we go for {mouse_name} {date} {session_type}")
    imaging_crashed = get_imaging_crashed(mouse_name, date)
    print(f"Imaging crashed: {imaging_crashed}")
    daq_crashed = get_daq_crashed(mouse_name, date)
    print(f"DAQ crashed: {daq_crashed}")

    if wheel_blocked:
        print("Wheel blocked")
        logger.info(f"Wheel blocked in session {mouse_name} {date}")
    session_sync = get_session_sync(
        tdms_path=tdms_path,
        mouse_name=mouse_name,
        date=date,
        tiff_directory=tiff_directory,
        trials=trials,
        imaging_crashed=imaging_crashed,
    )

    print("Got session sync")
    recorded_movement_during_freezes = "Session Number pre-freeze" in row

    manual = manual_wheel_freezes(mouse_name, date, session_sync)
    wheel_freeze = (
        manual
        if manual is not None
        else (
            get_wheel_freeze(session_sync, recorded_movement_during_freezes)
            if wheel_blocked
            else None
        )
    )

    trials = add_imaging_info_to_trials(
        trials=trials,
        session_sync=session_sync,
        wheel_freeze=wheel_freeze,
        daq_crashed=daq_crashed,
        is_freeze_session=False,
    )

    trials_pre_freeze, trials_post_freeze = load_synced_freeze_sessions(
        mouse_name=mouse_name,
        date=date,
        row=row,
        daq_crashed=daq_crashed,
        session_sync=session_sync,
        wheel_freeze=wheel_freeze,
    )

    if wheel_blocked and wheel_freeze is not None:
        movement_pre_freeze, movement_post_freeze, freeze_movement_type = (
            get_wheel_freeze_movement(
                wheel_freeze=wheel_freeze,
                row=row,
                mouse_name=mouse_name,
                date=date,
                trials_pre_freeze=trials_pre_freeze,
                trials_post_freeze=trials_post_freeze,
            )
        )

    with open(CACHE_PATH / f"{mouse_name}_{date}.json", "w") as f:
        json.dump(
            Cached2pSession(
                mouse_name=mouse_name,
                date=date,
                trials=trials,
                session_type=session_type,
                # Could it be more obvious that we changed the datastructure after the code was written?
                wheel_freeze=(
                    wheel_freeze.model_copy(
                        update={
                            "trials_pre_freeze": trials_pre_freeze,
                            "trials_post_freeze": trials_post_freeze,
                            "movement_pre_freeze": movement_pre_freeze.tolist(),
                            "movement_post_freeze": movement_post_freeze.tolist(),
                            "freeze_movement_type": freeze_movement_type,
                        }
                    )
                    if wheel_freeze is not None
                    else None
                ),
            ).model_dump(),
            f,
        )

    print(f"Done for {mouse_name} {date} {session_type}")


def load_synced_freeze_sessions(
    mouse_name: str,
    date: str,
    row: pd.Series,
    daq_crashed: bool,
    session_sync: SessionImagingInfo,
    wheel_freeze: WheelFreeze | None,
) -> tuple[list[TrialInfo] | None, list[TrialInfo] | None]:

    pre_freeze_trials, post_freeze_trials = load_freeze_trials(mouse_name, date, row)
    if pre_freeze_trials is None or post_freeze_trials is None:
        # TODO: case where one exists but one does not
        return None, None

    result = []
    for freeze_type, trials in zip(
        ["pre", "post"], [pre_freeze_trials, post_freeze_trials]
    ):

        num_spacers_per_trial = np.array([count_spacers(trial) for trial in trials])
        if freeze_type == "post":
            # I sometimes stop the daq before I stop the post-freeze task.
            # So some trials are recorded as jsons but not in the daq.
            # Should have the first 15 minutes though, so correct that here
            check, start = is_ordered_subset(
                num_spacers_per_trial[:30], session_sync.behaviour_chunk_lens
            )
            assert (
                check and start is not None
            ), "Spacers recorded in txt file do not match sync"
            num_trials_in_daq = len(session_sync.behaviour_chunk_lens) - start
            trials = trials[:num_trials_in_daq]
            print(f"Post-freeze trials truncated to {num_trials_in_daq}")

        else:
            check, start = is_ordered_subset(
                num_spacers_per_trial, session_sync.behaviour_chunk_lens
            )
            assert check, "Spacers recorded in txt file do not match sync"

        # Utterly filthy mutation of session sync,
        # but it saves a complete re-write or loading big files three times.
        # Doesn't mutate the original.
        session_sync_freeze = replace(session_sync, task_sync_start=start)

        trials = add_imaging_info_to_trials(
            trials,
            session_sync_freeze,
            wheel_freeze,
            daq_crashed,
            is_freeze_session=True,
        )
        result.append(trials)

    return result[0], result[1]


def check_against_suite2p_output(
    mouse: str, date: str, valid_frame_times: np.ndarray
) -> None:
    s2p_path = TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0"
    # Just check the fluoresence shape. We need to cache before we run
    # oasis now, so it doesn't make sense to look at the spikes too
    f = np.load(s2p_path / "F.npy")
    assert len(valid_frame_times) == f.shape[1]


ALL_MICE = [
    # "JB011",
    # "JB012",
    # "JB013",
    # "JB014",
    # "JB015",
    # "JB016",
    # "JB017",
    # "JB018",
    # "JB019",
    # "JB020",
    # "JB021",
    # "JB022",
    # "JB023",
    # "JB024",
    # "JB025",
    # "JB026",
    # "JB027",
    "JB030",
    "JB031",
    "JB032",
    "JB033",
    "JB034",
    "JB035",
    "JB036",
    "J030",
    "J031",
    "J032",
    "J035",
    "J034",
    "J036",
    "J037",
    "J038",
]


def main() -> None:
    redo = False
    # for mouse_name in ["JB030"]:
    # Toggle whether the try catch throws or not without commenting it
    debug = True

    for mouse_name in ALL_MICE:
        #######3333333###33333################## TAKE ME OUT ###################################################
        # if mouse_name == "JB033":
        #     continue
        metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
        for _, row in metadata.iterrows():
            try:
                print(f"The type is {row['Type']}")
                date = row["Date"]
                session_type = row["Type"].lower()
                if "Analyse" in row and row["Analyse"] == "FALSE":
                    print(
                        f"Skipping {mouse_name} {date} {session_type} as Analyse is FALSE"
                    )
                    continue

                try:
                    wheel_blocked = row["Wheel blocked?"].lower() in {"yes", "true"}
                except KeyError as e:
                    print(f"No column 'Wheel blocked?' Error is: {e}")
                    print("Wheel blocked set to None")
                    wheel_blocked = None

                if not redo and (CACHE_PATH / f"{mouse_name}_{date}.json").exists():
                    previous = Cached2pSession.model_validate_json(
                        (CACHE_PATH / f"{mouse_name}_{date}.json").read_text()
                    )
                    # Logic to get the new wheel freeze movement data out of existing sessions
                    if previous.wheel_freeze is None:
                        print(f"Skipping {mouse_name} {date} as already exists")
                        continue
                    elif previous.wheel_freeze.freeze_movement_type is not None:
                        print(f"Skipping {mouse_name} {date} as already exists")
                        continue

                if "learning" not in session_type:
                    print(f"Skipping {mouse_name} {date} {session_type}")
                    continue
                try:
                    wheel_blocked = row["Wheel blocked?"].lower() in {"yes", "true"}
                except KeyError as e:
                    print(f"No column 'Wheel blocked?' found: {e}")
                    print("Wheel blocked set to None")
                    wheel_blocked = None
                if not row["Sync file"]:
                    print(
                        f"Skipping {mouse_name} {date} {session_type} as no sync file"
                    )
                    continue
                session_numbers = parse_session_number(row["Session Number"])
                trials = []
                for session_number in session_numbers:
                    session_path = (
                        BEHAVIOUR_DATA_PATH / mouse_name / row["Date"] / session_number
                    )
                    trials.extend(load_data(session_path))
                logger.info("\n")
                logger.info(f"Processing {mouse_name} {date} {session_type}")
                process_session(
                    trials=trials,
                    tiff_directory=TIFF_UMBRELLA / date / mouse_name,
                    tdms_path=SYNC_FILE_PATH / Path(row["Sync file"]),
                    mouse_name=mouse_name,
                    session_type=session_type,
                    date=date,
                    wheel_blocked=wheel_blocked,
                    row=row,
                )
                logger.info(
                    f"Completed processing for {mouse_name} {date} {session_type}"
                )
            except Exception as e:
                if debug:
                    raise
                tb = traceback.extract_tb(e.__traceback__)
                last_trace = tb[
                    -1
                ]  # Get the last traceback entry (where the exception occurred)
                filename = last_trace.filename
                line_number = last_trace.lineno
                msg = f"Error processing {mouse_name} {date} {session_type} in {filename} on line {line_number}: {e}"
                logger.debug(msg)
                print(msg)
                full_tb = traceback.format_exc()  # Get full traceback as a string
                logger.debug(full_tb)
                print(full_tb)


if __name__ == "__main__":
    main()
