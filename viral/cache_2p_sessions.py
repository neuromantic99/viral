from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys
import time
import traceback
from typing import List, Tuple
import matplotlib.pyplot as plt

from nptdms import TdmsFile
from ScanImageTiffReader import ScanImageTiffReader
import numpy as np
import pandas as pd

# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


from viral.constants import (
    BEHAVIOUR_DATA_PATH,
    SPREADSHEET_ID,
    SYNC_FILE_PATH,
    TIFF_UMBRELLA,
)
from viral.gsheets_importer import gsheet2df
from viral.models import Cached2pSession, TrialInfo
from viral.multiple_sessions import parse_session_number
from viral.single_session import HERE, load_data
from viral.utils import (
    extract_TTL_chunks,
    find_chunk,
    get_sampling_rate,
    get_tiff_paths_in_directory,
    time_list_to_datetime,
    trial_is_imaged,
    degrees_to_cm,
)

import logging

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
) -> None:
    trial_spacer_daq_times = behaviour_times[
        np.sum(behaviour_chunk_lens[:trial_idx]) : np.sum(
            behaviour_chunk_lens[: trial_idx + 1]
        )
    ]
    # Sanity check the above logic
    assert len(trial_spacer_daq_times) == count_spacers(trial)

    # trial.valid_frame_times = valid_frame_times

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
    for state in trial.states_info:
        state.start_time_daq = bpod_to_daq(state.start_time)
        state.end_time_daq = bpod_to_daq(state.end_time)
        state.closest_frame_start = int(
            np.argmin(np.abs(valid_frame_times - state.start_time_daq))
        )
        state.closest_frame_end = int(
            np.argmin(np.abs(valid_frame_times - state.end_time_daq))
        )

        if state.name == "spacer_high_00":
            trial.trial_start_closest_frame = state.closest_frame_start

    last_state = trial.states_info[-1]
    trial.trial_end_closest_frame = last_state.closest_frame_end

    for event in trial.events_info:
        event.start_time_daq = float(bpod_to_daq(event.start_time))
        event.closest_frame = int(
            np.argmin(np.abs(valid_frame_times - event.start_time_daq))
        )


def add_imaging_info_to_trials(
    tdms_path: Path, tiff_directory: Path, trials: List[TrialInfo]
) -> List[TrialInfo]:

    logger.info("Adding imaging info to trials")
    t1 = time.time()

    tiff_paths = sorted(get_tiff_paths_in_directory(tiff_directory))

    stack_lengths_tiffs, epochs, all_tiff_timestamps = get_tiff_metadata(
        tiff_paths=tiff_paths
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

    print(f"Length of frame clock: {len(frame_clock)}")
    print(f"Length of behaviour clock: {len(behaviour_clock)}")
    print(f"Min behaviour clock: {np.min(behaviour_clock)}")
    print(f"Max behaviour clock: {np.max(behaviour_clock)}")

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

    frame_times_daq, chunk_lengths_daq = extract_TTL_chunks(frame_clock, sampling_rate)

    sanity_check_imaging_frames(frame_times_daq, sampling_rate, frame_clock)

    # Consistently, the number of triggers recorded is two more than the number of frames recorded.
    # This only occurs when the imaging is manually stopped before a grab is complete (confirmed by counting triggers
    # from a completed grab).
    # The reason for first extra frame is obvious (we stop the imaging mid-way through a frame so it is not saved).
    # The second happens for unclear reasons but must be at the end as there are no extra frames in the middle and the first
    # frame is relaibly correct
    # Possible we may see a recording with one extra frame if the imaging is stopped on flyback. The error below will catch this.
    assert np.all(
        chunk_lengths_daq - stack_lengths_tiffs == 2
    ), f"Chunk lengths do not match stack lengths. Chunk lengths: {chunk_lengths_daq}. Stack lengths: {stack_lengths_tiffs}. This will occcur especially on crashed recordings, think about a fix"

    # Remove the final two frames from valid frames
    valid_frame_times = np.array([])
    offset = 0
    for stack_len_tiff, chunk_len_daq in zip(
        stack_lengths_tiffs, chunk_lengths_daq, strict=True
    ):
        valid_frame_times = np.append(
            valid_frame_times, frame_times_daq[offset : offset + stack_len_tiff]
        )
        offset += chunk_len_daq

    assert (
        len(valid_frame_times)
        == sum(stack_lengths_tiffs)
        == len(frame_times_daq) - 2 * len(stack_lengths_tiffs)
    )

    for idx, trial in enumerate(trials):
        # Works in place, maybe not ideal
        add_daq_times_to_trial(
            trial,
            idx,
            valid_frame_times,
            behaviour_times,
            behaviour_chunk_lens,
            sampling_rate,
        )

    for trial in trials:
        check_timestamps(
            epochs=epochs,
            trial=trial,
            all_tiff_timestamps=all_tiff_timestamps,
            chunk_lens=chunk_lengths_daq,
            valid_frame_times=valid_frame_times,
            sampling_rate=sampling_rate,
            daq_start_time=daq_start_time,
        )

    return trials


def get_tiff_metadata(
    tiff_paths: List[Path], use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    mouse_name = tiff_paths[0].parent.name
    date = tiff_paths[0].parent.parent.name

    # For debugging, remove eventually
    if use_cache:
        temp_cache_path = Path("/home/josef/code/viral/data/temp_caches")
        if (temp_cache_path / f"{mouse_name}_{date}_stack_lengths.npy").exists():
            print("Using cached tiff metadata")
            stack_lengths = np.load(
                temp_cache_path / f"{mouse_name}_{date}_stack_lengths.npy"
            )
            epochs = np.load(temp_cache_path / f"{mouse_name}_{date}_epochs.npy")
            all_tiff_timestamps = np.load(
                temp_cache_path / f"{mouse_name}_{date}_all_tiff_timestamps.npy"
            )
            return stack_lengths, epochs, all_tiff_timestamps

    print("Could not find cached tiff metadata. Reading tiffs (takes a long time)")
    tiffs = [ScanImageTiffReader(str(tiff)) for tiff in tiff_paths]
    stack_lengths = [tiff.shape()[0] for tiff in tiffs]
    epochs = []
    all_tiff_timestamps = []
    for tiff in tiffs:
        tiff_timestamps = [
            float(
                re.search(
                    r"frameTimestamps_sec\s*=\s*(-?\d+\.\d+)",
                    tiff.description(idx),
                )[1]
            )
            for idx in range(tiff.shape()[0])
        ]
        all_tiff_timestamps.extend(tiff_timestamps)

        # Epoch is the same for all frames
        epochs.append(
            list(
                map(
                    float,
                    re.search(r"epoch\s*=\s*\[([^\]]+)\]", tiff.description(0))[
                        1
                    ].split(),
                )
            )
        )

        diffed = np.diff(tiff_timestamps)

        # Check no dropped frames in the middle
        assert (
            round(np.max(diffed), 3) == round(np.min(diffed), 3) == 0.033
        ), f"Dropped frames in the middle based on tiff timestamps. Min diffed = {np.min(diffed)}, max diffed = {np.max(diffed)}"

    if use_cache:
        for variable, name in zip(
            [stack_lengths, all_tiff_timestamps, epochs],
            ["stack_lengths", "all_tiff_timestamps", "epochs"],
        ):
            np.save(
                temp_cache_path / f"{mouse_name}_{date}_{name}.npy",
                variable,
            )

    return stack_lengths, epochs, all_tiff_timestamps


def check_timestamps(
    epochs: List[List[float]],
    trial: TrialInfo,
    all_tiff_timestamps: np.ndarray,
    chunk_lens: np.ndarray,
    valid_frame_times: np.ndarray,
    sampling_rate: int,
    daq_start_time: datetime,
) -> None:
    """Compares the timestamps in the tiff to the timestamps in the Daq (the time of the trigger, offset to the timestamp that the daq started)
    Currently works trial by trial which isn't really necessary.
    There is a bit of a drift here, frames at the start have a closer match in times to frames at the end. This is probably because the sampling rate
    is not exactly 10,000. But if they are off by less than 10ms that's fine.
    This will not process ITI frames in the original version of task.py but will after we added the position storage
    """

    if not trial_is_imaged(trial):
        return

    assert len(all_tiff_timestamps) == len(valid_frame_times)

    trial_start_state = [
        state for state in trial.states_info if "spacer_high" in state.name
    ][0]

    trial_end_state = trial.states_info[-1]

    first_frame_trial = trial_start_state.closest_frame_start
    last_frame_trial = trial_end_state.closest_frame_start

    epoch_trial = find_chunk(chunk_lens, first_frame_trial)
    chunk_start = time_list_to_datetime(epochs[epoch_trial])

    for frame in range(first_frame_trial, last_frame_trial):

        # The time in the tiff. Not sure if this is the end or the start of the tiff
        frame_datetime = chunk_start + timedelta(seconds=all_tiff_timestamps[frame])
        frame_daq_time = daq_start_time + timedelta(
            seconds=valid_frame_times[frame] / sampling_rate
        )

        offset = (frame_datetime - frame_daq_time).total_seconds()
        assert abs(offset) < 0.01, "Tiff timestamp does not match daq timestamp"


def process_session(
    trials: List[TrialInfo],
    tiff_directory: Path,
    tdms_path: Path,
    mouse_name: str,
    date: str,
    session_type: str,
) -> None:

    print(f"Off we go for {mouse_name} {date} {session_type}")
    trials = add_imaging_info_to_trials(
        tdms_path,
        tiff_directory,
        trials,
    )

    with open(
        HERE.parent / "data" / "cached_2p" / f"{mouse_name}_{date}.json", "w"
    ) as f:
        json.dump(
            Cached2pSession(
                mouse_name=mouse_name,
                date=date,
                trials=trials,
                session_type=session_type,
            ).model_dump(),
            f,
        )

    print(f"Done for {mouse_name} {date} {session_type}")


if __name__ == "__main__":

    # for mouse_name in ["JB017", "JB019", "JB020", "JB021", "JB022", "JB023"]:
    redo = True
    for mouse_name in ["JB020"]:
        metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
        for _, row in metadata.iterrows():
            try:
                print(f"the type is {row['Type']}")

                date = row["Date"]
                session_type = row["Type"].lower()

                if (
                    not redo
                    and (
                        HERE.parent / "data" / "cached_2p" / f"{mouse_name}_{date}.json"
                    ).exists()
                ):
                    print(f"Skipping {mouse_name} {date} as already exists")
                    continue

                if (
                    "learning day" not in session_type
                    and "reversal learning" not in session_type
                ):
                    print(f"Skipping {mouse_name} {date} {session_type}")
                    continue

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

                logger.info(f"\n")
                logger.info(f"Processing {mouse_name} {date} {session_type}")
                process_session(
                    trials=trials,
                    tiff_directory=TIFF_UMBRELLA / date / mouse_name,
                    tdms_path=SYNC_FILE_PATH / Path(row["Sync file"]),
                    mouse_name=mouse_name,
                    session_type=session_type,
                    date=date,
                )
                logger.info(
                    f"Completed processing for {mouse_name} {date} {session_type}"
                )

            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                line_number = tb[-1].lineno  # Get the line number of the exception
                msg = f"Error processing {mouse_name} {date} {session_type} on line {line_number}: {e}"
                logger.debug(msg)
                print(msg)
