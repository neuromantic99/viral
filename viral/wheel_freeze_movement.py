from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

from viral.constants import CACHE_PATH, TEMP_CACHE_PATH, TIFF_UMBRELLA
from viral.models import SessionImagingInfo, WheelFreeze
from viral.run_oasis import moving_average
from viral.utils import detect_events, motion_energy_in_chunks, subset_frames_mp4


def process_motion_energy(motion_energy: np.ndarray) -> np.ndarray:
    zscored = zscore(motion_energy)
    smoothed = gaussian_filter1d(zscored, sigma=3)
    events = detect_events(smoothed, lower_threshold=0.5, upper_threshold=2)
    return np.array(events)


def merge_close_events(events: np.ndarray, min_distance: int) -> np.ndarray:
    merged_events = [events[0]]
    for event in events[1:]:
        if event[0] - merged_events[-1][1] < min_distance:
            merged_events[-1][1] = max(merged_events[-1][1], event[1])
        else:
            merged_events.append(event)
    return np.array(merged_events)


def get_wheel_freeze_movement(
    session_sync: SessionImagingInfo,
    wheel_freeze: WheelFreeze,
    row: pd.Series,
    mouse_name: str,
    date: str,
) -> None:
    """Main driver function for processing on the videos"""
    # This gets the motion energy and saves it. Good as it takes a long time and probably won't need changing
    motion_energy_pre, motion_energy_post = get_motion_energy_video(
        wheel_freeze=wheel_freeze,
        row=row,
        mouse_name=mouse_name,
        date=date,
    )

    # Not sure this will work for all mice. May need adjusting
    event_boundaries_pre = merge_close_events(
        process_motion_energy(motion_energy_pre), 30
    )

    event_boundaries_post = merge_close_events(
        process_motion_energy(motion_energy_post), 30
    )

    pre_freeze_path, post_freeze_path = extract_freeze_paths(row, date, mouse_name)

    for name, event_boundaries, video_path in zip(
        ["pre", "post"],
        [event_boundaries_pre, event_boundaries_post],
        [pre_freeze_path, post_freeze_path],
    ):
        frames_in_events = []
        frames_outside_events = set(range(len(motion_energy_pre)))
        for start, end in event_boundaries:
            frames_in_events.extend(list(range(start, end + 1)))
            frames_outside_events -= set(range(start, end + 1))
        frames_outside_events = list(frames_outside_events)

        np.save(
            CACHE_PATH.parent
            / "wheel_freeze_motion_events"
            / f"{mouse_name}_{date}_{name}_in_events.npy",
            frames_in_events,
        )

        np.save(
            CACHE_PATH.parent
            / "wheel_freeze_motion_events"
            / f"{mouse_name}_{date}_{name}_outside_events.npy",
            frames_outside_events,
        )

        subset_frames_mp4(
            video_path,
            frames_in_events,
            Path(
                TEMP_CACHE_PATH
                / "motion_energy"
                / f"{mouse_name}_{date}_{name}_in_events.mp4"
            ),
        )

        subset_frames_mp4(
            video_path,
            frames_outside_events,
            Path(
                TEMP_CACHE_PATH
                / "motion_energy"
                / f"{mouse_name}_{date}_{name}_outside_events.mp4"
            ),
        )


def extract_freeze_paths(
    row: pd.Series, date: str, mouse_name: str
) -> tuple[Path, Path]:
    pre_freeze_path = (
        TIFF_UMBRELLA / date / mouse_name / row["Pupil pre-freeze"].values[0]
    )
    assert pre_freeze_path.exists(), f"Pre-freeze path {pre_freeze_path} does not exist"

    post_freeze_path = (
        TIFF_UMBRELLA / date / mouse_name / row["Pupil post-freeze"].values[0]
    )
    assert (
        post_freeze_path.exists()
    ), f"Post-freeze path {post_freeze_path} does not exist"

    return pre_freeze_path, post_freeze_path


def get_motion_energy_video(
    wheel_freeze: WheelFreeze,
    row: pd.Series,
    mouse_name: str,
    date: str,
) -> tuple[np.ndarray, np.ndarray]:
    use_cache = True
    if use_cache:
        print("Using cached motion energy")
        return np.load(
            TEMP_CACHE_PATH / "motion_energy" / f"{mouse_name}_{date}_pre.npy"
        ), np.load(TEMP_CACHE_PATH / "motion_energy" / f"{mouse_name}_{date}_post.npy")

    pre_freeze_path, post_freeze_path = extract_freeze_paths(row, date, mouse_name)
    chunk_size = 500
    motion_energy_pre = motion_energy_in_chunks(pre_freeze_path, chunk_size=chunk_size)
    motion_energy_post = motion_energy_in_chunks(
        post_freeze_path, chunk_size=chunk_size
    )

    # This is pure happy path, this will fail with the slightest error in the recording
    # but outlines what Should happen. The motion energy is two frames shorter than the number
    # of frames because it's diffed (1 frame less) and the trigger is on the low of the frame clock (so you miss the first one)
    assert (
        len(motion_energy_pre)
        == len(motion_energy_post)
        == 26998
        == wheel_freeze.pre_training_end_frame
        - wheel_freeze.pre_training_start_frame
        - 2
        == wheel_freeze.post_training_end_frame
        - wheel_freeze.post_training_start_frame
        - 2
    )
    np.save(
        TEMP_CACHE_PATH / "motion_energy" / f"{mouse_name}_{date}_pre.npy",
        motion_energy_pre,
    )
    np.save(
        TEMP_CACHE_PATH / "motion_energy" / f"{mouse_name}_{date}_post.npy",
        motion_energy_post,
    )

    return motion_energy_pre, motion_energy_post


def compare_to_suite2p_motion(motion_energy_pre: np.ndarray) -> None:
    s2p_path = Path("/Volumes/MarcBusche/Josef/2P/2026-05-13/J030/suite2p/plane0")
    stat = np.load(s2p_path / "stat.npy", allow_pickle=True)
    ops = np.load(s2p_path / "ops.npy", allow_pickle=True).item()
    motion_artifact = np.abs(np.diff(ops["xoff"]))
    motion_artifact = ops["corrXY"]

    motion_artifact_smoothed = moving_average(motion_artifact, window=10)
    motion_energy_pre = moving_average(motion_energy_pre, window=10)

    fig, ax1 = plt.subplots()

    # ax1.plot(motion_artifact_smoothed[: len(motion_energy_pre)])
    ax1.plot(motion_artifact[: len(motion_energy_pre)], color="blue", alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(motion_energy_pre, "--", color="red")
    ax1.set_ylabel("Motion Artifact (suite2p)", color="blue")
    ax2.set_ylabel("Motion Energy (video)", color="red")
