from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))


from viral.multiple_sessions import parse_session_number
from viral.single_session import load_data
from viral.gsheets_importer import gsheet2df
from viral.constants import (
    BEHAVIOUR_DATA_PATH,
    CACHE_PATH,
    SPREADSHEET_ID,
    TEMP_CACHE_PATH,
    TIFF_UMBRELLA,
)
from viral.models import Cached2pSession, SessionImagingInfo, TrialInfo, WheelFreeze
from viral.run_oasis import moving_average
from viral.utils import detect_events, motion_energy_in_chunks, subset_frames_mp4


def process_motion_energy(motion_energy: np.ndarray, plot: bool = False) -> np.ndarray:
    zscored = zscore(motion_energy)
    smoothed = gaussian_filter1d(zscored, sigma=10)
    events = np.array(detect_events(smoothed, lower_threshold=0.0, upper_threshold=0.5))
    events = merge_close_events(events, min_distance=30)

    if plot:
        plt.plot(smoothed)
        for start, end in events:
            plt.axvspan(start, end, color="red", alpha=0.3)
    return events


def merge_close_events(events: np.ndarray, min_distance: int) -> np.ndarray:
    merged_events = [events[0]]
    did_merge = False
    for event in events[1:]:
        if event[0] - merged_events[-1][1] < min_distance:
            merged_events[-1][1] = max(merged_events[-1][1], event[1])
            did_merge = True
        else:
            merged_events.append(event)
    if did_merge:
        return merge_close_events(np.array(merged_events), min_distance)
    return np.array(merged_events)


def get_wheel_freeze_movement_camera(
    wheel_freeze: WheelFreeze,
    row: pd.Series,
    mouse_name: str,
    date: str,
    save_mp4s: bool = False,
) -> None:
    """Main driver function for processing on the videos"""
    print("Processing wheel freeze movement for", mouse_name, date)
    # This gets the motion energy and saves it. Good as it takes a long time and probably won't need changing
    pre_freeze_path, post_freeze_path = extract_freeze_paths(row, date, mouse_name)
    if pre_freeze_path is None or post_freeze_path is None:
        # TODO: I think there are rare occurances where one exists but one doesn't.
        print(f"Missing pre or post freeze path for {mouse_name} on {date}")
        return

    motion_energy_pre, motion_energy_post = get_motion_energy_video(
        wheel_freeze=wheel_freeze,
        pre_freeze_path=pre_freeze_path,
        post_freeze_path=post_freeze_path,
        mouse_name=mouse_name,
        date=date,
    )

    if mouse_name == "J034" and date == "2026-06-11":
        # Screen flashes white at the start for some reason
        motion_energy_pre = motion_energy_pre[1:]

    # Not sure this will work for all mice. May need adjusting
    event_boundaries_pre = process_motion_energy(motion_energy_pre)
    event_boundaries_post = process_motion_energy(motion_energy_post)

    for name, event_boundaries, video_path in zip(
        ["pre", "post"],
        [event_boundaries_pre, event_boundaries_post],
        [pre_freeze_path, post_freeze_path],
    ):
        frames_in_events = []
        frames_outside_events = set(
            range(len(motion_energy_pre if name == "pre" else len(motion_energy_post)))
        )
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
            list(frames_outside_events),
        )

        if save_mp4s:
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
    print(f"Finished processing wheel freeze movement for {mouse_name} on {date}")


def extract_freeze_paths(
    row: pd.Series, date: str, mouse_name: str
) -> tuple[Path | None, Path | None]:
    pre_freeze_path = TIFF_UMBRELLA / date / mouse_name / row["Pupil pre-freeze"]
    if not pre_freeze_path.exists() or not row["Pupil pre-freeze"]:
        pre_freeze_path = None

    post_freeze_path = TIFF_UMBRELLA / date / mouse_name / row["Pupil post-freeze"]

    if not post_freeze_path.exists() or not row["Pupil post-freeze"]:
        post_freeze_path = None

    return pre_freeze_path, post_freeze_path


def get_motion_energy_video(
    wheel_freeze: WheelFreeze,
    pre_freeze_path: Path,
    post_freeze_path: Path,
    mouse_name: str,
    date: str,
) -> tuple[np.ndarray, np.ndarray]:
    use_cache = True
    pre_cache_path = TEMP_CACHE_PATH / "motion_energy" / f"{mouse_name}_{date}_pre.npy"
    post_cache_path = (
        TEMP_CACHE_PATH / "motion_energy" / f"{mouse_name}_{date}_post.npy"
    )

    if use_cache and pre_cache_path.exists() and post_cache_path.exists():
        print("Using cached motion energy")
        return np.load(pre_cache_path), np.load(post_cache_path)

    chunk_size = 500
    motion_energy_pre = motion_energy_in_chunks(pre_freeze_path, chunk_size=chunk_size)
    motion_energy_post = motion_energy_in_chunks(
        post_freeze_path, chunk_size=chunk_size
    )

    # 27000 is the number of frames we image for (15 mins)
    # The number of frames is usually - though not always - 1 less than this,
    # I think because the trigger is on the low of the frame clock, so you miss the first one.
    # The motion energy is probably 26998 as it results from a diff.
    acceptable = {26998, 26999, 27000}
    assert (
        len(motion_energy_pre) in acceptable
        and len(motion_energy_post) in acceptable
        and (
            wheel_freeze.pre_training_end_frame - wheel_freeze.pre_training_start_frame
        )
        in acceptable
        and (
            wheel_freeze.post_training_end_frame
            - wheel_freeze.post_training_start_frame
            - 2
        )
        in acceptable
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


def load_freeze_trials(
    mouse_name: str, date: str, row: pd.Series
) -> tuple[list[TrialInfo] | None, list[TrialInfo] | None]:

    try:
        session_number_pre = parse_session_number(row["Session Number pre-freeze"])[0]
        session_number_post = parse_session_number(row["Session Number post-freeze"])[0]
    except KeyError as e:
        print(
            f"Missing freeze session numbers for {mouse_name} on {date}. Error is: {e}"
        )
        return None, None

    return load_data(
        BEHAVIOUR_DATA_PATH / mouse_name / date / session_number_pre
    ), load_data(BEHAVIOUR_DATA_PATH / mouse_name / date / session_number_post)


# To use when computing freeze wheel speeds
# all_positions = np.array([], dtype="float")
# previous_rotary_end = 0
# for trial in trials:
#     position = np.array(trial.rotary_encoder_position)
#     all_positions = np.concatenate((all_positions, position + previous_rotary_end))
#     previous_rotary_end += position[-1]
def plot_motion_energy_results(mouse_name: str, date: str) -> None:

    motion_energy_pre = np.load(
        TEMP_CACHE_PATH / "motion_energy" / f"{mouse_name}_{date}_pre.npy"
    )
    motion_energy_pre = motion_energy_pre[1:]
    events_pre = process_motion_energy(motion_energy_pre, plot=True)

    # motion_energy_post = np.load(
    #     TEMP_CACHE_PATH / "motion_energy" / f"{mouse_name}_{date}_post.npy"
    # )
    # events_post = process_motion_energy(motion_energy_post, plot=True)


def mp4_saver() -> None:
    """Utilty to save motion energy videos without going through cache2psessions"""
    mouse_name = "J034"
    date = "2026-06-11"

    session_path = CACHE_PATH / f"{mouse_name}_{date}.json"
    session = Cached2pSession.model_validate_json(session_path.read_text())

    metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
    row = metadata[metadata["Date"] == date].iloc[0]

    get_wheel_freeze_movement_camera(
        wheel_freeze=session.wheel_freeze,
        row=row,
        mouse_name=mouse_name,
        date=date,
    )


if __name__ == "__main__":
    pass
