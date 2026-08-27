from pathlib import Path
import sys
from typing_extensions import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.ndimage import uniform_filter1d
from scipy.stats import median_abs_deviation
from scipy.ndimage import (
    uniform_filter1d,
    binary_propagation,
    binary_closing,
    binary_opening,
)

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
from viral.models import Cached2pSession, TrialInfo, WheelFreeze
from viral.run_oasis import moving_average
from viral.utils import (
    motion_energy_in_chunks,
    subset_frames_mp4,
)


def get_wheel_freeze_movement(
    wheel_freeze: WheelFreeze,
    row: pd.Series,
    mouse_name: str,
    date: str,
    trials_pre_freeze: list[TrialInfo] | None = None,
    trials_post_freeze: list[TrialInfo] | None = None,
) -> tuple[np.ndarray, np.ndarray, Literal["camera", "rotary_encoder", "suite2p"]]:
    """
    We have three options for freeze movement, the camera, the rotary encoder and the suite2p motion correction.
    I have looked at all three, the camera is the best because it captures both paw movement and grooming.
    The rotary encode is decent, does miss grooming but is pretty well correlated with the camera.
    The suite2p motion is ok it is correlated and will probably do but is a much more noisy signal.

    I tried combining metrics but camera alone was better, so the other two are treated as fallbacks
    """

    save_mp4s = False

    # Try the camera
    pre_freeze_path, post_freeze_path = extract_freeze_paths(row, date, mouse_name)
    if pre_freeze_path is not None and post_freeze_path is not None:
        if (
            pre_freeze_path.stat().st_size > 1e6
            and post_freeze_path.stat().st_size > 1e6
        ):
            movement_pre, movement_post = get_wheel_freeze_movement_camera(
                wheel_freeze=wheel_freeze,
                pre_freeze_path=pre_freeze_path,
                post_freeze_path=post_freeze_path,
                mouse_name=mouse_name,
                date=date,
                save_mp4s=save_mp4s,
            )
            return movement_pre, movement_post, "camera"
        else:
            print(
                f"Camera tiff files for {mouse_name} on {date} are too small, due to weird saving issue, falling back to encoder or suite2p"
            )

    # Try the rotary encoder
    # TODO: one might be present and the other not, in which case we should still return the one that is present
    if trials_post_freeze is not None and trials_pre_freeze is not None:
        return (
            get_wheel_freeze_movement_encoder(trials_pre_freeze),
            get_wheel_freeze_movement_encoder(trials_post_freeze),
            "rotary_encoder",
        )

    # Fallback to suite2p
    movement_pre, movement_post = get_wheel_freeze_movement_suite2p(
        mouse_name, date, wheel_freeze
    )
    return movement_pre, movement_post, "suite2p"


def get_wheel_freeze_movement_suite2p(
    mouse_name: str, date: str, wheel_freeze: WheelFreeze
) -> tuple[np.ndarray, np.ndarray]:

    s2p_path = TIFF_UMBRELLA / date / mouse_name / "suite2p" / "plane0"
    ops = np.load(s2p_path / "ops.npy", allow_pickle=True).item()
    # 10x as it just makes everything a bit easier to deal with
    motion_artifact_suite2p = np.abs(np.diff(ops["xoff"])) * 10

    movement, _ = camera_movement(
        motion_artifact_suite2p,
        fs=30,
        hi=1,
        lo=0.5,
        smooth_s=2,
        min_s=1,
        plot=True,
    )

    return (
        movement[
            wheel_freeze.pre_training_start_frame : wheel_freeze.pre_training_end_frame
        ],
        movement[
            wheel_freeze.post_training_start_frame : wheel_freeze.post_training_end_frame
        ],
    )


def get_wheel_freeze_movement_camera(
    wheel_freeze: WheelFreeze,
    pre_freeze_path: Path,
    post_freeze_path: Path,
    mouse_name: str,
    date: str,
    save_mp4s: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Main driver function for processing on the videos"""
    print("Processing wheel freeze movement for", mouse_name, date)

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

    movement_pre, _ = camera_movement(motion_energy_pre, fs=30, plot=False)
    movement_post, _ = camera_movement(motion_energy_post, fs=30, plot=False)
    if save_mp4s:
        for name, original_video_path in zip(
            ["pre", "post"], [pre_freeze_path, post_freeze_path]
        ):
            subset_frames_mp4(
                original_video_path,
                np.where(movement_pre)[0],
                TEMP_CACHE_PATH
                / "motion_energy"
                / "videos"
                / f"{mouse_name}_{date}_{name}_movement.mp4",
            )

            subset_frames_mp4(
                original_video_path,
                np.where(~movement_pre)[0],
                TEMP_CACHE_PATH
                / "motion_energy"
                / "videos"
                / f"{mouse_name}_{date}_{name}_no_movement.mp4",
            )

    return movement_pre, movement_post


def extract_freeze_paths(
    row: pd.Series, date: str, mouse_name: str
) -> tuple[Path | None, Path | None]:

    if "Pupil pre-freeze" not in row or "Pupil post-freeze" not in row:
        print(f"Missing pre or post freeze path for {mouse_name} on {date}")
        return None, None

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

    try:
        pre = load_data(BEHAVIOUR_DATA_PATH / mouse_name / date / session_number_pre)
    except FileNotFoundError as e:
        print(
            f"Missing pre-freeze session data for {mouse_name} on {date}. Error is: {e}",
            "This happens either because the session didn't save or because the file path is wrong. Check the path",
        )
        pre = None
    try:
        post = load_data(BEHAVIOUR_DATA_PATH / mouse_name / date / session_number_post)
    except FileNotFoundError as e:
        print(
            f"Missing post-freeze session data for {mouse_name} on {date}. Error is: {e}",
            "This happens either because the session didn't save or because the file path is wrong. Check the path",
        )
        post = None
    return pre, post


def encoder_to_frames(
    all_positions: np.ndarray, frame_positions: np.ndarray
) -> np.ndarray:
    """Upsample a slow/irregular rotary-encoder trace onto the camera clock.

    all_positions   : encoder position at each encoder sample (counts, radians,
                      metres -- anything cumulative and monotonic-ish).
    frame_positions : index of the camera frame matching each encoder sample.
                      Duplicates are allowed; the last sample of a frame wins.

    Returns position per camera frame, length n_frames.
    """
    pos = np.asarray(all_positions, float)
    frame = np.asarray(frame_positions)
    if pos.shape != frame.shape or pos.ndim != 1:
        raise ValueError(
            "all_positions and frame_positions must be 1-D and equal length"
        )

    # Sort by frame, keeping original order within a frame,
    # then take the last sample of each frame
    order = np.lexsort((np.arange(frame.size), frame))
    frame, pos = frame[order], pos[order]
    keep = np.r_[np.diff(frame) > 0, True]
    frame, pos = frame[keep], pos[keep]
    return np.interp(
        np.arange(np.min(frame_positions), np.max(frame_positions)), frame, pos
    )


def test_movement_extraction() -> None:
    """
    Test function to examine different freeze movement methods.
    Ideally use a mouse with camera encoder and suite2p,
    save the results of all of them and compare them to each other
    """
    mouse_name = "J034"
    date = "2026-06-10"

    session_path = CACHE_PATH / f"{mouse_name}_{date}.json"
    session = Cached2pSession.model_validate_json(session_path.read_text())
    metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
    row = metadata[metadata["Date"] == date].iloc[0]

    pre_freeze_path, post_freeze_path = extract_freeze_paths(row, date, mouse_name)

    save_pre = lambda movement_pre, name, is_moving: subset_frames_mp4(
        pre_freeze_path,
        np.where(movement_pre)[0] if is_moving else np.where(~movement_pre)[0],
        Path("/Users/jamesrowland/Code/viral/freeze_videos")
        / f"{mouse_name}_{date}_{name}_pre_{'moving' if is_moving else 'non-moving'}.mp4",
    )

    save_post = lambda movement_post, name, is_moving: subset_frames_mp4(
        post_freeze_path,
        np.where(movement_post)[0] if is_moving else np.where(~movement_post)[0],
        Path("/Users/jamesrowland/Code/viral/freeze_videos")
        / f"{mouse_name}_{date}_{name}_post_{'moving' if is_moving else 'non-moving'}.mp4",
    )

    movement_pre_camera, movement_post_camera = get_wheel_freeze_movement_camera(
        wheel_freeze=session.wheel_freeze,
        pre_freeze_path=pre_freeze_path,
        post_freeze_path=post_freeze_path,
        mouse_name=mouse_name,
        date=date,
        save_mp4s=False,
    )

    # Make sure the movement comes from the camera
    assert (movement_pre_camera == session.wheel_freeze.movement_pre_freeze).all()
    assert (movement_post_camera == session.wheel_freeze.movement_post_freeze).all()

    movement_pre_encoder = get_wheel_freeze_movement_encoder(
        session.wheel_freeze.trials_pre_freeze
    )

    movement_post_encoder = get_wheel_freeze_movement_encoder(
        session.wheel_freeze.trials_post_freeze
    )

    movement_pre_suite2p, movement_post_suite2p = get_wheel_freeze_movement_suite2p(
        mouse_name, date, session.wheel_freeze
    )

    for is_moving in [True, False]:
        save_pre(movement_pre_camera, "camera", is_moving=is_moving)
        save_post(movement_post_camera, "camera", is_moving=is_moving)

        save_pre(movement_pre_encoder, "encoder", is_moving=is_moving)
        save_post(movement_post_encoder, "encoder", is_moving=is_moving)

        save_pre(movement_pre_suite2p, "suite2p", is_moving=is_moving)
        save_post(movement_post_suite2p, "suite2p", is_moving=is_moving)


def get_wheel_freeze_movement_encoder(
    trials: list[TrialInfo],
) -> np.ndarray:
    all_positions = np.array([], dtype="float")
    previous_rotary_end = 0
    frame_positions = np.array([], dtype="float")

    for trial in trials:
        trial_position = np.array(trial.rotary_encoder_position)
        closest_frame = np.array(
            [
                state.closest_frame_start
                for state in trial.states_info
                if state.name == "store_encoder_position"
            ]
        )

        assert trial_position.shape == closest_frame.shape
        # Not imaged
        if np.min(closest_frame) == np.max(closest_frame):
            continue

        all_positions = np.concatenate(
            (all_positions, trial_position + previous_rotary_end)
        )
        frame_positions = np.concatenate((frame_positions, closest_frame))
        previous_rotary_end += trial_position[-1]

    encoder = encoder_to_frames(all_positions, frame_positions)

    return np.gradient(encoder, 1 / 30) != 0


def camera_movement(
    motion_energy: np.ndarray,
    fs: float,
    smooth_s: float = 0.25,
    hi: float = 2.0,
    lo: float = 1.0,
    min_s: float = 0.2,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify movement from video motion energy.

    Four steps: remove slow drift, smooth, rescale to the resting noise floor,
    threshold with hysteresis.

    motion_energy : 1-D array, one value per frame.
    fs            : frame rate (Hz).
    drift_s       : window for the rolling baseline that removes lighting and
                    settling drift. THE parameter that matters -- see below.
    smooth_s      : boxcar width, seconds.
    hi, lo        : threshold in MADs above the resting baseline. A sample is
                    movement if the trace exceeds `hi`, and neighbouring samples
                    stay movement while it remains above `lo`.
    min_s         : shortest allowed bout, and shortest allowed gap.

    Returns (moving, z): a boolean mask and the normalised trace it was
    thresholded from, so you can plot the trace and re-threshold without
    recomputing.

    """
    x = np.abs(np.asarray(motion_energy, dtype=np.float64))
    if x.ndim != 1:
        raise ValueError("motion_energy must be 1-D")

    x = uniform_filter1d(x, max(1, int(smooth_s * fs)))

    # Scale by the MAD, which the resting state dominates. Guard against a
    # degenerate estimate: smoothing leaves floating-point dust, so a "zero" MAD
    # comes back as ~1e-15 and would blow the trace up by 15 orders of magnitude.
    ref = float(np.percentile(x, 99))
    s = float(median_abs_deviation(x, scale="normal"))
    if s <= 1e-6 * ref:
        s = ref if ref > 0 else 1.0
    z = (x - np.median(x)) / s

    k = np.ones(max(1, int(min_s * fs)), dtype=bool)
    seeded = binary_propagation(z > hi, mask=z > lo)  # grow from hi down to lo
    moving = binary_opening(binary_closing(seeded, k), k)  # drop short gaps/bouts

    if plot:
        t = np.arange(z.size)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(
            t,
            0,
            1,
            where=moving.tolist(),
            transform=ax.get_xaxis_transform(),
            color="0.85",
            lw=0,
            zorder=0,
            label="movement",
        )
        ax.plot(t, z, lw=0.6, color="tab:orange")
        ax.axhline(hi, color="tab:red", ls=":", lw=0.8, label=f"hi={hi}")
        ax.axhline(lo, color="tab:red", ls="--", lw=0.5, label=f"lo={lo}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("motion energy (MADs above rest)")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

    return moving, z


if __name__ == "__main__":
    test_movement_extraction()
