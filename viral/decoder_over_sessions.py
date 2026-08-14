# Allow you to run the file directly, remove if exporting as a proper module
from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import pickle

from viral.gsheets_importer import gsheet2df

from viral.bayesian_decoder import cross_validate_same_day, decode_main
from viral.constants import CACHE_PATH, LOCAL_DFF_PATH, SPREADSHEET_ID
from viral.models import Cached2pSession, TrialInfo
from viral.utils import get_genotype, get_wheel_circumference_from_rig
from viral.imaging_utils import (
    activity_trial_position,
    get_online_position_and_frames,
    load_imaging_data,
    trial_is_imaged,
)


def frames_and_positions(
    trials: list[TrialInfo],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a list of trials. Returns the:
    position of the mouse on a given frame
    frame index of the given position
    lap id of the given position
    """

    all_positions = np.array([])
    all_frame_positions = np.array([])
    lap_id = np.array([])
    for idx, trial in enumerate(trials):
        if not trial_is_imaged(trial):
            continue
        position, frame_position = get_online_position_and_frames(
            trial=trial,
            wheel_circumference=get_wheel_circumference_from_rig("2P"),
            threshold_speed=False,
            speed_threshold=2,
        )
        keep = position < 180
        position = position[keep]
        frame_position = frame_position[keep]
        all_positions = np.concatenate([all_positions, position])
        all_frame_positions = np.concatenate([all_frame_positions, frame_position])
        lap_id = np.concatenate([lap_id, np.full(len(position), idx)])

    return all_positions, all_frame_positions, lap_id


def decode_single_session(
    session: Cached2pSession, spks: np.ndarray, rewarded: bool | None
) -> dict:

    all_positions, all_frame_positions, lap_id = frames_and_positions(session.trials)

    res = decode_main(
        all_frame_positions=all_frame_positions,
        all_positions=all_positions,
        lap_id=lap_id,
        spks=spks,
    )
    return res


def run_all_mice(rewarded: bool | None) -> dict[str, dict]:

    use_local_dff = False
    cache_files = list(CACHE_PATH.glob("*.json"))

    if (HERE / "decoding_results.pkl").exists():
        with open(HERE / "decoding_results.pkl", "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = {}

    for cache_file in cache_files:
        file_parts = cache_file.stem.split("_")
        date = file_parts[1]
        mouse = file_parts[0]

        key = f"{mouse}_{date}"
        if key in all_results:
            print(f"Skipping {mouse} on {date} as it is already processed.")
            continue

        session = Cached2pSession.model_validate_json(cache_file.read_text())

        if (LOCAL_DFF_PATH / f"{mouse}_{date}_dff.npy").exists() and use_local_dff:
            print(f"Loading local dff for {mouse} on {date}.")
            dff = np.load(LOCAL_DFF_PATH / f"{mouse}_{date}_dff.npy")
            spks = np.load(LOCAL_DFF_PATH / f"{mouse}_{date}_spks.npy")
            denoised = np.load(LOCAL_DFF_PATH / f"{mouse}_{date}_denoised.npy")
        else:
            try:
                dff, spks, denoised = load_imaging_data(mouse, date)
                if use_local_dff:
                    np.save(LOCAL_DFF_PATH / f"{mouse}_{date}_dff.npy", dff)
                    np.save(LOCAL_DFF_PATH / f"{mouse}_{date}_spks.npy", spks)
                    np.save(LOCAL_DFF_PATH / f"{mouse}_{date}_denoised.npy", denoised)
            except (FileNotFoundError, AssertionError):
                print(f"Imaging data not found for {mouse} on {date}. Skipping.")
                continue

        try:
            result = decode_single_session(
                session=session, spks=spks, rewarded=rewarded
            )

        except Exception as e:
            print(f"Error occurred while decoding {mouse} on {date}: {e}")
            continue

        all_results[key] = result
        with open(HERE / "decoding_results.pkl", "wb") as f:
            pickle.dump(all_results, f)

    return all_results


def parse_mouse_session_errors(
    mouse_results: dict, mouse: str
) -> tuple[list[int], list[float]]:
    metadata = gsheet2df(SPREADSHEET_ID, mouse, 1)
    dates = sorted([k.split("_")[1] for k in mouse_results.keys()])

    running_trial_count = 0
    n_trials_completed = []
    error = []

    for date in dates:
        session_name = metadata[metadata["Date"] == date].iloc[0]["Type"].lower()
        if not session_name.startswith("learning day"):
            continue
        session = Cached2pSession.model_validate_json(
            (CACHE_PATH / f"{mouse}_{date}.json").read_text()
        )
        running_trial_count += len(session.trials)
        if len(session.trials) < 25:
            print(f"Skipping {mouse} on {date} due to insufficient trials.")
            continue
        result = mouse_results[f"{mouse}_{date}"]["mean_error"]
        n_trials_completed.append(running_trial_count)
        error.append(result)
    return n_trials_completed, error


if __name__ == "__main__":
    # all_results = run_all_mice(rewarded=None)
    with open(HERE / "decoding_results.pkl", "rb") as f:
        all_results = pickle.load(f)

    mice = set([key.split("_")[0] for key in all_results.keys()])

    x = []
    y = []

    for mouse in mice:
        if get_genotype(mouse) != "NLGF":
            continue
        mouse_results = {k: v for k, v in all_results.items() if k.startswith(mouse)}
        n_trials_completed, error = parse_mouse_session_errors(mouse_results, mouse)
        x.extend(n_trials_completed)
        y.extend(error)

    plt.plot(x, y, "o")
    plt.show()
