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
from viral.multiple_sessions import parse_session_number
from viral.constants import (
    BEHAVIOUR_DATA_PATH,
    CACHE_PATH,
    LOCAL_DFF_PATH,
    SPREADSHEET_ID,
)
from viral.models import Cached2pSession, TrialInfo
from viral.single_session import load_data

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
    trials = [
        trial
        for trial in session.trials
        if rewarded is None or trial.texture_rewarded == rewarded
    ]

    all_positions, all_frame_positions, lap_id = frames_and_positions(trials)

    res = decode_main(
        all_frame_positions=all_frame_positions,
        all_positions=all_positions,
        lap_id=lap_id,
        spks=spks,
    )
    return res


def run_all_mice(rewarded: bool | None) -> dict[str, dict]:

    use_local_dff = True
    cache_files = list(CACHE_PATH.glob("*.json"))

    results_file = HERE / f"decoding_results_rewarded_{rewarded}.pkl"

    if results_file.exists():
        with open(results_file, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = {}

    for cache_file in cache_files:
        file_parts = cache_file.stem.split("_")
        date = file_parts[1]
        mouse = file_parts[0]

        if get_genotype(mouse) not in {"WT", "NLGF"}:
            continue

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
        with open(results_file, "wb") as f:
            pickle.dump(all_results, f)

    return all_results


def parse_mouse_session_errors(
    mouse_results: dict, mouse: str
) -> tuple[list[int], list[float], list[int]]:
    metadata = gsheet2df(SPREADSHEET_ID, mouse, 1)

    n_trials_in_session = []
    error = []
    decoded_session_idxs = []

    learning_day_idx = 0

    for _, row in metadata.iterrows():
        date = row["Date"]
        session_name = row["Type"].lower()
        if not session_name.startswith("learning day"):
            continue
        session_number = parse_session_number(row["Session Number"])[0]
        session_path = BEHAVIOUR_DATA_PATH / mouse / row["Date"] / session_number
        n_trials_in_session.append(len(load_data(session_path)))

        try:
            session = Cached2pSession.model_validate_json(
                (CACHE_PATH / f"{mouse}_{date}.json").read_text()
            )
        except FileNotFoundError:
            print(f"Cache file not found for {mouse} on {date}. Skipping.")
            continue
        if len(session.trials) < 25:
            print(f"Skipping {mouse} on {date} due to insufficient trials.")
            continue
        try:
            result = mouse_results[f"{mouse}_{date}"]["mean_error"]
        except KeyError:
            print(f"Decoder result not found for {mouse} on {date}. Skipping.")
            continue
        error.append(result)
        decoded_session_idxs.append(learning_day_idx)
        learning_day_idx += 1

    return n_trials_in_session, error, decoded_session_idxs


def parse_decoder_results(rewarded: bool) -> None:
    with open(HERE / f"decoding_results_rewarded_{rewarded}.pkl", "rb") as f:
        all_results = pickle.load(f)

    mice = set([key.split("_")[0] for key in all_results.keys()])

    save_file = HERE / f"decoder_parsed_rewarded_{rewarded}.pkl"
    if (save_file).exists():
        with open(save_file, "rb") as f:
            decoder_parsed = pickle.load(f)
    else:
        decoder_parsed = {}

    for mouse in mice:
        mouse_results = {k: v for k, v in all_results.items() if k.startswith(mouse)}
        n_trials_in_session, error, decoded_session_idxs = parse_mouse_session_errors(
            mouse_results, mouse
        )
        decoder_parsed[mouse] = {
            "n_trials_in_session": n_trials_in_session,
            "error": error,
            "decoded_session_idxs": decoded_session_idxs,
        }
        with open(save_file, "wb") as f:
            pickle.dump(decoder_parsed, f)


def fit_decoded(decoded_parsed: dict[str, dict]) -> None:

    plt.figure()
    for genotype in {"WT", "NLGF"}:

        for mouse, result in decoded_parsed.items():
            if get_genotype(mouse) != genotype:
                continue
            cum = np.cumsum(result["n_trials_in_session"])
            x = np.array(cum)[result["decoded_session_idxs"]]

            # x = np.array(result["decoded_session_idxs"])
            y = np.array(result["error"])
            plt.scatter(
                x, y, label=mouse, color="blue" if genotype == "WT" else "orange"
            )

    plt.ylim(0, 60)


if __name__ == "__main__":
    # rewarded = False
    # for rewarded in [True, False]:
    #     # all_results = run_all_mice(rewarded=rewarded)
    # parse_decoder_results(rewarded=rewarded)

    for rewarded in [True, False]:
        with open(HERE / f"decoder_parsed_rewarded_{rewarded}.pkl", "rb") as f:
            decoder_parsed = pickle.load(f)

        fit_decoded(decoder_parsed)
        plt.title(f"rewarded={rewarded}")
    plt.show()
