from pathlib import Path
import sys
from typing import Dict

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pydantic import ValidationError

from ensemble_reactivation import main as ensemble_main
from viral.grosmark_analysis import get_place_cells
from viral.sessions_keep import SESSIONS_KEEP
from viral.utils import shaded_line_plot

# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


from viral.models import Cached2pSession, GrosmarkConfig, Mouse2pSessions
from viral.cache_2p_sessions import process_session
from viral.constants import (
    BEHAVIOUR_DATA_PATH,
    CACHE_PATH,
    SPREADSHEET_ID,
    SYNC_FILE_PATH,
    TIFF_UMBRELLA,
)
from viral.gsheets_importer import gsheet2df
from viral.multiple_sessions import parse_session_number
from viral.single_session import load_data

## TODO: Do we want to include the first day of learning?
# There's likely a lot of interesting reactivated activtity there


def get_session(
    mouse_name: str, date: str, metadata: pd.DataFrame, stage: str
) -> Cached2pSession:

    path = CACHE_PATH / f"{mouse_name}_{date}.json"
    try:
        cached_session = Cached2pSession.model_validate_json(path.read_text())
        print(f"Loaded cached session for {mouse_name} {date} from {path}")
        return cached_session
    except (FileNotFoundError, ValidationError) as e:
        print(f"Cache missing for {mouse_name} {date}. Reprocessing session.")
        row = metadata[metadata["Date"] == date].squeeze(axis=0)
        session_type = row["Type"].lower()
        assert (
            "learning" in session_type if stage == "learned" else stage in session_type
        )
        session_numbers = parse_session_number(row["Session Number"])
        trials = []
        for session_number in session_numbers:
            session_path = (
                BEHAVIOUR_DATA_PATH / mouse_name / row["Date"] / session_number
            )
            trials.extend(load_data(session_path))
        print(f"Got error when loading {mouse_name} {date} from cache. Error is: {e}")

        try:
            wheel_blocked = row["Wheel blocked?"].lower() in {"yes", "true"}
        except KeyError as e:
            print(f"No column 'Wheel blocked?' found: {e}")
            print("Wheel blocked set to None")
            wheel_blocked = False

        process_session(
            trials=trials,
            tiff_directory=TIFF_UMBRELLA / date / mouse_name,
            tdms_path=SYNC_FILE_PATH / Path(row["Sync file"]),
            mouse_name=mouse_name,
            session_type=session_type,
            date=date,
            wheel_blocked=wheel_blocked,
        )

    return Cached2pSession.model_validate_json(path.read_text())


def get_completed_mouse_sessions(mouse_name: str) -> Mouse2pSessions:

    results = [None, None, None]
    for idx, stage in enumerate(["unsupervised", "learning", "learned"]):
        path = CACHE_PATH / f"{mouse_name}_{SESSIONS_KEEP[mouse_name][stage]}.json"
        try:
            results[idx] = Cached2pSession.model_validate_json(path.read_text())
            print(f"Loaded cached session for {mouse_name} {stage} from {path}")
        except (ValidationError, FileNotFoundError) as e:
            print(f"Error retrieving unsupervised session for {mouse_name}: {e}")

    return Mouse2pSessions(
        mouse_name=mouse_name,
        unsupervised=results[0],
        learning=results[1],
        learned=results[2],
    )


def get_mouse_sessions(mouse_name: str) -> Mouse2pSessions:
    metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
    stages = ["unsupervised", "learning", "learned"]
    sessions: dict[str, Cached2pSession | None] = {}
    for stage in stages:
        if SESSIONS_KEEP[mouse_name][stage] is None:
            sessions[stage] = None
            continue

        sessions[stage] = get_session(
            mouse_name,
            SESSIONS_KEEP[mouse_name][stage],
            metadata,
            stage=stage,
        )

    return Mouse2pSessions(
        mouse_name=mouse_name,
        unsupervised=sessions["unsupervised"],
        learning=sessions["learning"],
        learned=sessions["learned"],
    )


def place_cells_plot_learning_stages(
    mouse_name: str, date: str, config: GrosmarkConfig
) -> np.ndarray | None:
    with open(CACHE_PATH / f"{mouse_name}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())

    spks = np.load(
        TIFF_UMBRELLA
        / session.date
        / session.mouse_name
        / "suite2p"
        / "plane0"
        / "oasis_spikes.npy"
    )

    for rewarded in [False, True]:
        pcs_mask, smoothed_matrix, place_threshold = get_place_cells(
            session=session, spks=spks, rewarded=rewarded, config=config, plot=False
        )
    if pcs_mask is None:
        return None

    return collapse_smoothed_matrix(smoothed_matrix, pcs_mask)


def collapse_smoothed_matrix(
    smoothed_matrix: np.ndarray, pcs_mask: np.ndarray
) -> np.ndarray:
    return smoothed_matrix[pcs_mask, :].mean(axis=0)


def main() -> None:
    # for mouse_name in SESSIONS_KEEP.keys():
    #     mouse_sessions = get_mouse_sessions(mouse_name)

    stages = ["unsupervised", "learning", "learned"]

    config = GrosmarkConfig(
        bin_size=2,
        start=0,
        end=180,
    )

    all_collapsed_stages = {stage: [] for stage in stages}

    for mouse_name in SESSIONS_KEEP.keys():
        if mouse_name not in {"JB034", "JB035", "JB036"}:
            continue
        for stage in stages:
            if SESSIONS_KEEP[mouse_name][stage] is None:
                continue
            collapsed = place_cells_plot_learning_stages(
                mouse_name, date=SESSIONS_KEEP[mouse_name][stage], config=config
            )
            if collapsed is not None:
                all_collapsed_stages[stage].append(collapsed)

    colors = ["green", "blue", "orange"]
    color_idx = 0
    for stage_name, data in all_collapsed_stages.items():

        data_matrix = np.vstack(data)
        shaded_line_plot(
            arr=data_matrix,
            x_axis=np.linspace(config.start, config.end, data_matrix.shape[1]),
            color=colors[color_idx],
            label=stage_name,
        )
        color_idx += 1
        for landmark_center in [45, 90, 135]:
            plt.axvspan(
                landmark_center - 2.5, landmark_center + 2.5, color="red", alpha=0.5
            )

    1 / 0


if __name__ == "__main__":
    main()
