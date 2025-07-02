from pathlib import Path
import sys

import pandas as pd
from pydantic import ValidationError


# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


from viral.run_oasis import main as run_oasis
from viral.models import Cached2pSession, Mouse2pSessions
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


SESSIONS_KEEP = {
    # Imaging of poor quality, dont analyse
    # "JB011": {
    #     "unsupervised": "2024-10-22",
    #     "learning": "2024-10-25",
    #     "learned": "2024-10-30",
    # },
    "JB014": {  # LOOKS GOOD
        "unsupervised": "2024-10-24",
        "learning": "2024-10-31",
        "learned": "2024-11-04",
    },
    "JB015": {
        "unsupervised": "2024-10-24",
        "learning": "2024-10-31",
        "learned": "2024-11-19",
    },
    "JB016": {
        "unsupervised": "2024-10-24",
        "learning": "2024-10-31",
        "learned": "2024-11-05",
    },
    "JB018": {
        "unsupervised": "2024-11-20",
        "learning": "2024-11-28",
        "learned": "2024-12-03",
    },
    "JB019": {
        "unsupervised": "2024-11-19",
        "learning": "2024-11-20",
        "learned": "2024-11-22",
    },
    "JB020": {
        "unsupervised": "2024-11-19",
        "learning": "2024-11-20",
        "learned": "2024-11-22",
    },
    "JB021": {
        "unsupervised": "2024-11-29",
        "learning": "2024-12-06",
        "learned": "2024-12-09",
    },
    "JB022": {
        "unsupervised": "2024-12-04",
        "learning": "2024-12-10",
        "learned": "2024-12-12",
    },
    "JB026": {
        "unsupervised": "2024-12-10",
        "learning": "2024-12-13",
        "learned": "2024-12-15",
    },
    "JB030": {
        "unsupervised": "2025-03-07",
        "learning": "2025-03-13",
        "learned": "2025-03-14",
    },
    "JB033": {
        "unsupervised": "2025-03-13",
        "learning": "2025-03-17",
        "learned": "2025-03-19",
    },
}

# 26 is ok, 27 dont use,
# 30 is ok 33 is good 31 is ok not great


def oasis_runner() -> None:
    print("Running OASIS deconvolution on session data")
    for mouse_name, sessions in SESSIONS_KEEP.items():
        for stage, date in sessions.items():
            print(f"Processing {mouse_name} {date} for stage {stage}")
            run_oasis(mouse_name, date, grosmark=False)


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
        process_session(
            trials=trials,
            tiff_directory=TIFF_UMBRELLA / date / mouse_name,
            tdms_path=SYNC_FILE_PATH / Path(row["Sync file"]),
            mouse_name=mouse_name,
            session_type=session_type,
            date=date,
        )
    return Cached2pSession.model_validate_json(path.read_text())


def get_completed_mouse_sessions(mouse_name: str) -> list[Mouse2pSessions]:

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
    try:
        unsupervised = get_session(
            mouse_name,
            SESSIONS_KEEP[mouse_name]["unsupervised"],
            metadata,
            stage="unsupervised",
        )
    except Exception as e:
        print(f"Error retrieving unsupervised session for {mouse_name}: {e}")
        unsupervised = None

    try:
        learning = get_session(
            mouse_name,
            SESSIONS_KEEP[mouse_name]["learning"],
            metadata,
            stage="learning",
        )
    except Exception as e:
        print(f"Error retrieving learning session for {mouse_name}: {e}")
        learning = None
    try:
        learned = get_session(
            mouse_name,
            SESSIONS_KEEP[mouse_name]["learned"],
            metadata,
            stage="learned",
        )
    except Exception as e:
        print(f"Error retrieving learned session for {mouse_name}: {e}")
        learned = None

    return Mouse2pSessions(
        mouse_name=mouse_name,
        unsupervised=unsupervised,
        learning=learning,
        learned=learned,
    )


def main() -> None:
    mouse_name = "JB016"


if __name__ == "__main__":
    oasis_runner()
