# Allow you to run the file directly, remove if exporting as a proper module
from pathlib import Path
import sys

import pandas as pd
from pydantic import ValidationError

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


from models import Cached2pSession, Mouse2pSessions
from viral.cache_2p_sessions import process_session
from viral.constants import (
    BEHAVIOUR_DATA_PATH,
    SPREADSHEET_ID,
    SYNC_FILE_PATH,
    TIFF_UMBRELLA,
)
from viral.gsheets_importer import gsheet2df
from viral.multiple_sessions import parse_session_number
from viral.single_session import load_data

CACHE_PATH = HERE.parent / "data" / "cached_2p"


SESSIONS_KEEP = {
    "JB011": {
        "unsupervised": "2024-10-22",
        "learning": "2024-10-25",
        "learned": "2024-10-30",
    },
    "JB014": {
        "unsupervised": "2024-10-24",
        "learning": "2024-10-30",
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
        "unsupervised": "2024-11-19",
        "learning": ":2024-12-05",
        "learned": "2024-12-09",
    },
    "JB022": {
        "unsupervised": "2024-12-04",
        "learning": "2024-12-10",
        "learned": "2024-12-12",
    },
}


def get_session(
    mouse_name: str, date: str, metadata: pd.DataFrame, stage: str
) -> Cached2pSession:

    row = metadata[metadata["Date"] == date].squeeze(axis=0)
    session_type = row["Type"].lower()
    assert "learning" in session_type if stage == "learned" else stage in session_type

    session_numbers = parse_session_number(row["Session Number"])

    trials = []
    for session_number in session_numbers:
        session_path = BEHAVIOUR_DATA_PATH / mouse_name / row["Date"] / session_number
        trials.extend(load_data(session_path))

    path = CACHE_PATH / f"{mouse_name}_{date}.json"
    try:
        return Cached2pSession.model_validate_json(path.read_text())
    except (FileNotFoundError, ValidationError):
        process_session(
            trials=trials,
            tiff_directory=TIFF_UMBRELLA / date / mouse_name,
            tdms_path=SYNC_FILE_PATH / Path(row["Sync file"]),
            mouse_name=mouse_name,
            session_type=session_type,
            date=date,
        )
    return Cached2pSession.model_validate_json(path.read_text())


def main() -> None:
    mouse_name = "JB011"
    metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
    mouse_sessions = Mouse2pSessions(
        mouse_name=mouse_name,
        unsupervised=get_session(
            mouse_name,
            SESSIONS_KEEP[mouse_name]["unsupervised"],
            metadata,
            stage="unsupervised",
        ),
        learning=get_session(
            mouse_name,
            SESSIONS_KEEP[mouse_name]["learning"],
            metadata,
            stage="learning",
        ),
        learned=get_session(
            mouse_name, SESSIONS_KEEP[mouse_name]["learned"], metadata, stage="learned"
        ),
    )


if __name__ == "__main__":
    main()
