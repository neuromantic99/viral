import json
from pathlib import Path
import sys
from typing import List

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
from viral.utils import add_imaging_info_to_trials

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
    for mouse_name in ["JB014"]:
        # for mouse_name in ["JB011", "JB014", "JB015", "JB016"]:
        metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
        for _, row in metadata.iterrows():
            try:
                date = row["Date"]
                session_type = row["Type"].lower()

                if (
                    HERE.parent / "data" / "cached_2p" / f"{mouse_name}_{date}.json"
                ).exists():
                    print(f"Skipping {mouse_name} {date} as already exists")
                    continue

                if (
                    "learning day" not in session_type
                    and "reversal learning" not in session_type
                ):
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
                logger.debug(
                    f"Error processing {mouse_name} {date} {session_type}: {e}"
                )
