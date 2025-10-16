from pathlib import Path
import sys
from typing import Dict, List, Literal

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pydantic import ValidationError


# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))
from viral.imaging_utils import trial_is_imaged

from viral.grosmark_analysis import get_place_cells
from viral.sessions_keep import SESSIONS_KEEP
from viral.utils import (
    degrees_to_cm,
    get_speed_positions,
    get_wheel_circumference_from_rig,
    shaded_line_plot,
)


from viral.models import Cached2pSession, GrosmarkConfig, Mouse2pSessions
from viral.cache_2p_sessions import process_session
from viral.constants import (
    BEHAVIOUR_DATA_PATH,
    CACHE_PATH,
    SERVER_PATH,
    SPREADSHEET_ID,
    SYNC_FILE_PATH,
    TIFF_UMBRELLA,
)
from viral.gsheets_importer import gsheet2df
from viral.multiple_sessions import parse_session_number
from viral.single_session import load_data
from viral.ensemble_reactivation import main as ensemble_main

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


def store_place_cell_result(mouse_name: str, date: str, config: GrosmarkConfig) -> None:

    print("Processing", mouse_name, date)
    with open(CACHE_PATH / f"{mouse_name}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())
    spks_path = TIFF_UMBRELLA / session.date / session.mouse_name / "suite2p" / "plane0"

    assert (
        spks_path / "full_grosmark_oasis_preprocessed.npy"
    ).exists(), (
        f"File {spks_path / 'full_grosmark_oasis_preprocessed.npy'} does not exist"
    )
    spks = np.load(spks_path / "oasis_spikes.npy")

    for rewarded in [False, True]:
        pcs_mask, smoothed_matrix, place_threshold = get_place_cells(
            session=session, spks=spks, rewarded=rewarded, config=config, plot=False
        )


class PlaceCellResults:
    def __init__(self, cache_umbrella: Path) -> None:

        self.smoothed_matrix_files = list(
            (cache_umbrella / "smoothed_matrix").glob("*npy")
        )
        self.pcs_combined_files = list((cache_umbrella / "pcs_combined").glob("*npy"))
        self.place_threshold_files = list(
            (cache_umbrella / "place_threshold").glob("*npy")
        )
        self.unsupervised: Dict[str, List] = {"rewarded": [], "unrewarded": []}
        self.learning: Dict[str, List] = {"rewarded": [], "unrewarded": []}
        self.learned: Dict[str, List] = {"rewarded": [], "unrewarded": []}

    def load_file(
        self, file_list: List[Path], date: str, mouse: str, rewarded: bool | None
    ) -> np.ndarray:
        files_match = [
            file
            for file in file_list
            if f"{mouse}_{date}" in file.name and f"rewarded_{rewarded}" in file.name
        ]
        if not files_match:
            raise FileNotFoundError(
                f"No file found for {mouse} {date} rewarded {rewarded}"
            )

        assert (
            len(files_match) == 1
        ), f"more than one file found for {mouse} {date} rewarded {rewarded}"
        return np.load(files_match[0])

    def collapsed_matrix_result(
        self, mouse_name: str, stage: str, rewarded: bool | None
    ) -> np.ndarray:
        date = SESSIONS_KEEP[mouse_name][stage]

        smoothed_matrix = self.load_file(
            self.smoothed_matrix_files,
            date=date,
            mouse=mouse_name,
            rewarded=rewarded,
        )

        pcs_combined = self.load_file(
            self.pcs_combined_files,
            date=date,
            mouse=mouse_name,
            rewarded=rewarded,
        )
        place_threshold = self.load_file(
            self.place_threshold_files,
            date=date,
            mouse=mouse_name,
            rewarded=rewarded,
        )
        mask = smoothed_matrix[pcs_combined, :] > place_threshold[pcs_combined, :]
        return np.sum(mask, axis=0) / mask.shape[0]

        # return smoothed_matrix[pcs_combined, :].mean(axis=0)

    def driver(self) -> None:

        for stage, store in zip(
            ["unsupervised", "learning", "learned"],
            [self.unsupervised, self.learning, self.learned],
        ):
            for mouse_name in SESSIONS_KEEP.keys():
                if mouse_name not in {"JB034", "JB035", "JB036"}:
                    continue

                for rewarded in [False, True]:
                    try:
                        result = self.collapsed_matrix_result(
                            mouse_name, stage=stage, rewarded=rewarded
                        )
                    except FileNotFoundError:
                        continue
                    store["rewarded" if rewarded else "unrewarded"].append(result)

    def plot_result(
        self,
        stage_data: List,
        label: str,
        color: str,
    ) -> None:

        matrix = np.vstack(stage_data)
        shaded_line_plot(
            arr=matrix,
            x_axis=np.linspace(0, 180, matrix.shape[1]),
            color=color,
            label=label,
        )
        for landmark_center in [45, 90, 135]:
            plt.axvspan(
                landmark_center - 2.5, landmark_center + 2.5, color="red", alpha=0.5
            )


def get_speed_summary(
    session: Cached2pSession, rewarded: bool | None, config: GrosmarkConfig
) -> np.ndarray:
    return np.array(
        [
            np.array(
                [
                    speed.speed
                    for speed in get_speed_positions(
                        degrees_to_cm(
                            np.array(trial.rotary_encoder_position),
                            get_wheel_circumference_from_rig("2P"),
                        ),
                        config.start,
                        config.end,
                        config.bin_size,
                        sampling_rate=30,
                    )
                ]
            )
            for trial in session.trials
            if trial_is_imaged(trial)
            and (rewarded is None or trial.texture_rewarded == rewarded)
        ]
    )


def main() -> None:
    place_cell_result = PlaceCellResults(SERVER_PATH / "viral_caches" / "place_cells")
    place_cell_result.driver()

    for data, name in zip(
        [
            place_cell_result.unsupervised,
            place_cell_result.learning,
            place_cell_result.learned,
        ],
        ["unsupervised", "learning", "learned"],
    ):

        plt.figure()
        place_cell_result.plot_result(
            data["unrewarded"],
            "unrewarded",
            "green",
        )
        place_cell_result.plot_result(data["rewarded"], "rewarded", "blue")
        plt.ylim(0, 0.4)
        plt.legend()
        plt.xlabel("Corridor position (cm)")
        plt.ylabel("Proportion place cells\nsignificantly active")
        plt.title(name.capitalize())
        plt.tight_layout()
    1 / 0


def plot_speed_summary() -> None:

    config = GrosmarkConfig(
        start=0,
        end=180,
        bin_size=2,
    )

    result = {
        stage: {"rewarded": [], "unrewarded": []}
        for stage in ["unsupervised", "learning", "learned"]
    }

    for mouse_name, dates in SESSIONS_KEEP.items():
        if mouse_name not in {"JB034", "JB035", "JB036"}:
            continue

        for stage, date in dates.items():
            with open(CACHE_PATH / f"{mouse_name}_{date}.json", "r") as f:
                session = Cached2pSession.model_validate_json(f.read())
                for rewarded in [False, True]:
                    speed = get_speed_summary(session, rewarded=rewarded, config=config)
                    result[stage]["rewarded" if rewarded else "unrewarded"].append(
                        speed
                    )


if __name__ == "__main__":

    for mouse in SESSIONS_KEEP.keys():
        if mouse not in {"JB034", "JB035", "JB036"}:
            continue

        for date in SESSIONS_KEEP[mouse].values():
            for rewarded in [True, False, None]:
                print(f"Starting {mouse} {date} rewarded {rewarded}")
                ensemble_main(mouse, date, rewarded=rewarded, plot=False)
