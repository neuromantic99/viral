import json
from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from typing import List

from matplotlib import pyplot as plt
import numpy as np
from viral.gsheets_importer import gsheet2df
from viral.single_session import load_data, remove_bad_trials, summarise_trial
from viral.constants import DATA_PATH, HERE
from viral.models import MouseSummary, SessionSummary, TrialSummary
from viral.utils import d_prime


MOUSE = "J007"
SPREADSHEET_ID = "1fMnVXrDeaWTkX-TT21F4mFuAlaXIe6uVteEjv8mH0Q4"


def cache_mouse(mouse_name: str):
    metadata = gsheet2df(SPREADSHEET_ID, MOUSE, 1)
    session_summaries = []
    for _, row in metadata.iterrows():
        if "learning day" not in row["Type"].lower():
            continue

        session_number = (
            row["Session Number"]
            if len(row["Session Number"]) == 3
            else f"00{row['Session Number']}"
        )
        session_path = DATA_PATH / MOUSE / row["Date"] / session_number
        trials = load_data(session_path)
        trials = remove_bad_trials(trials)
        session_summaries.append(
            SessionSummary(
                name=row["Type"], trials=[summarise_trial(trial) for trial in trials]
            )
        )

    with open(HERE.parent / "data" / f"{mouse_name}.json", "w") as f:
        json.dump(
            MouseSummary(
                sessions=session_summaries,
                name=mouse_name,
            ).model_dump(),
            f,
        )


def load_cache(mouse_name: str) -> MouseSummary:
    with open(HERE.parent / "data" / f"{mouse_name}.json", "r") as f:
        return MouseSummary.model_validate_json(f.read())


def speed_difference(trials: List[TrialSummary]) -> float:

    rewarded = [trial.speed_AZ for trial in trials if trial.rewarded]
    unrewarded = [trial.speed_AZ for trial in trials if not trial.rewarded]

    dprime = (np.mean(unrewarded) - np.mean(rewarded)) / (
        (np.std(rewarded) + np.std(unrewarded)) / 2
    )
    return dprime.astype(float)


def licking_difference(trials: List[TrialSummary]) -> float:
    rewarded = [trial.licks_AZ > 0 for trial in trials if trial.rewarded]
    not_rewarded = [trial.licks_AZ > 0 for trial in trials if not trial.rewarded]

    return d_prime(sum(rewarded) / len(rewarded), sum(not_rewarded) / len(not_rewarded))

    # return sum(rewarded) / len(rewarded) - sum(not_rewarded) / len(not_rewarded)


def learning_metric(trials: List[TrialSummary]) -> float:
    return (speed_difference(trials) + licking_difference(trials)) / 2


def plot_performance_across_days(sessions: List[SessionSummary]) -> None:

    plt.plot(
        range(len(sessions)),
        [learning_metric(session.trials) for session in sessions],
    )

    plt.xticks(
        range(len(sessions)),
        [f"Day {idx + 1}" for idx in range(len(sessions))],
        rotation=90,
    )

    plt.axhline(0, color="black", linestyle="--")
    # plt.title(MOUSE)


if __name__ == "__main__":
    cache_mouse(MOUSE)
    # Assumes that the sheet name is the same as the mouse name
    mouse = load_cache(MOUSE)
    sessions = [session for session in mouse.sessions if len(session.trials) > 20]
    plot_performance_across_days(sessions)
    plt.show()
