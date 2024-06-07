from pathlib import Path
import sys

from matplotlib import pyplot as plt


HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

import seaborn as sns


sns.set_theme(context="talk", style="ticks")

from viral.single_session import (
    load_data,
    plot_rewarded_vs_unrewarded_licking,
    plot_speed,
    remove_bad_trials,
)
from viral.multiple_sessions import load_cache, plot_performance_across_days
from viral.constants import DATA_PATH, HERE


def before_learning_licking() -> None:
    MOUSE = "J005"
    DATE = "2024-04-26"
    SESSION_NUMBER = "002"
    SESSION_PATH = DATA_PATH / MOUSE / DATE / SESSION_NUMBER

    trials = load_data(SESSION_PATH)
    trials = remove_bad_trials(trials)

    plot_rewarded_vs_unrewarded_licking(trials)


def after_learning_licking() -> None:
    MOUSE = "J007"
    DATE = "2024-04-29"
    SESSION_NUMBER = "002"
    SESSION_PATH = DATA_PATH / MOUSE / DATE / SESSION_NUMBER

    trials = load_data(SESSION_PATH)
    trials = remove_bad_trials(trials)

    plot_rewarded_vs_unrewarded_licking(trials)


def before_learning_speed() -> None:

    MOUSE = "J005"
    DATE = "2024-04-26"
    SESSION_NUMBER = "002"
    SESSION_PATH = DATA_PATH / MOUSE / DATE / SESSION_NUMBER

    trials = load_data(SESSION_PATH)
    trials = remove_bad_trials(trials)
    plot_speed(trials, 30)


def after_learning_speed() -> None:

    MOUSE = "J005"
    DATE = "2024-05-02"
    SESSION_NUMBER = "001"
    SESSION_PATH = DATA_PATH / MOUSE / DATE / SESSION_NUMBER

    trials = load_data(SESSION_PATH)
    trials = remove_bad_trials(trials)
    plot_speed(trials, 30)


def across_days():
    plt.figure(figsize=(10, 4))
    # f, (ax, bx) = plt.subplots(, 1, sharey="row")
    for idx, mouse_name in enumerate(["J004", "J005", "J007"]):

        mouse = load_cache(mouse_name)
        sessions = [session for session in mouse.sessions if len(session.trials) > 30]
        plt.subplot(1, 3, idx + 1)
        plt.ylim(-0.7, 2.7)
        plot_performance_across_days(sessions)
        plt.title(mouse_name)
        if idx == 0:
            plt.ylabel("Learning Metric")

    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / "learning_metric_across_days.png")


if __name__ == "__main__":
    across_days()
