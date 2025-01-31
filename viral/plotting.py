import copy
import json
from pathlib import Path
import random
import re
import sys

from pydantic import ValidationError

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))
sys.path.append(str(HERE.parent))

from typing import List, Tuple, Callable

from matplotlib import pyplot as plt
import numpy as np
from viral.gsheets_importer import gsheet2df
from viral.single_session import (
    get_binned_licks,
    load_data,
    remove_bad_trials,
    summarise_trial,
)
from viral.constants import BEHAVIOUR_DATA_PATH, HERE, SPREADSHEET_ID
from viral.models import MouseSummary, SessionSummary, TrialSummary
from viral.utils import (
    average_different_lengths,
    d_prime,
    get_genotype,
    get_wheel_circumference_from_rig,
    shaded_line_plot,
    get_sex,
    get_setup,
    get_session_type,
)
from multiple_sessions import (
    flatten_sessions,
    rolling_performance,
    plot_metric,
    trials_per_session_metric,
    cache_mouse,
    load_cache,
)

import seaborn as sns

sns.set_theme(context="talk", style="ticks")


def plot_rolling_performance(
    sessions: List[SessionSummary],
    window: int,
    chance_level: Tuple[float, float],
    add_text_to_chance: bool = True,
) -> None:

    trials = flatten_sessions(sessions)
    rolling = rolling_performance(trials, window)

    plt.plot(rolling)
    plt.axhspan(chance_level[0], chance_level[1], color="gray", alpha=0.5)
    if add_text_to_chance:
        plt.text(
            len(rolling),
            0,
            "chance level",
            horizontalalignment="right",
            verticalalignment="center",
            color="gray",
            fontsize=12,
            weight="bold",
            clip_on=True,
        )
    plt.xlabel("Trial Number")
    plt.ylabel("Learning metric")


def plot_num_trials_summaries(
    mice: List[MouseSummary], condition: str = "pre_reversal"
) -> None:
    plot_metric(
        mice=mice,
        metric_fn=None,
        # bit of a hack, but as it is looking at a session-by-session basis, we don't need to compute a metric by trials
        condition=condition,
        y_label="Number of trials",
        metric_name="num_trials",
    )


def plot_running_speed_summaries(
    mice: List[MouseSummary],
    condition: str = "pre_reversal",
):

    running_speed_dict = {}
    for mouse in mice:
        pre_reversal = [
            session
            for session in mouse.sessions
            if "reversal" not in session.name.lower()
        ]
        post_reversal = [
            session for session in mouse.sessions if "reversal" in session.name.lower()
        ]
        assert len(pre_reversal) + len(post_reversal) == len(mouse.sessions)

        # overall speed
        # running_speed_dict[mouse.name] = {
        #     "pre_reversal_unrewarded": running_speed_overall(
        #         flatten_sessions(pre_reversal), False
        #     ),
        #     "post_reversal_unrewarded": running_speed_overall(
        #         flatten_sessions(post_reversal), False
        #     ),
        #     "pre_reversal_rewarded": running_speed_overall(
        #         flatten_sessions(pre_reversal), True
        #     ),
        #     "post_reversal_rewarded": running_speed_overall(
        #         flatten_sessions(post_reversal), True
        #     ),
        # }

        # # AZ speed
        # running_speed_dict[mouse.name] = {
        #     "pre_reversal_unrewarded": running_speed_AZ(
        #         flatten_sessions(pre_reversal), False
        #     ),
        #     "post_reversal_unrewarded": running_speed_AZ(
        #         flatten_sessions(post_reversal), False
        #     ),
        #     "pre_reversal_rewarded": running_speed_AZ(
        #         flatten_sessions(pre_reversal), True
        #     ),
        #     "post_reversal_rewarded": running_speed_AZ(
        #         flatten_sessions(post_reversal), True
        #     ),
        # }

        # non-AZ speed
        running_speed_dict[mouse.name] = {
            "pre_reversal_unrewarded": running_speed_nonAZ(
                flatten_sessions(pre_reversal), False
            ),
            "post_reversal_unrewarded": running_speed_nonAZ(
                flatten_sessions(post_reversal), False
            ),
            "pre_reversal_rewarded": running_speed_nonAZ(
                flatten_sessions(pre_reversal), True
            ),
            "post_reversal_rewarded": running_speed_nonAZ(
                flatten_sessions(post_reversal), True
            ),
        }

    to_plot = {
        "NLGF\nRewarded": [
            data[f"{condition}_rewarded"]
            for mouse, data in running_speed_dict.items()
            if data[f"{condition}_rewarded"] > 1
            if get_genotype(mouse) == "NLGF"
        ],
        "NLGF\nUnrewarded": [
            data[f"{condition}_unrewarded"]
            for mouse, data in running_speed_dict.items()
            if data[f"{condition}_unrewarded"] > 1
            if get_genotype(mouse) == "NLGF"
        ],
        "Oligo-Bace1\nKnockout\nRewarded": [
            data[f"{condition}_rewarded"]
            for mouse, data in running_speed_dict.items()
            if data[f"{condition}_rewarded"] > 1
            if get_genotype(mouse) == "Oligo-BACE1-KO"
        ],
        "Oligo-Bace1\nKnockout\nUnrewarded": [
            data[f"{condition}_unrewarded"]
            for mouse, data in running_speed_dict.items()
            if data[f"{condition}_unrewarded"] > 1
            if get_genotype(mouse) == "Oligo-BACE1-KO"
        ],
        "WT\nRewarded": [
            data[f"{condition}_rewarded"]
            for mouse, data in running_speed_dict.items()
            if data[f"{condition}_rewarded"] > 1
            if get_genotype(mouse) == "WT"
        ],
        "WT\nUnrewarded": [
            data[f"{condition}_unrewarded"]
            for mouse, data in running_speed_dict.items()
            if data[f"{condition}_unrewarded"] > 1
            if get_genotype(mouse) == "WT"
        ],
    }
    plt.ylabel(f"Running speed (cm/s)")
    plt.title(condition.replace("_", " ").capitalize())

    sns.boxplot(to_plot, showfliers=False)
    sns.stripplot(to_plot, edgecolor="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / f"behaviour-summaries-running-speed{condition}")
    plt.show()


def plot_trial_time_summaries(
    mice: List[MouseSummary], condition: str = "pre_reversal"
):
    trial_time_dict = {}
    for mouse in mice:
        pre_reversal = [
            session
            for session in mouse.sessions
            if "reversal" not in session.name.lower()
        ]
        post_reversal = [
            session for session in mouse.sessions if "reversal" in session.name.lower()
        ]
        assert len(pre_reversal) + len(post_reversal) == len(mouse.sessions)

        # overall speed
        trial_time_dict[mouse.name] = {
            "pre_reversal_unrewarded": trial_time(
                flatten_sessions(pre_reversal), False
            ),
            "post_reversal_unrewarded": trial_time(
                flatten_sessions(post_reversal), False
            ),
            "pre_reversal_rewarded": trial_time(flatten_sessions(pre_reversal), True),
            "post_reversal_rewarded": trial_time(flatten_sessions(post_reversal), True),
        }

    to_plot = {
        "NLGF\nRewarded": [
            data[f"{condition}_rewarded"]
            for mouse, data in trial_time_dict.items()
            if data[f"{condition}_rewarded"] > 1
            if get_genotype(mouse) == "NLGF"
        ],
        "NLGF\nUnrewarded": [
            data[f"{condition}_unrewarded"]
            for mouse, data in trial_time_dict.items()
            if data[f"{condition}_unrewarded"] > 1
            if get_genotype(mouse) == "NLGF"
        ],
        "Oligo-Bace1\nKnockout\nRewarded": [
            data[f"{condition}_rewarded"]
            for mouse, data in trial_time_dict.items()
            if data[f"{condition}_rewarded"] > 1
            if get_genotype(mouse) == "Oligo-BACE1-KO"
        ],
        "Oligo-Bace1\nKnockout\nUnrewarded": [
            data[f"{condition}_unrewarded"]
            for mouse, data in trial_time_dict.items()
            if data[f"{condition}_unrewarded"] > 1
            if get_genotype(mouse) == "Oligo-BACE1-KO"
        ],
        "WT\nRewarded": [
            data[f"{condition}_rewarded"]
            for mouse, data in trial_time_dict.items()
            if data[f"{condition}_rewarded"] > 1
            if get_genotype(mouse) == "WT"
        ],
        "WT\nUnrewarded": [
            data[f"{condition}_unrewarded"]
            for mouse, data in trial_time_dict.items()
            if data[f"{condition}_unrewarded"] > 1
            if get_genotype(mouse) == "WT"
        ],
    }
    plt.ylabel(f"Trial time (s)")
    plt.title(condition.replace("_", " ").capitalize())

    sns.boxplot(to_plot, showfliers=False)
    sns.stripplot(to_plot, edgecolor="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / f"behaviour-summaries-trial-time-{condition}")
    plt.show()


if __name__ == "__main__":

    mice = list()
    redo = False
    for mouse_name in [
        "JB011",
        "JB012",
        "JB013",
        "JB014",
        "JB015",
        "JB016",
        # "JB017",
        "JB018",
        "JB019",
        "JB020",
        "JB021",
        "JB022",
        "JB023",
        "JB024",
        "JB025",
        "JB026",
        "JB027",
        "JB025",
        "JB026",
        "JB027",
    ]:

        print(f"\nProcessing {mouse_name}...")
        if redo:
            cache_mouse(mouse_name)
            mice.append(load_cache(mouse_name))
            print(f"mouse_name {mouse_name} redone and cached")
        else:
            try:
                mice.append(load_cache(mouse_name))
                print(f"mouse_name {mouse_name} already cached")
            except (ValidationError, FileNotFoundError):
                print(f"mouse_name {mouse_name} not cached yet...")
                cache_mouse(mouse_name)
                mice.append(load_cache(mouse_name))
                print(f"mouse_name {mouse_name} cached now")
    plot_num_trials_summaries(mice, "pre_reversal")
