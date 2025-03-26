import sys
from pathlib import Path

from matplotlib import pyplot as plt
from typing import List, Tuple

# from viral.models import MouseSummary
from pydantic import ValidationError

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

import seaborn as sns
import numpy as np
import typing
import pandas as pd


sns.set_theme(context="talk", style="ticks")

from viral.constants import HERE
from viral.utils import get_genotype, get_sex
from viral.single_session import (
    load_data,
    plot_rewarded_vs_unrewarded_licking,
    plot_speed_reward_unrewarded,
    remove_bad_trials,
)
from viral.multiple_sessions import (
    cache_mouse,
    load_cache,
    plot_performance_across_days,
    get_chance_level,
    rolling_performance,
    plot_rolling_performance,
    learning_metric,
    flatten_sessions,
    plot_mouse_performance,
    plot_performance_summaries,
)
from viral.constants import BEHAVIOUR_DATA_PATH, HERE
from viral.models import TrialSummary, SessionSummary
from viral.gsheets_importer import gsheet2df


def get_mice_age(mouse_names: List[str]) -> dict:
    mouse_overview_df = gsheet2df(
        "1j1stQALfwqGnKht8Soltl-PlrM2W4s3uQl3S4mnREGM", "Sheet1", 3
    )
    mouse_age_columns = [
        "Name",
        "Age at learning commenced",
        "Age at last day of training",
    ]
    mouse_ages_df = mouse_overview_df[mouse_age_columns]

    mice_age_dict = dict()

    for _, row in mouse_ages_df.iterrows():
        if row["Name"] in mouse_names:
            mice_age_dict[row["Name"]] = {
                "Age at learning commenced 1": int(row[1]),
                "Age at last day of training 1": int(row[3]),
                "Age at learning commenced 2": int(row[2]),
                "Age at last day of training 2": int(row[4]),
            }

    return mice_age_dict


def plot_mice_age(mouse_names: List[str]) -> None:
    ages: dict = {
        "WT": {"male": {}, "female": {}},
        "NLGF": {"male": {}, "female": {}},
        "Oligo-BACE1-KO": {"male": {}, "female": {}},
    }

    mice_age_dict = get_mice_age(mouse_names)

    for mouse_name in mouse_names:
        genotype = get_genotype(mouse_name)
        sex = get_sex(mouse_name)

        if genotype in ages and sex in ages[genotype]:
            ages[genotype][sex][mouse_name] = mice_age_dict.get(mouse_name)

    data = list()
    for genotype, sex_dict in ages.items():
        for sex, mice in sex_dict.items():
            for mouse, age_data in mice.items():
                if age_data:
                    data.append(
                        {
                            "mouse": mouse,
                            "genotype": genotype,
                            "sex": sex,
                            "Age at learning commenced 1": age_data[
                                "Age at learning commenced 1"
                            ]
                            / 30,
                            "Age at last day of training 1": age_data[
                                "Age at last day of training 1"
                            ]
                            / 30,
                            "Age at learning commenced 2": age_data[
                                "Age at learning commenced 2"
                            ]
                            / 30,
                            "Age at last day of training 2": age_data[
                                "Age at last day of training 2"
                            ]
                            / 30,
                        }
                    )

    df = pd.DataFrame(data)
    male_data = df[df["sex"] == "male"]
    female_data = df[df["sex"] == "female"]

    yticks = np.arange(4, 6, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.boxplot(
        x="genotype", y="Age at learning commenced 1", hue="sex", data=df, ax=axes[0]
    )
    sns.scatterplot(
        x="genotype",
        y="Age at learning commenced 1",
        hue="sex",
        data=df,
        size=8,
        ax=axes[0],
        legend=False,
        zorder=10,
    )
    axes[0].set_yticks(yticks)
    axes[0].set_xlabel("Genotype")
    axes[0].set_ylabel("Age (months)")
    axes[0].set_title("Age at learning commenced 1")

    # Plot Age at last day of training 1
    sns.boxplot(
        x="genotype", y="Age at last day of training 1", hue="sex", data=df, ax=axes[1]
    )
    sns.scatterplot(
        x="genotype",
        y="Age at last day of training 1",
        hue="sex",
        data=df,
        size=8,
        ax=axes[1],
        legend=False,
        zorder=10,
    )

    sns.scatterplot(
        x="genotype",
        y="Age at last day of training 1",
        data=male_data,
        ax=axes[1],
        hue="sex",
        legend=False,
        zorder=10,
        s=80,
    )
    sns.scatterplot(
        x="genotype",
        y="Age at last day of training 1",
        data=female_data,
        ax=axes[1],
        hue="sex",
        legend=False,
        zorder=10,
        s=80,
    )

    axes[1].set_yticks(yticks)
    axes[1].set_xlabel("Genotype")
    axes[1].set_ylabel("Age (months)")
    axes[1].set_title("Age at last day of training 1")
    plt.tight_layout()
    plt.show()

    # Boxplot for genotype (and sex)
    # Scatterplot individually


if __name__ == "__main__":

    plot_mice_age(
        [
            "JB011",
            "JB012",
            "JB013",
            "JB014",
            "JB015",
            "JB016",
            "JB017",
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
            "JB030",
            "JB031",
            "JB032",
            "JB033",
        ]
    )

    # mice: List[MouseSummary] = []

    # redo = False
    # # cache_mouse("JB011")
    # # cache_mouse("JB023")
    # # cache_mouse("JB026")
    # # cache_mouse("JB027")

    # for mouse_name in [
    #     # "JB011",
    #     # "JB012",
    #     # "JB013",
    #     # "JB014",
    #     # "JB015",
    #     # "JB016",
    #     # "JB017",
    #     # "JB018",
    #     # "JB019",
    #     # "JB020",
    #     # "JB021",
    #     # "JB022",
    #     # "JB023",
    #     # "JB024",
    #     # "JB025",
    #     "JB026",
    #     # "JB027",
    # ]:

    #     print(f"\nProcessing {mouse_name}...")
    #     if redo:
    #         cache_mouse(mouse_name)
    #         mice.append(load_cache(mouse_name))
    #         print(f"mouse_name {mouse_name} redone and cached")
    #     else:
    #         try:
    #             mice.append(load_cache(mouse_name))
    #             print(f"mouse_name {mouse_name} already cached")
    #         except (ValidationError, FileNotFoundError):
    #             print(f"mouse_name {mouse_name} not cached yet...")
    #             cache_mouse(mouse_name)
    #             mice.append(load_cache(mouse_name))
    #             print(f"mouse_name {mouse_name} cached now")

    # # plot_performance_summaries(mice, "recall", ["genotype", "sex"], window=40)
    # plot_performance_summaries(mice, "recall_reversal", ["genotype", "sex"], window=40)
    # plot_mouse_performance(mice[0], window=50)

    # # plot_running_speed_summaries(mice, "recall", running_speed_AZ)
    # # plot_running_speed_summaries(mice, "recall_reversal", running_speed_AZ)
    # # plot_trial_time_summaries(mice, "learning")
    # # plot_num_trials_summaries(mice, "learning")
