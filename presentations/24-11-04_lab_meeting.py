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


sns.set_theme(context="talk", style="ticks")

from viral.constants import HERE
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
)
from viral.constants import BEHAVIOUR_DATA_PATH, HERE
from viral.models import TrialSummary, SessionSummary


def rolling_performance_cumulative(
    mouse_list: list[str], window: int
) -> Tuple[List[float], List[float], Tuple[float, float]]:
    """Return list with average rolling performances for an entire group and chance level"""
    redo = False
    mice = []
    for mouse_name in mouse_list:
        if redo:
            cache_mouse(mouse_name)
        try:
            mice.append(load_cache(mouse_name))
            print(f"mouse_name {mouse_name} already cached")
        except (ValidationError, FileNotFoundError):
            print(f"mouse_name {mouse_name} not cached yet...")
            cache_mouse(mouse_name)
            mice.append(load_cache(mouse_name))
            print(f"mouse_name {mouse_name} cached now")

    rolling_per_mouse = [
        rolling_performance(flatten_sessions(mouse.sessions), window) for mouse in mice
    ]
    rolling_per_mouse_padded = pad_to_max_length(rolling_per_mouse)

    rolling_performance_average = np.nanmean(rolling_per_mouse_padded, axis=0)

    chance = get_chance_level(mice=mice, window=window)
    chance_level = (
        np.percentile(chance, 1).astype(float),
        np.percentile(chance, 99).astype(float),
    )

    return rolling_per_mouse_padded, rolling_performance_average, chance_level


def plot_rolling_performance_cumulative(
    mouse_list: list[str], group_name: str, window: int, xlim: int = 500
) -> None:
    rolling_per_mouse_padded, rolling_performance_average, chance_level = (
        rolling_performance_cumulative(mouse_list=mouse_list, window=window)
    )
    x_values = np.arange(1, len(rolling_performance_average) + 1)
    plt.figure(figsize=(12, 6))
    for i, mouse_performance in enumerate(rolling_per_mouse_padded):
        plt.plot(
            x_values, mouse_performance, alpha=0.4, label=f"Mouse {i + 1}", linewidth=2
        )
    plt.plot(
        x_values,
        rolling_performance_average,
        label="Rolling Performance Average",
        color="C0",
        linewidth=4,
    )
    plt.axhspan(chance_level[0], chance_level[1], color="gray", alpha=0.5)
    plt.text(
        len(rolling_performance_average),
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
    plt.ylim(-0.8, 4.1)
    # TODO
    # Change the x lim back!!!
    plt.xlim(0, xlim)
    plt.title(group_name)
    plt.savefig(
        HERE.parent / "plots" / f"rolling_performance_group_{group_name}.png", dpi=1200
    )


def plot_rolling_performance_all_mice() -> None:

    mice_names = ["JB012"]
    mice = []
    redo = True
    for mouse_name in mice_names:
        if redo:
            cache_mouse(mouse_name)
        try:
            mice.append(load_cache(mouse_name))
            print(f"mouse_name {mouse_name} already cached")
        except (ValidationError, FileNotFoundError):
            print(f"mouse_name {mouse_name} not cached yet...")
            cache_mouse(mouse_name)
            mice.append(load_cache(mouse_name))
            print(f"mouse_name {mouse_name} cached now")

    # TODO: The range is about -0.5 to 0.5
    chance = get_chance_level(mice)

    plt.figure(figsize=(len(mice_names) * 3, 4))
    for idx, mouse in enumerate(mice):
        plt.subplot(1, len(mice_names), idx + 1)
        # plt.title(f"wild type {idx + 1}")
        plt.title(f"{mouse.name}")
        plot_rolling_performance(
            mouse.sessions,
            window=50 if mouse.name != "J020" else 20,
            chance_level=(
                np.percentile(chance, 1).astype(float),
                np.percentile(chance, 99).astype(float),
            ),
        )
        plt.ylim(-0.8, 4.1)
        # plt.xlim(0, 600)
        plt.xlabel("trial number")
        if idx == 0:
            plt.ylabel("learning metric")

    plt.tight_layout()
    # plt.savefig(HERE.parent / "plots" / "rolling_performance.png", dpi=1200)
    plt.show()


def plot_rolling_performance_genotype(genotype: str, mice_names: list) -> None:
    mice = []
    redo = True
    for mouse_name in mice_names:
        # mice = [load_cache(mouse_name) for mouse_name in ["JB011", "JB012", "JB013", "JB014", "JB015", "JB016"]]
        if redo:
            cache_mouse(mouse_name)
        try:
            mice.append(load_cache(mouse_name))
            print(f"mouse_name {mouse_name} already cached")
        except (ValidationError, FileNotFoundError):
            print(f"mouse_name {mouse_name} not cached yet...")
            cache_mouse(mouse_name)
            mice.append(load_cache(mouse_name))
            print(f"mouse_name {mouse_name} cached now")

        # TODO: The range is about -0.5 to 0.5
    chance = get_chance_level(mice=mice, window=50)

    plt.figure(figsize=(len(mice_names) * 3, 4))
    for idx, mouse in enumerate(mice):
        plt.subplot(1, len(mice_names), idx + 1)
        # plt.title(f"wild type {idx + 1}")
        plt.title(f"{genotype} {idx + 1}")
        plot_rolling_performance(
            mouse.sessions,
            window=50 if mouse.name != "J020" else 20,
            chance_level=(
                np.percentile(chance, 1).astype(float),
                np.percentile(chance, 99).astype(float),
            ),
        )
        plt.ylim(-0.8, 4.1)
        # plt.xlim(0, 600)
        plt.xlabel("trial number")
        if idx == 0:
            plt.ylabel("learning metric")

    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / f"rolling_performance{genotype}.png", dpi=1200)


if __name__ == "__main__":
    mouse_list = [
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
    ]
    for mouse in mouse_list:
        cache_mouse(mouse)
    # plot_rolling_performance_all_mice()
    # plot_rolling_performance_genotype(
    #     genotype="Oligo-KO", mice_names=["JB014", "JB015"]
    # )
    # plot_rolling_performance_genotype(
    #     genotype="APP-NLGF", mice_names=["JB011", "JB012", "JB013", "JB016"]
    # )
    # plot_rolling_performance_genotype(genotype="APP-NLGF", mice_names=["JB018"])

    plot_rolling_performance_cumulative(
        mouse_list=["JB014", "JB015", "JB018", "JB020", "JB022"],
        group_name="Oligo-BACE1-KO",
        window=50,
        xlim=900,
    )
    plot_rolling_performance_cumulative(
        mouse_list=[
            "JB011",
            "JB012",
            "JB013",
            "JB016",
            "JB017",
            "JB019",
            "JB021",
            "JB023",
        ],
        group_name="APP-NLGF",
        window=50,
        xlim=900,
    )
    plot_rolling_performance_cumulative(
        mouse_list=["JB024", "JB025", "JB026", "JB027"],
        group_name="Wild type",
        window=50,
        xlim=900,
    )

    # plot_rolling_performance_cumulative(
    #     mouse_list=["J018", "J019", "J020", "J021"],
    #     group_name="Wild type",
    #     window=50,
    # )
    # plot_rolling_performance_genotype(
    #     genotype="Wild type", mice_names=["J018", "J019", "J020", "J021"]
    # )

    # mice: List[MouseSummary] = []
    # redo = True
    # # for mouse_name in ["JB011"]:
    # for mouse_name in ["J018", "J019", "J020", "J021"]:
    #     # for mouse_name in ["JB011", "JB012", "JB013", "JB014", "JB015", "JB016"]:
    #     # for mouse_name in ["J015", "J016", "J004", "J005", "J007"]:
    #     # for mouse_name in ["J004", "J005", "J007"]:
    #     # for mouse_name in ["J004", "J005", "J007"]:

    #     print(f"\nProcessing {mouse_name}...")
    #     if redo:
    #         cache_mouse(mouse_name)

    #     try:
    #         mice.append(load_cache(mouse_name))
    #         print(f"mouse_name {mouse_name} already cached")
    #     except (ValidationError, FileNotFoundError):
    #         print(f"mouse_name {mouse_name} not cached yet...")
    #         cache_mouse(mouse_name)
    #         mice.append(load_cache(mouse_name))
    #         print(f"mouse_name {mouse_name} cached now")

    # chance = get_chance_level(mice)

    # for mouse in mice:
    #     plt.figure()
    #     plt.title(mouse.name)
    #     window = 50 if mouse.name != "J020" else 20
    #     plot_rolling_performance(
    #         mouse.sessions,
    #         window,
    #         (
    #             np.percentile(chance, 1).astype(float),
    #             np.percentile(chance, 99).astype(float),
    #         ),
    #         add_text_to_chance=True,
    #     )
    plt.show()
