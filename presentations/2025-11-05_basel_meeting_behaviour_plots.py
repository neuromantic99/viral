from pathlib import Path
import sys
from scipy import stats


from pydantic import ValidationError

from viral.multiple_sessions import (
    cache_mouse,
    create_metric_dict,
    flatten_sessions,
    get_chance_level,
    load_cache,
    rolling_performance,
)

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))
sys.path.append(str(HERE.parent))

from typing import List, Tuple, Dict, Literal

from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
import numpy as np
from viral.constants import HERE, SERVER_PATH
from viral.models import (
    MouseSummary,
    SessionSummary,
    MultipleSessionsConfig,
)
from viral.utils import (
    get_genotype,
    get_session_type,
)

import seaborn as sns
import pandas as pd

sns.set_theme(context="talk", style="ticks")


def plot_rolling_performance(
    sessions: List[SessionSummary],
    config: MultipleSessionsConfig,
    chance_level: Tuple[float, float],
    add_text_to_chance: bool = True,
) -> None:

    trials = flatten_sessions(sessions)
    rolling = rolling_performance(trials, config)

    plt.plot(rolling)
    plt.axhspan(chance_level[0], chance_level[1], color="gray", alpha=0.5)
    if add_text_to_chance:
        plt.text(
            40,
            0.2,
            "Chance level",
            horizontalalignment="right",
            verticalalignment="center",
            color="gray",
        )
    plt.xlabel("Trial Number")
    plt.ylabel("Learning metric")


def filter_sessions_by_session_type(
    mouse: MouseSummary, session_type: str = "learning"
) -> List[SessionSummary]:
    """Filter sessions by session type"""
    session_categories: dict = {
        "learning": [],
        "reversal": [],
        "recall": [],
        "recall_reversal": [],
    }

    assert session_type in session_categories.keys(), "Invalid session type provided"

    for session in mouse.sessions:
        type = get_session_type(session_name=session.name)
        if type == "learning":
            session_categories["learning"].append(session)
        elif type == "reversal":
            session_categories["reversal"].append(session)
        elif type == "recall":
            session_categories["recall"].append(session)
        elif type == "recall_reversal":
            session_categories["recall_reversal"].append(session)
        else:
            raise ValueError(f"Invalid session type: {type} for session {session.name}")

    assert sum(len(lst) for lst in session_categories.values()) == len(
        mouse.sessions
    ), "Length of session categories does not match total number of sessions"

    return session_categories[session_type]


def plot_performance_summaries(
    mice: List[MouseSummary],
    session_type: Literal["learning", "reversal", "recall", "recall_reversal"],
    group_by: list[str],
    config: MultipleSessionsConfig,
) -> float:
    rolling_performance_dict = create_metric_dict(
        mice,
        rolling_performance,
        config,
        True,
        False,
    )
    to_plot: Dict[str, list] = dict()

    for mouse in mice:
        data = rolling_performance_dict[mouse.name]
        group_label_parts = list()
        for attr in group_by:
            value = getattr(mouse, attr)
            if isinstance(value, dict):
                dict_value = value.get(session_type, "")
                group_label_parts.append(str(dict_value))
            else:
                group_label_parts.append(str(value))

        group_label = "\n".join(group_label_parts)

        if group_label not in to_plot:
            to_plot[group_label] = list()
        if session_type in data:
            try:
                session_data = np.array(data[session_type])
                first_threshold_idx = np.where(session_data > 1)[0][0] + config.window
                to_plot[group_label].append(first_threshold_idx)
            except IndexError:
                print(f"There is no valid data for {mouse.name} in {session_type}")
                continue

    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    plt.ylabel("Trials to criterion")
    plt.title(session_type.replace("_", " ").capitalize())
    sns.boxplot(
        to_plot,
        showfliers=False,
        hue_order=["WT", "NLGF"],
        palette=palette,
    )
    ax = plt.gca()
    new_labels = [
        label.get_text()
        .replace("Oligo-BACE1-KO", "Oligo-\nBACE1-KO")
        .replace("_", "\n")
        for label in ax.get_xticklabels()
    ]
    ax.set_xticklabels(new_labels)
    sns.stripplot(
        to_plot,
        edgecolor="black",
        linewidth=1,
        palette=palette,
        hue_order=["WT", "NLGF"],
    )

    nlgf_vs_wt = stats.ttest_ind(
        to_plot["NLGF"],
        to_plot["WT"],
    )

    sns.despine()
    plt.tight_layout()
    return nlgf_vs_wt.pvalue


def get_num_to_x(
    sessions: List[SessionSummary], excluded_session_types: List[str]
) -> int:
    return sum(
        len(session.trials)
        for session in sessions
        if get_session_type(session.name) not in excluded_session_types
    )


def plot_mouse_performance(mouse: MouseSummary, config: MultipleSessionsConfig) -> None:
    chance = get_chance_level([mouse], config)
    plt.figure(figsize=(12, 8))
    plot_rolling_performance(
        mouse.sessions,
        config,
        (
            np.percentile(chance, 1).astype(float),
            np.percentile(chance, 99).astype(float),
        ),
        add_text_to_chance=True,
    )
    phases = [
        {
            "name": "reversal",
            "label": "Reversal\nStarts",
            "excluded_session_types": ["reversal", "recall", "recall_reversal"],
            "colour": sns.color_palette()[0],
        },
    ]
    for phase in phases:
        num_to_x = get_num_to_x(
            sessions=mouse.sessions,
            excluded_session_types=phase["excluded_session_types"],
        )
        plt.axvline(
            num_to_x - config.window,
            color=phase["colour"],
            linestyle="--" if phase["name"] != "recall" else "solid",
        )
        plt.text(
            num_to_x - config.window + 10,
            2,
            phase["label"],
            color=phase["colour"],
            # fontsize=15,
        )
    plt.axhline(1, color="red", linestyle="dotted", alpha=0.7, linewidth=1.5)
    plt.title(mouse.name)
    plt.tight_layout()
    plt.xlim(None, 400)
    plt.savefig(HERE.parent / "plots" / f"{mouse.name}-performance.svg", dpi=300)


if __name__ == "__main__":

    mice: List[MouseSummary] = []

    redo = False

    config = MultipleSessionsConfig(speed=0.5, licking=0.5, window=50)

    for mouse_name in {
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
        "JB034",
        "JB035",
        "JB036",
    }:
        if get_genotype(mouse_name) not in {"WT", "NLGF"}:
            continue

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

    config = MultipleSessionsConfig(speed=0.5, licking=0.5, window=50)
    for mouse in mice:
        if mouse.name == "JB023":
            sns.set_context(context="talk", font_scale=1.2)
            plot_mouse_performance(mouse, config=config)

            plt.title("")
            sns.despine()
            plt.savefig(
                SERVER_PATH
                / "viral_plots"
                / "behaviour_summaries"
                / f"behaviour_summaries_{mouse.name}_performance_plot.png"
            )

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    configs = [
        ("Combined", MultipleSessionsConfig(speed=0.5, licking=0.5, window=50)),
        ("Speed only", MultipleSessionsConfig(speed=1, licking=0, window=50)),
        ("Licking only", MultipleSessionsConfig(speed=0, licking=1, window=50)),
    ]

    p_values = []
    for ax, (config_name, config) in zip(axs, configs):
        plt.sca(ax)  # make this subplot the current axes
        p_value = plot_performance_summaries(
            mice, "learning", ["genotype"], config=config
        )
        p_values.append(p_value)
        ax.set_title(config_name)
        if ax != axs[0]:
            ax.set_ylabel("")

    ymin_plot, ymax_plot = ax.get_ylim()
    plot_range = ymax_plot - ymin_plot
    for ax, p_value in zip(axs, p_values):
        text_y = (
            ymax_plot - plot_range * 0.1
        )  # place text just below the top of the axis
        p_text = f"P = {round(p_value, 2)}"
        ax.text(0.5, text_y, p_text, ha="center", va="top")

    plt.tight_layout()

    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "behaviour_summaries"
        / f"behaviour_summaries_comparison_plot.png"
    )
    plt.show()
