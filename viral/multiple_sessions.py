import copy
import json
from pathlib import Path
import random
import pandas as pd
import re
import sys
import inspect
from pydantic import ValidationError

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))
sys.path.append(str(HERE.parent))

from typing import List, Tuple, Dict, Callable, Optional

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
from viral.models import (
    MouseSummary,
    SessionSummary,
    TrialSummary,
    MultipleSessionsConfig,
)
from viral.utils import (
    d_prime,
    get_genotype,
    get_wheel_circumference_from_rig,
    shaded_line_plot,
    get_sex,
    get_setup,
    get_session_type,
    get_rewarded_texture,
)

import seaborn as sns

sns.set_theme(context="talk", style="ticks")


def parse_session_number(session_number: str) -> List[str]:
    """Deals with multiple session numbers (Must be indicated with a '+' or an 'and') and adding 00 to session numbers"""
    session_numbers = (
        [s.strip() for s in re.split("\+ |and |\*|\n", session_number)]
        if "and" in session_number or "+" in session_number
        else [session_number.strip()]
    )

    session_numbers = [
        (
            session_number
            if len(session_number) == 3
            else (
                f"00{session_number}"
                if len(session_number) == 1
                else f"0{session_number}"
            )
        )
        for session_number in session_numbers
    ]

    assert all(
        session_numbers and len(session_number) == 3
        for session_number in session_numbers
    ), f"Failed to parse session numbers. Original string {session_number}. Processed strings: {session_numbers}"

    return session_numbers


def cache_mouse(mouse_name: str) -> None:
    metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
    # Remove empty rows
    metadata = metadata[metadata["Date"].astype(bool)]
    session_summaries = []

    setup = dict()
    rewarded_textures = dict()
    encountered_session_types = set()

    for _, row in metadata.iterrows():
        type_check = row["Type"].lower()
        if "learning day" not in type_check or "unsupervised" in type_check:
            assert (
                "habituation" in type_check
                or "take bottle out" in type_check
                or "water in dish" in type_check
                or "unsupervised" in type_check
            ), f"type {type_check} not understood"
            continue

        session_type = get_session_type(session_name=type_check)
        encountered_session_types.add(session_type)
        if session_type not in rewarded_textures and not pd.isna(
            row["Rewarded texture"]
        ):
            rewarded_textures[session_type] = get_rewarded_texture(
                row["Rewarded texture"]
            )
            setup[session_type] = get_setup(row["Rig"])

        print(f"Processing session: {row['Type']}")

        session_numbers = parse_session_number(row["Session Number"])

        trials = []
        for session_number in session_numbers:
            session_path = (
                BEHAVIOUR_DATA_PATH / mouse_name / row["Date"] / session_number
            )
            trials.extend(load_data(session_path))

        assert trials[0].texture, "You're accidently processing a habituation"
        wheel_circumference = get_wheel_circumference_from_rig(row["Rig"])

        print(f"Total of {len(trials)} trials")

        trials = remove_bad_trials(trials, wheel_circumference=wheel_circumference)
        print(f"Total of {len(trials)} after bad removal")

        if not [trial for trial in trials if trial.texture_rewarded] or not [
            trial for trial in trials if not trial.texture_rewarded
        ]:
            print(
                f"Mouse {mouse_name}, date {row['Date']}, sessions {session_numbers} does not have both rewarded and unrewarded trials"
            )
            continue

        session_summaries.append(
            SessionSummary(
                name=row["Type"],
                trials=[
                    summarise_trial(trial, wheel_circumference=wheel_circumference)
                    for trial in trials
                ],
                rewarded_licks=get_binned_licks(
                    [trial for trial in trials if trial.texture_rewarded],
                    wheel_circumference=wheel_circumference,
                ),
                unrewarded_licks=get_binned_licks(
                    [trial for trial in trials if not trial.texture_rewarded],
                    wheel_circumference=wheel_circumference,
                ),
            )
        )
        print(f"Session summary for {row["Type"]} / {row["Date"]}")

    for session_type in encountered_session_types:
        if session_type not in rewarded_textures:
            raise ValueError(
                f"Missing rewarded texture information for {mouse_name}, session type: {session_type}"
            )
        if session_type not in setup:
            raise ValueError(
                f"Missing setup information for {mouse_name}, session type: {session_type}"
            )

    with open(
        HERE.parent / "data" / "behaviour_summaries" / f"{mouse_name}.json", "w"
    ) as f:
        json.dump(
            MouseSummary(
                sessions=session_summaries,
                name=mouse_name,
                genotype=get_genotype(mouse_name),
                sex=get_sex(mouse_name),
                setup=setup,
                rewarded_texture=rewarded_textures,
            ).model_dump(),
            f,
        )


def load_cache(mouse_name: str) -> MouseSummary:
    with open(
        HERE.parent / "data" / "behaviour_summaries" / f"{mouse_name}.json", "r"
    ) as f:
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


def learning_metric(
    trials: List[TrialSummary], config: MultipleSessionsConfig
) -> float:
    return (
        speed_difference(trials) * config.speed
        + licking_difference(trials) * config.licking
    )


def plot_binned_licking(sessions: List[SessionSummary]) -> None:
    shaded_line_plot(
        np.array([session.rewarded_licks for session in sessions]),
        (np.arange(0, 200, 5)[1:] + np.arange(0, 200, 5)[:-1]) / 2,
        "blue",
        "rewarded",
    )

    shaded_line_plot(
        np.array([session.unrewarded_licks for session in sessions]),
        (np.arange(0, 200, 5)[1:] + np.arange(0, 200, 5)[:-1]) / 2,
        "red",
        "unrewarded",
    )
    plt.xlim(50, 200)
    # plt.ylim(0, 10)
    plt.legend()


def plot_performance_across_days(
    sessions: List[SessionSummary], config: MultipleSessionsConfig
) -> None:

    plt.plot(
        range(len(sessions)),
        [learning_metric(session.trials, config) for session in sessions],
    )

    plt.xticks(
        range(len(sessions)),
        [f"Day {idx + 1}" for idx in range(len(sessions))],
        rotation=90,
    )

    plt.axhline(0, color="black", linestyle="--")
    # plt.title(MOUSE)


def flatten_sessions(sessions: List[SessionSummary]) -> List[TrialSummary]:
    return [trial for session in sessions for trial in session.trials]


def rolling_performance(
    trials: List[TrialSummary], config: MultipleSessionsConfig
) -> List[float]:
    return [
        learning_metric(trials[idx - config.window : idx], config)
        for idx in range(config.window, len(trials))
    ]


def running_speed_overall(trials: List[TrialSummary], rewarded: bool) -> float:
    speeds = [trial.trial_speed for trial in trials if trial.rewarded == rewarded]
    return np.mean(speeds)


def running_speed_AZ(trials: List[TrialSummary], rewarded: bool) -> float:
    speeds = [trial.speed_AZ for trial in trials if trial.rewarded == rewarded]
    return np.mean(speeds)


def running_speed_nonAZ(trials: List[TrialSummary], rewarded: bool) -> float:
    speeds = [trial.speed_nonAZ for trial in trials if trial.rewarded == rewarded]
    return np.mean(speeds)


def trial_time(trials: List[TrialSummary], rewarded: bool) -> float:
    trial_times = [
        trial.trial_time_overall for trial in trials if trial.rewarded == rewarded
    ]
    return np.mean(trial_times)


def trials_run(sessions: List[SessionSummary]) -> float:
    num_trials = [session.num_trials for session in sessions]
    return np.mean(num_trials)


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


def get_chance_level(
    mice: List[MouseSummary], config: MultipleSessionsConfig
) -> List[float]:
    """Rough permutation test / bootstrap for chance level. Needs to be formalised further."""

    # TODO: Maybe should compute this on a per mouse basis
    all_trials = [
        copy.deepcopy(trial)
        for mouse in mice
        for trial in flatten_sessions(mouse.sessions)
    ]

    result = []
    for _ in range(1000):
        sample = random.sample(all_trials, config.window)
        for trial in sample:
            trial.rewarded = random.choice([True, False])
        result.append(learning_metric(sample, config))

    return result


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


def create_metric_dict(
    mice: List[MouseSummary],
    metric_fn: Callable,
    config: Optional[MultipleSessionsConfig] = None,
    flat_sessions: bool = True,
    include_reward_status: bool = True,
) -> dict:
    """Create a dictionary for metric data to be plotted.

    Args:
        mice (List[MouseSummary]):                  A list of MouseSummary objects.
        metric_fn (Callable):                       A function which takes raw data and processes it for a metric, e.g. extracting speed.
        config (MultipleSessionsConfig):            A MultipleSessionsConfig object including window size, and weights for learning metric computation.
        flat_sessions (bool):                       Whether to flatten sessions. Is needed for metrices which are computed on a trial-by-trial basis. Defaults to true.
        include_reward_status (bool):               Whether to distinguish between rewarded and unrewarded trials. Defaults to True.
    Returns:
        dict:                           The dictionary of metric data to be plotted.
    """
    # Initiating the metric_dict dictionary
    metric_dict = dict()
    session_types = [
        "learning",
        "reversal",
        "recall",
        "recall_reversal",
    ]  # TODO Probably shouldn't be hardcoded here

    # Checks whether the given metric function takes 'window' as an argument to decide whether to use the 'window' input
    metric_fn_params = inspect.signature(metric_fn).parameters
    has_config_param = "config" in metric_fn_params
    use_config_param = has_config_param and config

    # Iterating through the list of MouseSummary objects
    for mouse in mice:
        # Initiating a dictionary for each mouse called 'mouse_metrics'
        mouse_metrics = dict()
        # Creates a dictionary of session types with the session type as key and a list of SessionSummary objects as values
        sessions = {
            s_type: filter_sessions_by_session_type(mouse, s_type)
            for s_type in session_types
        }
        reward_statuses = [True, False] if include_reward_status else [None]
        # Iterating through the different session types
        for s_type in session_types:
            # Flattening the sessions to be a list of TrialSummary objects if specified
            processed_sessions = (
                flatten_sessions(sessions[s_type])
                if flat_sessions
                else sessions[s_type]
            )
            if not processed_sessions:
                print(
                    f"Warning: No sessions found for session type {s_type} for mouse {mouse.name}"
                )
            # Iterating through the reward statuses
            for reward_status in reward_statuses:
                # Creating a key for each category in the mouse_metric dictionary
                # If reward status is to be included, the key will look like this: 'session_type_rewarded' / 'session_type_unrewarded'
                # Else, the key will look like this: 'session_type'
                key = (
                    f"{s_type}_{"rewarded" if reward_status else "unrewarded"}"
                    if include_reward_status
                    else f"{s_type}"
                )
                # Build the kwargs
                kwargs = {}
                if include_reward_status:
                    kwargs["rewarded"] = reward_status
                if flat_sessions:
                    kwargs["trials"] = processed_sessions
                if not flat_sessions:
                    kwargs["sessions"] = processed_sessions
                if use_config_param:
                    kwargs["config"] = config
                mouse_metrics[key] = metric_fn(**kwargs)
        # Adding the mouse_metrics dictionary as the value for the mouse_name in the overall metric_dict dictionary
        metric_dict[mouse.name] = mouse_metrics
    return metric_dict


def prepare_plot_data(
    metric_dict: dict,
    session_type: str,
    genotypes: list[str],
    include_reward_status: bool = True,
) -> dict:
    """Prepare plot data by session type and reward status"""
    if include_reward_status is False:
        return {
            f"{genotype}": [
                data[f"{session_type}"]
                for mouse, data in metric_dict.items()
                if get_genotype(mouse) == genotype
            ]
            for genotype in genotypes
        }
    else:
        return {
            f"{genotype}_{reward_status}": [
                data[f"{session_type}_{reward_status}"]
                for mouse, data in metric_dict.items()
                if data[f"{session_type}_{reward_status}"]
                and get_genotype(mouse) == genotype
            ]
            for genotype in genotypes
            for reward_status in ["rewarded", "unrewarded"]
        }


def plot_running_speed_summaries(
    mice: List[MouseSummary],
    session_type: str = "learning",
    speed_function: Callable = running_speed_overall,
) -> None:
    running_speed_dict = create_metric_dict(
        mice,
        speed_function,
        flat_sessions=True,
        include_reward_status=True,
    )

    to_plot = prepare_plot_data(
        running_speed_dict, session_type, ["NLGF", "Oligo-BACE1-KO", "WT"]
    )

    to_plot = {
        key: [value for value in values if value is not None]
        for key, values in to_plot.items()
    }

    plt.ylabel(f"Running speed (cm/s)")
    plt.title(session_type.replace("_", " ").capitalize())

    sns.boxplot(to_plot, showfliers=False)
    sns.stripplot(to_plot, edgecolor="black", linewidth=1)

    ax = plt.gca()
    new_labels = [
        label.get_text()
        .replace("Oligo-BACE1-KO", "Oligo-\nBACE1-KO")
        .replace("_", "\n")
        for label in ax.get_xticklabels()
    ]
    ax.set_xticklabels(new_labels, fontsize=10)

    plt.tight_layout()
    plt.savefig(
        HERE.parent
        / "plots"
        / f"behaviour-summaries-{speed_function.__name__}-{session_type}.svg",
        dpi=300,
    )
    plt.show()


def plot_trial_time_summaries(mice: List[MouseSummary], session_type: str = "learning"):
    trial_time_dict = create_metric_dict(
        mice, trial_time, flat_sessions=True, include_reward_status=True
    )

    to_plot = prepare_plot_data(
        trial_time_dict, session_type, ["NLGF", "Oligo-BACE1-KO", "WT"]
    )

    to_plot = {
        key: [value for value in values if value is not None]
        for key, values in to_plot.items()
    }

    plt.ylabel(f"Trial time (s)")
    plt.title(session_type.replace("_", "").capitalize())

    sns.boxplot(to_plot, showfliers=False)
    sns.stripplot(to_plot, edgecolor="black", linewidth=1)

    ax = plt.gca()
    new_labels = [
        label.get_text()
        .replace("Oligo-BACE1-KO", "Oligo-\nBACE1-KO")
        .replace("_", "\n")
        for label in ax.get_xticklabels()
    ]
    ax.set_xticklabels(new_labels, fontsize=10)

    plt.tight_layout()
    plt.savefig(
        HERE.parent / "plots" / f"behaviour-summaries-trial-time-{session_type}"
    )
    plt.show()


def plot_num_trials_summaries(mice: List[MouseSummary], session_type: str = "learning"):
    num_trials_dict = create_metric_dict(
        mice, trials_run, flat_sessions=False, include_reward_status=False
    )

    to_plot = prepare_plot_data(
        num_trials_dict, session_type, ["NLGF", "Oligo-BACE1-KO", "WT"], False
    )

    to_plot = {
        key: [value for value in values if value is not None]
        for key, values in to_plot.items()
    }
    plt.ylabel(f"# Trials per Sessions")
    plt.title(session_type.replace("_", " ").capitalize())

    sns.boxplot(to_plot, showfliers=False)
    sns.stripplot(to_plot, edgecolor="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(
        HERE.parent / "plots" / f"behaviour-summaries-num-trials-{session_type}"
    )
    plt.show()


def plot_performance_summaries(
    mice: List[MouseSummary],
    session_type: str,
    group_by: list[str],
    config: MultipleSessionsConfig,
):
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
    plt.figure()
    plt.ylabel("Trials to criterion")
    plt.title(session_type.replace("_", " ").capitalize())
    sns.boxplot(to_plot, showfliers=False)
    ax = plt.gca()
    new_labels = [
        label.get_text()
        .replace("Oligo-BACE1-KO", "Oligo-\nBACE1-KO")
        .replace("_", "\n")
        for label in ax.get_xticklabels()
    ]
    ax.set_xticklabels(new_labels, fontsize=12)
    sns.stripplot(to_plot, edgecolor="black", linewidth=1)
    plt.tight_layout()
    group_suffix = "-".join(group_by)
    plt.savefig(
        HERE.parent
        / "plots"
        / f"behaviour-summaries-{group_suffix}-{session_type}.svg",
        dpi=300,
    )
    plt.show()


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
        {
            "name": "recall",
            "label": "Memory\nRecall\nStarts",
            "excluded_session_types": ["recall", "recall_reversal"],
            "colour": sns.color_palette()[1],
        },
        {
            "name": "recall_reversal",
            "label": "Recall\nReversal\nStarts",
            "excluded_session_types": ["recall_reversal"],
            "colour": sns.color_palette()[2],
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
            fontsize=15,
        )
    plt.axhline(1, color="red", linestyle="dotted", alpha=0.7, linewidth=1.5)
    plt.title(mouse.name)
    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / f"{mouse.name}-performance.svg", dpi=300)
    plt.show()


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
    }:

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

    plot_performance_summaries(mice, "learning", ["genotype"], config=config)
    # plot_mouse_performance(mice[0], config=config)
    # plot_running_speed_summaries(mice, "recall", running_speed_AZ)
    # ## Probably not interesting as related to speed
    # plot_trial_time_summaries(mice, "learning")
    # plot_num_trials_summaries(mice, "reversal")
    # plot_num_trials_summaries(mice, "recall")
    # plot_num_trials_summaries(mice, "recall_reversal")
