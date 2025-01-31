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

import seaborn as sns
import pandas as pd

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

    with open(
        HERE.parent / "data" / "behaviour_summaries" / f"{mouse_name}.json", "w"
    ) as f:
        json.dump(
            MouseSummary(
                sessions=session_summaries,
                name=mouse_name,
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


def learning_metric(trials: List[TrialSummary]) -> float:
    return (speed_difference(trials) + licking_difference(trials)) / 2


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


def flatten_sessions(sessions: List[SessionSummary]) -> List[TrialSummary]:
    return [trial for session in sessions for trial in session.trials]


def rolling_performance(trials: List[TrialSummary], window: int) -> List[float]:
    return [
        learning_metric(trials[idx - window : idx])
        for idx in range(window, len(trials))
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


def get_rolling_performance(mouse: MouseSummary, window: int) -> dict:
    pre_reversal = [
        session for session in mouse.sessions if "reversal" not in session.name.lower()
    ]
    post_reversal = [
        session for session in mouse.sessions if "reversal" in session.name.lower()
    ]
    assert len(pre_reversal) + len(post_reversal) == len(mouse.sessions)

    return {
        "pre_reversal": np.array(
            rolling_performance(flatten_sessions(pre_reversal), window)
        ),
        "post_reversal": np.array(
            rolling_performance(flatten_sessions(post_reversal), window)
        ),
    }


def get_chance_level(mice: List[MouseSummary], window: int) -> List[float]:
    """Rough permutation test / bootstrap for chance level. Needs to be formalised further."""

    # TODO: Maybe should compute this on a per mouse basis
    all_trials = [
        copy.deepcopy(trial)
        for mouse in mice
        for trial in flatten_sessions(mouse.sessions)
    ]

    result = []
    for _ in range(1000):
        sample = random.sample(all_trials, window)
        for trial in sample:
            trial.rewarded = random.choice([True, False])
        result.append(learning_metric(sample))

    return result


def extract_metric(
    mice: List[MouseSummary], metric_fn: Callable, metric_name: str
) -> dict:
    metric_dict = {}

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
        # TODO Add the recall conditions

        if metric_name == "num_trials":
            metric_dict[mouse.name] = {
                "pre_reversal": [len(session.trials) for session in pre_reversal],
                "post_reversal": [len(session.trials) for session in post_reversal],
            }
        else:
            metric_dict[mouse.name] = {
                "pre_reversal_unrewarded": metric_fn(flatten_sessions(pre_reversal)),
                "post_reversal_unrewarded": metric_fn(flatten_sessions(post_reversal)),
                "pre_reversal_rewarded": metric_fn(flatten_sessions(pre_reversal)),
                "post_reversal_rewarded": metric_fn(flatten_sessions(post_reversal)),
            }

    return metric_dict


def prepare_plot_by_genotype_and_condition(
    metric_dict: dict, condition: str, metric_name: str
) -> pd.DataFrame:
    genotypes = [
        "WT",
        "NLGF",
        "Oligo-BACE1-KO",
    ]  # TODO: Probably not brilliant to hard-code it in here
    reward_conditions = ["unrewarded", "rewarded"]
    plot_data = list()

    if metric_name == "num_trials":
        # Session-level metrics
        for genotype in genotypes:
            for mouse, data in metric_dict.items():
                if get_genotype(mouse) == genotype:
                    for session_value in data[condition]:
                        label = f"{get_genotype(mouse)}_{condition}"
                        plot_data.append(
                            {
                                "genotype": genotype,
                                "value": session_value,
                                condition: condition,
                            }
                        )
    else:
        # Trial-level metrics
        for genotype in genotypes:
            for genotype in genotypes:
                for reward_condition in reward_conditions:
                    key = f"{condition}_{reward_condition}"
                    for mouse, data in metric_dict.items():
                        if get_genotype(mouse) == genotype:
                            plot_data.append(
                                {
                                    "genotype": genotype,
                                    "value": data[key],
                                    "condition": f"{condition}_{reward_condition}",
                                }
                            )
    return pd.DataFrame(plot_data)


def plot_metric(
    mice: List[MouseSummary],
    metric_fn: Callable,
    metric_name: str,
    y_label: str,
    condition: str = "pre_reversal",
) -> None:
    metric_dict = extract_metric(mice, metric_fn, metric_name)
    to_plot = prepare_plot_by_genotype_and_condition(
        metric_dict, condition, metric_name
    )
    plt.figure()
    sns.boxplot(data=to_plot, showfliers=False)
    sns.stripplot(data=to_plot, edgecolor="black", linewidth=1)
    plt.ylabel(y_label)
    plt.title(condition.replace("_", " ").capitalize())
    plt.tight_layout()
    plt.savefig(
        HERE.parent / "plots" / f"behaviour-summaries-{metric_name}_{condition}"
    )
    plt.show()


def trials_per_session_metric(sessions: list[SessionSummary]) -> float:
    num_trials = [session.num_trials for session in sessions]
    return np.mean(num_trials).astype(float)


def plot_num_trials_summaries(
    mice: List[MouseSummary], condition: str = "pre_reversal"
):
    num_trials_dict = {}
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

        num_trials_dict[mouse.name] = {
            "pre_reversal": trials_run(pre_reversal),
            "post_reversal": trials_run(post_reversal),
        }

    to_plot = {
        "NLGF": [
            data[condition]
            for mouse, data in num_trials_dict.items()
            if data[condition] > 1
            if get_genotype(mouse) == "NLGF"
        ],
        "Oligo-Bace1\nKnockout": [
            data[condition]
            for mouse, data in num_trials_dict.items()
            if data[condition] > 1
            if get_genotype(mouse) == "Oligo-BACE1-KO"
        ],
        "WT": [
            data[condition]
            for mouse, data in num_trials_dict.items()
            if data[condition] > 1
            if get_genotype(mouse) == "WT"
        ],
    }
    plt.ylabel(f"# Trials per Sessions")
    plt.title(condition.replace("_", " ").capitalize())

    sns.boxplot(to_plot, showfliers=False)
    sns.stripplot(to_plot, edgecolor="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / f"behaviour-summaries-num-trials-{condition}")
    plt.show()


def plot_performance_summaries(mice: List[MouseSummary], key: str):
    rolling_performance_dict = {}
    window = 40

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

        rolling_performance_dict[mouse.name] = {
            "pre_reversal": np.array(
                rolling_performance(flatten_sessions(pre_reversal), window)
            ),
            "post_reversal": np.array(
                rolling_performance(flatten_sessions(post_reversal), window)
            ),
        }

    to_plot = {
        "NLGF": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "NLGF"
        ],
        "Oligo-Bace1\nKnockout": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "Oligo-BACE1-KO"
        ],
        "WT": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "WT"
        ],
    }

    plt.ylabel(f"Trials to criterion")
    plt.title(key.replace("_", " ").capitalize())

    sns.boxplot(to_plot, showfliers=False)
    sns.stripplot(to_plot, edgecolor="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / f"behaviour-summaries-{key}")
    plt.show()


def plot_performance_summaries_genotype_sex(
    mice: List[MouseSummary], key: str = "pre_reversal"
):
    rolling_performance_dict = {}
    window = 40

    for mouse in mice:
        rolling_performance_dict[mouse.name] = get_rolling_performance(
            mouse=mouse, window=window
        )

    to_plot = {
        "NLGF\nMale": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "NLGF"
            if get_sex(mouse) == "male"
        ],
        "NLGF\nFemale": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "NLGF"
            if get_sex(mouse) == "female"
        ],
        "Oligo-Bace1\nKnockout\nMale": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "Oligo-BACE1-KO"
            if get_sex(mouse) == "male"
        ],
        "Oligo-Bace1\nKnockout\nFemale": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "Oligo-BACE1-KO"
            if get_sex(mouse) == "female"
        ],
        "WT\nMale": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "WT"
            if get_sex(mouse) == "male"
        ],
        "WT\nFemale": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "WT"
            if get_sex(mouse) == "female"
        ],
    }

    plt.ylabel(f"Trials to criterion")
    plt.title(key.replace("_", " ").capitalize())

    sns.boxplot(to_plot, showfliers=False)
    sns.stripplot(to_plot, edgecolor="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / f"behaviour-summaries-genotype-sex-{key}")
    plt.show()


def plot_performance_summaries_genotype_setup(
    mice: List[MouseSummary], key: str = "pre_reversal"
):
    rolling_performance_dict = {}
    window = 40

    for mouse in mice:
        rolling_performance_dict[mouse.name] = get_rolling_performance(
            mouse=mouse, window=window
        )

    to_plot = {
        "NLGF\nbox": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "NLGF"
            if get_setup(mouse) == "box"
        ],
        "NLGF\n2P": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "NLGF"
            if get_setup(mouse) == "2P"
        ],
        "Oligo-Bace1\nKnockout\nbox": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "Oligo-BACE1-KO"
            if get_setup(mouse) == "box"
        ],
        "Oligo-Bace1\nKnockout\n2P": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "Oligo-BACE1-KO"
            if get_setup(mouse) == "2P"
        ],
        "WT\nbox": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "WT"
            if get_setup(mouse) == "box"
        ],
        "WT\n2P": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_genotype(mouse) == "WT"
            if get_setup(mouse) == "2P"
        ],
    }

    plt.ylabel(f"Trials to criterion")
    plt.title(key.replace("_", " ").capitalize())

    sns.boxplot(to_plot, showfliers=False)
    sns.stripplot(to_plot, edgecolor="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / f"behaviour-summaries-genotype-setup-{key}")
    plt.show()


def plot_performance_summaries_sex(mice: List[MouseSummary], key: str = "pre_reversal"):
    rolling_performance_dict = {}
    window = 40

    for mouse in mice:
        for mouse in mice:
            rolling_performance_dict[mouse.name] = get_rolling_performance(
                mouse=mouse, window=window
            )

    to_plot = {
        "Male": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_sex(mouse) == "male"
        ],
        "Female": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_sex(mouse) == "female"
        ],
    }

    plt.ylabel(f"Trials to criterion")
    plt.title(key.replace("_", " ").capitalize())

    sns.boxplot(to_plot, showfliers=False)
    sns.stripplot(to_plot, edgecolor="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / f"behaviour-summaries-sex-{key}")
    plt.show()


def plot_performance_summaries_setup(
    mice: List[MouseSummary], key: str = "pre_reversal"
):
    rolling_performance_dict = {}
    window = 40

    for mouse in mice:
        for mouse in mice:
            rolling_performance_dict[mouse.name] = get_rolling_performance(
                mouse=mouse, window=window
            )

    to_plot = {
        "box": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_setup(mouse) == "box"
        ],
        "2P": [
            np.where(data[key] > 1)[0][0] + window
            for mouse, data in rolling_performance_dict.items()
            if get_setup(mouse) == "2P"
        ],
    }

    plt.ylabel(f"Trials to criterion")
    plt.title(key.replace("_", " ").capitalize())

    sns.boxplot(to_plot, showfliers=False)
    sns.stripplot(to_plot, edgecolor="black", linewidth=1)
    plt.tight_layout()
    plt.savefig(HERE.parent / "plots" / f"behaviour-summaries-setup-{key}")
    plt.show()


if __name__ == "__main__":

    mice: List[MouseSummary] = []

    redo = False
    cache_mouse("JB011")
    # cache_mouse("JB023")
    # cache_mouse("JB026")
    # cache_mouse("JB027")

    for mouse_name in [
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

    # plot_performance_summaries(mice, "pre_reversal")
    # plot_performance_summaries(mice, "post_reversal")
    # plot_running_speed_summaries(mice, "pre_reversal")
    # plot_running_speed_summaries(mice, "post_reversal")
    # plot_trial_time_summaries(mice, "post_reversal")
    # plot_trial_time_summaries(mice, "pre_reversal")
    # plot_num_trials_summaries(mice, "pre_reversal")
    # plot_num_trials_summaries(mice, "post_reversal")

    # plot_performance_summaries_setup(mice, "pre_reversal")
    # plot_performance_summaries_setup(mice, "post_reversal")
    # plot_performance_summaries_sex(mice, "pre_reversal")
    # plot_performance_summaries_sex(mice, "post_reversal")
    # plot_performance_summaries_genotype_setup(mice, "pre_reversal")
    # plot_performance_summaries_genotype_setup(mice, "post_reversal")
    # plot_performance_summaries_genotype_sex(mice, "pre_reversal")
    # plot_performance_summaries_genotype_sex(mice, "post_reversal")

    plt.show()

    # mouse = mice[1]
    window = 50

    chance = get_chance_level(mice, window=window)

    mouse = mice[0]
    plt.figure()
    plot_rolling_performance(
        mouse.sessions,
        window,
        (
            np.percentile(chance, 1).astype(float),
            np.percentile(chance, 99).astype(float),
        ),
        add_text_to_chance=True,
    )

    num_to_reversal = sum(
        len(session.trials)
        for session in mouse.sessions
        if get_session_type(session.name)
        not in {"reversal", "recall", "recall_reversal"}
    )

    num_to_recall = sum(
        len(session.trials)
        for session in mouse.sessions
        if get_session_type(session.name) not in {"recall", "recall_reversal"}
    )

    num_to_recall_reversal = sum(
        len(session.trials)
        for session in mouse.sessions
        if get_session_type(session.name) != "recall_reversal"
    )

    plt.axvline(num_to_reversal - window, color=sns.color_palette()[1], linestyle="--")
    plt.text(
        num_to_reversal - window + 10,
        2,
        "Reversal\nStarts",
        color=sns.color_palette()[1],
        fontsize=15,
    )

    plt.axvline(num_to_recall - window, color=sns.color_palette()[2], linestyle="--")
    plt.text(
        num_to_recall - window + 10,
        2,
        "Memory\nRecall\nStarts",
        color=sns.color_palette()[2],
        fontsize=15,
    )

    plt.axvline(
        num_to_recall_reversal - window, color=sns.color_palette()[3], linestyle="--"
    )
    plt.text(
        num_to_recall_reversal - window + 10,
        2,
        "Recall\nReversal\nStarts",
        color=sns.color_palette()[3],
        fontsize=15,
    )
    plt.tight_layout()
    # plt.savefig(HERE.parent / "plots" / "example-mouse-performance.png")
    plt.show()
