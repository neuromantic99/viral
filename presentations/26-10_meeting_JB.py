import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from typing import Dict, List, Literal
from scipy import stats
from matplotlib import pyplot as plt
from pydantic import ValidationError
import numpy as np
import seaborn as sns
from viral.gsheets_importer import gsheet2df
from viral.single_session import load_data, summarise_trial
from viral.constants import (
    BEHAVIOUR_DATA_PATH,
    SPREADSHEET_ID,
)
from viral.sessions_keep import SESSIONS_KEEP
from viral.utils import (
    get_genotype,
    get_wheel_circumference_from_rig,
)
from viral.multiple_sessions import (
    load_cache,
    parse_session_number,
    cache_mouse,
    create_metric_dict,
    load_cache,
    rolling_performance,
)
from viral.nlgf.three_group import holm
from viral.nlgf.style import GENOTYPE_COLOURS
from viral.models import MouseSummary, MultipleSessionsConfig

plt.rcParams["pdf.fonttype"] = 42


# from basic_plots.py (James's code)
def compute_anticipatory_licking(genotype: str) -> Dict[str, List[tuple[float, float]]]:
    result: Dict[str, List[tuple[float, float]]] = {
        "Baseline": [],
        "Trained": [],
    }

    for mouse, sessions in SESSIONS_KEEP.items():
        if not get_genotype(mouse) == genotype:
            continue
        print(mouse)
        date_trained = sessions["learned"]

        if date_trained is None:
            continue

        metadata = gsheet2df(SPREADSHEET_ID, mouse, 1)
        first_session_row = metadata[metadata["Type"].str.lower() == "learning day 1"]
        assert len(first_session_row) == 1
        first_session_date = first_session_row["Date"].values[0]

        for date, session_type in zip(
            [first_session_date, date_trained], ["First Session", "Final Session"]
        ):

            session_numbers = metadata[metadata["Date"] == date][
                "Session Number"
            ].values[0]
            # TODO: if I am not mistaken, this was causing a bug if session number in the spreadsheet is set to e.g. "2 + 3" as this code would only look at session "2"
            # session_path = (
            #     BEHAVIOUR_DATA_PATH
            #     / mouse
            #     / date
            #     / parse_session_number(session_number)[0]
            # )
            # trials = load_data(session_path)
            session_paths = [
                (BEHAVIOUR_DATA_PATH / mouse / date / session_number)
                for session_number in parse_session_number(session_numbers)
            ]
            trials = list()
            for i, session_path in enumerate(session_paths):
                trials.extend(load_data(session_path))
            summaries = [
                summarise_trial(trial, get_wheel_circumference_from_rig("2P"))
                for trial in trials
            ]
            rewarded = [trial.licks_AZ > 0 for trial in summaries if trial.rewarded]
            not_rewarded = [
                trial.licks_AZ > 0 for trial in summaries if not trial.rewarded
            ]

            result_tuple = (
                (sum(rewarded) / len(rewarded)) * 100,
                (sum(not_rewarded) / len(not_rewarded)) * 100,
            )

            if session_type == "First Session":
                result["Baseline"].append(result_tuple)
            else:
                result["Trained"].append(result_tuple)
    return result


# from basic_plots.py (James's code)
def basic_anticipatory_licking_plot(genotype: str) -> None:
    # if Path("AL_dict.pkl.npy").exists():
    #     data = np.load("AL_dict.pkl.npy", allow_pickle=True).item()
    # else:
    #     data = compute_anticipatory_licking()
    #     np.save("AL_dict.pkl.npy", data)
    data = compute_anticipatory_licking(genotype=genotype)

    def make_plot(tuples: List[tuple[float, float]], ax: plt.Axes) -> None:
        means = np.mean(tuples, axis=0)
        for session in tuples:
            ax.plot([1, 0], session, color="gray", alpha=0.3)

        ax.plot([1, 0], means, color="black", linewidth=2, marker="o")

        ax.set_xlim(-0.2, 1.2)
        ax.set_xticks([0, 1], ["Unrewarded\nTexture", "Rewarded\nTexture"])
        arr = np.array(tuples)
        p_value = stats.ttest_rel(arr[:, 0], arr[:, 1]).pvalue
        ax.text(
            0.5,
            80,
            f"P = {p_value:.2g}",
            ha="center",
            va="bottom",
            fontsize=12,
            font="arial",
        )

    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

    make_plot(data["Baseline"], axes[0])
    make_plot(data["Trained"], axes[1])
    axes[0].set_ylabel("Trials with licks in AZ (%)")
    axes[1].set_title("After Learning")
    axes[0].set_title("Before Learning")
    sns.despine()
    plt.suptitle(f"Anticipatory Licking in {genotype} mice")
    plt.tight_layout()

    plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        HERE.parent / "plots" / f"basic_anticipatory_licking_plot_{genotype}.pdf",
        bbox_inches="tight",
        transparent=True,
    )


# from viral.multiple_session.speed_difference (James's code)
# Jimmy said to use it from the MouseSummary directly (19/10/2026)

# def compute_anticipatory_speed(genotype: str) -> Dict[str, List[tuple[float, float]]]:
#     result: Dict[str, List[tuple[float, float]]] = {
#         "Baseline": [],
#         "Trained": [],
#     }

#     for mouse, sessions in SESSIONS_KEEP.items():
#         if not get_genotype(mouse) == genotype:
#             continue
#         print(mouse)
#         date_trained = sessions["learned"]

#         if date_trained is None:
#             continue

#         metadata = gsheet2df(SPREADSHEET_ID, mouse, 1)
#         first_session_row = metadata[metadata["Type"].str.lower() == "learning day 1"]
#         assert len(first_session_row) == 1
#         first_session_date = first_session_row["Date"].values[0]

#         for date, session_type in zip(
#             [first_session_date, date_trained], ["First Session", "Final Session"]
#         ):

#             session_numbers = metadata[metadata["Date"] == date][
#                 "Session Number"
#             ].values[0]
#             # TODO: if I am not mistaken, this was causing a bug if session number in the spreadsheet is set to e.g. "2 + 3" as this code would only look at session "2"
#             # session_path = (
#             #     BEHAVIOUR_DATA_PATH
#             #     / mouse
#             #     / date
#             #     / parse_session_number(session_number)[0]
#             # )
#             # trials = load_data(session_path)
#             session_paths = [
#                 (BEHAVIOUR_DATA_PATH / mouse / date / session_number)
#                 for session_number in parse_session_number(session_numbers)
#             ]
#             trials = list()
#             for i, session_path in enumerate(session_paths):
#                 trials.extend(load_data(session_path))
#             summaries = [
#                 summarise_trial(trial, get_wheel_circumference_from_rig("2P"))
#                 for trial in trials
#             ]

#             # TODO: this is unsatisfactory, but speed AZ can be infinite
#             # TODO: it might look weird as the basis for the AZ licking is different then!!!
#             # this solution is derived from viral.nlgf.behaviour session_behaviour()
#             # it will drop trials with infinite speed
#             rewarded = [trial.speed_AZ for trial in summaries if trial.rewarded]
#             rewarded_valid = [
#                 speed_AZ for speed_AZ in rewarded if np.isfinite(speed_AZ)
#             ]
#             not_rewarded = [trial.speed_AZ for trial in summaries if not trial.rewarded]
#             not_rewarded_valid = [
#                 speed_AZ for speed_AZ in not_rewarded if np.isfinite(speed_AZ)
#             ]

#             result_tuple = (np.mean(rewarded), np.mean(not_rewarded))

#             if session_type == "First Session":
#                 result["Baseline"].append(result_tuple)
#             else:
#                 result["Trained"].append(result_tuple)
#     return result


# hence altered function
def get_anticipatory_speed(genotype: str) -> Dict[str, List[tuple[float, float]]]:
    result: Dict[str, List[tuple[float, float]]] = {
        "Baseline": [],
        "Trained": [],
    }

    for mouse, sessions in SESSIONS_KEEP.items():
        if not get_genotype(mouse) == genotype:
            continue
        print(mouse)
        date_trained = sessions["learned"]

        if date_trained is None:
            continue

        metadata = gsheet2df(SPREADSHEET_ID, mouse, 1)
        first_session_row = metadata[metadata["Type"].str.lower() == "learning day 1"]
        assert len(first_session_row) == 1
        first_session_date = first_session_row["Date"].values[0]

        for date, session_type in zip(
            [first_session_date, date_trained], ["First Session", "Final Session"]
        ):

            session_numbers = metadata[metadata["Date"] == date][
                "Session Number"
            ].values[0]
            # TODO: if I am not mistaken, this was causing a bug if session number in the spreadsheet is set to e.g. "2 + 3" as this code would only look at session "2"
            # session_path = (
            #     BEHAVIOUR_DATA_PATH
            #     / mouse
            #     / date
            #     / parse_session_number(session_number)[0]
            # )
            # trials = load_data(session_path)
            session_paths = [
                (BEHAVIOUR_DATA_PATH / mouse / date / session_number)
                for session_number in parse_session_number(session_numbers)
            ]
            trials = list()
            for i, session_path in enumerate(session_paths):
                trials.extend(load_data(session_path))
            summaries = [
                summarise_trial(trial, get_wheel_circumference_from_rig("2P"))
                for trial in trials
            ]
            # dropping trials in which speed was infinite
            rewarded = [
                trial.speed_AZ
                for trial in summaries
                if trial.rewarded and np.isfinite(trial.speed_AZ)
            ]
            not_rewarded = [
                trial.speed_AZ
                for trial in summaries
                if not trial.rewarded and np.isfinite(trial.speed_AZ)
            ]

            # Jimmy told me to just ignore it for now
            # n_trials_dropped = len(
            #     [trial.speed_AZ for trial in summaries if np.isinf(trial.speed_AZ)]
            # )

            # print(f"Dropped {n_trials_dropped} trials")
            # # stop this if per mouse session, more than three trials are dropped
            # if n_trials_dropped > 3:
            #     1 / 0

            result_tuple = (np.mean(rewarded), np.mean(not_rewarded))

            if session_type == "First Session":
                result["Baseline"].append(result_tuple)
            else:
                result["Trained"].append(result_tuple)
    return result


def basic_anticipatory_speed_plot(genotype: str) -> None:
    data = get_anticipatory_speed(genotype=genotype)

    def make_plot(tuples: List[tuple[float, float]], ax: plt.Axes) -> None:
        means = np.mean(tuples, axis=0)
        for session in tuples:
            ax.plot([1, 0], session, color="gray", alpha=0.3)

        ax.plot([1, 0], means, color="black", linewidth=2, marker="o")

        ax.set_xlim(-0.2, 1.2)
        ax.set_xticks([0, 1], ["Unrewarded\nTexture", "Rewarded\nTexture"])
        arr = np.array(tuples)
        p_value = stats.ttest_rel(arr[:, 0], arr[:, 1]).pvalue
        ax.text(
            0.5,
            0.95,
            f"P = {p_value:.2g}",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=12,
            font="arial",
        )

    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

    make_plot(data["Baseline"], axes[0])
    make_plot(data["Trained"], axes[1])
    axes[0].set_ylabel("Mean speed in AZ (cm/s)")
    axes[1].set_title("After Learning")
    axes[0].set_title("Before Learning")
    sns.despine()
    plt.suptitle(f"Anticipatory Speed in {genotype} mice")
    plt.tight_layout()

    plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        HERE.parent / "plots" / f"basic_anticipatory_speed_plot_{genotype}.pdf",
        bbox_inches="tight",
        transparent=True,
    )


# got this from viral.presentations.2025-11-05_basel_meeting_behaviour_plots (commit ), which was derived from viral.multiple_sessions
def plot_performance_summaries(
    mice: List[MouseSummary],
    session_type: Literal["learning", "reversal", "recall", "recall_reversal"],
    group_by: list[str],
    config: MultipleSessionsConfig,
) -> Dict[str, float]:
    rolling_performance_dict = create_metric_dict(
        mice,
        rolling_performance,
        config,
        True,
        False,
    )
    to_plot: Dict[str, list] = dict()

    print(f"Analyzing {session_type} sessions")
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
        session_data = np.array(data[session_type])
        if session_data.size == 0:
            print(
                f"{session_type} session data empty for {mouse.name} ({mouse.genotype})"
            )
        elif np.all(np.isnan(session_data)):
            print(
                f"{session_type} session data is all NaN for {mouse.name} ({mouse.genotype})"
            )
        elif np.all(session_data < 1):
            print(
                f"{mouse.name} never reached criterion in {session_type} ({mouse.genotype})"
            )
        else:
            first_threshold_idx = np.where(session_data > 1)[0][0] + config.window
            to_plot[group_label].append(first_threshold_idx)
            print(f"mouse {mouse.name} ({mouse.genotype}) used")

    colors = sns.color_palette(n_colors=3)
    palette = {"WT": colors[0], "NLGF": colors[1], "Oligo-BACE1-KO": colors[2]}

    genotype_order = ["WT", "NLGF", "Oligo-BACE1-KO"]
    to_plot = {g: to_plot[g] for g in genotype_order}

    plt.ylabel("Trials to criterion")
    # add a warning that the Holm-adjusted p values should not be used to draw conclusions in stages other than learning
    # since we would have to decide on the questions asked and would have to decide on the Holm family
    plt.title(
        f"{session_type.replace("_", " ").capitalize()}{' (exploratory)' if session_type != "learning" else ''}"
    )
    sns.boxplot(
        to_plot,
        showfliers=False,
        hue_order=["WT", "NLGF", "Oligo-BACE1-KO"],
        palette=palette,
    )
    ax = plt.gca()
    ax.set_xticklabels(
        [
            f"{g.replace('Oligo-BACE1-KO', 'Oligo-' + chr(10) + 'BACE1-KO')}\n(n = {len(to_plot[g])})"
            for g in to_plot
        ]
    )
    sns.stripplot(
        to_plot,
        edgecolor="black",
        linewidth=1,
        palette=palette,
        hue_order=["WT", "NLGF", "Oligo-BACE1-KO"],
    )

    # what Claude suggested (19/10/2026, Opus 5.5 Medium)
    # doing a Mann Whitney U test instead
    # two-sided: tests if there is a difference between the two genotypes, we don't test for the direction specifically, e.g. "Are NLGFs slower than WTs?"
    # permutationmethod helps if ties occur, i.e. if two mice needed the exact same number of trials to criterion
    nlgf_vs_wt = stats.mannwhitneyu(
        to_plot["NLGF"],
        to_plot["WT"],
        alternative="two-sided",
        method=stats.PermutationMethod(random_state=0),
    )

    oligo_vs_nlgf = stats.mannwhitneyu(
        to_plot["Oligo-BACE1-KO"],
        to_plot["NLGF"],
        alternative="two-sided",
        method=stats.PermutationMethod(random_state=0),
    )

    # what Claude suggested (19/10/2026, Opus 5.5 Medium)
    # holm-adjust the pvalues
    p_nlgf_vs_wt, p_oligo_vs_nlgf = holm(ps=[nlgf_vs_wt.pvalue, oligo_vs_nlgf.pvalue])

    sns.despine()
    plt.tight_layout()
    return {"nlgf_vs_wt": p_nlgf_vs_wt, "oligo_vs_nlgf": p_oligo_vs_nlgf}


# got this from viral.presentations.2025-11-05_basel_meeting_behaviour_plots (it was the __name__ == "__main__" there)
def plot_behaviour(
    mice: List[MouseSummary],
    stage: Literal["learning", "reversal", "recall", "recall_reversal"],
) -> None:
    def group_of(text: str) -> str | None:
        t = text.lower().replace("-", "").replace("_", "")
        if "oligo" in t:
            return "Oligo-BACE1-KO"
        elif "nlgf" in t:
            return "NLGF"
        elif "wt" in t:
            return "WT"
        else:
            raise ValueError("Unknown/invalid group")
        # if text == "nlgf_vs_wt":
        #     return "NLGF"
        # elif text == "oligo_vs_nlgf":
        #     return "Oligo-BACE1-KO"
        # else:
        #     raise ValueError("Unknown/invalid group comparison")

    def format_p(p) -> str:
        p = float(p)
        if np.isnan(p):
            return "n/a"
        return "P < 0.001" if p < 0.001 else f"P = {p:.3f}"

    def add_brackets(ax, p_dict, y_start, step, inset=0.05):
        tick = step * 0.25
        pos = {
            group_of(lbl.get_text()): x
            for x, lbl in zip(ax.get_xticks(), ax.get_xticklabels())
        }
        items = []
        for key, p in p_dict.items():
            a, b = (group_of(s) for s in key.split("_vs_"))
            if a in pos and b in pos:
                lo, hi = sorted((pos[a], pos[b]))
                items.append((hi - lo, lo + inset, hi - inset, p))
        y = y_start
        for _, x1, x2, p in sorted(items):
            ax.plot(
                [x1, x1, x2, x2],
                [y, y + tick, y + tick, y],
                color="black",
                linewidth=2,
                clip_on=False,
            )
            ax.text(
                (x1 + x2) / 2,
                y + tick * 1.3,
                format_p(p),
                ha="center",
                va="bottom",
                fontsize=16,
            )
            y += 0  # adjacent comparisons share one height; see note below

    # one big axes, created BEFORE plotting so plot_performance_summaries draws into it
    fig, ax = plt.subplots(figsize=(10, 9))
    p_values = plot_performance_summaries(mice, stage, ["genotype"], config=CONFIG)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    rng = ymax - ymin
    add_brackets(ax, p_values, y_start=ymax + rng * 0.03, step=rng * 0.1)

    ax.tick_params(labelsize=16)
    ax.set_xlabel("")
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    fig.text(
        0.5,
        0.01,
        "Two-sided Mann-Whitney U (permutation p),\nHolm-corrected within stage (2 comparisons)",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        HERE.parent
        / "plots"
        / "behaviour_summaries"
        / f"behaviour_summaries_comparison_plot_stage-{stage}.pdf"
    )
    plt.show()


# took this from viral.multiple_sessions.py plot_performance_summaries()
# had to take it and alter as it wouldn't work with any group_by other than just genotype and I wanted to have sex and setup as well
def plot_performance_summaries_confounds(
    mice: List[MouseSummary],
    session_type: Literal["learning", "reversal", "recall", "recall_reversal"],
    group_by: list[str],
    config: MultipleSessionsConfig,
) -> None:

    GENOTYPE_COLOURS

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
    plt.figure(figsize=(max(6, 1.2 * len(to_plot)), 6), layout="constrained")
    plt.ylabel("Trials to criterion")
    plt.title(session_type.replace("_", " ").capitalize())
    sns.boxplot(
        to_plot,
        showfliers=False,
        # palette=GENOTYPE_COLOURS,
        # hue_order=["WT", "NLGF", "Oligo-BACE1-KO"],
    )
    ax = plt.gca()
    new_labels = [
        label.get_text()
        .replace("Oligo-BACE1-KO", "Oligo-\nBACE1-KO")
        .replace("_", "\n")
        for label in ax.get_xticklabels()
    ]
    ax.set_xticklabels(new_labels)
    ax.tick_params(axis="x", labelsize=9)
    sns.stripplot(
        to_plot,
        edgecolor="black",
        linewidth=1,
        # palette=GENOTYPE_COLOURS,
        # hue_order=["WT", "NLGF", "Oligo-BACE1-KO"],
    )

    sns.despine()
    # plt.tight_layout()
    group_suffix = "-".join(group_by)

    for extension in ["png", "pdf"]:
        plt.savefig(
            HERE.parent
            / "plots"
            / "behaviour_summaries_by_confound"
            / f"behaviour_summaries_{group_suffix}-{session_type}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


if __name__ == "__main__":
    CONFIG = MultipleSessionsConfig(speed=0.5, licking=0.5, window=50)
    redo = False
    mice: List[MouseSummary] = list()
    # got this list from viral.utils get_genotype
    # explicitly used all mice (also behaviour-only), dumped the neuronal BACE 1 KO because there are no of mine
    # CAVE: there are 5 mice I didnt experiment on ("J030", "J031", "J032", "J034", "J035")
    mouse_list = [
        "JB014",
        "JB015",
        "JB018",
        "JB020",
        "JB022",
        "JB011",
        "JB012",
        "JB013",
        "JB016",
        "JB017",
        "JB019",
        "JB021",
        "JB023",
        "JB036",
        "J030",
        "J031",
        "J032",
        "JB024",
        "JB025",
        "JB026",
        "JB027",
        "JB030",
        "JB031",
        "JB032",
        "JB033",
        "J034",
        "J035",
    ]

    for mouse_name in mouse_list:
        if get_genotype(mouse_name) not in {"WT", "NLGF", "Oligo-BACE1-KO"}:
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

    # CAUTION: this is an all imaging mice from get_genotype as you got these functions from Jimmy's code
    # CAUTION: the trial basis for licking and speed is not shared as for the speed plots, trials with infinite speed are filtered out
    # Jimmy said it would be fine
    # Claude said:
    # "Your plots each show one measure on its own: lick % before and after learning, and speed before and after learning.
    # No statistic combines them, so you don't need a shared trial set."
    basic_anticipatory_licking_plot(genotype="WT")
    basic_anticipatory_licking_plot(genotype="NLGF")
    basic_anticipatory_licking_plot(genotype="Oligo-BACE1-KO")
    basic_anticipatory_speed_plot(genotype="WT")
    basic_anticipatory_speed_plot(genotype="NLGF")
    basic_anticipatory_speed_plot(genotype="Oligo-BACE1-KO")

    # CAUTION: this is now using the mouse list
    plot_behaviour(mice=mice, stage="learning")
    plot_behaviour(mice=mice, stage="reversal")
    plot_behaviour(mice=mice, stage="recall")
    plot_behaviour(mice=mice, stage="recall_reversal")

    plot_performance_summaries_confounds(
        mice=mice,
        session_type="learning",
        group_by=["genotype", "sex"],
        config=CONFIG,
    )
    plot_performance_summaries_confounds(
        mice=mice,
        session_type="reversal",
        group_by=["genotype", "sex"],
        config=CONFIG,
    )
    plot_performance_summaries_confounds(
        mice=mice,
        session_type="recall",
        group_by=["genotype", "sex"],
        config=CONFIG,
    )
    plot_performance_summaries_confounds(
        mice=mice,
        session_type="recall_reversal",
        group_by=["genotype", "sex"],
        config=CONFIG,
    )

    plot_performance_summaries_confounds(
        mice=mice,
        session_type="learning",
        group_by=["setup"],
        config=CONFIG,
    )
    plot_performance_summaries_confounds(
        mice=mice,
        session_type="reversal",
        group_by=["setup"],
        config=CONFIG,
    )
    plot_performance_summaries_confounds(
        mice=mice,
        session_type="recall",
        group_by=["setup"],
        config=CONFIG,
    )
    plot_performance_summaries_confounds(
        mice=mice,
        session_type="recall_reversal",
        group_by=["setup"],
        config=CONFIG,
    )
