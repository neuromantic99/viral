"""
AI assistance:
Claude Opus 5.5 (medium effort), used for code suggestions and for
discussing analysis choices. All code was reviewed and checked by JB.
List not exhaustive:
- 19/10/2026: statistical tests in statistics_behavioural_performance
- 27-28/10/2026 and earlier (partly via 26-10_meeting_JB.py): parts of plot_behaviour and plot_performance_summaries_per_genotype (AI assistance was not recorded as thoroughly)
- 28/10/2026: pick_example (criterion check, median-based selection), figure saving and session lookup in plot_example_licking_and_running, tests.test_multiple_sessions.test_rolling_performance, more minor changes in the entire document,
help with adjusting plot_speed_reward_unrewarded
- 29/10/2026: help with fixing the SEM issue in viral.utils.shaded_line_plot, help with figuring out why speed can get infinite, helped make licking plots prettier
Substantial AI-written passages are marked inline.
Some git commit hashes might become out of date after this branch is merged.
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from typing import Dict, List, Literal, Tuple
from scipy import stats
from matplotlib import pyplot as plt
from pydantic import ValidationError
import numpy as np
import seaborn as sns
import csv
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
    shaded_line_plot,
    degrees_to_cm,
    get_speed_positions,
)
from viral.single_session import remove_bad_trials, plot_rewarded_vs_unrewarded_licking
from viral.multiple_sessions import (
    load_cache,
    parse_session_number,
    cache_mouse,
    create_metric_dict,
    load_cache,
    rolling_performance,
    learning_metric,
    filter_sessions_by_session_type,
)
from viral.nlgf.three_group import holm
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, SHORT_NAME
from viral.models import (
    SpeedPosition,
    TrialInfo,
    SessionSummary,
    MouseSummary,
    MultipleSessionsConfig,
)

plt.rcParams["pdf.fonttype"] = 42

# depending on which comparisons one wants to make, statistics have to be adjusted (one test/multiple tests, corrections/no corrections)
# TODO: you could obviously add the neuronal BACE1 knockout mice as well
COMPARISONS: Dict[str, Dict] = {
    "nlgf_vs_wt": {
        "genotypes": ["WT", "NLGF"],  # genotypes involved
        "tests": [("NLGF", "WT")],  # tests needed (tuple of type (a, b))
    },
    "nlgf_vs_wt_and_oligo_vs_nlgf": {
        "genotypes": ["WT", "NLGF", "Oligo-BACE1-KO"],
        "tests": [("NLGF", "WT"), ("Oligo-BACE1-KO", "NLGF")],
    },
}
short_name_for_key = {"WT": "wt", "NLGF": "nlgf", "Oligo-BACE1-KO": "oligo"}


def statistics_behavioural_performance(
    to_plot: Dict, comparison: Literal["nlgf_vs_wt", "nlgf_vs_wt_and_oligo_vs_nlgf"]
) -> Dict:
    """
    Performs Mann Whitney U test(s).
    If there is just one test in the comparison, p_values are returned uncorrected. If n_tests >= 2, p_values are Holm-corrected.
    """
    # TODO: Probably need to rethink the correction method if indeed we do more than two tests
    # what Claude suggested (19/10/2026, Opus 5.5 Medium)
    # doing a Mann Whitney U test instead
    # two-sided: tests if there is a difference between the two genotypes, we don't test for the direction specifically, e.g. "Are NLGFs slower than WTs?"
    # permutationmethod helps if ties occur, i.e. if two mice needed the exact same number of trials to criterion
    tests = COMPARISONS[comparison]["tests"]
    p_values_raw = dict()
    for a, b in tests:
        test_name = f"{short_name_for_key[a]}_vs_{short_name_for_key[b]}"
        p_values_raw[test_name] = stats.mannwhitneyu(
            to_plot[a],
            to_plot[b],
            alternative="two-sided",
            method=stats.PermutationMethod(random_state=0),
        ).pvalue
    # what Claude suggested (19/10/2026, Opus 5.5 Medium)
    # holm-adjust the pvalues
    if len(tests) >= 2:
        adjusted = holm(ps=list(p_values_raw.values()))
        return {name: p for name, p in zip(p_values_raw, adjusted)}
    return p_values_raw


# got this from viral.presentations.26-10_meeting_JB.py (used Claude as well) and altered it with Claude's help (before, was up-to-date with commit 3780eb2281e9cb31339cbf523d4f29088347baf3), which was derived from viral.presentations.2025-11-05_basel_meeting_behaviour_plots, which was derived from viral.multiple_sessions
def plot_performance_summaries_per_genotype(
    mice: List[MouseSummary],
    session_type: Literal["learning", "reversal", "recall", "recall_reversal"],
    config: MultipleSessionsConfig,
    comparison: Literal["nlgf_vs_wt", "nlgf_vs_wt_and_oligo_vs_nlgf"],
) -> Dict[str, float]:
    genotypes_used = COMPARISONS[comparison]["genotypes"]
    genotype_order = [g for g in GENOTYPE_ORDER if g in genotypes_used]
    mice = [mouse for mouse in mice if get_genotype(mouse.name) in genotype_order]
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
        genotype = get_genotype(mouse.name)
        if genotype not in to_plot:
            to_plot[genotype] = list()
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
            to_plot[genotype].append(first_threshold_idx)
            print(f"mouse {mouse.name} ({mouse.genotype}) used")

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
        hue_order=genotype_order,
        palette=GENOTYPE_COLOURS,
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
        palette=GENOTYPE_COLOURS,
        hue_order=genotype_order,
    )

    sns.despine()
    plt.tight_layout()

    p_values = statistics_behavioural_performance(
        to_plot=to_plot, comparison=comparison
    )

    return p_values


# got this from viral.presentations.26-10_meeting_JB.py (used Claude as well, was up-to-date with commit 3780eb2281e9cb31339cbf523d4f29088347baf3),
# which got it from viral.presentations.2025-11-05_basel_meeting_behaviour_plots (before, was up-to-date with commit 683071e5f0ec7de86ad71890247fc3b23ecd71cb, it was the __name__ == "__main__" there)
def plot_behaviour(
    mice: List[MouseSummary],
    stage: Literal["learning", "reversal", "recall", "recall_reversal"],
    config: MultipleSessionsConfig,
    comparison: Literal["nlgf_vs_wt", "nlgf_vs_wt_and_oligo_vs_nlgf"],
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
    p_values = plot_performance_summaries_per_genotype(
        mice, stage, config=config, comparison=comparison
    )
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    rng = ymax - ymin
    add_brackets(ax, p_values, y_start=ymax + rng * 0.03, step=rng * 0.1)

    ax.tick_params(labelsize=16)
    ax.set_xlabel("")
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    n_tests = len(COMPARISONS[comparison]["tests"])
    if n_tests >= 2:
        fig_text = f"Two-sided Mann-Whitney U (permutation p),\nHolm-corrected within stage ({n_tests} comparisons)"
    else:
        fig_text = "Two-sided Mann-Whitney U (permutation p)"
    fig.text(
        0.5,
        0.01,
        fig_text,
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
        / f"behaviour_summaries_{stage}_{comparison}.pdf"
    )
    plt.show()


# had to take this from viral.single_session rather than importing it as the function was using the MOUSE_NAME constant (before editing, was up-to-date with commit a0b8d5b100aa0af48c9c8bfac8801d0addcfdb6a)
# TODO: due to infinite speed in utils.get_speed_positions, there are gaps in the speed in the reward zone, needs dealing with
def plot_speed_reward_unrewarded(
    trials: List[TrialInfo], sampling_rate: int, wheel_circumference: float
) -> None:
    plt.figure(figsize=(10, 6))
    rewarded: List[List[SpeedPosition]] = []
    not_rewarded: List[List[SpeedPosition]] = []

    prev_rewarded = []
    prev_unrewarded = []

    first_position = 0
    # 28/10/2026 JB: in accordance with JR, I decided to plot speed up until 180cm only
    # last_position = 200
    last_position = 180
    step_size = 5

    for idx, trial in enumerate(trials):
        position = degrees_to_cm(
            np.array(trial.rotary_encoder_position),
            wheel_circumference=wheel_circumference,
        )

        speed = get_speed_positions(
            position=position,
            first_position=first_position,
            last_position=last_position,
            step_size=step_size,
            sampling_rate=sampling_rate,
        )
        for bin_idx in range(len(speed)):
            if np.isinf(speed[bin_idx].speed):
                if speed[bin_idx].position_start >= 180:
                    continue  # Doesn't matter
                print(
                    f"Got inf speed for position {speed[bin_idx].position_stop} in trial {idx}"
                )
                speed[bin_idx].speed = np.mean(
                    [bin.speed for bin in speed if not np.isinf(bin.speed)]
                )

        if trial.texture_rewarded:
            rewarded.append(speed)
        else:
            not_rewarded.append(speed)

        if trials[idx - 1].texture_rewarded:
            prev_rewarded.append(speed)
        else:
            prev_unrewarded.append(speed)

    # Should be the same for all trials
    # Use the bin_stop so there is no forward look ahead
    # plt.axvspan(180, 200, color="gray", alpha=0.5, zorder=-1)
    plt.axvspan(150, 180, color="lightgray", alpha=0.5, zorder=-1)
    # plt.xlim(0, 200)
    plt.xlim(0, 180)
    x_axis = [bin.position_stop for bin in rewarded[0]]
    shaded_line_plot(
        np.array([[bin.speed for bin in trial] for trial in rewarded]),
        x_axis,
        "green",
        "Rewarded",
    )

    shaded_line_plot(
        np.array([[bin.speed for bin in trial] for trial in not_rewarded]),
        x_axis,
        "red",
        "Unrewarded",
    )

    plt.xticks([0, 50, 100, 150, 180])
    plt.legend()
    plt.ylim(0, None)
    plt.xlabel("Distance (cm)")
    plt.ylabel("Speed (cm / s)")

    # Snippet by Claude Opus 5.5 (medium effort) 29/10/2026 to inspect the infinite speed issue
    # --start
    # Claude's reasoning:it is the state machine's sampling (position sampling lagging behind after entering the reward zone)
    # 29/10/2026 JB: in accordance with JR, I decided to plot speed up until 180cm only
    # with open(HERE.parent / "data" / "speed_debug.csv", "w", newline="") as f:
    #     w = csv.writer(f)
    #     w.writerow(["trial", "rewarded", "sample", "state", "time_s", "position_cm"])
    #     for idx, trial in enumerate(trials):  # trials after remove_bad_trials
    #         position = degrees_to_cm(
    #             np.array(trial.rotary_encoder_position), wheel_circumference
    #         )
    #         states = [
    #             s
    #             for s in trial.states_info
    #             if s.name
    #             in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
    #         ]
    #         for i, pos in enumerate(position):
    #             state = states[i] if i < len(states) else None
    #             w.writerow(
    #                 [
    #                     idx,
    #                     trial.texture_rewarded,
    #                     i,
    #                     state.name if state else "",
    #                     state.end_time if state else "",
    #                     round(pos, 3),
    #                 ]
    #             )
    # --end


def pick_example(
    mouse_list: List[str], config: MultipleSessionsConfig
) -> Tuple[str, SessionSummary]:
    """
    Decide on which mouse to plot for the example:
    1. Only look at wild-type mice
    2. Look at each mouse's last learning session
    3. Check that this session was above criterion
    5. Then check which mouse's last learning session is closest to the group median
    (if there is a tie, the session and thus mouse with more trials wins)
    6. Return the chosen mouse's last learning session to be plotted
    """
    # 28/10/2026 JB: structure and criterion check developed with Claude; median-selection block copied (marked below)
    # get wild-type mice's names from the mouse_list
    wt_mice = [
        mouse_name
        for mouse_name in mouse_list
        if get_genotype(mouse_name=mouse_name) == "WT"
    ]

    # get last learning sessions (mouse name: session)
    candidate_sessions = dict()
    for wt_mouse in wt_mice:
        learning_sessions = filter_sessions_by_session_type(
            mouse=load_cache(wt_mouse), session_type="learning"
        )
        last_learning_session = learning_sessions[-1]
        candidate_sessions[wt_mouse] = last_learning_session
    print(f"{len(candidate_sessions)} candidate sessions")

    # exploratory check to determine the above criterion check downstream
    print(f"window = {config.window}")
    cand_sess_lens = list()
    for wt_mouse in candidate_sessions.keys():
        candidate_session = candidate_sessions[wt_mouse]
        n_trials = candidate_session.num_trials
        print(f"{wt_mouse}: number of trials {n_trials}")
        cand_sess_lens.append(n_trials)
        n_cand_sess_short = (
            np.array(cand_sess_lens) <= config.window
        ).sum()  # rolling_performance is off by one (see tests.multiple_sessions.test_rolling_performance)
    print(
        f"fraction of candidate sessions with fewer trials than or equal to the criterion window of {config.window}: {n_cand_sess_short/len(candidate_sessions)}"
    )
    print(
        f"range of n_trials within candidate sessions: {min(cand_sess_lens)} - {max(cand_sess_lens)}"
    )
    # 28/10/2026 JB:
    # 10 candidate sessions
    # window = 50
    # JB024: number of trials 48
    # JB025: number of trials 43
    # JB026: number of trials 49
    # JB027: number of trials 51
    # JB030: number of trials 61
    # JB031: number of trials 75
    # JB032: number of trials 47
    # JB033: number of trials 47
    # J034: number of trials 45
    # J035: number of trials 65
    # fraction of candidate sessions with fewer trials than or equal to the criterion window of 50: 0.6
    # range of n_trials within candidate sessions: 43 - 75
    # with the n_trials being close to the current window size, I decided to look at the entire session instead (roughly fits one window)
    # an n_trials length check would arbitrarily cut off mice at or below the window size (e.g. JB026: 49) since they would receive no windows at all

    # candidate_sessions_above_min_length = dict()
    # for wt_mouse in candidate_sessions.keys():
    #     candidate_session = candidate_sessions[wt_mouse]
    #     n_trials = candidate_session.num_trials
    #     print(f"{wt_mouse}: number of trials {n_trials}")
    #     if (
    #         n_trials <= config.window
    #     ):  # rolling_performance is off by one (see tests.multiple_sessions.test_rolling_performance)
    #         print("Dropped candidate session (too few trials)")
    #     else:
    #         candidate_sessions_above_min_length[wt_mouse] = candidate_session
    # print(
    #     f"{len(candidate_sessions_above_min_length)} candidate sessions after length check"
    # )

    # check session was above criterion
    # trying to match the criterion mechanics in viral.presentations.26-11_poster_JB.plot_performance_summaries
    # decided against a check whether criterion > 1 happened in a session before the candidate session (sometimes, some mice only got there in the session before we switched stage)
    # so this checks if every rolling window in the candidate session is above criterion
    candidate_sessions_above_criterion = dict()
    learning_matrices = list()
    for wt_mouse in candidate_sessions.keys():
        candidate_session = candidate_sessions[wt_mouse]
        behavioural_performance = np.array(
            learning_metric(candidate_session.trials, config=config)
        )  # learning metric is not using windows, boils it down to one number per day
        above_criterion = np.all(behavioural_performance > 1)
        if above_criterion:
            candidate_sessions_above_criterion[wt_mouse] = {
                "session": candidate_session,
                "learning_metric": behavioural_performance,
            }
            learning_matrices.append(behavioural_performance)
        else:
            raise ValueError("Session not above criterion")

    # the following passage was copied from Claude Opus 5.5 Medium on 28/10/2026 with only minor adjustments
    # -- beginning
    group_median = np.median(learning_matrices)
    print(
        f"group median of the learning metric (n = {len(learning_matrices)} mice): {group_median:.2f}"
    )

    # distance of each mouse's learning metric to the group median, closest first
    distances = {
        wt_mouse: abs(float(entry["learning_metric"]) - group_median)
        for wt_mouse, entry in candidate_sessions_above_criterion.items()
    }

    # 28/10/2026 JB:
    # ties can happen regularly
    # to keep things simple, the tie is broken by the mouse/session with more trials (yes, even though it might be confounded)
    ranked_mice = sorted(
        distances,
        key=lambda m: (
            round(distances[m], 9),
            -candidate_sessions_above_criterion[m]["session"].num_trials,
        ),
    )

    ranked_distances = list()
    print("three mice closest to the group median:")
    for rank, wt_mouse in enumerate(ranked_mice[:3], start=1):
        entry = candidate_sessions_above_criterion[wt_mouse]
        ranked_distances.append(distances[wt_mouse])
        print(
            f"  {rank}. {wt_mouse} '{entry['session'].name}': "
            f"learning metric = {float(entry['learning_metric']):.2f}, "
            f"distance to median = {distances[wt_mouse]:.6f}, "
            f"n_trials = {entry['session'].num_trials}"
        )

    chosen_mouse = ranked_mice[0]
    print(
        f"Session chosen: {chosen_mouse} - {candidate_sessions_above_criterion[chosen_mouse]["session"].name}"
    )
    return chosen_mouse, candidate_sessions_above_criterion[chosen_mouse]["session"]
    # -- end


# got this from viral/single_session.py main() function (before, up-to-date with commit a0b8d5b100aa0af48c9c8bfac8801d0addcfdb6a)
def plot_example_licking_and_running(
    mouse_list: List[str], config: MultipleSessionsConfig
) -> None:
    # TODO: got a tie
    # 28/10/2026 JB:
    # group median of the learning metric (n = 10 mice): 1.72
    # three mice closest to the group median:
    # 1. JB030 'Learning day 5': learning metric = 1.87, distance to median = 0.15, n_trials = 61
    # 2. J035 'Learning day 3': learning metric = 1.56, distance to median = 0.15, n_trials = 65
    # 3. JB027 'Learning day 5': learning metric = 1.56, distance to median = 0.15, n_trials = 51
    # Session chosen: JB030 - Learning day 5
    mouse_name, session = pick_example(mouse_list=mouse_list, config=config)
    metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)

    date = metadata[metadata["Type"] == session.name]["Date"].values[0]
    session_number_raw = metadata[metadata["Date"] == date]["Session Number"].values[0]
    session_number = parse_session_number(session_number_raw)
    if len(session_number) > 1:
        raise ValueError("More than one session number in the spreadsheet")
    else:
        session_number = session_number[0]

    session_path = BEHAVIOUR_DATA_PATH / mouse_name / date / session_number

    trials = load_data(session_path=session_path)
    try:
        rig = metadata[metadata["Date"] == date]["Rig"].values[0]
    except IndexError as e:
        raise ValueError("You need to fill out the rig in the spreadsheet m8") from e

    wheel_circumference = get_wheel_circumference_from_rig(rig)
    print(f"Wheel circumference: {wheel_circumference}")

    # Is habituation
    if not trials[0].texture:
        raise ValueError("You are trying to plot habituation!")

    # Is task
    else:
        total_number_trials = len(trials)
        print(f"Number of trials: {total_number_trials}")
        trials = remove_bad_trials(trials, wheel_circumference=wheel_circumference)
        print(f"Number of after bad removal: {len(trials)}")
        print(
            f"Percent Timed Out: {(total_number_trials -  len(trials) ) / total_number_trials}"
        )

    # figure saving added with Claude (28/10/2026)
    plot_directory = HERE.parent / "plots" / "example_licking_and_running"
    plot_directory.mkdir(parents=True, exist_ok=True)
    plot_names = [
        "lick_raster_rewarded",
        "lick_raster_unrewarded",
        "anticipatory_licking",
        "speed_rewarded_vs_unrewarded",
    ]
    fig_nums_before = set(plt.get_fignums())
    plot_rewarded_vs_unrewarded_licking(trials, wheel_circumference=wheel_circumference)
    plot_speed_reward_unrewarded(
        trials, sampling_rate=30, wheel_circumference=wheel_circumference
    )
    new_figs = [n for n in plt.get_fignums() if n not in fig_nums_before]
    for fig_name, fig_num in zip(plot_names, new_figs):
        plt.figure(fig_num).savefig(
            plot_directory / f"{mouse_name}_{date}_{fig_name}.pdf"
        )
    plt.show()


if __name__ == "__main__":
    # TODO: decide on which mice to use: all or just imaged mice!

    CONFIG = MultipleSessionsConfig(speed=0.5, licking=0.5, window=50)
    # this is the comparison used downstream
    COMPARISON: Literal["nlgf_vs_wt", "nlgf_vs_wt_and_oligo_vs_nlgf"] = (
        "nlgf_vs_wt_and_oligo_vs_nlgf"
    )

    # All of this code is from "26-10_meeting_JB.py"
    redo = False
    mice: List[MouseSummary] = list()

    # got this list from viral.utils get_genotype
    # explicitly used all mice (also behaviour-only), dumped the neuronal BACE 1 KO because there are none of mine
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

    # pick_example(mouse_list=mouse_list, config=CONFIG)
    plot_example_licking_and_running(mouse_list=mouse_list, config=CONFIG)

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
    # basic_anticipatory_licking_plot(genotype="WT")
    # basic_anticipatory_licking_plot(genotype="NLGF")
    # basic_anticipatory_licking_plot(genotype="Oligo-BACE1-KO")
    # basic_anticipatory_speed_plot(genotype="WT")
    # basic_anticipatory_speed_plot(genotype="NLGF")
    # basic_anticipatory_speed_plot(genotype="Oligo-BACE1-KO")

    # CAUTION: this is now using the mouse list
    # plot_behaviour(mice=mice, stage="learning", config=CONFIG, comparison=COMPARISON)
    # plot_behaviour(mice=mice, stage="reversal", config=CONFIG, comparison=COMPARISON)
    # plot_behaviour(mice=mice, stage="recall", config=CONFIG, comparison=COMPARISON)
    # plot_behaviour(
    #     mice=mice, stage="recall_reversal", config=CONFIG, comparison=COMPARISON
    # )
