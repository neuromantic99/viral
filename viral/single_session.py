import sys
from pathlib import Path

# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind

from viral.constants import DATA_PATH, ENCODER_TICKS_PER_TURN, WHEEL_CIRCUMFERENCE
from viral.models import SpeedPosition, TrialInfo, TrialSummary
from viral.utils import (
    degrees_to_cm,
    get_speed_positions,
    licks_to_position,
    shaded_line_plot,
)

sns.set_theme(context="talk", style="ticks")

MOUSE = "J015"
DATE = "2024-06-06"
SESSION_NUMBER = "001"
SESSION_PATH = DATA_PATH / MOUSE / DATE / SESSION_NUMBER


def load_data(session_path: Path) -> List[TrialInfo]:
    trial_files = list(session_path.glob("trial*.json"))
    if not trial_files:
        raise FileNotFoundError(f"No trial files found in path {session_path}")
    trials: List[TrialInfo] = []
    for trial_file in trial_files:
        with open(trial_file) as f:
            trials.append(TrialInfo.model_validate_json(f.read()))
    return trials


def plot_lick_raster(
    lick_positions: List[np.ndarray[float]],
    title: str,
    rolling_y_lim: float | None = None,
    jitter: float = 0.0,
    x_label: str = "Position (cm)",
) -> float | None:

    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    f.suptitle(title)
    # f.suptitle(f"{title}. Number of trials: {len(lick_positions)}")

    for idx, lick_trial in enumerate(lick_positions):
        a0.scatter(
            lick_trial,
            [idx + random.random() * jitter for _ in range(len(lick_trial))],
            marker=".",
            c="black",
        )

    all_trials = np.concatenate(lick_positions)
    if len(all_trials) == 0:
        return None

    a0.set_ylabel("Trial number")
    a0.set_ylim(-1, len(lick_positions))
    a0.axvspan(180, 200, color="gray", alpha=0.5)
    a0.set_xlim(0, max(all_trials) + 0.1 * max(all_trials))
    bins = np.arange(0, 200, 5)
    n, _, _ = a1.hist(all_trials, bins)
    a1.set_ylabel("Total # licks")
    a1.set_xlabel(x_label)
    a1.set_ylim(0, rolling_y_lim)
    a1.set_xlim(0, 200)
    a1.axvspan(180, 200, color="gray", alpha=0.5)
    return max(n)


def get_anticipatory_licking(lick_positions: np.ndarray[float]) -> int:
    return np.sum(np.logical_and(lick_positions > 80, lick_positions < 180))


def plot_trial_length(trials: List[TrialInfo]):
    plt.figure()
    sns.boxplot(
        {
            "rewarded": [
                trial.trial_end_time - trial.trial_start_time
                for trial in trials
                if trial.texture_rewarded
            ],
            "unrewarded": [
                trial.trial_end_time - trial.trial_start_time
                for trial in trials
                if not trial.texture_rewarded
            ],
        },
        # inner="point",
    )
    plt.show()


def get_percent_timedout(trials: List[TrialInfo]) -> Dict[str, float]:
    rewarded = [trial for trial in trials if trial.texture_rewarded]
    unrewarded = [trial for trial in trials if not trial.texture_rewarded]

    return {
        "rewarded": round(
            sum(
                trial.trial_end_time - trial.trial_start_time > 120
                for trial in rewarded
            )
            / len(rewarded)
            * 100,
            2,
        ),
        "unrewarded": round(
            sum(
                trial.trial_end_time - trial.trial_start_time > 120
                for trial in unrewarded
            )
            / len(unrewarded)
            * 100,
            2,
        ),
    }


def disparity(trials: List[TrialInfo]) -> None:
    for trial in trials:
        if degrees_to_cm(trial.rotary_encoder_position[-1]) < 180:
            print(degrees_to_cm(trial.rotary_encoder_position[-1]))
            print(trial.trial_end_time - trial.trial_start_time)
            print("\n")


def plot_previous_trial_dependent_licking(trials: List[TrialInfo]) -> None:
    prev_rewarded = [
        licks_to_position(trials[idx])
        for idx in range(len(trials))
        if trials[idx - 1].texture_rewarded
    ]
    prev_unrewarded = [
        licks_to_position(trials[idx])
        for idx in range(len(trials))
        if not trials[idx - 1].texture_rewarded
    ]
    jitter = 0
    y_max = plot_lick_raster(prev_rewarded, "prev_rewaredd", None, jitter=jitter)
    plot_lick_raster(prev_unrewarded, "prev_unrewarded", None, jitter=jitter)

    plt.figure()
    plt.title("Anticipatory licking")
    sns.boxplot(
        {
            "rewarded": [get_anticipatory_licking(trial) for trial in prev_rewarded],
            "unrewarded": [
                get_anticipatory_licking(trial) for trial in prev_unrewarded
            ],
        }
    )
    plt.show()


def plot_licking_all_trials(trials: List[TrialInfo]) -> None:
    plot_lick_raster(
        [licks_to_position(trial) for trial in trials],
        "All trials",
        None,
        jitter=0,
    )
    plt.show()


def plot_rewarded_vs_unrewarded_licking(trials: List[TrialInfo]) -> None:
    rewarded = [licks_to_position(trial) for trial in trials if trial.texture_rewarded]
    unrewarded = [
        licks_to_position(trial) for trial in trials if not trial.texture_rewarded
    ]

    jitter = 0.2
    y_max = plot_lick_raster(rewarded, "Rewarded Trials", None, jitter=jitter)
    plot_lick_raster(unrewarded, "Unrewarded Trials", y_max, jitter=jitter)

    plt.figure()
    plt.title("Anticipatory licking")

    rewarded_anticipatory = [get_anticipatory_licking(trial) for trial in rewarded]
    unrewarded_anticipatory = [get_anticipatory_licking(trial) for trial in unrewarded]

    sns.boxplot(
        {
            "rewarded": rewarded_anticipatory,
            "unrewarded": unrewarded_anticipatory,
        }
    )

    plt.title(
        f"Percent of trials with anticipatory licking:\n"
        f"Rewarded: {round(sum(trial > 0 for trial in rewarded_anticipatory) / len(rewarded_anticipatory), 2)}\n"
        f"Unrewarded: {round(sum(trial  > 0 for trial in unrewarded_anticipatory) / len(unrewarded_anticipatory), 2)}"
    )

    plt.show()


def plot_licking_habituation(trials: List[TrialInfo]) -> None:
    licks = [np.array(trial.lick_start) for trial in trials]
    plot_lick_raster(licks, "Licking Habituation", x_label="Time (s)", jitter=0)
    plt.show()


def plot_position_whole_session(trials: List[TrialInfo], sampling_rate: int) -> None:
    all_positions = np.array([], dtype="float")
    previous_rotary_end = 0
    for trial in trials:
        position = np.array(trial.rotary_encoder_position)
        all_positions = np.concatenate((all_positions, position + previous_rotary_end))
        previous_rotary_end += position[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(len(all_positions)) / sampling_rate,
        (all_positions / ENCODER_TICKS_PER_TURN) * WHEEL_CIRCUMFERENCE / 100,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.show()


def plot_speed_all_trials(trials: List[TrialInfo], sampling_rate: int) -> None:
    plt.figure(figsize=(10, 6))

    first_position = 0
    last_position = 200
    step_size = 5

    speeds = []

    for trial in trials:
        position = degrees_to_cm(np.array(trial.rotary_encoder_position))

        speeds.append(
            get_speed_positions(
                position=position,
                first_position=first_position,
                last_position=last_position,
                step_size=step_size,
                sampling_rate=sampling_rate,
            )
        )

    # Should be the same for all trials
    # Use the bin_stop so there is no forward look ahead
    x_axis = [bin.position_stop for bin in speeds[0]]

    shaded_line_plot(
        np.array([[bin.speed for bin in trial] for trial in speeds]),
        x_axis,
        "Black",
        "All Trials",
    )

    plt.axvspan(180, 200, color="gray", alpha=0.5)
    plt.legend()
    plt.ylim(0, None)
    plt.xlabel("Distance (cm)")
    plt.ylabel("Speed (cm / s)")
    plt.show()


def plot_speed(trials: List[TrialInfo], sampling_rate: int) -> None:
    plt.figure(figsize=(10, 6))
    rewarded: List[List[SpeedPosition]] = []
    not_rewarded: List[List[SpeedPosition]] = []

    prev_rewarded = []
    prev_unrewarded = []

    first_position = 0
    last_position = 200
    step_size = 5
    for idx, trial in enumerate(trials):

        position = degrees_to_cm(np.array(trial.rotary_encoder_position))

        speed = get_speed_positions(
            position=position,
            first_position=first_position,
            last_position=last_position,
            step_size=step_size,
            sampling_rate=sampling_rate,
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

    plt.axvspan(180, 200, color="gray", alpha=0.5)
    plt.legend()
    plt.ylim(0, None)
    plt.xlabel("Distance (cm)")
    plt.ylabel("Speed (cm / s)")
    plt.title(MOUSE)
    plt.show()


def remove_bad_trials(trials: List[TrialInfo]) -> List[TrialInfo]:
    """Remove the first trial as putting the mouse on the wheel, and timed out trials"""
    trials = trials[1:]
    return [
        trial
        for trial in trials
        if trial.rotary_encoder_position
        and degrees_to_cm(trial.rotary_encoder_position[-1]) >= 180
    ]


def summarise_trial(trial: TrialInfo) -> TrialSummary:

    position = degrees_to_cm(np.array(trial.rotary_encoder_position))

    return TrialSummary(
        speed_AZ=get_speed_positions(
            position=position,
            first_position=170,
            last_position=180,
            step_size=10,
            sampling_rate=30,
        )[0].speed,
        licks_AZ=get_anticipatory_licking(licks_to_position(trial)),
        rewarded=trial.texture_rewarded,
    )


def az_speed_histogram(trial_summaries: List[TrialSummary]) -> None:

    rewarded = [trial.speed_AZ for trial in trial_summaries if trial.rewarded]
    not_rewarded = [trial.speed_AZ for trial in trial_summaries if not trial.rewarded]
    plt.hist(
        rewarded,
        bins=10,
        color="green",
        alpha=0.5,
    )

    plt.hist(
        not_rewarded,
        bins=10,
        color="red",
        alpha=0.5,
    )

    plt.title(ttest_ind(rewarded, not_rewarded))
    plt.show()


if __name__ == "__main__":
    trials = load_data(SESSION_PATH)
    # disparity(trials)

    # plot_licking_habituation(trials)

    # print(f"Number of trials: {len(trials)}")
    # print(f"Percent Timed Out: {get_percent_timedout(trials)}")
    trials = remove_bad_trials(trials)

    # az_speed_histogram([summarise_trial(trial) for trial in trials])

    # print(f"Number of trials after removing timed out: {len(trials)}")

    plot_rewarded_vs_unrewarded_licking(trials)
    # plot_speed(trials, sampling_rate=30)
    # # plot_trial_length(trials)

    # plot_previous_trial_dependent_licking(trials)
