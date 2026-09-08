import copy
import json
from pathlib import Path
import random
import pandas as pd
import re
import sys
from scipy import stats
from natsort import natsorted

from statsmodels.formula.api import mixedlm

import inspect
from pydantic import ValidationError
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))
sys.path.append(str(HERE.parent))

from typing import List, Tuple, Dict, Callable, Optional, Literal

from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
import numpy as np
from viral.nlgf.style import GENOTYPE_COLOURS
from viral.gsheets_importer import gsheet2df
from viral.single_session import (
    get_binned_licks,
    load_data,
    remove_bad_trials,
    summarise_trial,
)
from viral.constants import BEHAVIOUR_DATA_PATH, CACHE_PATH, HERE, SPREADSHEET_ID
from viral.models import (
    Cached2pSession,
    MouseSummary,
    SessionSummary,
    TrialInfo,
    TrialSummary,
    MultipleSessionsConfig,
)
from viral.imaging_utils import (
    get_ITI_start_frame,
    get_session_n_frames,
    trial_is_imaged,
)
from viral.utils import (
    SessionType,
    below_threshold_for_n_consecutive_samples,
    d_prime,
    degrees_to_cm,
    get_genotype,
    get_wheel_circumference_from_rig,
    shaded_line_plot,
    get_sex,
    get_setup,
    get_session_type,
    get_rewarded_texture,
)
from viral.imaging_utils import compute_speed_grosmark

import seaborn as sns
import pandas as pd

sns.set_theme(context="poster", style="ticks")


def get_iti_still_frames(
    session: Cached2pSession,
    speed_threshold: float = 3.0,
    n_consecutive_samples: int = 3 * 30,
    exclude_start_seconds: float = 3.0,
    lick_pad_frames: int = 15,
    fs: int = 30,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Frames during the inter-trial interval in which the mouse was genuinely still.

    The ITI is 20 s (598-600 frames), so a session of 40-100 trials carries 13-33
    minutes of in-task quiescence - as much offline data as a freeze block, and
    interleaved with the experience rather than bracketing it. That makes it usable for
    reactivation analyses in the many sessions with no wheel freeze.

    Three exclusions, all of which matter:

    exclude_start_seconds drops the beginning of each ITI. GCaMP decays with a time
    constant near 0.7 s, so running-evoked calcium bleeds several seconds past the end of
    the trial; without this, apparent offline activity is partly the tail of the run.
    Three seconds is about four time constants. It also removes reward delivery and the
    arousal transient that follows it, which sit at the ITI start.

    Licking is excluded with a pad either side, since lick bouts carry their own motor
    and reward signals.

    Stillness uses Grosmark's definition, matching get_resting_position_and_frames:
    velocity below 3 cm/s for at least 3 consecutive seconds. It is applied per frame
    rather than as a per-trial exclusion, because a few moving frames are far more
    common than a whole ITI of running and a few is all it takes to contaminate a
    correlation.

    Worth running once at a stricter threshold as a sensitivity check - if the result
    moves, movement is driving it.

    The mask spans the whole session, so it indexes the neural array directly:

        dff, spks, _ = load_imaging_data(mouse, date)
        mask, per_trial = get_iti_still_frames(session)
        quiescent = spks[:, mask]

    Its length is read from the .npy header on the server by get_session_n_frames,
    which costs one small read rather than pulling a multi-gigabyte array across the
    network - so the frame budget can be checked across the whole cohort without
    loading any imaging data.

    The length is not inferred from the trials, because the recording continues past the
    last trial - a long way past on a freeze session - so a trial-derived length would
    be short and every downstream index would misalign silently.

    Returns a boolean mask over the session's frames, and a per-trial frame budget so
    you can see what was retained. Check the retained fraction by genotype before
    comparing groups: if one group fidgets more it contributes less data AND different
    data.
    """
    n_frames = get_session_n_frames(session.mouse_name, session.date)

    mask = np.zeros(n_frames, dtype=bool)
    records = []
    max_frame_seen = -1

    wheel_circumference = get_wheel_circumference_from_rig("2P")
    exclude_start_frames = int(exclude_start_seconds * fs)

    for idx, trial in enumerate(session.trials):
        if not trial_is_imaged(trial):
            continue

        try:
            iti_start = get_ITI_start_frame(trial)
        except ValueError:
            continue

        iti_end = trial.trial_end_closest_frame
        if iti_end is None:
            continue
        iti_end = int(iti_end)

        position = degrees_to_cm(
            np.array(trial.rotary_encoder_position), wheel_circumference
        )
        frame_position = np.array(
            [
                state.closest_frame_start
                for state in trial.states_info
                if state.name
                in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
            ]
        )
        if len(position) != len(frame_position) or len(position) < 2:
            continue

        still = below_threshold_for_n_consecutive_samples(
            compute_speed_grosmark(position),
            threshold=speed_threshold,
            n_samples=n_consecutive_samples,
        )

        # Frames sampled during the ITI, past the calcium-bleed exclusion
        in_iti = (frame_position >= iti_start + exclude_start_frames) & (
            frame_position <= iti_end
        )

        max_frame_seen = max(max_frame_seen, iti_end)

        # Every count below is unique IMAGING FRAMES, so the columns subtract from one
        # another. The behavioural samples these come from can be denser than the frame
        # rate, so counting samples for one column and frames for another would make the
        # per-stage costs uninterpretable.
        def in_bounds(frames: np.ndarray) -> np.ndarray:
            frames = np.unique(frames).astype(int)
            return frames[(frames >= 0) & (frames < n_frames)]

        scored_frames = in_bounds(frame_position[in_iti])
        still_frames = in_bounds(frame_position[in_iti & still])

        lick_frames = _lick_frames(trial, pad=lick_pad_frames)
        keep_frames = (
            still_frames[~np.isin(still_frames, lick_frames)]
            if lick_frames.size
            else still_frames
        )

        mask[keep_frames] = True

        records.append(
            {
                "trial": idx,
                "rewarded": trial.texture_rewarded,
                "iti_frames": iti_end - iti_start,
                "scored_frames": scored_frames.size,
                "still_frames": still_frames.size,
                "retained_frames": keep_frames.size,
                "retained_seconds": keep_frames.size / fs,
                # kept so the retained frames can be split by the trial they follow
                "frames": keep_frames,
            }
        )

    # Catches a session whose behavioural frame indices run past the imaging data,
    # which means the cache and the suite2p output disagree about the recording
    assert max_frame_seen < n_frames, (
        f"{session.mouse_name} {session.date}: trial frames run to {max_frame_seen} but "
        f"the imaging data has only {n_frames} frames"
    )

    return mask, pd.DataFrame(records)


def _lick_frames(trial: TrialInfo, pad: int) -> np.ndarray:
    """Frames spanned by lick bouts, padded either side.

    Port1In and Port1Out are not guaranteed to pair up - a trial can end mid-lick - so
    they are zipped only as far as the shorter of the two rather than with strict=True.
    """
    onsets = [
        event.closest_frame for event in trial.events_info if event.name == "Port1In"
    ]
    offsets = [
        event.closest_frame for event in trial.events_info if event.name == "Port1Out"
    ]

    frames: List[int] = []
    for onset, offset in zip(onsets, offsets):
        if onset is None or offset is None:
            continue
        frames.extend(range(int(onset) - pad, int(offset) + pad + 1))

    return np.array(sorted(set(frames)), dtype=int)


def iti_masks_by_trial_type(
    per_trial: pd.DataFrame, n_frames: int
) -> Dict[str, np.ndarray]:
    """Split retained ITI frames by whether the trial they FOLLOW was rewarded.

    This is the frame-split design, and it exists because the template split does not
    survive contact with the data: each split template is built from half the running
    bouts, so it frequently yields zero significant components and the session drops out
    of the comparison entirely.

    Splitting the frames instead costs nothing - the ensembles are still built from all
    the running data - and it asks a question generic online-offline coupling cannot
    answer. Coupling is a static property of the cells and is identical in both frame
    subsets, so it cannot produce a difference between them. A difference means the
    offline period is expressing something about the trial that just happened.

    Note what each contrast buys. With the "all" template this is a main effect: is
    reactivation stronger after reward? (Real and published - Singer and Frank - but
    explicable by arousal.) The content-specificity claim needs the interaction, which
    needs the split templates too, so both are emitted where both exist.
    """
    masks = {}
    for label, is_rewarded in (("rewarded", True), ("unrewarded", False)):
        mask = np.zeros(n_frames, dtype=bool)
        for frames in per_trial.loc[per_trial["rewarded"] == is_rewarded, "frames"]:
            mask[frames] = True
        masks[label] = mask
    return masks


def report_iti_still_frames(
    session: Cached2pSession,
    speed_threshold: float = 3.0,
    n_consecutive_samples: int = 3 * 30,
    exclude_start_seconds: float = 3.0,
    lick_pad_frames: int = 15,
    fs: int = 30,
) -> pd.DataFrame:
    """Print the ITI frame budget for one session and return the per-trial breakdown."""
    mask, per_trial = get_iti_still_frames(
        session,
        speed_threshold=speed_threshold,
        n_consecutive_samples=n_consecutive_samples,
        exclude_start_seconds=exclude_start_seconds,
        lick_pad_frames=lick_pad_frames,
        fs=fs,
    )

    if per_trial.empty:
        print(f"{session.mouse_name} {session.date}: no imaged trials with an ITI")
        return per_trial

    total_iti = per_trial["iti_frames"].sum()
    scored = per_trial["scored_frames"].sum()
    still = per_trial["still_frames"].sum()
    retained = per_trial["retained_frames"].sum()

    minutes = lambda frames: frames / fs / 60
    print(
        f"{session.mouse_name} {session.date} ({session.session_type}): "
        f"{len(per_trial)} imaged trials\n"
        f"  ITI total            {minutes(total_iti):6.1f} min\n"
        f"  after start cut      {minutes(scored):6.1f} min  "
        f"(-{minutes(total_iti - scored):.1f} min to calcium bleed and reward)\n"
        f"  still                {minutes(still):6.1f} min  "
        f"(-{minutes(scored - still):.1f} min to movement)\n"
        f"  retained             {minutes(retained):6.1f} min  "
        f"(-{minutes(still - retained):.1f} min to licking)\n"
        f"  = {retained / total_iti:.1%} of the ITI, median "
        f"{per_trial['retained_seconds'].median():.1f} s per trial of "
        f"{per_trial['iti_frames'].median() / fs:.0f} s"
    )
    return per_trial


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
                or "do not analyse" in type_check
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

        if len(session_numbers) == 1:
            assert sorted([trial.trial_start_time for trial in trials]) == [
                trial.trial_start_time for trial in trials
            ]
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
            fontsize=18,
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
    plt.figure()
    plt.ylabel("Trials to criterion")
    plt.title(session_type.replace("_", " ").capitalize())
    sns.boxplot(
        to_plot, showfliers=False, palette=GENOTYPE_COLOURS, hue_order=["WT", "NLGF"]
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
        palette=GENOTYPE_COLOURS,
        hue_order=["WT", "NLGF"],
    )

    nlgf_vs_wt = stats.ttest_ind(
        to_plot["NLGF"],
        to_plot["WT"],
    )
    print(f"NLGF vs WT: {nlgf_vs_wt.pvalue:.3f}")

    # Add statistical significance annotations
    offset = 30
    plt.text(
        0.5,
        max(max(to_plot["NLGF"]), max(to_plot["WT"])) + offset,
        f"p = {nlgf_vs_wt.pvalue:.2f}",
        ha="center",
        fontsize=18,
    )

    # Add the horizontaol line underneath the significance annotation
    plt.hlines(
        y=max(max(to_plot["NLGF"]), max(to_plot["WT"])) + offset - 5,
        xmin=0,
        xmax=1,
        color="black",
        linewidth=1,
    )

    # and the vertical lines connecting the boxes to the horizontal line
    plt.vlines(
        x=0,
        ymin=max(max(to_plot["NLGF"]), max(to_plot["WT"])) + offset - 5,
        ymax=max(max(to_plot["NLGF"]), max(to_plot["WT"])) + offset - 15,
        color="black",
        linewidth=1,
    )
    plt.vlines(
        x=1,
        ymin=max(max(to_plot["NLGF"]), max(to_plot["WT"])) + offset - 5,
        ymax=max(max(to_plot["NLGF"]), max(to_plot["WT"])) + offset - 15,
        color="black",
        linewidth=1,
    )

    # plt.ylim(0, max(max(to_plot["NLGF"]), max(to_plot["WT"])) + offset + 30)
    plt.ylim(0, 470)

    sns.despine()
    plt.tight_layout()
    group_suffix = "-".join(group_by)

    for extension in ["png", "pdf"]:
        plt.savefig(
            HERE.parent
            / "plots"
            / f"behaviour-summaries-{group_suffix}-{session_type}.{extension}",
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
        # {
        #     "name": "recall",
        #     "label": "Memory\nRecall\nStarts",
        #     "excluded_session_types": ["recall", "recall_reversal"],
        #     "colour": sns.color_palette()[1],
        # },
        # {
        #     "name": "recall_reversal",
        #     "label": "Recall\nReversal\nStarts",
        #     "excluded_session_types": ["recall_reversal"],
        #     "colour": sns.color_palette()[2],
        # },
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
            fontsize=18,
        )
    plt.axhline(1, color="red", linestyle="dotted", alpha=0.7, linewidth=1.5)
    plt.text(
        5,
        1.05,
        "Criterion",
        color="red",
        fontsize=18,
    )
    plt.title(mouse.name)
    plt.tight_layout()
    sns.despine()
    plt.savefig(HERE.parent / "plots" / f"{mouse.name}-performance.svg", dpi=300)
    plt.savefig(HERE.parent / "plots" / f"{mouse.name}-performance.png", dpi=300)
    plt.show()


def learning_metric_first_x_trials(
    trials: List[TrialSummary], config: MultipleSessionsConfig, x: int
) -> float:
    return learning_metric(trials[:x], config)


def plot_learning_metric_first_x_trials(
    mice: List[MouseSummary],
    session_type: Literal["learning", "reversal", "recall", "recall_reversal"],
    group_by: list[str],
    config: MultipleSessionsConfig,
    x: int,
) -> None:

    to_plot: Dict = {"genotype": [], "performance": [], "mouse": []}

    for mouse in mice:
        session = natsorted(
            filter_sessions_by_session_type(mouse, "learning"),
            key=lambda session: session.name,
        )[-1]
        # if session.name.lower() in ["learning day 1", "learning day 2"]:
        #     print(
        #         f"Mouse {mouse.name} has a session with name {session.name} which is likely a learning session but does not match the expected format. Please check the session naming for this mouse."
        #     )
        #     continue
        if get_session_type(session.name) == session_type:
            to_plot["genotype"].append(get_genotype(mouse.name))
            to_plot["performance"].append(
                learning_metric_first_x_trials(session.trials, config, x)
            )
            to_plot["mouse"].append(mouse.name)

    to_plot = pd.DataFrame(to_plot)
    # fit linear mixed effects for WT vs NLGF, controlling for mouse as a random effect

    to_mixed_effect = to_plot[to_plot["genotype"].isin(["WT", "NLGF"])]
    model = mixedlm(
        "performance ~ genotype",
        to_mixed_effect,
        groups="mouse",
    )
    result = model.fit()
    print(result.summary())
    # now do ttest
    wt = to_plot[to_plot["genotype"] == "WT"]["performance"]
    nlgf = to_plot[to_plot["genotype"] == "NLGF"]["performance"]
    ttest = stats.ttest_ind(wt, nlgf)
    print(
        f"T-test WT vs NLGF: p-value = {ttest.pvalue:.3f}, t-statistic = {ttest.statistic:.3f}"
    )

    plt.figure()
    plt.ylabel(f"Learning metric (first {x} trials)")
    plt.title(session_type.replace("_", " ").capitalize())
    sns.boxplot(to_plot, hue="genotype", y="performance", showfliers=False)
    ax = plt.gca()
    new_labels = [
        label.get_text()
        .replace("Oligo-BACE1-KO", "Oligo-\nBACE1-KO")
        .replace("_", "\n")
        for label in ax.get_xticklabels()
    ]
    ax.set_xticklabels(new_labels, fontsize=12)
    sns.stripplot(
        to_plot,
        hue="genotype",
        y="performance",
        edgecolor="black",
        linewidth=1,
        dodge=True,
    )

    sns.despine()
    plt.tight_layout()
    # remove stripplot legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title="Genotype")

    # group_suffix = "-".join(group_by)
    # plt.savefig(
    #     HERE.parent
    #     / "plots"
    #     / f"behaviour-summaries-first-{x}-trials-{group_suffix}-{session_type}.pdf",
    #     dpi=300,
    # )
    plt.show()


def session_counter() -> None:
    possible_mice = [f"JB{i:03d}" for i in range(1, 39)] + [
        f"J{i:03d}" for i in range(30, 39)
    ]
    all_mice = {"WT": [], "NLGF": []}
    session_types = [session_type.value for session_type in SessionType]
    for mouse_name in possible_mice:
        try:
            genotype = get_genotype(mouse_name)
            if genotype in all_mice:
                all_mice[genotype].append(mouse_name)
        except ValueError:
            pass

    freeze_counts = {
        "WT": {k: 0 for k in session_types},
        "NLGF": {k: 0 for k in session_types},
    }
    for genotype, mice in all_mice.items():
        for mouse in mice:
            metadata = gsheet2df(SPREADSHEET_ID, mouse, 1)
            if "Wheel blocked?" not in metadata.columns:
                print(f"Mouse {mouse} does not have 'Wheel blocked?' column")
                continue
            for _, row in metadata.iterrows():
                type_check = row["Type"].lower()
                try:
                    session_type = get_session_type(session_name=type_check)
                except ValueError:
                    print(
                        f"Mouse {mouse} has an unrecognized session type: {type_check}"
                    )
                    continue

                wheel_blocked = row["Wheel blocked?"].lower() in {"yes", "true"}
                freeze_counts[genotype][session_type] += 1


def iti_still_frames_all_mice() -> None:
    cache_files = list(CACHE_PATH.glob("*.json"))

    summary = {}

    for file in cache_files:
        session = Cached2pSession.model_validate_json(file.read_text())

        if "trigger_panda_ITI" not in set(
            [state.name for state in session.trials[0].states_info]
        ):
            continue
        print(f"Processing {session.mouse_name} {session.date}")

        per_trial = report_iti_still_frames(session)
        genotype = get_genotype(session.mouse_name)
        if genotype not in summary:
            summary[genotype] = []
        summary[genotype].append(per_trial)

    collapsed = {k: [] for k in summary.keys()}
    for genotype, per_trial_list in summary.items():
        collapsed[genotype].extend(
            [
                per_trial["retained_seconds"].median()
                for per_trial in per_trial_list
                if "retained_seconds" in per_trial
            ]
        )
    1 / 0


if __name__ == "__main__":

    mice: List[MouseSummary] = []

    redo = False

    config = MultipleSessionsConfig(speed=0.5, licking=0.5, window=50)

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
        "JB030",
        "JB031",
        "JB032",
        "JB033",
        "JB034",
        "JB035",
        "JB036",
        "J030",
        "J031",
        "J032",
        "J035",
        "J034",
        "J036",
        "J037",
        "J038",
    ]:

        print(f"\nProcessing {mouse_name}...")
        # if not get_genotype(mouse_name) in {"WT", "NLGF"}:
        # continue

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

    # for s in ["learning", "reversal"]:
    #     plot_performance_summaries(mice, s, ["genotype"], config=config)

    plot_mouse_performance(mice[-7], config=config)

    # plot_learning_metric_first_x_trials(mice, "learning", ["genotype"], config, x=10)

    # plot_mouse_performance(mice[0], config=config)
    # plot_performance_summaries(mice, "learning", ["genotype"], config=config)
    # for mouse in mice:
    #     plot_mouse_performance(mouse, config=config)
    # plot_running_speed_summaries(mice, "recall", running_speed_AZ)
    # ## Probably not interesting as related to speed
    # plot_trial_time_summaries(mice, "learning")
    # plot_num_trials_summaries(mice, "reversal")
    # plot_num_trials_summaries(mice, "recall")
    # plot_num_trials_summaries(mice, "recall_reversal")
