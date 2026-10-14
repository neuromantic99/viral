"""Behaviour across the full training history, from the cached MouseSummary files.

behaviour_summaries/ covers every training session for 32 mice, not just the imaged
ones, so it is the best-powered part of this study: 12 NLGF against 10 WT, with sex
balanced 6M/6F and 5M/5F. It is also the lab's own pipeline output, so the metrics
here are the same numbers the existing behavioural figures plot - speed_difference and
licking_difference are imported from multiple_sessions rather than reimplemented.

Session order within a stage is the experience axis. Absolute date is not usable
because the two cohorts (JB0xx in 2024-25, J0xx in 2026) ran on different calendars.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.models import MouseSummary, SessionSummary
from viral.multiple_sessions import licking_difference, speed_difference
from viral.nlgf.paths import HD, RESULTS
from viral.utils import get_session_type, get_sex

BEHAVIOUR_SUMMARIES = HD / "behaviour_summaries"

MIN_TRIALS_PER_TYPE = 5
STAGE_ORDER = ["learning", "reversal", "recall", "recall_reversal"]


def load_mice() -> List[MouseSummary]:
    return [
        MouseSummary.model_validate_json(p.read_text())
        for p in sorted(BEHAVIOUR_SUMMARIES.glob("*.json"))
        if not p.name.startswith("._")
    ]


def _session_row(mouse: MouseSummary, session: SessionSummary) -> Dict:
    trials = session.trials
    rew = [t for t in trials if t.rewarded]
    unrew = [t for t in trials if not t.rewarded]

    stage = get_session_type(session.name)
    row = dict(
        mouse=mouse.name,
        genotype=mouse.genotype,
        sex=get_sex(mouse.name),
        cohort="J" if not mouse.name.startswith("JB") else "JB",
        session_name=session.name,
        stage=stage,
        # Rig matters: speeds are in cm/s via the rig's wheel circumference (2P 34.7,
        # box 53.4), and rig is not balanced across genotype, so any speed comparison
        # has to be made within rig
        rig=mouse.setup.get(stage),
        n_trials=len(trials),
        n_rewarded=len(rew),
        n_unrewarded=len(unrew),
    )

    if len(rew) >= MIN_TRIALS_PER_TYPE and len(unrew) >= MIN_TRIALS_PER_TYPE:
        row["licking_dprime"] = float(licking_difference(trials))
        row["speed_dprime"] = float(speed_difference(trials))
        row["lick_rate_rewarded"] = float(np.mean([t.licks_AZ > 0 for t in rew]))
        row["lick_rate_unrewarded"] = float(np.mean([t.licks_AZ > 0 for t in unrew]))
        row["speed_AZ_rewarded"] = float(np.mean([t.speed_AZ for t in rew]))
        row["speed_AZ_unrewarded"] = float(np.mean([t.speed_AZ for t in unrew]))
    else:
        for k in (
            "licking_dprime", "speed_dprime", "lick_rate_rewarded",
            "lick_rate_unrewarded", "speed_AZ_rewarded", "speed_AZ_unrewarded",
        ):
            row[k] = np.nan

    row["mean_trial_speed"] = float(np.mean([t.trial_speed for t in trials]))
    row["median_trial_time"] = float(np.median([t.trial_time_overall for t in trials]))
    row["frac_reward_drunk"] = (
        float(np.mean([t.reward_drunk for t in rew])) if rew else np.nan
    )
    return row


def behaviour_table() -> pd.DataFrame:
    rows = []
    for mouse in load_mice():
        for session in mouse.sessions:
            rows.append(_session_row(mouse, session))
    df = pd.DataFrame(rows)

    # Session index within stage, per mouse: the experience axis. Sessions are stored
    # in the order they were run, so position within the stage is the day number.
    df["session_in_stage"] = df.groupby(["mouse", "stage"]).cumcount()
    df["session_overall"] = df.groupby("mouse").cumcount()
    df.to_csv(RESULTS / "behaviour_full.csv", index=False)
    return df


if __name__ == "__main__":
    df = behaviour_table()
    print(f"{len(df)} sessions, {df.mouse.nunique()} mice\n")
    print(
        df.groupby("genotype")
        .agg(
            mice=("mouse", "nunique"),
            sessions=("session_name", "size"),
            trials=("n_trials", "sum"),
            lick_dprime=("licking_dprime", "median"),
            speed_dprime=("speed_dprime", "median"),
        )
        .to_string()
    )
