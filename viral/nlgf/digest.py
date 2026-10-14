"""Compact per-session digests, built once from the 13 GB of cached JSON.

A Cached2pSession JSON is ~54 MB, almost all of it per-sample rotary encoder traces.
Every downstream question here needs the same handful of derived quantities, so they
are extracted once into a ~100 kB npz per session. A full pass over cached_2p takes
minutes; a full pass over the digests takes seconds.

Nothing is recomputed differently from the existing code: positions come from
get_online_position_and_frames and trial summaries from summarise_trial. This module
only chooses what to keep. The one exception is freeze movement, which comes from
viral.nlgf.freeze.movement_bools rather than get_movement_bool - see that module for
the four malformed cases in the cache and which of them are recoverable.
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.imaging_utils import get_online_position_and_frames, trial_is_imaged
from viral.models import Cached2pSession, TrialInfo
from viral.nlgf.freeze import movement_bools
from viral.nlgf.paths import digest_path, session_json_paths
from viral.single_session import summarise_trial
from viral.utils import (
    degrees_to_cm,
    get_genotype,
    get_wheel_circumference_from_rig,
)

WHEEL = get_wheel_circumference_from_rig("2P")

# Corridor geometry, as used throughout the existing analyses
CORRIDOR_END_CM = 180.0
AZ_START_CM = 150.0
LANDMARKS_CM = (45.0, 90.0, 135.0)


def _lick_positions(trial: TrialInfo) -> np.ndarray:
    """Position in cm at each lick onset.

    trial.lick_start holds bpod times; the encoder is sampled at the trigger_panda
    states, so a lick is placed at the position of the nearest preceding sample. This
    is the same alignment get_anticipatory_licking uses, kept explicit here so lick
    position is available for the whole corridor rather than only the reward zone.
    """
    position = degrees_to_cm(np.array(trial.rotary_encoder_position), WHEEL)
    sample_times = np.array(
        [
            state.start_time
            for state in trial.states_info
            if state.name
            in ("trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI")
        ]
    )
    if sample_times.size == 0 or position.size == 0:
        return np.array([])
    n = min(sample_times.size, position.size)
    sample_times, position = sample_times[:n], position[:n]

    licks = np.array(trial.lick_start, dtype=float)
    if licks.size == 0:
        return np.array([])
    idx = np.searchsorted(sample_times, licks, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    return position[idx]


def build_digest(session: Cached2pSession) -> Dict[str, Any]:
    trial_rows = []
    lick_pos_all, lick_pos_trial = [], []
    pos_all, frame_all, lap_all, pos_all_nothresh, frame_all_nothresh = (
        [],
        [],
        [],
        [],
        [],
    )

    for idx, trial in enumerate(session.trials):
        try:
            imaged = trial_is_imaged(trial)
        except (AssertionError, IndexError):
            imaged = False

        with warnings.catch_warnings():
            # get_speed_positions warns on empty position bins, which is common and
            # already handled by the nan it produces
            warnings.simplefilter("ignore")
            try:
                summary = summarise_trial(trial, WHEEL)
                speed_az, speed_nonaz = summary.speed_AZ, summary.speed_nonAZ
                trial_speed, licks_az = summary.trial_speed, summary.licks_AZ
                reward_drunk_ = summary.reward_drunk
            except Exception:
                speed_az = speed_nonaz = trial_speed = np.nan
                licks_az, reward_drunk_ = -1, False

        lp = _lick_positions(trial)
        lick_pos_all.append(lp)
        lick_pos_trial.append(np.full(lp.size, idx))

        if imaged:
            for thresh, (pl, fl) in (
                (True, (pos_all, frame_all)),
                (False, (pos_all_nothresh, frame_all_nothresh)),
            ):
                try:
                    p, f = get_online_position_and_frames(
                        trial=trial,
                        wheel_circumference=WHEEL,
                        threshold_speed=thresh,
                        speed_threshold=5 if thresh else None,
                    )
                except (AssertionError, ValueError):
                    continue
                keep = p < CORRIDOR_END_CM
                pl.append(p[keep])
                fl.append(f[keep])
                if thresh:
                    lap_all.append(np.full(int(keep.sum()), idx))

        trial_rows.append(
            (
                idx,
                float(trial.trial_start_closest_frame or -1),
                float(trial.trial_end_closest_frame or -1),
                float(trial.texture_rewarded),
                float(imaged),
                float(licks_az),
                float(reward_drunk_),
                speed_az,
                speed_nonaz,
                trial_speed,
                trial.trial_end_time - trial.trial_start_time,
                float(len(trial.lick_start)),
                float(len(trial.reward_on) > 0),
            )
        )

    cat = lambda xs: np.concatenate(xs) if xs else np.array([])

    out: Dict[str, Any] = dict(
        trials=np.array(trial_rows, dtype=float),
        trial_columns=np.array(
            [
                "idx",
                "start_frame",
                "end_frame",
                "texture_rewarded",
                "imaged",
                "licks_AZ",
                "reward_drunk",
                "speed_AZ",
                "speed_nonAZ",
                "trial_speed",
                "trial_time",
                "n_licks",
                "reward_on",
            ]
        ),
        run_position_cm=cat(pos_all),
        run_frame=cat(frame_all),
        run_lap=cat(lap_all),
        run_position_cm_all=cat(pos_all_nothresh),
        run_frame_all=cat(frame_all_nothresh),
        lick_position_cm=cat(lick_pos_all),
        lick_trial=cat(lick_pos_trial),
    )

    wf = session.wheel_freeze
    meta = dict(
        mouse=session.mouse_name,
        date=session.date,
        session_type=session.session_type,
        genotype=get_genotype(session.mouse_name),
        n_trials=len(session.trials),
        n_trials_imaged=int(sum(r[4] for r in trial_rows)),
        has_freeze=wf is not None,
        freeze_type=(wf.freeze_movement_type if wf is not None else None),
    )

    if wf is not None:
        movement = movement_bools(wf)
        meta["repair_pre"] = movement.repair_pre
        meta["repair_post"] = movement.repair_post
        meta["freeze_usable"] = movement.usable
        out["freeze_bounds"] = np.array(
            [
                wf.pre_training_start_frame,
                wf.pre_training_end_frame,
                wf.post_training_start_frame,
                wf.post_training_end_frame,
            ],
            dtype=np.int64,
        )
        if movement.usable:
            out["movement_pre"] = movement.movement_pre
            out["movement_post"] = movement.movement_post

    out["meta"] = np.array(json.dumps(meta))
    return out


def build_all(overwrite: bool = False) -> None:
    paths = session_json_paths()
    n_ok = n_skip = n_fail = 0
    for i, p in enumerate(paths, 1):
        mouse, date = p.stem.split("_")[0], p.stem.split("_")[1]
        target = digest_path(mouse, date)
        if target.exists() and not overwrite:
            n_skip += 1
            continue
        try:
            session = Cached2pSession.model_validate_json(p.read_text())
            np.savez_compressed(target, **build_digest(session))
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"FAIL {p.name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        if i % 10 == 0:
            print(f"  [{i}/{len(paths)}] ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
    print(f"done: ok={n_ok} skipped={n_skip} failed={n_fail}", flush=True)


if __name__ == "__main__":
    build_all(overwrite="--overwrite" in sys.argv)
