"""General immobility masks: any period the animal was still, wherever it was.

get_iti_still_frames scores stillness only inside the inter-trial interval, which
requires the encoder to be sampled during the ITI. That sampling arrived with the
trigger_panda_ITI state between the 2024 and 2025 cohorts: a JB011 or JB016 trial has
trigger_panda_ITI = 0, a JB030 trial has 598. In the older sessions the ITI therefore
looks like zero stillness when in fact stillness there is unmeasurable - and because
the old cohort is mostly NLGF, that absence is aligned with genotype and would have
removed most NLGF mice from every offline comparison.

Nothing about the question needs the ITI specifically. Immobility is immobility: what
is required is that speed is sampled while the animal is still, and the encoder is
sampled throughout the trial in every cohort. Mice stop inside the corridor often,
particularly on unrewarded trials, so within-trial immobility recovers the old cohort.

Definition, applied per trial over all sampled frames:

  still      Grosmark's criterion, matching get_resting_position_and_frames and
             get_iti_still_frames: speed below 3 cm/s for at least 3 consecutive
             seconds.
  settle     the first 3 s of each immobility bout is dropped. GCaMP decays with a
             time constant near 0.7 s, so running-evoked calcium bleeds several
             seconds into the stationary period; this is the same exclusion
             get_iti_still_frames applies at the ITI start, generalised to apply
             wherever the movement actually stopped.
  licking    lick frames with a 15 frame pad either side are dropped, since lick
             bouts carry their own motor and reward signals.

The retained fraction is returned alongside the mask. It is a covariate and a possible
confound in its own right - a group that fidgets more contributes less data and
different data - so it is checked by genotype before any comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.imaging_utils import compute_speed_grosmark, trial_is_imaged
from viral.models import Cached2pSession, TrialInfo
from viral.multiple_sessions import _lick_frames
from viral.nlgf.iti import local_n_frames
from viral.nlgf.paths import CACHED_2P, DERIVED
from viral.utils import (
    below_threshold_for_n_consecutive_samples,
    degrees_to_cm,
    get_wheel_circumference_from_rig,
)

IMMOBILITY_MASKS = DERIVED / "immobility_masks"
IMMOBILITY_MASKS.mkdir(parents=True, exist_ok=True)

FS = 30
SPEED_THRESHOLD = 3.0
N_CONSECUTIVE = 3 * FS
SETTLE_FRAMES = 3 * FS
LICK_PAD = 15

STATE_NAMES = ("trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI")


def _trial_still_frames(trial: TrialInfo, wheel_circumference: float) -> np.ndarray:
    position = degrees_to_cm(np.array(trial.rotary_encoder_position), wheel_circumference)
    frame_position = np.array(
        [s.closest_frame_start for s in trial.states_info if s.name in STATE_NAMES]
    )
    if len(position) != len(frame_position) or len(position) < N_CONSECUTIVE:
        return np.array([], dtype=int)

    still = below_threshold_for_n_consecutive_samples(
        compute_speed_grosmark(position),
        threshold=SPEED_THRESHOLD,
        n_samples=N_CONSECUTIVE,
    )
    if not still.any():
        return np.array([], dtype=int)

    # Drop the settling period at the start of each immobility bout
    edges = np.flatnonzero(np.diff(np.r_[0, still.astype(np.int8), 0]))
    keep = np.zeros_like(still)
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start > SETTLE_FRAMES:
            keep[start + SETTLE_FRAMES : end] = True

    frames = np.unique(frame_position[keep]).astype(int)
    lick = _lick_frames(trial, pad=LICK_PAD)
    if lick.size:
        frames = frames[~np.isin(frames, lick)]
    return frames


def immobility_mask(
    mouse: str, date: str, use_cache: bool = True
) -> Tuple[np.ndarray, pd.DataFrame]:
    cache = IMMOBILITY_MASKS / f"{mouse}_{date}.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["mask"], pd.DataFrame(z["records"], columns=list(z["record_columns"]))

    n_frames = local_n_frames(mouse, date)
    session = Cached2pSession.model_validate_json(
        (CACHED_2P / f"{mouse}_{date}.json").read_text()
    )
    wheel = get_wheel_circumference_from_rig("2P")

    mask = np.zeros(n_frames, dtype=bool)
    records = []
    for idx, trial in enumerate(session.trials):
        try:
            if not trial_is_imaged(trial):
                continue
        except (AssertionError, IndexError):
            continue
        frames = _trial_still_frames(trial, wheel)
        frames = frames[(frames >= 0) & (frames < n_frames)]
        mask[frames] = True
        n_sampled = sum(1 for s in trial.states_info if s.name in STATE_NAMES)
        records.append(dict(trial=idx, rewarded=float(trial.texture_rewarded),
                            sampled_frames=float(n_sampled),
                            retained_frames=float(frames.size)))

    df = pd.DataFrame(records)
    cols = list(df.columns) if len(df) else ["trial", "rewarded", "sampled_frames",
                                             "retained_frames"]
    np.savez_compressed(
        cache, mask=mask,
        records=df.to_numpy(dtype=float) if len(df) else np.zeros((0, len(cols))),
        record_columns=np.array(cols),
    )
    return mask, df


def build_all(overwrite: bool = False) -> None:
    from viral.nlgf.load import ledger

    led = ledger()
    led = led[led.has_spks]
    ok = fail = skip = 0
    for i, r in enumerate(led.itertuples(), 1):
        if (IMMOBILITY_MASKS / f"{r.mouse}_{r.date}.npz").exists() and not overwrite:
            skip += 1
            continue
        try:
            immobility_mask(r.mouse, r.date, use_cache=False)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        if i % 20 == 0:
            print(f"  [{i}/{len(led)}] ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"done: ok={ok} skipped={skip} failed={fail}", flush=True)


if __name__ == "__main__":
    build_all(overwrite="--overwrite" in sys.argv)
