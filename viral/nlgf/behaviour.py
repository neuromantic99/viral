"""Task performance by genotype.

This runs first and needs no imaging. Every neural claim later rests on the two
genotypes doing the same task to the same standard - if NLGF mice run less, lick less,
or discriminate worse, then a difference in RSC coding is a difference in what the
animal is doing, not in how the cortex represents it.

Measures are the ones already used in multiple_sessions.py, so numbers here are
comparable to the existing behavioural figures:

  licking d'  d_prime(P(anticipatory lick | rewarded), P(anticipatory lick | unrewarded))
              where an anticipatory lick is one in 150-180 cm before reward delivery
  speed d'    (mean AZ speed unrewarded - rewarded) / mean of the two SDs, i.e. how
              much the animal slows for the rewarded texture

Both are signed so that positive means the animal discriminates in the correct
direction.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.load import Digest, ledger, load_digest
from viral.nlgf.paths import RESULTS
from viral.utils import d_prime

MIN_TRIALS_PER_TYPE = 5


def session_behaviour(d: Digest) -> Dict:
    t = d.trials
    # licks_AZ is set to -1 where summarise_trial raised, which happens when a trial
    # has no reward_on1 state to time anticipatory licks against
    valid = t[(t.licks_AZ >= 0) & np.isfinite(t.speed_AZ)]
    rew = valid[valid.texture_rewarded == 1]
    unrew = valid[valid.texture_rewarded == 0]

    out = dict(
        mouse=d.mouse,
        date=d.date,
        genotype=d.genotype,
        session_type=d.meta["session_type"],
        n_trials=len(t),
        n_valid=len(valid),
        n_rewarded=len(rew),
        n_unrewarded=len(unrew),
        n_trials_dropped=int((t.licks_AZ < 0).sum()),
    )

    if len(rew) < MIN_TRIALS_PER_TYPE or len(unrew) < MIN_TRIALS_PER_TYPE:
        out.update(licking_dprime=np.nan, speed_dprime=np.nan)
    else:
        hit = float((rew.licks_AZ > 0).mean())
        fa = float((unrew.licks_AZ > 0).mean())
        out["licking_dprime"] = d_prime(hit, fa)
        out["lick_rate_rewarded"] = hit
        out["lick_rate_unrewarded"] = fa

        sd = (rew.speed_AZ.std() + unrew.speed_AZ.std()) / 2
        out["speed_dprime"] = (
            float((unrew.speed_AZ.mean() - rew.speed_AZ.mean()) / sd)
            if sd > 0
            else np.nan
        )

    # Gross performance and motivation, which are confounds rather than the question
    out["mean_trial_speed"] = float(valid.trial_speed.mean()) if len(valid) else np.nan
    out["median_trial_time"] = (
        float(valid.trial_time.median()) if len(valid) else np.nan
    )
    out["licks_per_trial"] = float(valid.n_licks.mean()) if len(valid) else np.nan
    out["frac_reward_drunk"] = float(rew.reward_drunk.mean()) if len(rew) else np.nan

    # Distance run during the session, from the speed-thresholded running frames
    out["running_frames"] = int(d.run_frame.size)

    return out


def behaviour_table() -> pd.DataFrame:
    led = ledger()
    rows: List[Dict] = []
    for r in led.itertuples():
        try:
            rows.append(session_behaviour(load_digest(r.mouse, r.date)))
        except (
            Exception
        ) as e:  # noqa: BLE001 - one bad session should not stop the pass
            print(f"  behaviour failed {r.mouse} {r.date}: {type(e).__name__}: {e}")
    df = pd.DataFrame(rows)
    df = df.merge(
        led[["mouse", "date", "stage", "day", "has_freeze", "has_spks"]],
        on=["mouse", "date"],
        how="left",
    )
    df.to_csv(RESULTS / "behaviour_sessions.csv", index=False)
    return df


def lick_profile(d: Digest, bin_size_cm: float = 5.0) -> Dict[str, np.ndarray]:
    """Licks per trial as a function of corridor position, split by texture.

    This is the raw form of the discrimination measure: where the animal licks, rather
    than whether it licked in the zone. A genotype that discriminates but licks in the
    wrong place looks identical on d' and different here.
    """
    edges = np.arange(0, 185, bin_size_cm)
    t = d.trials
    out = {"edges": edges}
    for name, want in (("rewarded", 1.0), ("unrewarded", 0.0)):
        trial_ids = set(t.index[t.texture_rewarded == want])
        n = len(trial_ids)
        if n == 0:
            out[name] = np.full(edges.size - 1, np.nan)
            continue
        keep = np.isin(d.lick_trial, list(trial_ids))
        counts, _ = np.histogram(d.lick_position_cm[keep], bins=edges)
        out[name] = counts / n
    return out


if __name__ == "__main__":
    df = behaviour_table()
    print(f"{len(df)} sessions, {df.mouse.nunique()} mice")
    print(
        df.groupby("genotype")
        .agg(
            mice=("mouse", "nunique"),
            sessions=("date", "size"),
            licking_dprime=("licking_dprime", "median"),
            speed_dprime=("speed_dprime", "median"),
        )
        .to_string()
    )
