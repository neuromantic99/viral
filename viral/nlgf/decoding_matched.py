"""Decoding error with trial count and cell count matched across sessions.

The unmatched Tier 2 result said NLGF have more place cells, wider fields and better
decoding. All three track how much data the session contains rather than anything
about the animal:

    field width        rho = +0.75 with n_trials
    place cell yield   rho = +0.59
    reliability        rho = +0.58
    decoding error     rho = -0.33 with n_trials, -0.42 with n_cells

and NLGF run 65.5 trials per session against WT's 46.6, with 556 cells against 497.
The genotype difference is what more data predicts, so it cannot be read as coding.

Yield, width and reliability cannot be de-confounded here: place_threshold is the 99th
percentile of a shuffle distribution computed over the session's OWN laps, so its
variance falls with lap count and it is only valid for the exact trial set it was built
from. Subsampling trials and reusing the cached threshold would compare a noisy mean
map against a threshold calibrated for a quieter one. Fixing that needs the 2000
shuffles re-run per subsample.

Decoding error has no such dependency - it needs no threshold and no place-field
criterion - so it can be matched directly, and it is the measure to weight anyway.
Both nuisance variables are fixed by construction:

    trials   the first N_TRIALS imaged trials, in order. First rather than random, so
             within-session drift and learning are matched too; parity still
             alternates, so the odd/even template/test split stays balanced.
    cells    N_CELLS drawn at random without replacement, repeated N_REPEATS times and
             the median taken, so the result does not depend on one draw.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.chang import ChangConfig, decoding_error, run_rate_map
from viral.imaging_utils import trial_is_imaged
from viral.models import Cached2pSession
from viral.nlgf.paths import CACHED_2P, RESULTS, spks_path
from viral.utils import get_genotype

CHANG = ChangConfig()
N_TRIALS = 24
N_CELLS = 100
N_REPEATS = 10


def _safe_imaged(trial) -> bool:
    try:
        return trial_is_imaged(trial)
    except (AssertionError, IndexError):
        return False


def session_matched_decoding(mouse: str, date: str) -> Optional[Dict]:
    spath = spks_path(mouse, date)
    if not spath.exists():
        return None
    session = Cached2pSession.model_validate_json(
        (CACHED_2P / f"{mouse}_{date}.json").read_text()
    )
    imaged = [t for t in session.trials if _safe_imaged(t)]
    if len(imaged) < N_TRIALS:
        return None

    spks_full = np.asarray(np.load(spath, mmap_mode="r"), dtype=np.float32)
    n_cells = spks_full.shape[0]
    if n_cells < N_CELLS:
        return None

    matched = session.model_copy(update={"trials": imaged[:N_TRIALS]})

    rng = np.random.default_rng(0)
    errors = []
    for _ in range(N_REPEATS):
        cells = rng.choice(n_cells, N_CELLS, replace=False)
        spks = spks_full[cells]
        template, _, valid = run_rate_map(matched, spks, CHANG)
        err = decoding_error(matched, spks, template, valid, CHANG)
        if np.isfinite(err):
            errors.append(err)

    if not errors:
        return None

    return dict(
        mouse=mouse, date=date, genotype=get_genotype(mouse),
        session_type=session.session_type,
        n_cells_total=n_cells, n_trials_total=len(imaged),
        decoding_error_matched=float(np.median(errors)),
        decoding_error_iqr=float(np.percentile(errors, 75) - np.percentile(errors, 25)),
        n_repeats=len(errors),
    )


def matched_table() -> pd.DataFrame:
    from viral.nlgf.cohort import describe, sessions

    led = sessions()
    print(f"cohort: {describe(led)}", flush=True)
    rows = []
    for i, r in enumerate(led.itertuples(), 1):
        try:
            row = session_matched_decoding(r.mouse, r.date)
            if row is not None:
                rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        if i % 10 == 0:
            print(f"  [{i}/{len(led)}] kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "decoding_matched.csv", index=False)
    return df


if __name__ == "__main__":
    df = matched_table()
    print(f"\n{len(df)} sessions, {df.mouse.nunique()} mice")
    print(df.groupby("genotype").agg(
        mice=("mouse", "nunique"), sessions=("date", "size"),
        err=("decoding_error_matched", "median")).to_string())
