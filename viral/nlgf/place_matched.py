"""Place cells detected on a MATCHED number of laps.

Answering "should we use a less permissive method": the method is not the problem, and
swapping it would not fix this. Any criterion that compares an observed rate map to a
per-session shuffle distribution gets more permissive as laps accumulate, because the
null tightens as 1/n_laps while the real map does not. That is true of the Grosmark
criterion, of a spatial-information shuffle test, and of a split-half reliability test
alike - it is a property of testing against a session's own null, not of which statistic
is tested. Yield then reports statistical power rather than biology, which is why it
correlated rho = +0.59 with lap count and produced an apparent genotype difference.

The fix is to hold laps constant, which was previously impractical: the cached
place_threshold is only valid for the exact trial set it was built from, so subsampling
laps meant re-running 2000 shuffles per session. viral.nlgf.place_threshold makes that
cost ~30 s, because the circular permutation acts on the already-binned map and the
binning does not have to be repeated. So the threshold is regenerated here from the
same N_LAPS laps the rate map is built from, and every session is scored at the same
statistical power.

Laps are taken in order rather than at random, so within-session drift and learning are
matched too.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.constants import grosmark_config
from viral.grosmark_analysis import filter_additional_check
from viral.imaging_utils import activity_trial_position, trial_is_imaged
from viral.models import Cached2pSession
from viral.nlgf.cohort import describe, sessions
from viral.nlgf.paths import CACHED_2P, RESULTS, spks_path
from viral.nlgf.place import _skaggs
from viral.nlgf.place_threshold import shuffle_threshold
from viral.utils import (
    find_n_consecutive_trues_extent,
    get_genotype,
    get_wheel_circumference_from_rig,
    has_n_consecutive_trues,
)

CONFIG = grosmark_config
WHEEL = get_wheel_circumference_from_rig("2P")
N_LAPS = 30
N_CELLS = 250
N_SHUFFLES = 2000


def session_place_matched(mouse: str, date: str) -> Optional[Dict]:
    spath = spks_path(mouse, date)
    jpath = CACHED_2P / f"{mouse}_{date}.json"
    if not spath.exists() or not jpath.exists():
        return None
    session = Cached2pSession.model_validate_json(jpath.read_text())

    trials = []
    for t in session.trials:
        try:
            if trial_is_imaged(t):
                trials.append(t)
        except (AssertionError, IndexError):
            continue
    if len(trials) < N_LAPS:
        return None
    trials = trials[:N_LAPS]

    spks_full = np.asarray(np.load(spath, mmap_mode="r"), dtype=np.float32)
    if spks_full.shape[0] < N_CELLS:
        return None
    rng = np.random.default_rng(0)
    cells = np.sort(rng.choice(spks_full.shape[0], N_CELLS, replace=False))
    spks = spks_full[cells]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_trials = np.array([
            activity_trial_position(
                trial=t, flu=spks, wheel_circumference=WHEEL, bin_size=CONFIG.bin_size,
                start=CONFIG.start, max_position=CONFIG.end, verbose=False,
                do_shuffle=False, threshold_speed=True)
            for t in trials
        ], dtype=np.float32)
        smoothed = gaussian_filter1d(
            np.nanmean(all_trials, 0), sigma=7.5 / CONFIG.bin_size, axis=1)

    threshold = shuffle_threshold(all_trials, n_shuffles=N_SHUFFLES)

    n = int((2 / CONFIG.bin_size) * 5)
    if n % 2 == 0:
        n += 1
    supra = smoothed > threshold
    pcs = has_n_consecutive_trues(supra, n)
    row: Dict = dict(mouse=mouse, date=date, genotype=get_genotype(mouse),
                     session_type=session.session_type,
                     n_cells_total=int(spks_full.shape[0]), n_laps=len(trials))
    if pcs.sum() == 0:
        row.update(frac_place_cells=0.0, spatial_info_pc=np.nan,
                   field_width_cm=np.nan, reliability_pc=np.nan)
        return row

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extra = filter_additional_check(
            all_trials=all_trials[:, pcs, :], place_threshold=threshold[pcs, :],
            smoothed_matrix=smoothed[pcs, :], n_consecutive_trues=n)
    combined = pcs.copy()
    combined[pcs] = extra
    row["frac_place_cells"] = float(combined.mean())
    row["n_place_cells"] = int(combined.sum())

    occupancy = np.nan_to_num(
        np.sum(~np.isnan(all_trials[:, 0, :]), axis=0).astype(float))
    if occupancy.sum() > 0 and combined.any():
        si = _skaggs(np.clip(smoothed, 0, None), occupancy)
        row["spatial_info_pc"] = float(np.nanmedian(si[combined]))
        extents = find_n_consecutive_trues_extent(supra[combined], n)
        row["field_width_cm"] = float(np.median(extents.sum(axis=1)) * CONFIG.bin_size)
        odd = np.nanmean(all_trials[1::2][:, combined, :], axis=0)
        even = np.nanmean(all_trials[0::2][:, combined, :], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rel = np.array([np.corrcoef(o, e)[0, 1] for o, e in zip(odd, even)])
        row["reliability_pc"] = float(np.nanmedian(rel))
    return row


def place_matched_table() -> pd.DataFrame:
    coh = sessions(min_cells=N_CELLS)
    print(f"cohort: {describe(coh)}", flush=True)
    rows = []
    for i, r in enumerate(coh.itertuples(), 1):
        try:
            row = session_place_matched(r.mouse, r.date)
            if row is not None:
                row.update(stage=r.stage, day=r.day)
                rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        if i % 10 == 0:
            print(f"  [{i}/{len(coh)}] kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "place_lap_matched.csv", index=False)
    return df


if __name__ == "__main__":
    df = place_matched_table()
    print(f"\n{len(df)} sessions, {df.mouse.nunique()} mice")
    print(df.groupby("genotype").agg(
        mice=("mouse", "nunique"), sessions=("date", "size"),
        frac_pc=("frac_place_cells", "median"), si=("spatial_info_pc", "median"),
        width=("field_width_cm", "median"), rel=("reliability_pc", "median")).to_string())
