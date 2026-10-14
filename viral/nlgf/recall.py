"""Long-term stability across the ~2 month gap, without cell registration.

The recall sessions ran 63-71 days after each mouse's last task session - a remarkably
consistent interval, and the same in both genotypes. That gap is the only handle in this
dataset on LONG-TERM representational stability, which is where the amyloid literature
most often reports an effect and which nothing else here tests: every other tier is a
within-session snapshot.

No cellRegistered.mat exists, so the definitive measure - whether cell i holds the same
field two months later - is not available. What is available are three measures that do
not need cells matched across days:

  quality     cell count, laps, event rate and matched decoding error at recall against
              the SAME mouse's pre-gap sessions. This runs first and gates the rest: if
              recall imaging is systematically poorer, degradation and data quality are
              confounded, which is the artefact class that has consumed every positive
              result in this study so far.

  drift       population vector correlation between laps as a function of lap
              separation, within a session. Same cells throughout a session, so no
              matching is needed. Gives a stability level (lag 1) and a decay rate.

  geometry    the position x position correlation matrix of the population - the shape
              of the representation rather than which cell carries what. It is
              n_bins x n_bins whatever the cell set, so it can be compared across days.
              Preserved geometry with unmatched cells would be drift with the job still
              being done; degraded geometry is a different claim.

Measures 2 and 3 ask whether the code still WORKS after two months. Only registration
could ask whether it is the SAME code. Nothing here substitutes for that.

Laps and cells are matched by construction, as everywhere else.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.constants import grosmark_config
from viral.imaging_utils import activity_trial_position, trial_is_imaged
from viral.models import Cached2pSession
from viral.nlgf.cohort import sessions
from viral.nlgf.paths import CACHED_2P, RESULTS, spks_path
from viral.utils import get_genotype, get_wheel_circumference_from_rig

CONFIG = grosmark_config
WHEEL = get_wheel_circumference_from_rig("2P")
# 20 rather than the 30 used elsewhere. The quality audit showed cell count is
# essentially UNCHANGED across the gap (within-mouse post/pre ratio 0.9-1.4, median
# 1.0), so the imaging is not degraded - but lap count is highly variable in both
# directions (-39 to +41 within mouse), and a 30-lap floor drops half the post-gap
# sessions. At 20 laps the comparison is 4 NLGF and 4 WT mice with both epochs, which
# supports the within-mouse pre-versus-post contrast that is the point of this tier.
N_LAPS = 20
N_CELLS = 250
MAX_LAG = 12
STAGES = ["learning", "reversal", "recall"]


def _lap_maps(session: Cached2pSession, spks: np.ndarray) -> Optional[np.ndarray]:
    trials = []
    for t in session.trials:
        try:
            if trial_is_imaged(t):
                trials.append(t)
        except (AssertionError, IndexError):
            continue
    if len(trials) < N_LAPS:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array([
            activity_trial_position(
                trial=t, flu=spks, wheel_circumference=WHEEL, bin_size=CONFIG.bin_size,
                start=CONFIG.start, max_position=CONFIG.end, verbose=False,
                do_shuffle=False, threshold_speed=True)
            for t in trials[:N_LAPS]
        ], dtype=np.float32)


def _pv_correlation_by_lag(lap_maps: np.ndarray) -> Dict[str, float]:
    """Population vector correlation between laps, as a function of lap separation.

    Each lap is flattened to a single cells x bins vector, so this measures whether the
    whole map is reproduced from lap to lap, not whether individual cells are.
    """
    n_laps = lap_maps.shape[0]
    flat = lap_maps.reshape(n_laps, -1)
    flat = np.nan_to_num(flat)
    keep = flat.std(axis=1) > 0
    flat, idx = flat[keep], np.flatnonzero(keep)
    if flat.shape[0] < 10:
        return {}

    with np.errstate(invalid="ignore"):
        c = np.corrcoef(flat)
    lags, vals = [], []
    # idx holds the ORIGINAL lap numbers of the laps that survived, so the separation
    # between two rows of c is idx[b] - idx[a], not b - a
    for lag in range(1, min(MAX_LAG, flat.shape[0] - 1) + 1):
        d = np.array([c[a, b] for a in range(len(idx)) for b in range(len(idx))
                      if idx[b] - idx[a] == lag])
        d = d[np.isfinite(d)]
        if d.size:
            lags.append(lag)
            vals.append(float(d.mean()))
    if len(lags) < 5:
        return {}
    fit = linregress(lags, vals)
    return dict(pv_lag1=vals[0], pv_slope=float(fit.slope),
                pv_mean=float(np.mean(vals)))


def _geometry(lap_maps: np.ndarray) -> np.ndarray:
    """Position x position correlation matrix of the trial-averaged population."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean_map = gaussian_filter1d(
            np.nanmean(lap_maps, axis=0), sigma=7.5 / CONFIG.bin_size, axis=1)
    mean_map = np.nan_to_num(mean_map)
    with np.errstate(invalid="ignore"):
        g = np.corrcoef(mean_map.T)
    return np.nan_to_num(g)


def session_recall(mouse: str, date: str) -> Optional[Dict]:
    spath = spks_path(mouse, date)
    jpath = CACHED_2P / f"{mouse}_{date}.json"
    if not spath.exists() or not jpath.exists():
        return None
    spks_full = np.asarray(np.load(spath, mmap_mode="r"), dtype=np.float32)
    if spks_full.shape[0] < N_CELLS:
        return None
    rng = np.random.default_rng(0)
    cells = np.sort(rng.choice(spks_full.shape[0], N_CELLS, replace=False))

    session = Cached2pSession.model_validate_json(jpath.read_text())
    lap_maps = _lap_maps(session, spks_full[cells])
    if lap_maps is None:
        return None

    row: Dict = dict(mouse=mouse, date=date, genotype=get_genotype(mouse),
                     session_type=session.session_type,
                     n_cells_total=int(spks_full.shape[0]))
    row.update(_pv_correlation_by_lag(lap_maps))
    if "pv_lag1" not in row:
        return None

    g = _geometry(lap_maps)
    np.save(RESULTS / f"geometry_{mouse}_{date}.npy", g.astype(np.float32))
    # Summary of the geometry: how fast similarity falls with position separation
    n = g.shape[0]
    sep, val = [], []
    for d in range(1, n // 2):
        v = np.diagonal(g, offset=d)
        sep.append(d * CONFIG.bin_size)
        val.append(float(np.mean(v)))
    fit = linregress(sep, val)
    row["geom_slope"] = float(fit.slope)
    row["geom_at_20cm"] = float(np.interp(20.0, sep, val))
    return row


def recall_table() -> pd.DataFrame:
    coh = sessions(min_cells=N_CELLS, stages=STAGES)
    print(f"{len(coh)} sessions across {STAGES}", flush=True)
    rows = []
    for i, r in enumerate(coh.itertuples(), 1):
        try:
            row = session_recall(r.mouse, r.date)
            if row is not None:
                row.update(stage=r.stage, n_trials=r.n_trials_imaged)
                rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        if i % 10 == 0:
            print(f"  [{i}/{len(coh)}] kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df["epoch"] = np.where(df.stage == "recall", "post-gap", "pre-gap")
    df.to_csv(RESULTS / "recall_stability.csv", index=False)
    return df


if __name__ == "__main__":
    df = recall_table()
    print(f"\n{len(df)} sessions, {df.mouse.nunique()} mice")
    print(df.groupby(["genotype", "epoch"]).agg(
        mice=("mouse", "nunique"), sessions=("date", "size"),
        cells=("n_cells_total", "median"), trials=("n_trials", "median"),
        pv_lag1=("pv_lag1", "median"), pv_slope=("pv_slope", "median"),
        geom=("geom_at_20cm", "median")).to_string())
