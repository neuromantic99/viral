"""Stability of the spatial map across days, on registered cells.

The cross-session texture analysis asks whether a trial-type discrimination transfers.
This asks the different and more standard question: does the same cell have the same
place field a day or two later? That is the Ziv-style drift measure, and the direct
analogue of the "offline map stabilization" that Shipley et al. report reduced in this
mouse model - measured here in cortex, over days rather than across a rest period.

Three measures, all on cells matched across the pair:

  field correlation   per cell, the correlation of its rate map on day A with day B.
                      Median over cells. This is stability of individual fields.
  PV correlation      per position bin, the correlation of the population vector on
                      day A with day B. Median over bins. This is stability of the
                      population code, and can hold even if individual cells drift.
  centroid shift      median distance the peak of a cell's field moves, in cm.

Each is reported against a WITHIN-SESSION split-half baseline: odd laps versus even
laps in the same session, on the same cells. Without that ceiling a low cross-day
correlation cannot be told from a noisy map, and both genotypes have plenty of noise -
the snake plots make that obvious. The quantity to read is the cross-day value relative
to its own within-day baseline.

Matching follows the rest of the package: 30 laps per session, per-cell residual
filtering to drop bad matches, and an equalised matched-cell pool, since NLGF pairs
have more matched cells and transfer measures track registration quality.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.constants import grosmark_config
from viral.imaging_utils import activity_trial_position, trial_is_imaged
from viral.models import Cached2pSession
from viral.nlgf.cross_session import MAX_CELL_RESIDUAL, N_POOL
from viral.nlgf.paths import CACHED_2P, RESULTS, spks_path
from viral.nlgf.registration_qc import _centroids, _fit_affine
from viral.utils import get_genotype, get_wheel_circumference_from_rig

CONFIG = grosmark_config
WHEEL = get_wheel_circumference_from_rig("2P")
N_LAPS = 30
SIGMA = 7.5 / CONFIG.bin_size


def _lap_maps(mouse: str, date: str, rows: np.ndarray) -> Optional[np.ndarray]:
    """(laps, cells, bins) for the given spks rows, over the first N_LAPS imaged laps."""
    path = spks_path(mouse, date)
    if not path.exists():
        return None
    session = Cached2pSession.model_validate_json(
        (CACHED_2P / f"{mouse}_{date}.json").read_text())
    trials = []
    for t in session.trials:
        try:
            if trial_is_imaged(t):
                trials.append(t)
        except (AssertionError, IndexError):
            continue
    if len(trials) < N_LAPS:
        return None
    spks = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if rows.max(initial=-1) >= spks.shape[0]:
        return None
    spks = spks[rows]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array([
            activity_trial_position(
                trial=t, flu=spks, wheel_circumference=WHEEL,
                bin_size=CONFIG.bin_size, start=CONFIG.start, max_position=CONFIG.end,
                verbose=False, do_shuffle=False, threshold_speed=True)
            for t in trials[:N_LAPS]], dtype=np.float32)


def _smooth(maps: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gaussian_filter1d(np.nan_to_num(np.nanmean(maps, axis=0)),
                                 sigma=SIGMA, axis=1)


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (a * b).sum(axis=1) / denom, np.nan)


def pair_stability(mouse: str, date_a: str, date_b: str) -> Optional[Dict]:
    mpath = RESULTS / f"matches_{mouse}_{date_a}_{date_b}.csv"
    if not mpath.exists():
        return None
    matches = pd.read_csv(mpath)

    try:
        ca, cb = _centroids(mouse, date_a), _centroids(mouse, date_b)
        if matches.row_a.max() < len(ca) and matches.row_b.max() < len(cb):
            _, resid = _fit_affine(ca[matches.row_a.to_numpy()],
                                   cb[matches.row_b.to_numpy()])
            matches = matches[resid <= MAX_CELL_RESIDUAL]
    except Exception:  # noqa: BLE001
        pass
    if len(matches) < N_POOL:
        return None
    rng = np.random.default_rng(0)
    matches = matches.iloc[np.sort(rng.choice(len(matches), N_POOL, replace=False))]

    ma = _lap_maps(mouse, date_a, matches.row_a.to_numpy())
    mb = _lap_maps(mouse, date_b, matches.row_b.to_numpy())
    if ma is None or mb is None:
        return None

    a, b = _smooth(ma), _smooth(mb)
    # within-session ceilings, odd versus even laps on the same cells
    a_odd, a_even = _smooth(ma[1::2]), _smooth(ma[0::2])
    b_odd, b_even = _smooth(mb[1::2]), _smooth(mb[0::2])

    field_cross = np.nanmedian(_rowwise_corr(a, b))
    field_within = np.nanmedian(np.r_[_rowwise_corr(a_odd, a_even),
                                      _rowwise_corr(b_odd, b_even)])
    pv_cross = np.nanmedian(_rowwise_corr(a.T, b.T))
    pv_within = np.nanmedian(np.r_[_rowwise_corr(a_odd.T, a_even.T),
                                   _rowwise_corr(b_odd.T, b_even.T)])
    shift = np.abs(np.argmax(a, axis=1) - np.argmax(b, axis=1)) * CONFIG.bin_size

    return dict(
        mouse=mouse, date_a=date_a, date_b=date_b, genotype=get_genotype(mouse),
        gap_days=int((pd.to_datetime(date_b) - pd.to_datetime(date_a)).days),
        n_cells=int(len(matches)),
        field_corr_cross=float(field_cross), field_corr_within=float(field_within),
        field_corr_ratio=float(field_cross / field_within) if field_within else np.nan,
        pv_corr_cross=float(pv_cross), pv_corr_within=float(pv_within),
        pv_corr_ratio=float(pv_cross / pv_within) if pv_within else np.nan,
        centroid_shift_cm=float(np.median(shift)),
    )


def stability_table() -> pd.DataFrame:
    qc = pd.read_csv(RESULTS / "registration_qc.csv")
    good = qc[qc.usable]
    print(f"{len(good)} QC-passing pairs", flush=True)
    rows: List[Dict] = []
    for i, r in enumerate(good.itertuples(), 1):
        try:
            out = pair_stability(r.mouse, r.date_a, r.date_b)
            if out is not None:
                out["affine_rms_px"] = r.affine_rms_px
                rows.append(out)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date_a} {r.date_b}: {type(e).__name__}: {e}",
                  flush=True)
        if i % 10 == 0:
            print(f"  [{i}/{len(good)}] kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "spatial_stability.csv", index=False)
    return df


if __name__ == "__main__":
    df = stability_table()
    print(f"\n{len(df)} pairs, {df.mouse.nunique()} mice")
    print(df.groupby("genotype").agg(
        pairs=("mouse", "size"),
        field_cross=("field_corr_cross", "median"),
        field_within=("field_corr_within", "median"),
        field_ratio=("field_corr_ratio", "median"),
        pv_cross=("pv_corr_cross", "median"),
        pv_ratio=("pv_corr_ratio", "median"),
        shift_cm=("centroid_shift_cm", "median")).round(3).to_string())
