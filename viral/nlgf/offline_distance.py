"""Grosmark figure 4 e/f in RSC: resting correlation as a function of place field distance.

The question is whether the offline correlation structure knows about the map. If cells
whose fields sit close together on the track are more correlated during immobility than
cells whose fields sit far apart, then what recurs at rest carries spatial information
and not merely a general level of coupling. That distance dependence is the minimal
signature of sequence-related reactivation available without decoding, and it is the
one measure here that cannot be produced by a global change in excitability: a uniform
increase in coupling lifts the whole curve and leaves its slope alone.

Everything that does the work is imported unchanged from viral.grosmark_analysis -
get_offline_correlation_matrix (150 ms Gaussian kernel on the immobility frames, then a
pairwise Pearson matrix) and correlations_vs_peak_distance (20 cm sliding bins over
|peak_i - peak_j|). Only the inputs are sourced locally, for the two reasons given in
reactivation.py: get_place_cells writes its caches to the server, and get_movement_bool
raises on the nine sessions whose movement vectors are malformed.

Two decisions carried over from the rest of this package rather than re-litigated:

  pool-matched   250 cells are drawn from the full population FIRST and place cells are
                 detected within that fixed pool. Place cells are selected FROM the
                 population, so a fixed number drawn from a large field of view is a
                 more selective set than the same number drawn from a small one, and
                 NLGF fields are larger here for trial-count reasons. Matching the pool
                 is what equalises the selectivity; matching the place cell count does
                 not. This is the step that removed the apparent genotype effect in
                 reactivation.py and it removes it from the correlation LEVEL here too.
  graded         spks are used as the denoised amplitudes, not binarised to spks > 0.
                 Binarising costs roughly 3-5x in sensitivity for co-activation.

Two nulls, because the level and the shape are different claims.

  shuf_   get_offline_correlation_matrix's own do_shuffle, which is shuffle_rows: each
          cell's samples are permuted in time independently, before the 150 ms kernel.
          That destroys cross-cell timing AND each cell's own autocorrelation, so it is
          a floor rather than a matched null - it says the correlations are not an
          artefact of the smoothing, nothing more. It is kept because it is the null in
          the Grosmark reproduction these curves are being compared against.
  perm_   the correlation matrix is left exactly as measured and the PEAK POSITIONS are
          permuted across cells, N_PEAK_PERMS times. Every correlation, every cell's
          rate and the whole distribution of pair correlations are preserved; only the
          correspondence between a pair's correlation and its field distance is broken.
          This is the null for the shape, and slope_z is scored against it.

The level itself is not tested here - a uniform change in coupling is what
activity_graded.py measures, against a circular-shift null.

The immobility frames are concatenated across bouts before correlating, which splices
discontiguous times together. This follows grosmark_analysis.offline_correlations
unchanged, and at a 150 ms kernel the edge each splice introduces is a few frames out of
tens of thousands - but it is a real approximation and both nulls are computed on the
same spliced matrix, so neither is credited for it.

NEAR_CM / FAR_CM collapse each curve to one number, near minus far, so the slope can
enter a per-mouse test. Both are read off the curve the figure plots.
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
from viral.grosmark_analysis import (
    correlations_vs_peak_distance,
    get_offline_correlation_matrix,
)
from viral.imaging_utils import activity_trial_position, trial_is_imaged
from viral.models import Cached2pSession
from viral.nlgf.load import load_digest
from viral.nlgf.paths import CACHED_2P, RESULTS, spks_path
from viral.nlgf.reactivation import (
    FS,
    MIN_EPOCH_SECONDS,
    N_TOTAL_CELLS,
    WHEEL,
    local_place_cells,
)
from viral.utils import get_genotype

CONFIG = grosmark_config

MIN_PLACE_CELLS = 15   # pairs, not cells, are the unit; 15 cells is 105 pairs
N_PEAK_PERMS = 50      # for the shape null; the slope SD over these gives slope_z
NEAR_CM = 20.0         # the first bin of the curve
FAR_CM = 60.0          # bins from here up are "far"


def _smoothed_map(session: Cached2pSession, spks: np.ndarray) -> Optional[np.ndarray]:
    """Trial-averaged, smoothed rate map, for the field PEAKS only.

    local_place_cells builds this internally but returns only the boolean mask. It is
    recomputed here rather than copied out of it so that the place cell definition stays
    single-sourced - a second copy of the thresholding is how the two would drift apart.
    """
    trials = []
    for t in session.trials:
        try:
            if trial_is_imaged(t):
                trials.append(t)
        except (AssertionError, IndexError):
            continue
    if len(trials) < 10:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_trials = np.array([
            activity_trial_position(
                trial=t, flu=spks, wheel_circumference=WHEEL, bin_size=CONFIG.bin_size,
                start=CONFIG.start, max_position=CONFIG.end, verbose=False,
                do_shuffle=False, threshold_speed=True)
            for t in trials
        ])
        return gaussian_filter1d(
            np.nanmean(all_trials, 0), sigma=7.5 / CONFIG.bin_size, axis=1)


def _summarise(x: np.ndarray, y: np.ndarray, prefix: str) -> Dict:
    """One curve as near, far, their difference, and the curve itself."""
    near = float(np.nanmean(y[x < NEAR_CM]))
    far = float(np.nanmean(y[x >= FAR_CM]))
    return {
        f"{prefix}near": near,
        f"{prefix}far": far,
        f"{prefix}near_minus_far": near - far,
        f"{prefix}curve": ";".join(f"{v:.6g}" for v in y),
    }


def session_curves(mouse: str, date: str) -> List[Dict]:
    spath = spks_path(mouse, date)
    if not spath.exists():
        return []
    digest = load_digest(mouse, date)
    if not digest.meta.get("freeze_usable", False):
        return []

    spks_full = np.asarray(np.load(spath, mmap_mode="r"), dtype=np.float64)
    if spks_full.shape[0] < N_TOTAL_CELLS:
        return []
    # Same seed and same draw as reactivation.py, so the two analyses describe the same
    # cells and their numbers are directly comparable
    pool = np.sort(np.random.default_rng(0).choice(
        spks_full.shape[0], N_TOTAL_CELLS, replace=False))
    spks = spks_full[pool]

    masks = digest.immobility_masks(n_frames=spks.shape[1])
    usable = [e for e, m in masks.items() if m.sum() / FS >= MIN_EPOCH_SECONDS]
    if not usable:
        return []

    session = Cached2pSession.model_validate_json(
        (CACHED_2P / f"{mouse}_{date}.json").read_text())
    pcs = local_place_cells(session, spks, pool=pool)
    if pcs is None or pcs.sum() < MIN_PLACE_CELLS:
        return []
    smoothed = _smoothed_map(session, spks)
    if smoothed is None:
        return []

    peaks = np.argmax(smoothed[pcs], axis=1) * CONFIG.bin_size + CONFIG.start
    place_spks = spks[pcs]

    records = []
    for epoch in usable:
        frames = place_spks[:, masks[epoch]]
        row: Dict = dict(
            mouse=mouse, date=date, genotype=get_genotype(mouse),
            session_type=session.session_type, epoch=epoch,
            n_place_cells=int(pcs.sum()), n_cells_pool=int(N_TOTAL_CELLS),
            n_cells=int(spks_full.shape[0]), n_trials=len(digest.trials),
            freeze_type=digest.meta.get("freeze_type"),
            epoch_seconds=float(masks[epoch].sum() / FS),
        )
        for kind, shuffle in (("", False), ("shuf_", True)):
            corrs = get_offline_correlation_matrix(
                offline=frames, wheel_freeze=True, do_shuffle=shuffle, plot=False)
            x, y = correlations_vs_peak_distance(corrs, peak_position_cm=peaks)
            row.update(_summarise(x, y, kind))
            if kind:
                continue
            # The shape null: same matrix, permuted peaks
            rng = np.random.default_rng(1)
            perm = np.array([
                correlations_vs_peak_distance(
                    corrs, peak_position_cm=rng.permutation(peaks))[1]
                for _ in range(N_PEAK_PERMS)
            ])
            slopes = np.array([_summarise(x, c, "")["near_minus_far"] for c in perm])
            row.update(_summarise(x, np.nanmean(perm, axis=0), "perm_"))
            sd = float(np.nanstd(slopes))
            row["perm_slope_sd"] = sd
            row["slope_z"] = ((row["near_minus_far"] - float(np.nanmean(slopes))) / sd
                              if sd > 0 else np.nan)
        records.append(row)
    return records


def curve_of(row: pd.Series, column: str = "curve") -> np.ndarray:
    return np.array([float(v) for v in str(row[column]).split(";")])


def distance_bins() -> np.ndarray:
    """The x axis correlations_vs_peak_distance produces: 20 cm sliding bin STARTS."""
    return np.arange(120)


def table() -> pd.DataFrame:
    from viral.nlgf.cohort import describe, sessions

    led = sessions()
    led = led[led.has_freeze & led.freeze_usable]
    print(f"cohort: {describe(led)}", flush=True)
    rows: List[Dict] = []
    for i, r in enumerate(led.itertuples(), 1):
        try:
            recs = session_curves(r.mouse, r.date)
            for rec in recs:
                rec.update(stage=r.stage)
            rows.extend(recs)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        print(f"  [{i}/{len(led)}] {r.mouse} {r.date} -> {len(rows)} records",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "offline_distance.csv", index=False)
    return df


if __name__ == "__main__":
    df = table()
    if len(df):
        print(f"\n{df.groupby(['mouse','date']).ngroups} sessions, "
              f"{df.mouse.nunique()} mice")
        print(df.groupby(["genotype", "epoch"]).agg(
            sessions=("date", "nunique"),
            near=("near", "median"), far=("far", "median"),
            slope=("near_minus_far", "median"),
            perm_slope=("perm_near_minus_far", "median"),
            slope_z=("slope_z", "median"),
            shuf_slope=("shuf_near_minus_far", "median")).to_string())
