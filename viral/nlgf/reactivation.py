"""Tier 5: offline reactivation in the freeze epochs, by genotype.

freeze_ensemble_reactivation cannot be called here for two reasons: it obtains place
cells through get_place_cells, which writes its caches to SERVER_PATH, and it builds
its immobility masks through get_movement_bool, which raises on the nine sessions whose
movement vectors are malformed. Everything that does the actual work -
build_offline_matrix_from_mask, build_run_ensembles, score_offline_reactivation - is
imported and used unchanged; only the two inputs are sourced locally.

The question is LEVEL, not the pre/post delta. The WT work established that
reactivation strength is present but does not increase from pre to post on a familiar
track, on two independent decoder-free measures. Asking whether that flat delta differs
by genotype is asking whether two nulls differ, which this n cannot answer. Asking
whether NLGF reactivate their run ensembles as strongly at all is a level question, it
is better powered, and it is the one that maps onto a consolidation deficit.

excess_r (observed minus the shuffled null) is reported rather than mean_rz, following
score_offline_reactivation's own docstring: the z divides by a term that shrinks as
place cells are added, and place cell yield differs by genotype here for reasons that
are an artefact of trial count (see fig_confound).

That is necessary but NOT sufficient. Measured across this dataset, excess_r itself
correlates rho = +0.80 with place cell count and +0.73 with total cell count, and NLGF
sessions carry 663 place cells against WT's 273 - so the unmatched contrast (NLGF 2.25
vs WT 0.70) is the Tier 2 artefact again. The docstring's stability check varied the
number of OTHER place cells around a FIXED assembly; here the assembly itself grows
with the cell set, ICA finds more and larger components, and mean_r rises with it.

Matching the ASSEMBLY is still not enough, which took three passes to pin down. With
100 place cells drawn per session the genotype effect persisted (post epoch p = 0.005,
the exact-permutation floor at n = 4 v 6) but collapsed to p = 0.955 once total
field-of-view cell count entered the model, and excess_r still correlated with it
(rho = +0.22). The reason is selection: place cells are chosen FROM the full population,
so 100 drawn from a field of 900 are a more selective set than 100 drawn from 400, and
NLGF fields are larger.

So the match happens one step earlier. N_TOTAL_CELLS cells are drawn from the whole
population FIRST, place cells are then detected within that fixed-size pool, and the
assembly is built from those. Every session therefore selects from an identically sized
population, which is the only way the selectivity is equalised rather than merely the
count. Sessions below N_TOTAL_CELLS are dropped.
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
from viral.ensemble_reactivation import (
    build_offline_matrix_from_mask,
    build_run_ensembles,
    score_offline_reactivation,
)
from viral.grosmark_analysis import filter_additional_check
from viral.imaging_utils import activity_trial_position, trial_is_imaged
from viral.models import Cached2pSession
from viral.nlgf.load import load_digest
from viral.nlgf.paths import CACHED_2P, RESULTS, spks_path
from viral.nlgf.place import threshold_path
from viral.utils import get_genotype, get_wheel_circumference_from_rig, has_n_consecutive_trues

CONFIG = grosmark_config
WHEEL = get_wheel_circumference_from_rig("2P")
FS = 30
MIN_EPOCH_SECONDS = 60.0
N_SHUFFLES = 500
N_TOTAL_CELLS = 250   # pool that place cells are selected FROM, matched across sessions
N_PLACE_CELLS = 60    # assembly size, drawn from the place cells found in that pool
N_REPEATS = 3


def local_place_cells(session: Cached2pSession, spks: np.ndarray,
                      pool: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """pcs_combined, reproducing get_place_cells against the cached shuffle threshold.

    `pool` is the row indices of `spks` within the FULL session, needed when spks has
    been subsampled. place_threshold is stored per cell per bin and each cell's row
    comes from np.nanpercentile over that cell's own 2000 shuffled maps, so it does not
    depend on which other cells were present: threshold[pool] is exactly the threshold
    those cells would have had. Without this the shapes disagree and the session is
    silently dropped."""
    tpath = threshold_path(session.mouse_name, session.date, None)
    if not tpath.exists():
        return None
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
        smoothed = gaussian_filter1d(
            np.nanmean(all_trials, 0), sigma=7.5 / CONFIG.bin_size, axis=1)

    threshold = np.load(tpath)
    if pool is not None:
        if threshold.shape[0] <= pool.max():
            raise ValueError(
                f"{session.mouse_name} {session.date}: threshold has "
                f"{threshold.shape[0]} cells but pool indexes up to {pool.max()}")
        threshold = threshold[pool]
    if threshold.shape != smoothed.shape:
        # Loud, not silent: a shape disagreement here drops the session, and dropping
        # every session still looks like a clean run that simply found nothing
        raise ValueError(
            f"{session.mouse_name} {session.date}: place_threshold is "
            f"{threshold.shape} but the rate map is {smoothed.shape}")

    n = int((2 / CONFIG.bin_size) * 5)
    if n % 2 == 0:
        n += 1
    pcs = has_n_consecutive_trues(smoothed > threshold, n)
    if pcs.sum() == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extra = filter_additional_check(
            all_trials=all_trials[:, pcs, :], place_threshold=threshold[pcs, :],
            smoothed_matrix=smoothed[pcs, :], n_consecutive_trues=n)
    out = pcs.copy()
    out[pcs] = extra
    return out if out.sum() >= 10 else None


def session_reactivation(mouse: str, date: str, match_cells: bool = True) -> List[Dict]:
    spath = spks_path(mouse, date)
    if not spath.exists():
        return []
    digest = load_digest(mouse, date)
    if not digest.meta.get("freeze_usable", False):
        return []

    spks_full = np.asarray(np.load(spath, mmap_mode="r"), dtype=np.float64)
    if spks_full.shape[0] < N_TOTAL_CELLS:
        return []
    pool_rng = np.random.default_rng(0)
    pool = np.sort(pool_rng.choice(spks_full.shape[0], N_TOTAL_CELLS, replace=False))
    spks = spks_full[pool]
    masks = digest.immobility_masks(n_frames=spks.shape[1])
    if len(masks) < 2:
        return []

    session = Cached2pSession.model_validate_json(
        (CACHED_2P / f"{mouse}_{date}.json").read_text())
    pcs = local_place_cells(session, spks, pool=pool)
    if pcs is None:
        return []
    pcs_idx = np.flatnonzero(pcs)
    if match_cells and pcs_idx.size < N_PLACE_CELLS:
        return []

    usable = [e for e, m in masks.items() if m.sum() / FS >= MIN_EPOCH_SECONDS]
    if len(usable) < 2:
        return []

    rng = np.random.default_rng(0)
    repeats = N_REPEATS if match_cells else 1
    per_epoch: Dict[str, List[float]] = {e: [] for e in usable}
    per_epoch_rz: Dict[str, List[float]] = {e: [] for e in usable}
    n_components_seen = []

    for _ in range(repeats):
        chosen = (rng.choice(pcs_idx, N_PLACE_CELLS, replace=False)
                  if match_cells else pcs_idx)
        place_cells = spks[np.sort(chosen), :]
        ensembles = build_run_ensembles(
            session, place_cells, rewarded=None, n_component_method="circular_shift")
        if ensembles.shape[1] == 0:
            continue
        n_components_seen.append(int(ensembles.shape[1]))
        for epoch in usable:
            offline_z = build_offline_matrix_from_mask(place_cells, masks[epoch])
            mean_r, mean_rz, null_mean, _ = score_offline_reactivation(
                offline_z, ensembles, n_shuffles=N_SHUFFLES)
            # Median over components within a draw: they share one ICA and one cell set
            per_epoch[epoch].append(float(np.median(mean_r - null_mean)))
            per_epoch_rz[epoch].append(float(np.median(mean_rz)))

    if not n_components_seen:
        return []

    records = []
    for epoch in usable:
        if not per_epoch[epoch]:
            continue
        records.append(dict(
            mouse=mouse, date=date, genotype=get_genotype(mouse),
            session_type=session.session_type, epoch=epoch,
            n_components=float(np.median(n_components_seen)),
            n_place_cells_available=int(pcs.sum()),
            n_place_cells_used=int(N_PLACE_CELLS if match_cells else pcs.sum()),
            n_cells_pool=int(N_TOTAL_CELLS), n_cells=int(spks_full.shape[0]),
            n_trials=len(digest.trials),
            freeze_type=digest.meta.get("freeze_type"),
            epoch_seconds=float(masks[epoch].sum() / FS),
            excess_r=float(np.median(per_epoch[epoch])),
            mean_rz=float(np.median(per_epoch_rz[epoch])),
            n_repeats=len(per_epoch[epoch]),
        ))
    return records


def reactivation_table() -> pd.DataFrame:
    from viral.nlgf.cohort import describe, sessions

    led = sessions()
    led = led[led.has_freeze & led.freeze_usable]
    print(f"cohort: {describe(led)}", flush=True)
    rows: List[Dict] = []
    for i, r in enumerate(led.itertuples(), 1):
        try:
            recs = session_reactivation(r.mouse, r.date)
            for rec in recs:
                rec.update(stage=r.stage)
            rows.extend(recs)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        print(f"  [{i}/{len(led)}] {r.mouse} {r.date} -> {len(rows)} records", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "reactivation_freeze_poolmatched.csv", index=False)
    return df


if __name__ == "__main__":
    df = reactivation_table()
    if len(df):
        print(f"\n{df.groupby(['mouse','date']).ngroups} sessions, {df.mouse.nunique()} mice")
        print(df.groupby(["genotype", "epoch"]).agg(
            sessions=("date", "nunique"), excess_r=("excess_r", "median"),
            mean_rz=("mean_rz", "median")).to_string())
