"""Chang et al. (2020) on the local copy, for both genotypes.

chang.py already implements the paper. It cannot be called from this package because
three of its inputs go to the server: get_place_cells writes its shuffle caches there,
load_spks validates against suite2p's iscell.npy there, and get_df_summary iterates the
server's session JSONs. Every function that does the actual work - detect_sce,
cluster_ensembles, explained_variance, run_rate_map, decoding_error,
decode_sce_positions, landmark_regression - is imported and used unchanged; only those
three inputs are sourced locally, which is the same arrangement reactivation.py uses.

It is also extended in the one way that matters for this project: chang.py was run on WT
only, and the question here is whether the genotypes differ.

WHAT THIS MEASURES, AND WHY IT IS NOT THE SAME AS fig43.

Our pre/post contrast elsewhere is reactivation STRENGTH, and it is flat. Chang et al.
never report a strength difference either. Their claim is about CONTENT: the positions
that resting synchronous events decode to sit closer to spatial landmarks in REST2 than
in REST1. That is a different measurement on the same epochs, and it is the one
published result from a design that matches ours almost exactly - dysgranular RSC layers
II-III, two-photon, a FAMILIAR linear treadmill with tactile landmarks, and the belt
clamped for a rest epoch either side of the run.

Two design choices come with it and are kept:

  ensembles are clustered WITHIN each rest epoch rather than carried over from running.
  That sidesteps the circularity in the run-template approach, where cells correlated
  during running are correlated offline for anatomical and neuropil reasons that no
  weight shuffle can separate from reinstatement. Here membership comes from rest and
  spatial tuning from run, so the link between them is the thing being tested.

  explained variance (Kudrimoti et al. 1999) asks whether REST2's correlation structure
  resembles RUN more than REST1's does, with REST1 partialled out. EV > REV is the
  reactivation claim; EV == REV means the structure was already there, which is what a
  familiar track predicts.

DENOISED BY DEFAULT. Chang et al. correlate continuous deconvolved rates. Our spks are
binarised and thinned, which discards amplitude, and amplitude carries most of the
signal in every co-activation measure in this package. Place cells and the decoding
template still come from spks, exactly as chang.py does it when passed `activity`.

NO POOL MATCHING, deliberately, unlike reactivation.py. The primary claim is within
session - pre versus post on one population - so cell count cancels. n_place_cells is
recorded so the between-genotype panels can be checked against it, and it should be,
because those panels do NOT get that protection.

The decoding error on held-out even trials is the gate for everything downstream. If the
template cannot decode running position, the offline decoded positions are noise and the
landmark measure says nothing either way. Chance is (end - start) / 3 = 46.7 cm.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.chang import (
    ChangConfig,
    cluster_ensembles,
    decode_sce_positions,
    decoding_error,
    detect_sce,
    distance_to_nearest_landmark,
    explained_variance,
    landmark_regression,
    run_rate_map,
    running_frames,
    smoothed_rates,
)
from viral.models import Cached2pSession
from viral.nlgf.load import load_digest
from viral.nlgf.paths import CACHED_2P, DFF, RESULTS, spks_path
from viral.nlgf.reactivation import local_place_cells
from viral.utils import get_genotype

CONFIG = ChangConfig()
FS = CONFIG.fs
MIN_EPOCH_SECONDS = 60.0
USE_DENOISED = True

# A session takes ~90 s, so the full ledger is an hour and a half in one process.
# VIRAL_SHARD="i/n" runs every n'th session starting at i and writes its own CSV, which
# is merged by merge_shards(). Defaults to the whole ledger and a single output file, so
# nothing changes for a plain run.
SHARD = os.environ.get("VIRAL_SHARD", "")


def _denoised(mouse: str, date: str, n_cells: int, n_frames: int) -> Optional[np.ndarray]:
    path = DFF / f"{mouse}_{date}_denoised.npy"
    if not path.exists():
        return None
    arr = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if arr.shape != (n_cells, n_frames):
        return None
    return arr


def session_chang(mouse: str, date: str) -> List[Dict]:
    spath = spks_path(mouse, date)
    if not spath.exists():
        return []
    digest = load_digest(mouse, date)
    if not digest.meta.get("freeze_usable", False):
        return []

    spks = np.asarray(np.load(spath, mmap_mode="r"), dtype=np.float64)
    masks = digest.immobility_masks(n_frames=spks.shape[1])
    usable = {e: m for e, m in masks.items()
              if m.sum() / FS >= MIN_EPOCH_SECONDS}
    if len(usable) < 2:
        return []

    session = Cached2pSession.model_validate_json(
        (CACHED_2P / f"{mouse}_{date}.json").read_text())
    pcs = local_place_cells(session, spks)
    if pcs is None:
        return []
    place_cells = spks[pcs]

    activity = _denoised(mouse, date, spks.shape[0], spks.shape[1]) if USE_DENOISED \
        else None
    if USE_DENOISED and activity is None:
        return []
    rates = smoothed_rates(
        place_cells if activity is None else activity[pcs], CONFIG)

    template, run_frames, valid_bins = run_rate_map(session, place_cells, CONFIG)
    if run_frames.size == 0:
        return []
    error = decoding_error(session, place_cells, template, valid_bins, CONFIG)

    ev_run_frames = running_frames(session, CONFIG)
    ev, rev = explained_variance(
        run=rates[:, ev_run_frames],
        rest_pre=rates[:, usable["pre"]],
        rest_post=rates[:, usable["post"]],
    )

    edges = np.arange(CONFIG.position_start_cm,
                      CONFIG.position_end_cm + CONFIG.position_bin_cm,
                      CONFIG.position_bin_cm)
    records = []
    for epoch, mask in usable.items():
        epoch_frames = np.flatnonzero(mask)
        epoch_rates = rates[:, epoch_frames]
        events = detect_sce(epoch_rates, CONFIG)
        if events:
            ensembles = cluster_ensembles(epoch_rates, CONFIG)
            decoded = decode_sce_positions(
                place_cells, events, epoch_frames, template, valid_bins, CONFIG)
            distances = distance_to_nearest_landmark(
                decoded, list(CONFIG.landmarks_cm))
            slope, slope_p = landmark_regression(decoded, CONFIG)
        else:
            ensembles, decoded = [], np.array([], dtype=float)
            distances = np.array([], dtype=float)
            slope, slope_p = np.nan, np.nan

        # The spatial profile itself, so the figure can plot what the summary
        # statistics are summarising rather than only the summary
        if decoded.size:
            counts, _ = np.histogram(decoded, bins=edges)
            profile = counts / counts.sum()
        else:
            profile = np.full(edges.size - 1, np.nan)

        records.append(dict(
            mouse=mouse, date=date, genotype=get_genotype(mouse),
            session_type=session.session_type, epoch=epoch,
            epoch_seconds=float(mask.sum() / FS),
            n_place_cells=int(pcs.sum()), n_cells=int(spks.shape[0]),
            n_trials=len(digest.trials),
            freeze_type=digest.meta.get("freeze_type"),
            n_sce=len(events),
            sce_rate_hz=len(events) / (mask.sum() / FS),
            n_ensembles=len(ensembles),
            mean_ensemble_size=(float(np.mean([e.size for e in ensembles]))
                                if ensembles else np.nan),
            n_decoded_bins=int(decoded.size),
            median_landmark_distance=(float(np.median(distances))
                                      if distances.size else np.nan),
            landmark_slope=slope, landmark_slope_p=slope_p,
            median_decoded_position=(float(np.median(decoded))
                                     if decoded.size else np.nan),
            decoding_error_cm=error,
            explained_variance=ev, reverse_explained_variance=rev,
            profile=";".join(f"{v:.6g}" for v in profile),
        ))
    return records


def profile_of(row: pd.Series) -> np.ndarray:
    return np.array([float(v) for v in str(row["profile"]).split(";")])


def bin_centres() -> np.ndarray:
    return CONFIG.bin_centres_cm


def chance_landmark_distance() -> float:
    """Median distance to the nearest landmark for uniformly decoded positions."""
    positions = np.linspace(CONFIG.position_start_cm, CONFIG.position_end_cm, 200_000)
    return float(np.median(
        distance_to_nearest_landmark(positions, list(CONFIG.landmarks_cm))))


def merge_shards() -> pd.DataFrame:
    """Concatenate the shard CSVs into the single table the figures read."""
    parts = sorted(RESULTS.glob("chang_local_shard*.csv"))
    if not parts:
        raise FileNotFoundError(f"no shard CSVs in {RESULTS}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.sort_values(["mouse", "date", "epoch"]).reset_index(drop=True)
    df.to_csv(RESULTS / "chang_local.csv", index=False)
    print(f"merged {len(parts)} shards -> {len(df)} rows")
    return df


def table() -> pd.DataFrame:
    from viral.nlgf.cohort import describe, sessions

    led = sessions()
    led = led[led.has_freeze & led.freeze_usable]
    out = RESULTS / "chang_local.csv"
    if SHARD:
        index, total = (int(v) for v in SHARD.split("/"))
        led = led.iloc[index::total]
        out = RESULTS / f"chang_local_shard{index}.csv"
        print(f"shard {index}/{total}: {len(led)} sessions", flush=True)
    print(f"cohort: {describe(led)}", flush=True)
    rows: List[Dict] = []
    for i, r in enumerate(led.itertuples(), 1):
        try:
            recs = session_chang(r.mouse, r.date)
            for rec in recs:
                rec.update(stage=r.stage)
            rows.extend(recs)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        print(f"  [{i}/{len(led)}] {r.mouse} {r.date} -> {len(rows)} records",
              flush=True)
        # Written as it goes: the full ledger is ~90 minutes in one process, and the
        # external drive is the bottleneck, so a crash near the end is expensive
        if rows and i % 5 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    df = table()
    if len(df):
        print(f"\n{df.groupby(['mouse','date']).ngroups} sessions, "
              f"{df.mouse.nunique()} mice   chance landmark distance "
              f"{chance_landmark_distance():.1f} cm")
        print(df.groupby(["genotype", "epoch"]).agg(
            sessions=("date", "nunique"),
            err=("decoding_error_cm", "median"),
            n_sce=("n_sce", "median"),
            lm=("median_landmark_distance", "median"),
            slope=("landmark_slope", "median"),
            ev=("explained_variance", "median"),
            rev=("reverse_explained_variance", "median")).to_string())
