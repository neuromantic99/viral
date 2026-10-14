"""Tier 2: spatial coding fidelity.

Does RSC still represent the corridor as well in NLGF? Four things, from cheapest to
most integrative:

  yield        fraction of cells with a significant place field
  information  Skaggs spatial information per cell, in bits per event
  field width  extent of the supra-threshold region, in cm
  reliability  correlation between the odd-lap and even-lap rate map, per cell
  decoding     median absolute error of a Bayesian decoder on held-out even trials,
               which is the population-level readout and the one that does not depend
               on any place-field criterion

Place cells are recomputed rather than reloaded, because get_place_cells writes its
caches to SERVER_PATH unconditionally and cannot be called here. The expensive half of
it - 2000 per-lap circular shuffles - is NOT recomputed: place_threshold on the drive
is exactly that shuffle distribution's 99th percentile, so the detection is reproduced
by comparing a freshly binned rate map against the cached threshold. Everything that
touches the definition (activity_trial_position, has_n_consecutive_trues,
filter_additional_check) is imported unchanged, so the place cells here are the same
cells get_place_cells would return.

Running speed differs by genotype (see fig_behaviour), and speed affects occupancy,
sampling and rate. It is carried through to the comparison as a covariate rather than
being assumed away.
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

from viral.chang import ChangConfig, decoding_error, run_rate_map
from viral.constants import grosmark_config
from viral.grosmark_analysis import filter_additional_check
from viral.imaging_utils import activity_trial_position, trial_is_imaged
from viral.models import Cached2pSession
from viral.nlgf.paths import CACHED_2P, PLACE_THRESHOLD, RESULTS, spks_path
from viral.utils import (
    find_n_consecutive_trues_extent,
    get_genotype,
    get_wheel_circumference_from_rig,
    has_n_consecutive_trues,
)

CONFIG = grosmark_config
CHANG = ChangConfig()
WHEEL = get_wheel_circumference_from_rig("2P")
SIGMA_CM = 7.5
FS = 30
MIN_TRIALS = 10


def threshold_path(mouse: str, date: str, rewarded: Optional[bool]) -> Path:
    return (
        PLACE_THRESHOLD
        / f"{mouse}_{date}_rewarded_{rewarded}_{CONFIG}_place_threshold_BOD_IGNORE!.npy"
    )


def _skaggs(rate_map_: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """Skaggs spatial information, bits per event, per cell.

    sum_i p_i (lambda_i / lambda) log2(lambda_i / lambda), with p_i the occupancy
    probability of bin i. Bits per EVENT rather than per second, so it is not carried
    by an overall rate difference - which matters here, since the genotypes could
    differ in rate without differing in tuning.
    """
    p = occupancy / occupancy.sum()
    lam = rate_map_
    lam_bar = (p[np.newaxis, :] * lam).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = lam / lam_bar[:, np.newaxis]
        term = p[np.newaxis, :] * ratio * np.log2(ratio)
    return np.nansum(term, axis=1)


def session_place(mouse: str, date: str, rewarded: Optional[bool] = None
                  ) -> Optional[Dict]:
    tpath = threshold_path(mouse, date, rewarded)
    spath = spks_path(mouse, date)
    if not tpath.exists() or not spath.exists():
        return None

    session = Cached2pSession.model_validate_json(
        (CACHED_2P / f"{mouse}_{date}.json").read_text()
    )
    spks = np.asarray(np.load(spath, mmap_mode="r"), dtype=np.float32)

    trials = [
        t for t in session.trials
        if _safe_imaged(t) and (rewarded is None or t.texture_rewarded == rewarded)
    ]
    if len(trials) < MIN_TRIALS:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # empty position bins give all-NaN slices
        all_trials = np.array([
            activity_trial_position(
                trial=t, flu=spks, wheel_circumference=WHEEL,
                bin_size=CONFIG.bin_size, start=CONFIG.start, max_position=CONFIG.end,
                verbose=False, do_shuffle=False, threshold_speed=True,
            )
            for t in trials
        ])
        smoothed = gaussian_filter1d(
            np.nanmean(all_trials, 0), sigma=SIGMA_CM / CONFIG.bin_size, axis=1
        )

    place_threshold = np.load(tpath)
    if place_threshold.shape != smoothed.shape:
        # Cell count changed since the threshold was cached (re-run suite2p); the
        # threshold is per cell per bin so it cannot be reused
        return None

    n_consecutive = int((2 / CONFIG.bin_size) * 5)
    if n_consecutive % 2 == 0:
        n_consecutive += 1

    supra = smoothed > place_threshold
    pcs = has_n_consecutive_trues(supra, n_consecutive)
    if pcs.sum() == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        additional = filter_additional_check(
            all_trials=all_trials[:, pcs, :],
            place_threshold=place_threshold[pcs, :],
            smoothed_matrix=smoothed[pcs, :],
            n_consecutive_trues=n_consecutive,
        )
    pcs_combined = pcs.copy()
    pcs_combined[pcs] = additional

    n_cells = spks.shape[0]
    row: Dict = dict(
        mouse=mouse, date=date, genotype=get_genotype(mouse),
        session_type=session.session_type, n_cells=n_cells, n_trials=len(trials),
        n_place_cells=int(pcs_combined.sum()),
        frac_place_cells=float(pcs_combined.mean()),
    )

    # Occupancy across the analysed bins, from the same speed-thresholded frames the
    # rate maps were built from
    occupancy = np.nansum(~np.isnan(all_trials[:, 0, :]), axis=0).astype(float)
    occupancy = np.where(occupancy > 0, occupancy, np.nan)
    valid_occ = np.nan_to_num(occupancy, nan=0.0)

    if valid_occ.sum() > 0 and pcs_combined.any():
        si = _skaggs(np.clip(smoothed, 0, None), valid_occ)
        row["spatial_info_pc"] = float(np.nanmedian(si[pcs_combined]))
        row["spatial_info_all"] = float(np.nanmedian(si))

        extents = find_n_consecutive_trues_extent(supra[pcs_combined], n_consecutive)
        row["field_width_cm"] = float(np.median(extents.sum(axis=1)) * CONFIG.bin_size)

        odd = np.nanmean(all_trials[1::2][:, pcs_combined, :], axis=0)
        even = np.nanmean(all_trials[0::2][:, pcs_combined, :], axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rel = np.array([
                np.corrcoef(o, e)[0, 1] for o, e in zip(odd, even)
            ])
        row["reliability_pc"] = float(np.nanmedian(rel))

    # Population decoding, independent of the place-field criterion
    try:
        template, _, valid_bins = run_rate_map(session, spks, CHANG)
        row["decoding_error_cm"] = decoding_error(
            session, spks, template, valid_bins, CHANG
        )
        row["n_valid_bins"] = int(valid_bins.sum())
    except Exception:
        row["decoding_error_cm"] = np.nan
        row["n_valid_bins"] = 0

    return row


def _safe_imaged(trial) -> bool:
    try:
        return trial_is_imaged(trial)
    except (AssertionError, IndexError):
        return False


def place_table(rewarded: Optional[bool] = None) -> pd.DataFrame:
    from viral.nlgf.load import ledger

    led = ledger()
    led = led[led.has_spks]
    rows = []
    for i, r in enumerate(led.itertuples(), 1):
        try:
            row = session_place(r.mouse, r.date, rewarded)
            if row is not None:
                rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        if i % 10 == 0:
            print(f"  [{i}/{len(led)}] kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"place_sessions_rewarded_{rewarded}.csv", index=False)
    return df


if __name__ == "__main__":
    df = place_table(None)
    print(f"\n{len(df)} sessions, {df.mouse.nunique()} mice")
    print(df.groupby("genotype").agg(
        mice=("mouse", "nunique"), sessions=("date", "size"),
        frac_pc=("frac_place_cells", "median"),
        si=("spatial_info_pc", "median"),
        width=("field_width_cm", "median"),
        rel=("reliability_pc", "median"),
        dec=("decoding_error_cm", "median"),
    ).to_string())
