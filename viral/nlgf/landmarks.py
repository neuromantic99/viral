"""Tier 3: is the corridor code anchored to the landmarks, and is that anchoring lost?

RSC's specialisation is landmark and cue coding, and retrosplenial hypometabolism is
the earliest functional signature of preclinical AD, so "landmark anchoring is lost" is
the most specific RSC-flavoured hypothesis available here.

This is done on ONLINE decoding, not the offline SCE decoding of Chang et al. That
measure was already shown, in the WT work, to flip sign between 11.8 and 14.4 cm of
decoder error and to be uninterpretable at the accuracy this preparation reaches. The
online version has no such problem: position is known, so decoding error can be
computed per spatial bin directly.

Two measures, both criterion-free (no place-field threshold anywhere):

  error profile   median absolute decoding error per spatial bin, regressed on distance
                  to the nearest landmark. If landmarks anchor the code, error should be
                  LOWER near them, i.e. a POSITIVE slope of error against distance.
  peak density    fraction of cells whose rate map peaks in each bin, regressed the same
                  way. A NEGATIVE slope means peaks cluster near landmarks.

Both slopes are reported as z-scores against a PHASE-SHIFTED null, not as raw slopes.
The landmarks sit at 45/90/135 cm, mid-track, so distance-to-nearest-landmark is small
in the middle and large at both ends - and decoding error is intrinsically lowest at
the ends of a linear track, where position is least ambiguous. A raw slope therefore
measures the track's edge profile far more than it measures landmark anchoring: the
first session tested gave -0.41, the opposite sign to the anchoring prediction, purely
from that geometry. The null shifts the whole landmark set by every offset from 1 to 44
cm, preserving the 45 cm spacing and hence the shape of the distance profile, and
changing only its phase. The z-score asks whether the code is anchored to the landmarks
specifically rather than to any comb of that spacing.

Trial and cell counts are matched exactly as in decoding_matched, because Tier 2
established that unmatched versions of all these quantities track data quantity rather
than genotype.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.bayesian_decoder import decode, map_estimate
from viral.chang import (
    ChangConfig,
    distance_to_nearest_landmark,
    run_rate_map,
    running_frames_and_positions,
)
from viral.imaging_utils import trial_is_imaged
from viral.models import Cached2pSession
from viral.nlgf.decoding_matched import N_CELLS, N_REPEATS, _safe_imaged
from viral.nlgf.paths import CACHED_2P, RESULTS, spks_path
from viral.utils import get_genotype

CHANG = ChangConfig()
LANDMARK_SPACING_CM = 45.0

# A per-BIN error profile needs many more decode bins than a single median error does.
# At 24 trials only ~14 of the 70 spatial bins collect the 2+ decoded samples they need,
# below the 15 the regression requires. 40 trials fills ~40 bins and costs sessions but
# not mice: the floor moves from 108 sessions to 85, with 7 NLGF and 8 WT either way.
N_TRIALS = 40
MIN_SAMPLES_PER_BIN = 2


def _phase_shifted_z(values: np.ndarray, keep: np.ndarray) -> float:
    """Slope of `values` on distance-to-landmark, z-scored against phase-shifted combs.

    Returns the observed slope's z against the null of every 1 cm phase shift of the
    landmark set, which holds the geometry fixed and moves only where the landmarks sit.
    """
    centres = CHANG.bin_centres_cm
    observed = None
    null = []
    for shift in range(0, int(LANDMARK_SPACING_CM)):
        marks = [m + shift for m in CHANG.landmarks_cm]
        d = distance_to_nearest_landmark(centres, marks)
        slope = linregress(d[keep], values[keep]).slope
        if shift == 0:
            observed = slope
        else:
            null.append(slope)
    null = np.asarray(null)
    sd = null.std(ddof=1)
    return float((observed - null.mean()) / sd) if sd > 0 else np.nan


def _per_bin_error(
    session: Cached2pSession, spks: np.ndarray
) -> Optional[Dict[str, np.ndarray]]:
    """Median absolute decoding error in each spatial bin, on held-out even trials."""
    template, _, valid = run_rate_map(session, spks, CHANG)
    frames, positions = running_frames_and_positions(session, CHANG, parity=0)
    per_bin = CHANG.decode_frames_per_bin_online
    if frames.size < per_bin:
        return None

    n = frames.size // per_bin
    usable = n * per_bin
    counts = spks[:, frames[:usable]].reshape(spks.shape[0], n, per_bin).sum(axis=2).T
    true_position = positions[:usable].reshape(n, per_bin).mean(axis=1)

    keep = (counts > 0).sum(axis=1) >= CHANG.min_active_cells
    if keep.sum() < 20:
        return None

    posterior = np.where(
        valid[np.newaxis, :], decode(counts[keep], template, tau=per_bin / CHANG.fs),
        -np.inf,
    )
    decoded = map_estimate(posterior) * CHANG.position_bin_cm + CHANG.position_start_cm
    truth = true_position[keep]
    error = np.abs(decoded - truth)

    bins = np.clip(
        ((truth - CHANG.position_start_cm) / CHANG.position_bin_cm).astype(int),
        0, CHANG.n_position_bins - 1,
    )
    profile = np.full(CHANG.n_position_bins, np.nan)
    for b in range(CHANG.n_position_bins):
        m = bins == b
        if m.sum() >= MIN_SAMPLES_PER_BIN:
            profile[b] = np.median(error[m])
    return dict(profile=profile, valid=valid, template=template)


def session_landmarks(mouse: str, date: str) -> Optional[Dict]:
    spath = spks_path(mouse, date)
    if not spath.exists():
        return None
    session = Cached2pSession.model_validate_json(
        (CACHED_2P / f"{mouse}_{date}.json").read_text())
    imaged = [t for t in session.trials if _safe_imaged(t)]
    if len(imaged) < N_TRIALS:
        return None
    spks_full = np.asarray(np.load(spath, mmap_mode="r"), dtype=np.float32)
    if spks_full.shape[0] < N_CELLS:
        return None

    matched = session.model_copy(update={"trials": imaged[:N_TRIALS]})
    distance = distance_to_nearest_landmark(
        CHANG.bin_centres_cm, list(CHANG.landmarks_cm))

    rng = np.random.default_rng(0)
    err_slopes, peak_slopes, profiles = [], [], []
    for _ in range(N_REPEATS):
        cells = rng.choice(spks_full.shape[0], N_CELLS, replace=False)
        spks = spks_full[cells]
        out = _per_bin_error(matched, spks)
        if out is None:
            continue
        profile, valid = out["profile"], out["valid"]

        ok = np.isfinite(profile) & valid
        if ok.sum() >= 15:
            err_slopes.append(_phase_shifted_z(np.nan_to_num(profile), ok))
            profiles.append(profile)

        # Peak density: where each cell's rate map is maximal
        template = out["template"]
        active = template.sum(axis=1) > 0
        if active.sum() >= 20:
            peaks = np.argmax(template[active][:, valid], axis=1)
            idx = np.flatnonzero(valid)[peaks]
            density = np.bincount(idx, minlength=CHANG.n_position_bins) / active.sum()
            peak_slopes.append(_phase_shifted_z(density, valid))

    if not err_slopes:
        return None

    return dict(
        mouse=mouse, date=date, genotype=get_genotype(mouse),
        session_type=session.session_type,
        error_landmark_z=float(np.median(err_slopes)),
        peak_landmark_z=float(np.median(peak_slopes)) if peak_slopes else np.nan,
        mean_error_cm=float(np.nanmedian([np.nanmedian(p) for p in profiles])),
        n_repeats=len(err_slopes),
    )


def landmark_table() -> pd.DataFrame:
    from viral.nlgf.cohort import describe, sessions

    led = sessions()
    print(f"cohort: {describe(led)}", flush=True)
    rows = []
    for i, r in enumerate(led.itertuples(), 1):
        try:
            row = session_landmarks(r.mouse, r.date)
            if row is not None:
                rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        if i % 10 == 0:
            print(f"  [{i}/{len(led)}] kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "landmarks.csv", index=False)
    return df


if __name__ == "__main__":
    df = landmark_table()
    print(f"\n{len(df)} sessions, {df.mouse.nunique()} mice")
    print(df.groupby("genotype").agg(
        mice=("mouse", "nunique"), sessions=("date", "size"),
        err_z=("error_landmark_z", "median"),
        peak_z=("peak_landmark_z", "median")).to_string())
