"""Do the animals change speed at the landmarks?

This decides whether Tier 3's landmark result means anything. Decoding runs in fixed
TIME bins, so if the animal slows at a landmark it covers less track within a bin, the
within-bin positional smear shrinks, and decoding error falls there - with no change in
coding at all. That is exactly the positive slope reported as evidence of anchoring.

The phase-shifted null does not catch this. It controls for profiles that vary with
position but are not aligned to the landmark comb; a speed change CAUSED by the
landmarks is phase-locked to them by construction and survives every shift.

Speed is computed from the UNTHRESHOLDED position samples. The speed-thresholded arrays
drop frames below 5 cm/s, which is precisely the slowing this is looking for - using
them would hide the effect and replace it with a hole in the occupancy. Laps are
recovered from position resets, since the unthresholded arrays carry no lap index.

Reports the same phase-shifted z as landmarks.py, so the two are directly comparable: if
speed_z is as large as error_landmark_z, the coding result is not separable from the
behaviour.
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

from viral.chang import ChangConfig, distance_to_nearest_landmark
from viral.nlgf.cohort import describe, sessions
from viral.nlgf.load import load_digest
from viral.nlgf.paths import RESULTS

CHANG = ChangConfig()
FS = 30
LANDMARK_SPACING_CM = 45.0
LAP_RESET_CM = -40.0     # a drop this large in position means a new lap
MAX_SPEED = 120.0        # cm/s; above this is an encoder glitch, not an animal


def _phase_shifted_z(values: np.ndarray, keep: np.ndarray) -> float:
    centres = CHANG.bin_centres_cm
    observed, null = None, []
    for shift in range(0, int(LANDMARK_SPACING_CM)):
        d = distance_to_nearest_landmark(centres, [m + shift for m in CHANG.landmarks_cm])
        slope = linregress(d[keep], values[keep]).slope
        if shift == 0:
            observed = slope
        else:
            null.append(slope)
    null = np.asarray(null)
    sd = null.std(ddof=1)
    return float((observed - null.mean()) / sd) if sd > 0 else np.nan


def session_speed_profile(mouse: str, date: str) -> Optional[Dict]:
    d = load_digest(mouse, date)
    pos, frame = d.run_position_cm_all, d.run_frame_all
    if pos.size < 200:
        return None

    # Laps from position resets, since the unthresholded arrays carry no lap index
    lap = np.zeros(pos.size, dtype=int)
    lap[1:] = np.cumsum(np.diff(pos) < LAP_RESET_CM)

    speeds, positions = [], []
    for l in np.unique(lap):
        m = lap == l
        p, f = pos[m], frame[m].astype(float)
        if p.size < 10:
            continue
        dp, df = np.diff(p), np.diff(f)
        ok = df > 0
        if not ok.any():
            continue
        v = dp[ok] / df[ok] * FS
        at = (p[:-1][ok] + p[1:][ok]) / 2
        good = (v > 0) & (v < MAX_SPEED)
        speeds.append(v[good])
        positions.append(at[good])
    if not speeds:
        return None
    v = np.concatenate(speeds)
    at = np.concatenate(positions)

    edges = np.arange(CHANG.position_start_cm,
                      CHANG.position_end_cm + CHANG.position_bin_cm,
                      CHANG.position_bin_cm)
    idx = np.digitize(at, edges) - 1
    n_bins = CHANG.n_position_bins
    profile = np.full(n_bins, np.nan)
    occupancy = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        occupancy[b] = m.sum()
        if m.sum() >= 5:
            profile[b] = np.median(v[m])

    keep = np.isfinite(profile)
    if keep.sum() < 20:
        return None
    return dict(
        mouse=mouse, date=date, genotype=d.genotype,
        session_type=d.meta["session_type"],
        speed_landmark_z=_phase_shifted_z(np.nan_to_num(profile), keep),
        occupancy_landmark_z=_phase_shifted_z(occupancy, np.ones(n_bins, bool)),
        mean_speed=float(np.nanmean(profile)),
        speed_cv=float(np.nanstd(profile) / np.nanmean(profile)),
        profile=profile, occupancy=occupancy,
    )


def speed_table() -> pd.DataFrame:
    coh = sessions()
    print(f"cohort: {describe(coh)}", flush=True)
    rows, profiles = [], {}
    for r in coh.itertuples():
        try:
            out = session_speed_profile(r.mouse, r.date)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}")
            continue
        if out is None:
            continue
        profiles[f"{out['mouse']}_{out['date']}"] = (out.pop("profile"),
                                                     out.pop("occupancy"))
        out["stage"] = r.stage
        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "speed_profile.csv", index=False)
    np.savez_compressed(RESULTS / "speed_profiles.npz",
                        **{k: np.vstack(v) for k, v in profiles.items()})
    return df


if __name__ == "__main__":
    df = speed_table()
    print(f"\n{len(df)} sessions, {df.mouse.nunique()} mice")
    print(df.groupby("genotype")[["speed_landmark_z", "occupancy_landmark_z",
                                  "mean_speed", "speed_cv"]].median().round(3).to_string())
