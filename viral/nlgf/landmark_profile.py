"""Tier 3, said plainly: what does firing rate do as the animal passes a landmark?

The existing landmark analysis (landmarks.py) regresses decoding error and peak density
on distance-to-nearest-landmark and reports a z-score against a comb of phase-shifted
pseudo-landmark sets. It is the right test, but it takes a paragraph to explain and
reads as overengineered - the objection raised in lab meeting.

This is the same question with nothing between the reader and the data: take every
landmark crossing, line them up, and plot the population firing rate against distance
from the landmark. If RSC anchors its code to the landmarks there is a bump at zero.

Three things are needed to make that bump mean something.

  ANTI-PHASE CONTROL. The landmarks sit at 45/90/135 cm - a regular 45 cm comb - so any
  periodic structure in the corridor produces a bump at zero whether or not it has
  anything to do with the landmarks. The same profile is therefore built around
  pseudo-landmarks at 67.5/112.5 cm, exactly out of phase, where there is nothing. Real
  anchoring is the difference between the two.

  SPEED. Animals slow at landmarks, deconvolved rate rises with speed, so a rate bump
  can be a speed bump wearing a hat. The speed profile is built from the same frames and
  plotted underneath, and the modulation is recomputed on rate residuals after
  regressing rate on speed across bins within each session.

  MATCHING. Trials and cells are subsampled to fixed counts, because Tier 2 established
  that unmatched versions of every quantity in this study track data quantity.

Modulation index is (near - far) / (near + far), near being within NEAR_CM of the
landmark and far beyond FAR_CM. Zero means flat, positive means rate is elevated at the
landmark. It is bounded, unitless and needs one sentence to explain.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.cohort import sessions
from viral.nlgf.landmarks import CHANG
from viral.nlgf.load import load_digest
from viral.nlgf.paths import RESULTS, spks_path

FS = 30
LANDMARKS_CM = (45.0, 90.0, 135.0)
PSEUDO_CM = (67.5, 112.5)          # exactly out of phase with the real comb
HALF_WINDOW_CM = 22.0              # half the 45 cm spacing: windows just touch
NEAR_CM, FAR_CM = 6.0, 16.0
N_TRIALS = 40
N_CELLS = 250
MIN_FRAMES_PER_BIN = 3


def _rate_and_speed(mouse: str, date: str, rng: np.random.Generator) -> Optional[tuple]:
    """Per-bin population rate (each cell normalised to its own mean) and median speed."""
    path = spks_path(mouse, date)
    if not path.exists():
        return None
    d = load_digest(mouse, date)
    spks = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if spks.shape[0] < N_CELLS:
        return None
    spks = spks[rng.choice(spks.shape[0], N_CELLS, replace=False)]

    pos, frame, lap = d.run_position_cm, d.run_frame, d.run_lap
    laps = np.unique(lap)
    if laps.size < N_TRIALS:
        return None
    keep_laps = rng.choice(laps, N_TRIALS, replace=False)
    m = np.isin(lap, keep_laps) & (frame < spks.shape[1])
    pos, frame, lap = pos[m], frame[m].astype(int), lap[m]
    if pos.size < 200:
        return None

    edges = np.arange(CHANG.position_start_cm,
                      CHANG.position_end_cm + CHANG.position_bin_cm,
                      CHANG.position_bin_cm)
    centres = (edges[:-1] + edges[1:]) / 2
    idx = np.digitize(pos, edges) - 1
    n_bins = CHANG.n_position_bins

    # speed from consecutive samples within a lap, so a lap reset is never a velocity
    speed = np.full(pos.size, np.nan)
    for l in keep_laps:
        s = np.flatnonzero(lap == l)
        if s.size < 3:
            continue
        dp, df = np.diff(pos[s]), np.diff(frame[s]).astype(float)
        ok = df > 0
        v = np.full(s.size, np.nan)
        v[:-1][ok] = dp[ok] / df[ok] * FS
        speed[s] = v

    rate = np.full((N_CELLS, n_bins), np.nan)
    spd = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() < MIN_FRAMES_PER_BIN:
            continue
        rate[:, b] = spks[:, frame[sel]].mean(axis=1)
        with np.errstate(invalid="ignore"):
            spd[b] = np.nanmedian(speed[sel])

    # each cell normalised to its own mean, so a few high-rate cells cannot set the shape
    means = np.nanmean(rate, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(means > 0, rate / means, np.nan)
    return np.nanmean(rate, axis=0), spd, centres


def _align(profile: np.ndarray, centres: np.ndarray, anchors) -> tuple:
    """Average the profile over windows centred on each anchor position."""
    offsets = np.arange(-HALF_WINDOW_CM, HALF_WINDOW_CM + CHANG.position_bin_cm,
                        CHANG.position_bin_cm)
    stack = []
    for a in anchors:
        vals = np.interp(a + offsets, centres, profile, left=np.nan, right=np.nan)
        stack.append(vals)
    return offsets, np.nanmean(np.vstack(stack), axis=0)


def _modulation(offsets: np.ndarray, values: np.ndarray) -> float:
    near = np.nanmean(values[np.abs(offsets) <= NEAR_CM])
    far = np.nanmean(values[np.abs(offsets) >= FAR_CM])
    if not np.isfinite(near) or not np.isfinite(far) or (near + far) == 0:
        return np.nan
    return float((near - far) / (near + far))


def session_profile(mouse: str, date: str) -> Optional[Dict]:
    rng = np.random.default_rng(abs(hash((mouse, date))) % (2**32))
    got = _rate_and_speed(mouse, date, rng)
    if got is None:
        return None
    rate, spd, centres = got
    if np.isfinite(rate).sum() < 30:
        return None

    # rate with the speed relationship removed, fitted across bins within this session
    ok = np.isfinite(rate) & np.isfinite(spd)
    resid = rate.copy()
    if ok.sum() > 10:
        b, a = np.polyfit(spd[ok], rate[ok], 1)
        resid = rate - (a + b * spd)

    offsets, r_real = _align(rate, centres, LANDMARKS_CM)
    _, r_pseudo = _align(rate, centres, PSEUDO_CM)
    _, s_real = _align(spd, centres, LANDMARKS_CM)
    _, s_pseudo = _align(spd, centres, PSEUDO_CM)
    _, q_real = _align(resid, centres, LANDMARKS_CM)
    _, q_pseudo = _align(resid, centres, PSEUDO_CM)

    d = load_digest(mouse, date)
    return dict(
        mouse=mouse, date=date, genotype=d.genotype,
        session_type=d.meta["session_type"],
        rate_mod=_modulation(offsets, r_real),
        rate_mod_pseudo=_modulation(offsets, r_pseudo),
        speed_mod=_modulation(offsets, s_real),
        speed_mod_pseudo=_modulation(offsets, s_pseudo),
        resid_mod=_modulation(offsets, q_real),
        resid_mod_pseudo=_modulation(offsets, q_pseudo),
        offsets=offsets, rate_real=r_real, rate_pseudo=r_pseudo,
        speed_real=s_real, speed_pseudo=s_pseudo,
        resid_real=q_real, resid_pseudo=q_pseudo,
    )


def profile_table() -> pd.DataFrame:
    led = sessions()
    rows, curves = [], {}
    for i, r in enumerate(led.itertuples(), 1):
        try:
            out = session_profile(r.mouse, r.date)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
            continue
        if out is None:
            continue
        key = f"{out['mouse']}_{out['date']}"
        for k in ("offsets", "rate_real", "rate_pseudo", "speed_real", "speed_pseudo",
                  "resid_real", "resid_pseudo"):
            curves[f"{key}_{k}"] = out.pop(k)
        out.update(stage=r.stage)
        rows.append(out)
        if i % 10 == 0:
            print(f"  [{i}/{len(led)}] kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "landmark_profile.csv", index=False)
    np.savez_compressed(RESULTS / "landmark_profiles.npz", **curves)
    print(f"wrote {len(df)} sessions")
    return df


if __name__ == "__main__":
    profile_table()
