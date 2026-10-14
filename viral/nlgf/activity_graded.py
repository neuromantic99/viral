"""Tier 1's synchrony measures, binary and graded, against the same fair null.

Everything in `_synchrony` is computed on `spks > 0`. The SCE structure work showed
that binarising costs roughly five-fold sensitivity for co-activation (co-participation
z of 15.6 binary versus 73.0 graded), so Tier 1's synchrony nulls are weaker than they
were reported to be. This recomputes them both ways on the same sessions, same cells,
same frames, so the question "would the genotype conclusion change" gets an answer
rather than an argument.

WHY GRADED NEEDS A DIFFERENT NULL. Denoised calcium traces are temporally smooth, so
any two of them correlate simply through shared slow structure - brain state, neuropil,
bleaching. That is the real reason the field deconvolves before measuring synchrony,
and it is a good reason. A raw graded correlation is NOT comparable to a binary one.
The circular-shift null fixes this: rolling each cell's trace independently preserves
its autocorrelation and amplitude distribution exactly and destroys only cross-cell
timing, so the z it produces is directly comparable between representations. Graded
without that null would be meaningless; with it, it is the more sensitive measure.

MEAN PAIRWISE CORRELATION, CHEAPLY. For z-scored cells, var(sum over cells) = n +
n(n-1) * mean_r, so the mean pairwise correlation follows from the variance of the
population sum alone - O(cells x frames) instead of O(cells^2 x frames). That is what
makes a 100-shuffle circular-shift null affordable here at all; the pairwise form would
take hours.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.activity import FS, MIN_FRAMES, N_CELLS, N_FRAMES_EQUALISED
from viral.nlgf.cohort import sessions
from viral.nlgf.immobility import immobility_mask
from viral.nlgf.load import load_digest
from viral.nlgf.paths import DFF, RESULTS, spks_path

N_SHUFFLES = 100

# QC. A mean pairwise correlation this high across 250 cortical cells is not
# physiological - the bulk of sessions sit at 0.05-0.15 - and the offending sessions
# fail in two distinguishable ways, both artefactual:
#
#   a brief global excursion   78% of the population-sum variance in the top 1% of
#                              frames, max |z| = 12.6, while the BINARY correlation
#                              stays at 0.008. No extra spikes occurred, so this is
#                              motion or z-drift, not co-firing.
#   a sustained global mode    variance spread normally across frames, yet r = 0.76 for
#                              the whole epoch. One shared component dominating the FOV.
#
# The graded measure is exposed to both because a shared artefact IS shared signal - the
# circular-shift null cannot help, since it correctly reports the artefact as
# non-random. The binary measure is immune, which is what identifies these sessions.
#
# Applied blind to genotype. It drops 7 of 103 sessions, all WT, and since the genotype
# effect is WT > NLGF, removing them is conservative rather than favourable.
MAX_PLAUSIBLE_R = 0.20


def _mean_r_from_sum(X: np.ndarray) -> float:
    """Mean pairwise correlation of z-scored rows, via the variance of their sum."""
    n = X.shape[0]
    if n < 2:
        return np.nan
    s = X.sum(axis=0)
    return float((s.var() - n) / (n * (n - 1)))


def _cofluctuation(X: np.ndarray) -> float:
    """var(population sum) / sum(per-cell var). 1.0 if cells were independent."""
    denom = X.var(axis=1).sum()
    return float(X.sum(axis=0).var() / denom) if denom > 0 else np.nan


def _z_against_shift(X: np.ndarray, fn, rng: np.random.Generator) -> tuple:
    obs = fn(X)
    n_frames = X.shape[1]
    null = np.empty(N_SHUFFLES)
    for i in range(N_SHUFFLES):
        shifts = rng.integers(0, n_frames, size=X.shape[0])
        null[i] = fn(np.stack([np.roll(X[j], int(k)) for j, k in enumerate(shifts)]))
    mu, sd = float(np.nanmean(null)), float(np.nanstd(null))
    return obs, mu, (obs - mu) / sd if sd > 0 else np.nan


def _zscore_rows(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return np.divide(X - mu, sd, out=np.zeros_like(X), where=sd > 0)


def session_graded(mouse: str, date: str) -> Optional[Dict]:
    p, dn = spks_path(mouse, date), DFF / f"{mouse}_{date}_denoised.npy"
    if not (p.exists() and dn.exists()):
        return None
    spks_all = np.load(p, mmap_mode="r")
    if spks_all.shape[0] < N_CELLS:
        return None
    n_frames = spks_all.shape[1]

    rng = np.random.default_rng(abs(hash((mouse, date))) % (2**32))
    keep = np.sort(rng.choice(spks_all.shape[0], N_CELLS, replace=False))
    binary = (np.asarray(spks_all)[keep] > 0).astype(np.float32)
    den = np.asarray(np.load(dn, mmap_mode="r"))[keep].astype(np.float32)
    if den.shape[1] != n_frames:
        return None

    try:
        still, _ = immobility_mask(mouse, date)
    except Exception:  # noqa: BLE001
        return None
    d = load_digest(mouse, date)
    run = np.zeros(n_frames, dtype=bool)
    rf = d.run_frame[d.run_frame < n_frames].astype(int)
    run[rf] = True
    if still.size != n_frames:
        return None

    row: Dict = dict(mouse=mouse, date=date, genotype=d.genotype,
                     session_type=d.meta["session_type"], n_cells=N_CELLS)
    for epoch, mask in (("run", run), ("still", still)):
        if mask.sum() < MIN_FRAMES:
            continue
        take = np.sort(rng.choice(int(mask.sum()),
                                  min(N_FRAMES_EQUALISED, int(mask.sum())),
                                  replace=False))
        idx = np.flatnonzero(mask)[take]
        row[f"{epoch}_n_frames"] = float(idx.size)
        for name, X in (("binary", binary[:, idx]), ("graded", den[:, idx])):
            Z = _zscore_rows(X)
            obs, mu, z = _z_against_shift(Z, _mean_r_from_sum, rng)
            row[f"{epoch}_{name}_mean_r"] = obs
            row[f"{epoch}_{name}_mean_r_null"] = mu
            row[f"{epoch}_{name}_mean_r_z"] = z
            obs, mu, z = _z_against_shift(Z, _cofluctuation, rng)
            row[f"{epoch}_{name}_cofluct"] = obs
            row[f"{epoch}_{name}_cofluct_z"] = z
    return row


def graded_table() -> pd.DataFrame:
    led = sessions()
    rows = []
    for i, r in enumerate(led.itertuples(), 1):
        try:
            out = session_graded(r.mouse, r.date)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
            continue
        if out is not None:
            out.update(stage=r.stage)
            rows.append(out)
        if i % 10 == 0:
            print(f"  [{i}/{len(led)}] kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    for epoch in ("run", "still"):
        col = f"{epoch}_graded_mean_r"
        if col in df:
            bad = df[col] > MAX_PLAUSIBLE_R
            if bad.any():
                print(f"QC: {int(bad.sum())} sessions fail {col} <= {MAX_PLAUSIBLE_R}",
                      flush=True)
                print(df.loc[bad, ["mouse", "date", "genotype", col]]
                      .round(3).to_string(index=False), flush=True)
            df.loc[bad, [c for c in df.columns if c.startswith(f"{epoch}_graded")]] = np.nan
            df[f"{epoch}_qc_fail"] = bad
    df.to_csv(RESULTS / "activity_graded.csv", index=False)
    print(f"wrote {len(df)} sessions")
    return df


if __name__ == "__main__":
    df = graded_table()
    if df.empty:
        sys.exit()
    from viral.nlgf.stats import exact_permutation

    print("\n=== sensitivity: how far above its own null does each representation sit? ===")
    for epoch in ("run", "still"):
        for name in ("binary", "graded"):
            col = f"{epoch}_{name}_mean_r_z"
            if col in df:
                print(f"  {epoch:5s} {name:6s} mean_r   median z = "
                      f"{df[col].median():8.1f}")
    print("\n=== genotype, on the z scale ===")
    for epoch in ("run", "still"):
        for name in ("binary", "graded"):
            for meas in ("mean_r_z", "cofluct_z"):
                col = f"{epoch}_{name}_{meas}"
                if col not in df:
                    continue
                s = df[df.genotype.isin(["WT", "NLGF"])].dropna(subset=[col])
                pm = s.groupby(["mouse", "genotype"], as_index=False)[col].median()
                if pm.groupby("genotype").size().min() < 3:
                    continue
                r = exact_permutation(pm, col)
                print(f"  {epoch:5s} {name:6s} {meas:10s} NLGF {r['mean_a']:8.1f}  "
                      f"WT {r['mean_b']:8.1f}   p = {r['p']:.3f}")
