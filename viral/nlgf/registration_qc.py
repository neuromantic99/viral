"""Sharper registration quality metrics, computed from the saved match tables.

residual_rms_px in the diagnostics removes only a median TRANSLATION before measuring
how much matched pairs disagree. That conflates three things: genuine non-rigid
deformation of the field of view, centroid jitter from the two days' masks differing,
and actual mismatches. Only the third is a reason to distrust a pair.

Fitting a full affine (translation, rotation, scale, shear) to the matched centroids
before taking the residual removes the first source, so what remains is jitter plus
mismatches. Two numbers come out of it:

    affine_rms_px    residual after the affine fit. Lower than the translation-only
                     figure by however much real deformation there was.
    outlier_frac     fraction of pairs more than OUTLIER_PX from the affine prediction.
                     Closer to a mismatch rate than any RMS, because RMS squares and a
                     few bad pairs dominate it.

The fit is RANSAC-like: it is estimated on inliers only, iteratively, so that the bad
pairs do not drag the transform toward themselves and hide.

Recomputed from the match CSVs, so no re-registration is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.paths import DFF, RESULTS
from viral.utils import get_genotype

OUTLIER_PX = 6.0
N_ITER = 5
MIN_MATCHED = 100
MAX_AFFINE_RMS = 8.0


def _centroids(mouse: str, date: str) -> np.ndarray:
    stat = np.load(DFF / f"{mouse}_{date}_stat.npy", allow_pickle=True)
    iscell = np.load(DFF / f"{mouse}_{date}_iscell.npy")[:, 0].astype(bool)
    med = np.array([r["med"] for r in stat[iscell]], dtype=float)
    return med[:, ::-1]  # suite2p med is (y, x)


def _fit_affine(a: np.ndarray, b: np.ndarray) -> tuple:
    """Least-squares affine a -> b, refit on inliers so outliers cannot hide."""
    keep = np.ones(len(a), dtype=bool)
    A = np.hstack([a, np.ones((len(a), 1))])
    for _ in range(N_ITER):
        coef, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
        resid = np.linalg.norm(A @ coef - b, axis=1)
        new = resid <= max(OUTLIER_PX, np.median(resid) * 3)
        if new.sum() < 20 or (new == keep).all():
            break
        keep = new
    return coef, np.linalg.norm(A @ coef - b, axis=1)


def pair_qc(mouse: str, date_a: str, date_b: str) -> Optional[Dict]:
    path = RESULTS / f"matches_{mouse}_{date_a}_{date_b}.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path)
    if len(m) < 20:
        return None
    ca, cb = _centroids(mouse, date_a), _centroids(mouse, date_b)
    if m.row_a.max() >= len(ca) or m.row_b.max() >= len(cb):
        return None
    a, b = ca[m.row_a.to_numpy()], cb[m.row_b.to_numpy()]

    d = b - a
    shift = np.median(d, axis=0)
    trans_rms = float(np.sqrt(((d - shift) ** 2).sum(axis=1).mean()))

    _, resid = _fit_affine(a, b)
    return dict(
        mouse=mouse, date_a=date_a, date_b=date_b, genotype=get_genotype(mouse),
        n_matched=int(len(m)),
        translation_rms_px=trans_rms,
        affine_rms_px=float(np.sqrt((resid ** 2).mean())),
        affine_median_px=float(np.median(resid)),
        outlier_frac=float(np.mean(resid > OUTLIER_PX)),
        deformation_px=float(trans_rms - np.sqrt((resid ** 2).mean())),
    )


def qc_table() -> pd.DataFrame:
    rows = []
    for f in sorted(RESULTS.glob("matches_*.csv")):
        mouse, date_a, date_b = f.stem[len("matches_"):].split("_")
        try:
            r = pair_qc(mouse, date_a, date_b)
            if r:
                rows.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {f.stem}: {type(e).__name__}: {e}")
    df = pd.DataFrame(rows)
    if len(df):
        df["usable"] = (df.n_matched >= MIN_MATCHED) & (df.affine_rms_px <= MAX_AFFINE_RMS)
        df.to_csv(RESULTS / "registration_qc.csv", index=False)
    return df


if __name__ == "__main__":
    df = qc_table()
    print(f"{len(df)} pairs\n")
    print(df[["translation_rms_px", "affine_rms_px", "outlier_frac",
              "deformation_px"]].describe().round(3).to_string())
    print("\n=== by genotype ===")
    print(df.groupby("genotype")[["translation_rms_px", "affine_rms_px",
                                  "outlier_frac", "n_matched"]].median().round(3).to_string())
    print(f"\nusable pairs (>= {MIN_MATCHED} matched, affine RMS <= {MAX_AFFINE_RMS} px):")
    print(df.groupby(["genotype", "usable"]).size().to_string())
