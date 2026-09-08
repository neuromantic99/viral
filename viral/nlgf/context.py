"""Tier 4: does RSC update when the reward contingency reverses?

RSC is required for contextual updating, and reversal is where a contextual deficit
should appear even when acquisition is intact - which, behaviourally, it is (Tier 0).

What can and cannot be asked here. Within a session the two textures map deterministically
onto rewarded and unrewarded, so "texture identity" and "reward value" are the same
partition of trials and cannot be separated. Separating them needs a decoder trained on
learning sessions and tested on reversal sessions, which needs cells matched across days,
and no cell registration output is available locally.

What is well posed is the TRAJECTORY. The textures are physically unchanged by a reversal;
only their meaning changes. So:

  if RSC codes the texture (sensory)   discriminability is unaffected by reversal
  if RSC codes the context (value)     discriminability collapses on reversal day 1
                                       and recovers as the new contingency is learned

and the genotype question is whether that recovery differs - a contextual updating deficit
predicts a slower or incomplete return in NLGF.

Two position windows, deliberately kept apart:

  early     20-100 cm. Before anticipatory licking diverges, so discriminability here is
            about the representation rather than the behavioural response it drives.
  approach  100-150 cm. Up to but excluding the reward zone, so reward delivery and
            consummatory licking are never in the window. Anticipatory licking IS, which
            is why it is reported second - a difference here can be motor.

Cells and trials are matched by construction, following Tier 2 and Tier 5, where
unmatched versions of every measure tracked data quantity rather than genotype.

The decoder runs on N_CELLS = 20, not on the full population, because the full
population SATURATES. With 250 cells the two textures separate perfectly in every
session tested (balanced accuracy 1.000), and a ceiling cannot show a group difference.
Measured accuracy against cell count on four sessions spanning 127 to 808 cells:

    cells      5     10     20     40     80    250
    accuracy  .68-.85  .73-.85  .83-.95  .93-.98  .97-1.0  1.000

20 cells sits below the ceiling with headroom in both directions. The whole curve is
recorded as well, since where a session saturates is itself a coding-efficiency measure.

That the ceiling is real rather than a leak was checked three ways on JB031 2025-04-03:
shuffling the labels gives exactly 0.500; trial index alone predicts the label at 0.439,
so there is no slow-drift or block-order leak (trials are interleaved, longest run 3);
and mean running speed is 6.8 vs 7.3 cm/s between trial types, so the separation is not
a speed difference. RSC simply represents which texture the animal is running through.

Because the decoder needs only 20 cells, this tier uses the lower MIN_CELLS floor of
150, which keeps all 8 NLGF mice rather than 7. A higher floor would buy nothing here.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.cohort import MIN_CELLS, describe, sessions
from viral.nlgf.load import load_digest
from viral.nlgf.paths import RESULTS, spks_path

WINDOWS: Dict[str, Tuple[float, float]] = {
    "early": (20.0, 100.0),
    "approach": (100.0, 150.0),
}
N_CELLS = 20
CELL_CURVE = (5, 10, 20, 40, 80, 160)
MIN_CELLS_TIER4 = 150
N_PER_CLASS = 15
N_REPEATS = 10
N_FOLDS = 5


def trial_features(digest, spks: np.ndarray, lo: float, hi: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Mean activity per cell per trial, over the running frames inside [lo, hi)."""
    pos, frame, lap = digest.run_position_cm, digest.run_frame, digest.run_lap
    keep = (pos >= lo) & (pos < hi) & (frame < spks.shape[1])
    pos, frame, lap = pos[keep], frame[keep].astype(int), lap[keep].astype(int)
    if frame.size == 0:
        return np.zeros((0, spks.shape[0])), np.array([])

    trials = digest.trials
    labels_by_idx = dict(zip(trials["idx"].astype(int), trials["texture_rewarded"]))

    X, y = [], []
    for trial_idx in np.unique(lap):
        f = frame[lap == trial_idx]
        # A trial needs enough frames in the window for its mean to mean anything
        if f.size < 15 or trial_idx not in labels_by_idx:
            continue
        X.append(spks[:, f].mean(axis=1))
        y.append(labels_by_idx[trial_idx])
    if not X:
        return np.zeros((0, spks.shape[0])), np.array([])
    return np.asarray(X), np.asarray(y)


def _decode(X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
            n_cells: int = N_CELLS) -> Optional[float]:
    """Balanced accuracy from stratified CV, on matched cells and trials.

    LDA with automatic shrinkage: with 250 cells and ~30 trials the problem is
    p >> n, where an unregularised discriminant separates any labelling perfectly.
    """
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if pos_idx.size < N_PER_CLASS or neg_idx.size < N_PER_CLASS:
        return None
    if X.shape[1] < n_cells:
        return None

    take = np.concatenate([
        rng.choice(pos_idx, N_PER_CLASS, replace=False),
        rng.choice(neg_idx, N_PER_CLASS, replace=False),
    ])
    cells = rng.choice(X.shape[1], n_cells, replace=False)
    Xs, ys = X[np.ix_(take, cells)], y[take]

    preds = np.zeros_like(ys)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train, test in StratifiedKFold(N_FOLDS, shuffle=True,
                                           random_state=0).split(Xs, ys):
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            model.fit(Xs[train], ys[train])
            preds[test] = model.predict(Xs[test])
    return float(balanced_accuracy_score(ys, preds))


def session_context(mouse: str, date: str) -> Optional[Dict]:
    spath = spks_path(mouse, date)
    if not spath.exists():
        return None
    digest = load_digest(mouse, date)
    spks = np.asarray(np.load(spath, mmap_mode="r"), dtype=np.float32)
    if spks.shape[0] < N_CELLS:
        return None

    row: Dict = dict(mouse=mouse, date=date, genotype=digest.genotype,
                     session_type=digest.meta["session_type"], n_cells=spks.shape[0])
    any_ok = False
    for name, (lo, hi) in WINDOWS.items():
        X, y = trial_features(digest, spks, lo, hi)
        if X.shape[0] == 0:
            row[f"acc_{name}"] = np.nan
            continue
        rng = np.random.default_rng(0)
        scores = [s for s in (_decode(X, y, rng) for _ in range(N_REPEATS))
                  if s is not None]
        row[f"acc_{name}"] = float(np.median(scores)) if scores else np.nan
        row[f"n_trials_{name}"] = int(X.shape[0])
        any_ok |= bool(scores)
        if name == "early":
            for n in CELL_CURVE:
                vals = [s for s in (_decode(X, y, rng, n_cells=n) for _ in range(4))
                        if s is not None]
                row[f"curve_{n}"] = float(np.median(vals)) if vals else np.nan
    return row if any_ok else None


def context_table() -> pd.DataFrame:
    coh = sessions(min_cells=MIN_CELLS_TIER4)
    print(f"cohort: {describe(coh)}", flush=True)
    rows = []
    for i, r in enumerate(coh.itertuples(), 1):
        try:
            row = session_context(r.mouse, r.date)
            if row is not None:
                row.update(stage=r.stage, day=r.day)
                rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        if i % 10 == 0:
            print(f"  [{i}/{len(coh)}] kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "context_decoding.csv", index=False)
    return df


if __name__ == "__main__":
    df = context_table()
    print(f"\n{len(df)} sessions, {df.mouse.nunique()} mice")
    print(df.groupby(["genotype", "stage"]).agg(
        sessions=("date", "size"), early=("acc_early", "median"),
        approach=("acc_approach", "median")).to_string())
