"""Is the RSC corridor code sensory or value-based? The registered-cell test.

Within a session the two textures map deterministically onto rewarded and unrewarded,
so texture identity and reward value are the same partition and cannot be separated
(see context.py). Across a reversal they come apart: the textures are physically
unchanged and only their meaning flips. With the same cells tracked through the
transition, one decoder answers it.

    train on session A, labels = TEXTURE IDENTITY, test on session B

For a pair that does NOT cross a reversal, texture and value are aligned, so this is
just a day-to-day stability measure and acts as the control. For a pair that DOES cross
one, they are anti-aligned, and:

    transfer stays high   ->  the code follows the texture. Sensory.
    transfer collapses    ->  the code followed the meaning, which just changed. Value.

Note the value-labelled version carries no extra information on a reversal pair: with
the labels anti-aligned it is exactly 1 - the texture score, the same predictions read
upside down. So texture transfer is the whole test, and it is reported against the
within-session baseline of each pair rather than against 0.5, since a pair whose cells
decode poorly to begin with cannot transfer well either.

Matched-cell count varies by pair AND by genotype - among QC-passing pairs the median
is 437 for NLGF against 265 for WT - so two levels of matching are needed, exactly as
in Tier 5 where fixing the assembly alone was not enough.

    pool      N_POOL matched cells are drawn first, so every pair selects from an
              identically sized set. Without this, drawing 60 from 437 gives a more
              selective sample than drawing 60 from 265, and cells that register well
              are plausibly the brighter and more active ones.
    decoder   N_CELLS are then drawn from that pool for each repeat.

Pairs are also filtered on registration quality before any of this: at least
MIN_MATCHED matched cells and an affine residual under MAX_AFFINE_RMS px, from
registration_qc.csv. A pair whose cells are mismatched would show as failed transfer
and be indistinguishable from a genuinely remapped code.

Pair-level filtering is not enough. Transfer correlates with registration quality at
rho = -0.55 (p = 1e-5), and NLGF register measurably worse - affine residual 3.14 vs
2.44 px, outlier fraction 0.063 vs 0.004 - so a genotype difference in transfer is
confounded with a genotype difference in match quality. Adjusting statistically halves
the effect and removes its significance, which settles nothing.

So bad matches are removed at SOURCE rather than modelled. Each matched pair has a
residual from the affine fit of that session pair's centroids; cells further than
MAX_CELL_RESIDUAL px from the fit are individually dropped before the pool is drawn.
This targets the mechanism - a mismatched cell contributes activity from the wrong
neuron and reads as a remapped code - instead of covarying away its consequence.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.load import load_digest
from viral.nlgf.registration_qc import _centroids, _fit_affine
from viral.nlgf.paths import RESULTS, spks_path
from viral.nlgf.textures import texture_labels
from viral.utils import get_genotype

WINDOW = (20.0, 100.0)      # early corridor, before anticipatory licking diverges
N_POOL = 150                # equalised matched-cell pool
MAX_CELL_RESIDUAL = 4.0     # px from the affine fit; drops individually bad matches
N_CELLS = 60                # decoder size, drawn from that pool
N_PER_CLASS = 12
N_REPEATS = 12
N_FOLDS = 5


def _features(mouse: str, date: str, rows: np.ndarray) -> Optional[tuple]:
    """Per-trial mean activity over the window, for the given spks rows."""
    path = spks_path(mouse, date)
    if not path.exists():
        return None
    digest = load_digest(mouse, date)
    spks = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if rows.max(initial=-1) >= spks.shape[0]:
        return None
    spks = spks[rows]

    pos, frame, lap = digest.run_position_cm, digest.run_frame, digest.run_lap
    keep = (pos >= WINDOW[0]) & (pos < WINDOW[1]) & (frame < spks.shape[1])
    frame, lap = frame[keep].astype(int), lap[keep].astype(int)
    if frame.size == 0:
        return None

    idx, code, meta = texture_labels(mouse, date)
    texture_by_trial = dict(zip(idx.tolist(), code.tolist()))
    rewarded_by_trial = dict(
        zip(digest.trials["idx"].astype(int), digest.trials["texture_rewarded"]))

    X, y_tex, y_rew = [], [], []
    for trial in np.unique(lap):
        f = frame[lap == trial]
        if f.size < 15 or trial not in texture_by_trial:
            continue
        X.append(spks[:, f].mean(axis=1))
        y_tex.append(texture_by_trial[trial])
        y_rew.append(rewarded_by_trial.get(trial, np.nan))
    if not X:
        return None
    return (np.asarray(X), np.asarray(y_tex), np.asarray(y_rew),
            meta["rewarded_code"])


def _balanced_draw(y: np.ndarray, rng: np.random.Generator) -> Optional[np.ndarray]:
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if pos.size < N_PER_CLASS or neg.size < N_PER_CLASS:
        return None
    return np.concatenate([rng.choice(pos, N_PER_CLASS, replace=False),
                           rng.choice(neg, N_PER_CLASS, replace=False)])


def _fit_predict(Xtr, ytr, Xte):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        m.fit(Xtr, ytr)
        return m.predict(Xte)


def _within(X, y, rng) -> float:
    take = _balanced_draw(y, rng)
    if take is None:
        return np.nan
    Xs, ys = X[take], y[take]
    pred = np.zeros_like(ys)
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=0).split(Xs, ys):
        pred[te] = _fit_predict(Xs[tr], ys[tr], Xs[te])
    return float(balanced_accuracy_score(ys, pred))


def pair_transfer(mouse: str, date_a: str, date_b: str) -> Optional[Dict]:
    mpath = RESULTS / f"matches_{mouse}_{date_a}_{date_b}.csv"
    if not mpath.exists():
        return None
    matches = pd.read_csv(mpath)
    n_before = len(matches)

    # Drop individually badly-matched cells: refit the affine on the pair's centroids
    # and keep only cells close to it. A cell far from the fit is very likely matched
    # to the wrong neuron, and would read as a remapped code.
    try:
        ca, cb = _centroids(mouse, date_a), _centroids(mouse, date_b)
        if matches.row_a.max() < len(ca) and matches.row_b.max() < len(cb):
            _, resid = _fit_affine(ca[matches.row_a.to_numpy()],
                                   cb[matches.row_b.to_numpy()])
            matches = matches[resid <= MAX_CELL_RESIDUAL]
    except Exception:  # noqa: BLE001 - fall back to pair-level filtering only
        pass
    n_after_filter = len(matches)

    if len(matches) < N_POOL:
        return None
    # Equalise the pool BEFORE drawing the decoder cells, so selectivity matches too
    pool_rng = np.random.default_rng(0)
    matches = matches.iloc[
        np.sort(pool_rng.choice(len(matches), N_POOL, replace=False))]

    fa = _features(mouse, date_a, matches.row_a.to_numpy())
    fb = _features(mouse, date_b, matches.row_b.to_numpy())
    if fa is None or fb is None:
        return None
    Xa, ya_tex, _, rew_a = fa
    Xb, yb_tex, _, rew_b = fb

    rng = np.random.default_rng(0)
    transfer, base_a, base_b = [], [], []
    for _ in range(N_REPEATS):
        cells = rng.choice(Xa.shape[1], N_CELLS, replace=False)
        ta, tb = _balanced_draw(ya_tex, rng), _balanced_draw(yb_tex, rng)
        if ta is None or tb is None:
            continue
        pred = _fit_predict(Xa[np.ix_(ta, cells)], ya_tex[ta], Xb[np.ix_(tb, cells)])
        transfer.append(float(balanced_accuracy_score(yb_tex[tb], pred)))
        base_a.append(_within(Xa[:, cells], ya_tex, rng))
        base_b.append(_within(Xb[:, cells], yb_tex, rng))
    if not transfer:
        return None

    return dict(
        mouse=mouse, date_a=date_a, date_b=date_b, genotype=get_genotype(mouse),
        n_matched_raw=int(n_before), n_matched_clean=int(n_after_filter),
        frac_cells_kept=float(n_after_filter / n_before),
        n_cells_used=N_CELLS, n_pool=N_POOL,
        gap_days=int((pd.to_datetime(date_b) - pd.to_datetime(date_a)).days),
        rewarded_code_a=rew_a, rewarded_code_b=rew_b,
        crosses_reversal=bool(rew_a != rew_b and rew_a >= 0 and rew_b >= 0),
        transfer_texture=float(np.median(transfer)),
        within_a=float(np.nanmedian(base_a)), within_b=float(np.nanmedian(base_b)),
        n_repeats=len(transfer),
    )


def transfer_table() -> pd.DataFrame:
    qc_path = RESULTS / "registration_qc.csv"
    if not qc_path.exists():
        raise FileNotFoundError(
            "run viral.nlgf.registration_qc first - pairs must be quality filtered "
            "before transfer is interpretable")
    qc = pd.read_csv(qc_path)
    good = set(zip(qc.loc[qc.usable, "mouse"], qc.loc[qc.usable, "date_a"],
                   qc.loc[qc.usable, "date_b"]))
    print(f"{len(qc)} registered pairs, {len(good)} pass QC", flush=True)

    files = sorted(RESULTS.glob("matches_*.csv"))
    rows: List[Dict] = []
    for i, f in enumerate(files, 1):
        stem = f.stem[len("matches_"):]
        mouse, date_a, date_b = stem.split("_")
        if (mouse, date_a, date_b) not in good:
            continue
        try:
            r = pair_transfer(mouse, date_a, date_b)
            if r is not None:
                rows.append(r)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {stem}: {type(e).__name__}: {e}", flush=True)
        if i % 10 == 0:
            print(f"  [{i}/{len(files)}] kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    if len(df):
        df["transfer_vs_baseline"] = df.transfer_texture - df[["within_a", "within_b"]].mean(axis=1)
        df.to_csv(RESULTS / "cross_session_transfer.csv", index=False)
    return df


if __name__ == "__main__":
    df = transfer_table()
    if len(df):
        print(f"\n{len(df)} pairs, {df.mouse.nunique()} mice")
        print(df.groupby(["genotype", "crosses_reversal"]).agg(
            pairs=("mouse", "size"), transfer=("transfer_texture", "median"),
            within=("within_a", "median"),
            vs_base=("transfer_vs_baseline", "median")).to_string())
