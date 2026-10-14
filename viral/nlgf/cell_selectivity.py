"""Per-cell texture selectivity before and after a reversal.

The population decoder answers "what does the population as a whole carry", and it
answers it with whatever dominates - if texture information is strong, LDA will use it
and be blind to a value-coding subpopulation sitting alongside. RSC is an integrator,
so separate populations are a real possibility and the population measure cannot see
them.

This asks the question cell by cell. Each matched cell gets a texture selectivity in
the pre-reversal session and again in the post-reversal session, both defined against
the SAME fixed reference: texture identity, coded 0/1 by sorted filename and therefore
consistent across sessions (see textures.py). The textures are physically unchanged by
a reversal; only which one is rewarded flips. So:

    a cell coding TEXTURE   keeps its sign          sel_A and sel_B agree
    a cell coding VALUE     reverses with the        sel_A and sel_B are opposite
                            contingency

Plotting sel_A against sel_B therefore separates the two populations geometrically:
texture cells lie along the positive diagonal, value cells along the negative one. A
unimodal cloud on the positive diagonal means one population that partly remaps; two
clusters mean two populations, which is the hypothesis the population decoder cannot
address.

Selectivity is AUC of the trial-wise activity for texture 1 versus texture 0, centred
so 0 means no preference and +/-0.5 means perfect separation. A cell counts as selective
when its AUC falls outside the central 95% of a within-session label shuffle, so the
classification does not rest on an arbitrary cutoff.

IMPORTANT: interpretation requires the animal to have actually re-evaluated. On
reversal day 1 most have not - only 3 of 22 mice reach licking d' > 1.0 - and across
the day-1 pairs texture transfer tracked how much the animal had learned
(rho = -0.72). Each pair therefore carries the post-session licking d', and pairs below
D_PRIME_MIN are reported separately rather than pooled.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.cross_session import MAX_CELL_RESIDUAL, N_POOL, _features
from viral.nlgf.paths import RESULTS
from viral.nlgf.registration_qc import _centroids, _fit_affine
from viral.utils import get_genotype

N_SHUFFLES = 200
D_PRIME_MIN = 1.0
MIN_TRIALS_PER_TEXTURE = 8


def _auc_from_ranks(ranks: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUC per cell for label 1 vs 0, centred on zero, from precomputed ranks."""
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    r1 = ranks[y == 1].sum(axis=0)
    return (r1 - n1 * (n1 + 1) / 2) / (n1 * n0) - 0.5


def _selectivity(x: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Centred AUC per cell, plus a boolean 'selective' from a label shuffle.

    Ranks are invariant to the labels, so they are computed once and the shuffle only
    reshuffles which rows are summed.
    """
    from scipy.stats import rankdata

    ranks = rankdata(x, axis=0)
    obs = _auc_from_ranks(ranks, y)
    null = np.stack([_auc_from_ranks(ranks, rng.permutation(y))
                     for _ in range(N_SHUFFLES)])
    lo, hi = np.percentile(null, [2.5, 97.5], axis=0)
    return obs, (obs < lo) | (obs > hi)


def _splithalf(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Selectivity on two interleaved halves of one session's trials.

    Used as the noise floor. Within a single session the reward contingency cannot have
    changed, so any sign flip between the two halves is measurement noise. Comparing
    that to the across-session flip rate is what turns "15% of cells flipped" into a
    statement about value coding rather than about trial counts.
    """
    h1, h2 = np.arange(0, len(y), 2), np.arange(1, len(y), 2)
    s1, g1 = _selectivity(X[h1], y[h1], rng)
    s2, g2 = _selectivity(X[h2], y[h2], rng)
    return (s1, g1), (s2, g2)


def pair_selectivity(mouse: str, date_a: str, date_b: str) -> Optional[pd.DataFrame]:
    mpath = RESULTS / f"matches_{mouse}_{date_a}_{date_b}.csv"
    if not mpath.exists():
        return None
    matches = pd.read_csv(mpath)
    try:
        ca, cb = _centroids(mouse, date_a), _centroids(mouse, date_b)
        if matches.row_a.max() < len(ca) and matches.row_b.max() < len(cb):
            _, resid = _fit_affine(ca[matches.row_a.to_numpy()],
                                   cb[matches.row_b.to_numpy()])
            matches = matches[resid <= MAX_CELL_RESIDUAL]
    except Exception:  # noqa: BLE001
        pass
    if len(matches) < N_POOL:
        return None

    fa = _features(mouse, date_a, matches.row_a.to_numpy())
    fb = _features(mouse, date_b, matches.row_b.to_numpy())
    if fa is None or fb is None:
        return None
    Xa, ya, _, rew_a = fa
    Xb, yb, _, rew_b = fb
    if min((ya == 0).sum(), (ya == 1).sum(),
           (yb == 0).sum(), (yb == 1).sum()) < MIN_TRIALS_PER_TEXTURE:
        return None

    rng = np.random.default_rng(0)
    sel_a, sig_a = _selectivity(Xa, ya, rng)
    sel_b, sig_b = _selectivity(Xb, yb, rng)

    # half-trial estimates, so the within-session noise floor and the across-session
    # measurement rest on the same number of trials and are directly comparable
    (a1, ga1), (a2, ga2) = _splithalf(Xa, ya, rng)
    (b1, gb1), _ = _splithalf(Xb, yb, rng)

    return pd.DataFrame(dict(
        mouse=mouse, date_a=date_a, date_b=date_b, genotype=get_genotype(mouse),
        crosses_reversal=bool(rew_a != rew_b and rew_a >= 0 and rew_b >= 0),
        cell=np.arange(len(sel_a)), sel_a=sel_a, sel_b=sel_b,
        sig_a=sig_a, sig_b=sig_b,
        half_a1=a1, half_a2=a2, half_b1=b1,
        hsig_a1=ga1, hsig_a2=ga2, hsig_b1=gb1))


def selectivity_table() -> pd.DataFrame:
    qc = pd.read_csv(RESULTS / "registration_qc.csv")
    beh = pd.read_csv(RESULTS / "behaviour_full.csv")
    beh["key"] = beh.mouse + "|" + beh.session_name.str.lower().str.strip()
    from viral.nlgf.load import ledger

    led = ledger()[["mouse", "date", "session_type"]]

    frames: List[pd.DataFrame] = []
    good = qc[qc.usable]
    for i, r in enumerate(good.itertuples(), 1):
        try:
            out = pair_selectivity(r.mouse, r.date_a, r.date_b)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date_a} {r.date_b}: {type(e).__name__}: {e}")
            continue
        if out is not None:
            frames.append(out)
        if i % 20 == 0:
            print(f"  [{i}/{len(good)}] pairs kept {len(frames)}", flush=True)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # attach the post-session behaviour, so unreversed animals can be held apart
    df = df.merge(led, left_on=["mouse", "date_b"], right_on=["mouse", "date"],
                  how="left")
    df["key"] = df.mouse + "|" + df.session_type.str.lower().str.strip()
    df = df.merge(beh[["key", "licking_dprime"]].drop_duplicates("key"), on="key",
                  how="left")
    df["reversed_behaviourally"] = df.licking_dprime >= D_PRIME_MIN
    df.to_csv(RESULTS / "cell_selectivity.csv", index=False)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Classify cells selective in BOTH sessions as texture-like or value-like."""
    both = df[df.sig_a & df.sig_b].copy()
    both["kind"] = np.where(np.sign(both.sel_a) == np.sign(both.sel_b),
                            "texture (sign kept)", "value (sign flipped)")
    return both


def flip_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Sign-flip rate within a session (noise) versus across the reversal (noise + value).

    Both use half the trials, so the two rates are on the same footing.
    """
    rows = []
    for (mouse, date_a, date_b), g in df.groupby(["mouse", "date_a", "date_b"]):
        w = g[g.hsig_a1 & g.hsig_a2]
        x = g[g.hsig_a1 & g.hsig_b1]
        if len(w) < 20 or len(x) < 20:
            continue
        rows.append(dict(
            mouse=mouse, date_a=date_a, date_b=g.date_b.iloc[0],
            genotype=g.genotype.iloc[0],
            crosses_reversal=bool(g.crosses_reversal.iloc[0]),
            licking_dprime=g.licking_dprime.iloc[0],
            reversed_behaviourally=bool(g.reversed_behaviourally.iloc[0]),
            flip_within=float((np.sign(w.half_a1) != np.sign(w.half_a2)).mean()),
            flip_across=float((np.sign(x.half_a1) != np.sign(x.half_b1)).mean()),
            n_within=len(w), n_across=len(x)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = selectivity_table()
    if df.empty:
        print("no reversal-crossing pairs available")
        sys.exit()
    n_rev = df[df.crosses_reversal].groupby(['mouse','date_a']).ngroups
    print(f"\n{df.groupby(['mouse','date_a']).ngroups} pairs "
          f"({n_rev} crossing a reversal), {len(df)} matched cells")
    both = summarise(df[df.crosses_reversal])
    for label, sub in [("animals that reversed behaviourally",
                        both[both.reversed_behaviourally]),
                       ("animals that did NOT", both[~both.reversed_behaviourally])]:
        if sub.empty:
            print(f"\n{label}: none"); continue
        counts = sub.kind.value_counts(normalize=True)
        print(f"\n{label}: {sub.mouse.nunique()} mice, {len(sub)} doubly-selective cells")
        print(counts.round(3).to_string())
        print(sub.groupby("genotype").kind.value_counts(normalize=True).round(3).to_string())
    fr = flip_rates(df)
    fr.to_csv(RESULTS / "cell_flip_rates.csv", index=False)
    print("\nsign-flip rate, half-trial estimates:")
    print(fr.groupby(["crosses_reversal", "reversed_behaviourally"])
          [["flip_within", "flip_across"]].agg(["mean", "size"]).round(3).to_string())
