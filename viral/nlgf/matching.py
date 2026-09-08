"""Matching by construction, for the covariates that keep eating this study's effects.

Two different things get called "matching" and they are not interchangeable:

COMMON SUPPORT (`matched_subset`) drops rows outside the range both groups occupy. It
answers "is there a positivity violation" - the failure mode found in Tier 5, where
every NLGF mouse had more cells than every WT mouse and a covariate-adjusted model was
extrapolating rather than comparing. What it does NOT do is fix a difference in the
distribution WITHIN a shared range: if both groups span 1-5 days but one averages 2.0
and the other 1.7, every row survives and nothing is controlled.

DISTRIBUTION MATCHING (`histogram_matched`) subsamples so both groups have an identical
histogram on the covariate. That is what removes the bias when the ranges already
overlap - which is the situation for session gap in the stability analysis, and the
reason the raw estimate there overstated the effect by about a third.

Use common support to check the comparison is possible at all; use histogram matching
to make it fair. Both cost pairs, so both cost power, and the honest report is the
estimate under each rather than the most favourable one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.stats import exact_permutation

N_DRAWS = 400


def _two_groups(d: pd.DataFrame, group_a: Optional[str], group_b: Optional[str]):
    """The two genotypes to contrast, taken from the data unless named explicitly.

    Deriving them matters: an earlier version hardcoded NLGF and WT, so on an
    Oligo-vs-NLGF frame the WT slice was empty, its min/max came back NaN, and
    `max(x, nan)` / `min(x, nan)` quietly returned x - the filter reported matching
    while passing every row through.
    """
    if group_a and group_b:
        return group_a, group_b
    present = list(pd.unique(d.genotype))
    if len(present) != 2:
        raise ValueError(f"need exactly two genotypes, got {present}; "
                         "pass group_a and group_b explicitly")
    return present[0], present[1]


def matched_subset(d: pd.DataFrame, on: List[str], group_a: Optional[str] = None,
                   group_b: Optional[str] = None) -> pd.DataFrame:
    """Rows inside the range BOTH groups occupy on every variable in `on`."""
    a_name, b_name = _two_groups(d, group_a, group_b)
    m = d.copy()
    for v in on:
        a = m.loc[m.genotype == a_name, v].dropna()
        b = m.loc[m.genotype == b_name, v].dropna()
        if a.empty or b.empty:
            raise ValueError(f"no rows for one group on {v!r}")
        lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
        m = m[(m[v] >= lo) & (m[v] <= hi)]
    return m


def histogram_matched(d: pd.DataFrame, value: str, on: str,
                      group_a: Optional[str] = None, group_b: Optional[str] = None,
                      n_draws: int = N_DRAWS, seed: int = 0,
                      collapse: str = "median") -> Dict:
    """Contrast with both groups forced to an identical histogram on `on`.

    At each level of `on`, min(n_a, n_b) rows are drawn from each group, so the
    covariate distributions are the same by construction rather than adjusted for. The
    draw is repeated and the estimate averaged, because a single subsample is itself a
    coin flip.

    Returns the mean difference over draws, the median p, and the fraction of draws
    reaching p < 0.05 - the last being the honest summary of how fragile the result is.
    """
    a_name, b_name = _two_groups(d, group_a, group_b)
    rng = np.random.default_rng(seed)
    A, B = d[d.genotype == a_name], d[d.genotype == b_name]
    levels = sorted(set(A[on]) & set(B[on]))
    if not levels:
        raise ValueError(f"groups share no level of {on!r}")

    diffs, ps, sizes = [], [], []
    for _ in range(n_draws):
        keep = []
        for lvl in levels:
            ga, gb = A[A[on] == lvl], B[B[on] == lvl]
            k = min(len(ga), len(gb))
            if k == 0:
                continue
            keep.append(ga.sample(k, random_state=int(rng.integers(1 << 31))))
            keep.append(gb.sample(k, random_state=int(rng.integers(1 << 31))))
        if not keep:
            continue
        s = pd.concat(keep)
        per_mouse = (s.groupby(["mouse", "genotype"], as_index=False)[value]
                     .agg(collapse))
        if per_mouse.groupby("genotype").size().min() < 3:
            continue
        r = exact_permutation(per_mouse, value, group_a=a_name, group_b=b_name)
        if np.isfinite(r["diff"]):
            diffs.append(r["diff"]); ps.append(r["p"]); sizes.append(len(s))
    if not diffs:
        return dict(diff=np.nan, p=np.nan, n_pairs=0, frac_sig=np.nan, n_draws=0)
    return dict(diff=float(np.mean(diffs)), p=float(np.median(ps)),
                n_pairs=float(np.mean(sizes)),
                frac_sig=float(np.mean(np.array(ps) < 0.05)), n_draws=len(diffs))


def ladder(d: pd.DataFrame, value: str, group_a: str, group_b: str,
           support: List[str], match_on: str) -> pd.DataFrame:
    """The same contrast under progressively stricter control, as a table.

    Ordered from no control to most conservative. The gap-ADJUSTED row is the headline
    estimate - it removes the covariate trend while keeping every pair - and the
    histogram-MATCHED row is the conservative bound, unbiased in principle but paying
    for it in discarded pairs and unstable mouse composition.
    """
    sub = d[d.genotype.isin([group_a, group_b])]
    rows = []

    def add(label, frame, hist=False, col=None):
        col = col or value
        if hist:
            r = histogram_matched(frame, col, match_on, group_a, group_b)
            rows.append(dict(control=label, diff=r["diff"], p=r["p"],
                             n_pairs=r["n_pairs"], frac_sig=r["frac_sig"]))
            return
        pm = frame.groupby(["mouse", "genotype"], as_index=False)[col].median()
        r = exact_permutation(pm, col, group_a=group_a, group_b=group_b)
        rows.append(dict(control=label, diff=r["diff"], p=r["p"],
                         n_pairs=len(frame), frac_sig=np.nan))

    reg = matched_subset(sub, support, group_a, group_b)
    add("no control", sub)
    add("common support\n" + ", ".join(support), reg)
    adj, _ = gap_adjusted(sub, value, match_on)
    add(f"{match_on}\nadjusted", adj, col="adj")
    adj_reg, _ = gap_adjusted(reg, value, match_on)
    add(f"{match_on} adjusted\n+ common support", adj_reg, col="adj")
    add(f"{match_on}\nhistogram matched", sub, hist=True)
    return pd.DataFrame(rows)


def gap_adjusted(d: pd.DataFrame, value: str, on: str = "gap_days",
                 out: str = "adj") -> pd.DataFrame:
    """Remove a covariate's linear trend while keeping every row.

    The slope is the MEAN OF THE WITHIN-GROUP SLOPES, not a slope fitted to the pooled
    data. Fitting on the pooled data lets the between-group difference leak into the
    slope whenever the groups differ on the covariate, which would partly subtract the
    very effect being measured. (In practice the three choices agree to the third
    decimal here, which is itself worth knowing.)

    Prefer this to `histogram_matched` when the groups share the covariate's range and
    differ only in distribution within it. Matching is the right tool for a positivity
    violation - no common support, where a model would extrapolate - and the wrong one
    here: it discarded a third of the pairs and, worse, changed WHICH MICE contributed,
    since a mouse with one pair drops out of most draws entirely. With 7 and 8 animals
    losing one is a larger perturbation than the bias being removed.
    """
    slopes = []
    for g, s in d.groupby("genotype"):
        s = s.dropna(subset=[value, on])
        if len(s) > 2 and s[on].nunique() > 1:
            slopes.append(np.polyfit(s[on], s[value], 1)[0])
    if not slopes:
        return d.assign(**{out: d[value]}), 0.0
    slope = float(np.mean(slopes))
    centred = d[on] - d[on].mean()
    return d.assign(**{out: d[value] - slope * centred}), slope
