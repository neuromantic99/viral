"""Genotype tests with the animal as the unit.

dec.py already does this for decoder accuracy with a genotype x experience LMM; this
is the same permutation scheme factored out so any per-session measure can use it. The
permutation is over MICE - each mouse's sessions travel with its label - and with 8 WT
against 6 NLGF it enumerates all C(14, 6) = 3003 assignments, which makes it exact.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd


def collapse_to_mice(df: pd.DataFrame, value: str, how: str = "median") -> pd.DataFrame:
    """One number per mouse. Sessions within a mouse are not independent, so they are
    collapsed before the test rather than treated as replicates."""
    out = (
        df.groupby(["genotype", "mouse"])[value]
        .agg(how)
        .reset_index()
        .dropna(subset=[value])
    )
    return out


def exact_permutation(
    per_mouse: pd.DataFrame,
    value: str,
    group_a: str = "NLGF",
    group_b: str = "WT",
    statistic: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> dict:
    """Difference in means between genotypes, with an exact permutation p.

    Returns the observed difference (a - b), the two-sided p, and the number of
    assignments enumerated. If the two groups together exceed 22 mice the enumeration
    is replaced by 100k random assignments, which is not reached by this dataset.
    """
    d = per_mouse[per_mouse.genotype.isin([group_a, group_b])].copy()
    v = d[value].to_numpy(dtype=float)
    is_a = (d.genotype == group_a).to_numpy()
    keep = np.isfinite(v)
    v, is_a = v[keep], is_a[keep]

    n, k = v.size, int(is_a.sum())
    if n < 3 or k == 0 or k == n:
        return dict(diff=np.nan, p=np.nan, n_perm=0, n_a=k, n_b=n - k)

    default_statistic = statistic is None
    if default_statistic:
        statistic = lambda a, b: float(a.mean() - b.mean())

    obs = statistic(v[is_a], v[~is_a])

    from math import comb

    if comb(n, k) <= 200_000:
        assignments = list(combinations(range(n), k))
        n_perm = comb(n, k)
        exact = True
    else:
        rng = np.random.default_rng(0)
        assignments = [rng.choice(n, k, replace=False) for _ in range(100_000)]
        n_perm = 100_000
        exact = False

    # Build the assignment matrix once. For the default difference-in-means this makes
    # the whole null one matrix product rather than a Python loop over ~24k
    # assignments, which matters because the sensitivity analysis re-runs the test
    # dozens of times per metric.
    membership = np.zeros((n_perm, n), dtype=bool)
    rows = np.repeat(np.arange(n_perm), k)
    membership[rows, np.asarray(assignments, dtype=int).ravel()] = True

    if default_statistic:
        counts_a, counts_b = k, n - k
        sums_a = membership @ v
        null = sums_a / counts_a - (v.sum() - sums_a) / counts_b
    else:
        null = np.array([statistic(v[m], v[~m]) for m in membership])

    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return dict(
        diff=obs,
        p=float(p),
        n_perm=n_perm,
        exact=exact,
        n_a=k,
        n_b=n - k,
        mean_a=float(v[is_a].mean()),
        mean_b=float(v[~is_a].mean()),
    )


def compare(
    df: pd.DataFrame, value: str, how: str = "median", **kwargs
) -> Tuple[pd.DataFrame, dict]:
    per_mouse = collapse_to_mice(df, value, how)
    return per_mouse, exact_permutation(per_mouse, value, **kwargs)


def minimum_detectable_difference(
    per_mouse: pd.DataFrame, value: str, group_a: str = "NLGF", alpha: float = 0.05
) -> float:
    """Smallest TOTAL group difference that would reach p < alpha at this n.

    Found by adding a constant to every group_a mouse and re-running the exact test.
    Note the return is the total difference (observed + added shift), not the shift:
    a null is only interpretable against the effect the design could have seen.
    """
    base = exact_permutation(per_mouse, value, group_a=group_a)
    if not np.isfinite(base["diff"]):
        return np.nan
    scale = abs(base["mean_b"]) if base["mean_b"] else 1.0
    lo, hi = 0.0, scale * 8 + 1e-9
    for _ in range(24):
        mid = (lo + hi) / 2
        shifted = per_mouse.copy()
        shifted.loc[shifted.genotype == group_a, value] += mid
        if exact_permutation(shifted, value, group_a=group_a)["p"] < alpha:
            hi = mid
        else:
            lo = mid
    return float(base["diff"] + hi)
