"""Inference when there are three genotypes and one of them is an intervention.

TWO QUESTIONS, WITH DIFFERENT ANSWERS.

1. How to get p-values across three groups.

   Not ANOVA as such. Classical one-way ANOVA assumes normal residuals and equal
   variances, and with 4, 7 and 8 mice neither can be checked, let alone relied on. The
   distribution-free equivalent is a PERMUTATION ANOVA: compute the same F statistic,
   then rebuild its null by shuffling genotype labels over mice. That keeps the
   backbone the rest of this package uses - the animal is the unit, the null comes from
   relabelling animals - and drops the parametric assumptions.

   But the omnibus test is not really the question here. An omnibus F asks "are these
   three groups all the same", which is an exploratory question about exchangeable
   groups. This design is not exploratory and the groups are not exchangeable: it is a
   disease model, an intervention on that model, and a healthy reference, with two
   PLANNED contrasts fixed before the data were seen -

       NLGF vs WT      is there a deficit
       Oligo vs NLGF   does the intervention move it

   Planned orthogonal-in-intent contrasts do not need an omnibus gate; the F test is
   reported here for completeness and because reviewers ask for it, not because a
   non-significant F should stop you looking at a planned contrast. The real
   multiplicity problem is not the three genotypes, it is the eight measures, and that
   is what the Holm correction below addresses.

2. Is it valid to test WT against Oligo to examine rescue?

   No, and this is the trap the whole design invites. The argument "Oligo differs from
   NLGF (p < 0.05) and does NOT differ from WT (p > 0.05), therefore rescued" is
   invalid twice over:

     - a non-significant difference is not evidence of equivalence. With four Oligo
       mice almost nothing reaches significance, so "no difference from WT" is what
       this comparison returns whether or not the groups are truly alike. Absence of
       evidence, presented as evidence of absence.
     - the difference between "significant" and "not significant" is not itself
       significant (Gelman & Stern 2006). Comparing two p-values on either side of 0.05
       is not a test of anything.

   And here there is a third, local reason: Oligo vs WT is the era-confounded contrast
   (see era.py). It is the one comparison in this dataset we already know is not clean.

   What IS valid:
     - Oligo vs NLGF, the intervention contrast. One test, time-matched, a real null.
     - if the claim is specifically "restored to WT levels", that is an EQUIVALENCE
       claim and needs an equivalence test (TOST) against a margin chosen in advance -
       not a failed difference test. `tost` below runs it, and at this n it will
       usually be inconclusive, which is the correct answer rather than a
       disappointing one.
     - the rescue index with a bootstrap interval, which estimates HOW FAR toward WT
       the intervention got and shows honestly how little the data pin it down. This is
       more informative than any of the p-values.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.stats import exact_permutation

N_PERM = 20000
N_BOOT = 5000


def _f_stat(groups: Sequence[np.ndarray]) -> float:
    groups = [g for g in groups if g.size]
    k = len(groups)
    allv = np.concatenate(groups)
    n = allv.size
    if k < 2 or n <= k:
        return np.nan
    grand = allv.mean()
    ssb = sum(g.size * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    if ssw <= 0:
        return np.inf
    return (ssb / (k - 1)) / (ssw / (n - k))


def permutation_anova(per_mouse: pd.DataFrame, value: str,
                      n_perm: int = N_PERM, seed: int = 0) -> Dict:
    """Omnibus F over all genotypes, with the null built by relabelling mice."""
    d = per_mouse.dropna(subset=[value])
    labels = d.genotype.to_numpy()
    v = d[value].to_numpy(dtype=float)
    names = list(pd.unique(labels))
    obs = _f_stat([v[labels == g] for g in names])
    if not np.isfinite(obs):
        return dict(F=np.nan, p=np.nan, n=len(v), groups=names)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        count += _f_stat([v[perm == g] for g in names]) >= obs
    return dict(F=float(obs), p=float((count + 1) / (n_perm + 1)), n=len(v),
                groups=names)


def tost(per_mouse: pd.DataFrame, value: str, group_a: str, group_b: str,
         margin: float, n_perm: int = N_PERM, seed: int = 0) -> Dict:
    """Two one-sided tests: is |a - b| smaller than `margin`?

    This is how an equivalence claim is actually made. The null is that the groups
    differ by AT LEAST the margin; rejecting it in both directions supports "these are
    alike to within `margin`". A large p here means inconclusive - never "equivalent".

    The margin must be chosen on scientific grounds before looking. Here the natural
    choice is a fraction of the disease effect: if the intervention leaves a residual
    gap smaller than, say, a third of the NLGF-WT deficit, that is a meaningful rescue.
    """
    d = per_mouse.dropna(subset=[value])
    d = d[d.genotype.isin([group_a, group_b])]
    a = d.loc[d.genotype == group_a, value].to_numpy(dtype=float)
    b = d.loc[d.genotype == group_b, value].to_numpy(dtype=float)
    if a.size < 2 or b.size < 2:
        return dict(p=np.nan, diff=np.nan, margin=margin, conclusive=False)
    obs = a.mean() - b.mean()
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    k = a.size
    # one-sided permutation p for (diff > -margin) and for (diff < +margin)
    lo_count = hi_count = 0
    for _ in range(n_perm):
        s = rng.permutation(pooled)
        d0 = s[:k].mean() - s[k:].mean()
        lo_count += d0 <= obs + margin
        hi_count += d0 >= obs - margin
    p_lo = (lo_count + 1) / (n_perm + 1)
    p_hi = (hi_count + 1) / (n_perm + 1)
    p = max(1 - p_lo, 1 - p_hi)
    return dict(p=float(p), diff=float(obs), margin=float(margin),
                conclusive=bool(p < 0.05))


def rescue_index_ci(per_mouse: pd.DataFrame, value: str, disease: str, healthy: str,
                    treated: str, n_boot: int = N_BOOT, seed: int = 0) -> Dict:
    """(treated - disease) / (healthy - disease), with a bootstrap-over-mice interval.

    The denominator is itself estimated and can approach zero, which sends the ratio to
    infinity - so the interval is reported on the percentile scale and will be very
    wide whenever the disease effect is not well separated from zero. That width IS the
    result: a rescue index quoted without it is meaningless.
    """
    d = per_mouse.dropna(subset=[value])
    arrs = {g: d.loc[d.genotype == g, value].to_numpy(dtype=float)
            for g in (disease, healthy, treated)}
    if any(a.size < 2 for a in arrs.values()):
        return dict(index=np.nan, lo=np.nan, hi=np.nan, frac_undefined=np.nan)
    denom = arrs[healthy].mean() - arrs[disease].mean()
    point = ((arrs[treated].mean() - arrs[disease].mean()) / denom
             if denom else np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        rs = {g: rng.choice(a, a.size, replace=True) for g, a in arrs.items()}
        den = rs[healthy].mean() - rs[disease].mean()
        if den == 0:
            continue
        vals.append((rs[treated].mean() - rs[disease].mean()) / den)
    vals = np.asarray(vals)
    # a denominator that changes sign under resampling means the disease effect itself
    # is not established, and the index is not interpretable at all
    sign_flips = float(np.mean(np.sign(
        [rng.choice(arrs[healthy], arrs[healthy].size).mean()
         - rng.choice(arrs[disease], arrs[disease].size).mean()
         for _ in range(1000)]) != np.sign(denom)))
    return dict(index=float(point), lo=float(np.percentile(vals, 2.5)),
                hi=float(np.percentile(vals, 97.5)), denom_sign_flips=sign_flips)


def holm(ps: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(ps, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    idx = np.argsort(p[ok])
    vals = p[ok][idx]
    m = vals.size
    adj = np.maximum.accumulate(vals * (m - np.arange(m)))
    res = np.empty(m)
    res[idx] = np.minimum(adj, 1.0)
    out[ok] = res
    return out
