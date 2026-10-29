"""Inference when there are three genotypes and one of them is an intervention.

EDITED 29/10/2026 by Claude Opus 5.5 (medium effort), at JB's request, after a review:
  - Removed `tost` and `rescue_index_ci`. Neither was used anywhere in the repo, and
    both were found to be incorrect: `tost` built its null from unshifted data and
    inverted the +1 correction, so its p-values were too small and could be 0;
    `rescue_index_ci` returned different keys on different paths ('frac_undefined' vs
    'denom_sign_flips') and computed its sign-flip check on separate resamples. They
    can be recovered from git history (any commit before this change).
  - Removed imports that were only needed by them or not used at all.
  - Updated the references to them in this docstring (marked "29/10/2026" below).
  `holm` (used by presentations/26-10_meeting_JB.py and 26-11_poster_JB.py) was
  checked and left unchanged; `permutation_anova` is unchanged and currently unused.

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
   is what the Holm correction below addresses. (29/10/2026: fig_activity.py does not
   currently apply it across measures; it gates the rescue contrast per measure instead.)

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
       not a failed difference test. At this n it will usually be inconclusive, which
       is the correct answer rather than a disappointing one. (29/10/2026: the earlier
       `tost` implementation here was removed as incorrect; none is currently
       available. Note it would still compare Oligo with WT, the era-confounded
       contrast above - a correct equivalence test does not remove that confound.)
     - the rescue index with a bootstrap interval, which estimates HOW FAR toward WT
       the intervention got and shows honestly how little the data pin it down. This is
       more informative than any of the p-values. (29/10/2026: the earlier
       `rescue_index_ci` was removed as incorrect; not currently implemented.)
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

N_PERM = 20000


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


def permutation_anova(
    per_mouse: pd.DataFrame, value: str, n_perm: int = N_PERM, seed: int = 0
) -> Dict:
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
    return dict(
        F=float(obs), p=float((count + 1) / (n_perm + 1)), n=len(v), groups=names
    )


# 29/10/2026: `tost` and `rescue_index_ci` were removed here (unused and incorrect; see the
# module docstring). Recover them from git history if needed.


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
