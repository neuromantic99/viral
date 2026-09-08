"""Session-level mixed models as an independent route to the same answers.

Everywhere else the confounds are controlled BY CONSTRUCTION - subsample to 250 cells,
30 laps, 24 trials - and sessions are collapsed to one number per mouse before testing.
That is robust but wasteful: it discards within-mouse variation, which is often the
cleanest evidence about whether a covariate actually moves the measure.

This fits the complementary model: every session is a row, mouse is a random intercept,
and covariates enter DECOMPOSED into their between- and within-mouse parts.

    y ~ genotype + x_between + x_within + (1 | mouse)

    x_between   the mouse's own mean of x. Carries the between-animal confound, which
                is the one that mattered in Tiers 2 and 5.
    x_within    the session's deviation from that mean. Carries the within-animal
                evidence, which nothing so far has tested.

Decomposing matters because a raw session-level covariate mixes the two, and they can
point in opposite directions - the covariate then absorbs genotype variance in a way
that is impossible to interpret.

Two things this does NOT do:

  It does not raise n. Genotype is a between-mouse predictor, so its standard error
  still comes from between-mouse variance: the effective n stays at the number of
  animals however many sessions each contributes. What the sessions buy is a
  better-estimated mean per mouse and sensible weighting of unequal session counts.

  It does not trust its own p-value. statsmodels reports asymptotic z-tests with no
  finite-sample correction, which are anticonservative at 15 groups. So the model gives
  the estimate and the genotype label is permuted ACROSS MICE for inference, exactly as
  dec.py does. Enumerated when the number of assignments allows, sampled otherwise.
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.paths import RESULTS

MAX_ENUMERATE = 1500
N_SAMPLED = 1500


def decompose(df: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    d = df.copy()
    for c in covariates:
        m = d.groupby("mouse")[c].transform("mean")
        d[f"{c}_between"] = (m - m.mean()) / (m.std() or 1.0)
        d[f"{c}_within"] = (d[c] - m) / (d[c].std() or 1.0)
    return d


def _fit(d: pd.DataFrame, formula: str) -> Optional[float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = smf.mixedlm(formula, d, groups=d["mouse"]).fit(method="lbfgs")
        except Exception:  # noqa: BLE001
            return None
    key = [k for k in res.params.index if k.startswith("genotype[T.")]
    return float(res.params[key[0]]) if key else None


def check_overlap(df: pd.DataFrame, covariates: List[str],
                  group_a: str = "NLGF") -> Dict[str, float]:
    """Common support between genotypes on each mouse-level covariate.

    Adjusting for a covariate assumes the groups overlap on it. They do not always:
    in the Tier 5 freeze set every NLGF mouse has MORE cells (804-928) than every WT
    mouse (362-621), correlation +0.92 with genotype and zero overlap. Conditioning on
    a covariate that perfectly separates the groups extrapolates the fit into a region
    neither group occupies, and the "adjusted" genotype coefficient jumped 27-fold and
    turned significant purely from that.

    Permutation does not rescue it either: permuting genotype destroys the
    genotype-covariate collinearity, so the null is built from well-conditioned models
    while the observed one is degenerate, which inflates significance.

    Returns the fraction of mice inside the overlap region per covariate, and the
    correlation with genotype. Overlap near zero means match by construction instead.
    """
    pm = df.groupby(["genotype", "mouse"])[covariates].mean().reset_index()
    g = (pm.genotype == group_a).to_numpy(dtype=float)
    out = {}
    for c in covariates:
        a, b = pm.loc[pm.genotype == group_a, c], pm.loc[pm.genotype != group_a, c]
        lo, hi = max(a.min(), b.min()), min(a.max(), b.max())
        inside = ((pm[c] >= lo) & (pm[c] <= hi)).mean() if hi >= lo else 0.0
        out[f"{c}_overlap"] = float(inside)
        out[f"{c}_corr_genotype"] = float(np.corrcoef(g, pm[c])[0, 1])
    return out


def lmm_permutation(df: pd.DataFrame, value: str, covariates: List[str],
                    group_a: str = "NLGF") -> Dict:
    d = decompose(df.dropna(subset=[value] + covariates), covariates)
    d["genotype"] = pd.Categorical(d.genotype, categories=["WT", group_a])

    terms = [f"{c}_between" for c in covariates] + [f"{c}_within" for c in covariates]
    formula = f"{value} ~ genotype" + ("" if not terms else " + " + " + ".join(terms))

    diagnostics = check_overlap(d, covariates, group_a) if covariates else {}
    poor = [c for c in covariates if diagnostics.get(f"{c}_overlap", 1.0) < 0.2]
    if poor:
        print(f"    WARNING: no common support on {poor} - the adjusted genotype "
              f"coefficient is an extrapolation and should not be interpreted",
              flush=True)

    obs = _fit(d, formula)
    if obs is None:
        return dict(value=value, coef=np.nan, p=np.nan, n_perm=0)

    labels = d.groupby("mouse")["genotype"].first()
    mice = labels.index.to_numpy()
    n, k = len(mice), int((labels == group_a).sum())
    total = comb(n, k)
    if total <= MAX_ENUMERATE:
        assignments, exact = list(combinations(range(n), k)), True
    else:
        rng = np.random.default_rng(0)
        assignments = [tuple(rng.choice(n, k, replace=False)) for _ in range(N_SAMPLED)]
        exact = False

    null = []
    perm = d.copy()
    for combo in assignments:
        lab = np.full(n, "WT", dtype=object)
        lab[list(combo)] = group_a
        perm["genotype"] = pd.Categorical(
            perm["mouse"].map(dict(zip(mice, lab))), categories=["WT", group_a])
        c = _fit(perm, formula)
        if c is not None:
            null.append(c)
    null = np.asarray(null)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (null.size + 1)
    return dict(value=value, coef=obs, p=float(p), n_perm=int(null.size),
                exact=exact, n_mice=n, n_sessions=len(d),
                covariates=",".join(covariates) or "none",
                min_overlap=min([v for k, v in diagnostics.items()
                                 if k.endswith("_overlap")], default=np.nan),
                interpretable=not poor, **diagnostics)


def run() -> pd.DataFrame:
    R = RESULTS
    jobs = []

    react = pd.read_csv(R / "reactivation_freeze_poolmatched.csv")
    react = react[react.genotype.isin(["WT", "NLGF"])]
    post = react[react.epoch == "post"]
    jobs += [("Tier 5 reactivation (post)", post, "excess_r", []),
             ("Tier 5 reactivation (post)", post, "excess_r",
              ["n_cells", "epoch_seconds"])]

    dec = pd.read_csv(R / "decoding_matched.csv")
    dec = dec[dec.genotype.isin(["WT", "NLGF"])]
    jobs += [("Tier 2 decoding error", dec, "decoding_error_matched", []),
             ("Tier 2 decoding error", dec, "decoding_error_matched",
              ["n_cells_total", "n_trials_total"])]

    place = pd.read_csv(R / "place_lap_matched.csv")
    jobs += [("Tier 2 place yield (lap-matched)", place, "frac_place_cells", []),
             ("Tier 2 place yield (lap-matched)", place, "frac_place_cells",
              ["n_cells_total"])]

    land = pd.read_csv(R / "landmarks.csv")
    land = land[land.genotype.isin(["WT", "NLGF"])]
    speed = pd.read_csv(R / "speed_profile.csv")
    both = land.merge(speed[["mouse", "date", "speed_landmark_z"]],
                      on=["mouse", "date"], how="inner")
    jobs += [("Tier 3 landmark anchoring", both, "error_landmark_z", []),
             ("Tier 3 landmark anchoring", both, "error_landmark_z",
              ["speed_landmark_z"])]

    rows = []
    for label, df, value, covs in jobs:
        print(f"  {label}  ({value}, covariates: {covs or 'none'})", flush=True)
        out = lmm_permutation(df, value, covs)
        out["analysis"] = label
        rows.append(out)
    res = pd.DataFrame(rows)
    cols = ["analysis", "value", "covariates", "coef", "p", "min_overlap",
            "interpretable", "n_mice", "n_sessions", "n_perm", "exact"]
    res = res[[c for c in cols if c in res.columns]]
    res.to_csv(R / "lmm_crosscheck.csv", index=False)
    return res


if __name__ == "__main__":
    print(run().to_string(index=False))
