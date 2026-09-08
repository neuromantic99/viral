from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FORMULA = "y ~ genotype * exper + cells + tis"


def prepare(df, logit_y=True):
    """Build the model frame: log experience axis, z-scored predictors."""
    d = df.copy()

    if logit_y:  # DV is a proportion
        p = np.clip(d["acc"], 1e-3, 1 - 1e-3)
        d["y"] = np.log(p / (1 - p))
    else:
        d["y"] = d["acc"]

    # z-scoring puts the intercept at the grand mean, not at zero trials
    def z(v):
        return (v - v.mean()) / v.std()

    d["exper"] = z(np.log(d["cum_trials"] + 1))
    d["cells"] = z(np.log(d["n_cells"]))
    d["tis"] = z(d["trials_in_session"])

    d["genotype"] = pd.Categorical(d["genotype"], categories=["WT", "NLGF"])
    return d


def fit(d):
    """Session-level LMM, random intercept and slope by mouse."""
    return smf.mixedlm(FORMULA, d, groups=d["mouse"], re_formula="~exper").fit()


def _coefs(res):
    """(genotype main effect, genotype x experience interaction)."""
    return res.params["genotype[T.NLGF]"], res.params["genotype[T.NLGF]:exper"]


def permutation_test(d, n_perm=None):
    """Permute the genotype label across MICE (each mouse's sessions travel
    with it) and refit. n_perm=None enumerates every possible assignment,
    which makes the test exact. 14 mice split 8/6 is 3003 fits, a few minutes.
    """
    labels = d.groupby("mouse")["genotype"].first()
    mice = labels.index.to_numpy()
    n = len(mice)
    k = int((labels == "NLGF").sum())

    if n_perm is None:
        assignments = list(combinations(range(n), k))
    else:
        rng = np.random.default_rng(0)
        assignments = [rng.choice(n, k, replace=False) for _ in range(n_perm)]

    obs_main, obs_int = _coefs(fit(d))

    null = []
    d_perm = d.copy()
    for combo in assignments:
        lab = np.full(n, "WT", dtype=object)
        lab[list(combo)] = "NLGF"
        d_perm["genotype"] = pd.Categorical(
            d_perm["mouse"].map(dict(zip(mice, lab))), categories=["WT", "NLGF"]
        )
        null.append(_coefs(fit(d_perm)))

    null = np.array(null)

    def p(obs, dist):
        return (np.sum(np.abs(dist) >= abs(obs)) + 1) / (len(dist) + 1)

    return pd.DataFrame(
        {
            "term": ["genotype (level)", "genotype x experience"],
            "coef": [obs_main, obs_int],
            "p_perm": [p(obs_main, null[:, 0]), p(obs_int, null[:, 1])],
            "n_perm": len(null),
            "exact": n_perm is None,
        }
    )


def two_stage(d):
    """One slope per mouse, then permute those 14 numbers. Model-free check
    on the LMM, with the animal as the unmistakable unit of analysis."""
    ref_x = d["exper"].median()

    rows = []
    for mouse, g in d.groupby("mouse"):
        slope, intercept = np.polyfit(g["exper"], g["y"], 1)
        rows.append(
            {
                "mouse": mouse,
                "genotype": g["genotype"].iloc[0],
                "slope": slope,
                "level_at_ref": intercept + slope * ref_x,
                "n_sessions": len(g),
            }
        )
    per_mouse = pd.DataFrame(rows)

    is_nlgf = (per_mouse["genotype"] == "NLGF").to_numpy()
    n, k = len(per_mouse), int(is_nlgf.sum())
    assignments = list(combinations(range(n), k))

    out = []
    for stat in ("slope", "level_at_ref"):
        v = per_mouse[stat].to_numpy()
        obs = v[is_nlgf].mean() - v[~is_nlgf].mean()
        null = np.array(
            [
                v[list(c)].mean() - v[[i for i in range(n) if i not in c]].mean()
                for c in assignments
            ]
        )
        out.append(
            {
                "stat": stat,
                "diff_NLGF_minus_WT": obs,
                "p_perm": (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1),
                "n_perm": len(null),
            }
        )

    return per_mouse, pd.DataFrame(out)
