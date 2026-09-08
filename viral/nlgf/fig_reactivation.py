"""Figure 7: offline reactivation, and why its apparent effect does not survive.

Level rather than pre/post delta: the WT work established the delta is flat on a
familiar track, and comparing two nulls is not a test this n can pass.

Three panels of the same measure under progressively better matching, because it took
three attempts to get it right:

  unmatched        all available place cells. NLGF reactivate ~3x more strongly.
  assembly-matched 100 place cells drawn per session. Still +65% in the post epoch at
                   p = 0.005, which at 4 v 6 mice is the exact-permutation FLOOR - a
                   perfect separation of all ten animals - and still collapsed to
                   p = 0.955 once field-of-view cell count entered the model.
  pool-matched     250 cells drawn FIRST, place cells detected within that fixed pool,
                   assembly built from those. The effect goes to -6%, p = 0.858, and
                   excess_r no longer tracks cell count (rho +0.06, was +0.22).

The reason the middle step was not enough is selection, not count: place cells are
chosen FROM the population, so 100 drawn from a field of 900 are a more selective set
than 100 drawn from 400, and NLGF fields are larger. Equalising the pool equalises the
selectivity; equalising the assembly does not.

What survives is that reactivation is real: excess_r is above the shuffled null in WT
(+0.23, one-sided p = 0.031) with NLGF indistinguishable (+0.26, p = 0.118, where 4 mice
cap the sign-flip test at 0.059).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, set_style, superplot


def figure() -> None:
    matched = pd.read_csv(RESULTS / "reactivation_freeze_poolmatched.csv")
    matched = matched[matched.genotype.isin(GENOTYPE_ORDER)]
    raw = pd.read_csv(RESULTS / "reactivation_freeze.csv")
    raw = raw[raw.genotype.isin(GENOTYPE_ORDER)]
    assembly = pd.read_csv(RESULTS / "reactivation_freeze_matched.csv")
    assembly = assembly[assembly.genotype.isin(GENOTYPE_ORDER)]

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.3))

    # Unmatched, for contrast
    ax = axes[0]
    rs = raw.groupby(["genotype", "mouse", "date"])["excess_r"].median().reset_index()
    pm, res = compare(rs, "excess_r")
    superplot(ax, rs, "excess_r", legend=True)
    ax.set_ylabel("excess_r (unmatched)")
    ax.set_title(f"Unmatched assembly\ndiff {res['diff']:+.2f}, p = {res['p']:.3f}",
                 fontsize=10.5)
    ax.text(0.5, -0.21, "all available place cells", transform=ax.transAxes,
            ha="center", fontsize=8.5, color="#868E96")

    # Assembly-matched post, where the artefact peaked
    ax = axes[1]
    sub = assembly[assembly.epoch == "post"]
    pm, res = compare(sub, "excess_r")
    superplot(ax, sub, "excess_r")
    ax.axhline(0, color="#CED4DA", lw=0.9, ls="--", zorder=0)
    ax.set_ylabel("excess_r (100 place cells)")
    ax.set_title(f"Assembly matched, post\ndiff {res['diff']:+.2f}, p = {res['p']:.3f}",
                 fontsize=10.5)
    ax.text(0.5, -0.21, "still tracks FOV cell count", transform=ax.transAxes,
            ha="center", fontsize=8.5, color="#868E96")

    # Pool-matched, pre and post
    for ax, epoch in zip(axes[2:3], ["post"]):
        sub = matched[matched.epoch == epoch]
        pm, res = compare(sub, "excess_r")
        superplot(ax, sub, "excess_r")
        ax.axhline(0, color="#CED4DA", lw=0.9, ls="--", zorder=0)
        ax.set_ylabel("excess_r (250-cell pool)")
        ax.set_title(f"Pool matched, {epoch}\ndiff {res['diff']:+.2f}, "
                     f"p = {res['p']:.3f}", fontsize=10.5)

    # The covariate that removed it at the assembly-matched step
    ax = axes[3]
    post = assembly[assembly.epoch == "post"]
    pm2 = post.groupby(["genotype", "mouse"])[
        ["excess_r", "epoch_seconds", "n_cells"]].median().reset_index()
    pm2["genotype"] = pd.Categorical(pm2.genotype, categories=["WT", "NLGF"])
    models = [
        ("genotype only", "excess_r ~ genotype"),
        ("+ epoch length", "excess_r ~ genotype + epoch_seconds"),
        ("+ cells in FOV", "excess_r ~ genotype + epoch_seconds + n_cells"),
    ]
    coefs, ps, labels = [], [], []
    for label, formula in models:
        fit = smf.ols(formula, data=pm2).fit()
        coefs.append(fit.params["genotype[T.NLGF]"])
        ps.append(fit.pvalues["genotype[T.NLGF]"])
        labels.append(label)
    ys = np.arange(len(models))[::-1]
    ax.barh(ys, coefs, height=0.5, color=GENOTYPE_COLOURS["NLGF"], alpha=0.85)
    for y, c, p in zip(ys, coefs, ps):
        ax.text(c + 0.006, y, f"p = {p:.3f}", va="center", fontsize=9,
                color="#495057")
    ax.axvline(0, color="#495057", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.02, max(coefs) * 1.55)
    ax.set_xlabel("NLGF − WT coefficient")
    ax.set_title("Adjusting for cell count\nremoves the effect", fontsize=10.5)

    n_n = matched[matched.genotype == "NLGF"].mouse.nunique()
    n_w = matched[matched.genotype == "WT"].mouse.nunique()
    fig.suptitle(
        f"Offline reactivation of run ensembles  ·  {n_n} NLGF / {n_w} WT mice  ·  "
        f"the apparent difference is a property of the recording, not the animal",
        y=1.04, fontsize=12.5, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig7_reactivation.png")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    figure()
    print(f"wrote figure to {PLOTS}")
