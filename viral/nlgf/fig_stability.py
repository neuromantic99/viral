"""Figure 19: place field stability across days.

The strongest candidate finding in the study, so the figure is built to let a sceptic
check it rather than to assert it.

  A  the measure itself - cross-day field correlation against the within-day split-half
     ceiling, per pair. A point on the diagonal is a field as reproducible between
     sessions as within one.
  B  the confound: stability tracks registration residual, and NLGF register worse.
  C  the effect under five levels of control. Field correlation falls ~0.06 per day
     and NLGF pairs average a slightly longer gap (2.0 vs 1.7 d), so the raw estimate
     is inflated - but only by about 20%, not more: the arithmetic says 0.30 days at
     -0.058/day is a bias of 0.019. Adjusting the trend out while keeping every pair
     gives -0.148, p = 0.016, and it does not matter whether the slope is fitted on WT
     alone, within genotypes, or pooled (all agree to the third decimal).

     The paler bar, histogram matching, is a conservative bound rather than a better
     estimate. It moves the number three times further than the gap bias can account
     for, because it discards a third of the pairs and changes WHICH MICE contribute -
     JB021 has a single pair and drops out of 72% of draws. With 7 and 8 animals that
     perturbation is larger than the bias being removed.
  D  the dissociation: individual fields drift more, but the population vector
     correlation does not differ. The geometry survives while the cells carrying it
     shuffle.

Panel C is the argument. In every earlier tier an apparent genotype effect collapsed
entirely when its confound was controlled; this one shrinks by about a fifth, to -0.148
at p = 0.016, and holds under every control. Note which tool belongs where: matching by
construction was right for Tiers 2 and 5 because those had positivity violations - no
common support at all - whereas session gap has full overlap and only a distributional
shift, where adjustment is appropriate and matching merely discards data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.matching import ladder
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, set_style, superplot


def figure() -> None:
    d = pd.read_csv(RESULTS / "spatial_stability.csv")
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6))

    # A: cross-day against the within-day ceiling
    ax = axes[0]
    for g in GENOTYPE_ORDER:
        s = d[d.genotype == g]
        ax.scatter(s.field_corr_within, s.field_corr_cross, s=34,
                   color=GENOTYPE_COLOURS[g], alpha=0.7, edgecolor="none", label=g)
    lim = [0.2, 0.85]
    ax.plot(lim, lim, color="#495057", lw=1.2, ls="--", zorder=0)
    ax.text(0.62, 0.66, "as stable across days\nas within one", fontsize=8,
            color="#868E96", rotation=38, ha="center")
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel("Within-session field correlation\n(odd vs even laps)")
    ax.set_ylabel("Across-day field correlation")
    ax.set_title("Fields fall below their own ceiling\nmore in NLGF", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper left")

    # B: the confound
    ax = axes[1]
    for g in GENOTYPE_ORDER:
        s = d[d.genotype == g]
        ax.scatter(s.affine_rms_px, s.field_corr_ratio, s=32,
                   color=GENOTYPE_COLOURS[g], alpha=0.7, edgecolor="none", label=g)
    ok = d.dropna(subset=["affine_rms_px", "field_corr_ratio"])
    rho, p = spearmanr(ok.affine_rms_px, ok.field_corr_ratio)
    b = np.polyfit(ok.affine_rms_px, ok.field_corr_ratio, 1)
    xs = np.linspace(ok.affine_rms_px.min(), ok.affine_rms_px.max(), 20)
    ax.plot(xs, np.polyval(b, xs), color="#495057", lw=1.4, ls="--", zorder=3)
    lo = max(d.loc[d.genotype == "NLGF", "affine_rms_px"].min(),
             d.loc[d.genotype == "WT", "affine_rms_px"].min())
    hi = min(d.loc[d.genotype == "NLGF", "affine_rms_px"].max(),
             d.loc[d.genotype == "WT", "affine_rms_px"].max())
    ax.axvspan(lo, hi, color="#CED4DA", alpha=0.35, lw=0, zorder=0)
    ax.text((lo + hi) / 2, ax.get_ylim()[1], "matched band", ha="center", va="top",
            fontsize=8, color="#868E96")
    ax.set_xlabel("Registration residual (px)")
    ax.set_ylabel("Field stability ratio")
    ax.set_title(f"The confound\nrho = {rho:+.2f}, p = {p:.1g}", fontsize=10.5)

    # C: effect under increasing control - the argument
    ax = axes[2]
    dw = d[d.genotype.isin(GENOTYPE_ORDER)]
    lad = ladder(dw, "field_corr_ratio", "NLGF", "WT", ["affine_rms_px"], "gap_days")
    # the adjusted rows are the headline; the histogram-matched row is a conservative
    # bound that costs a third of the pairs, so it is drawn paler
    alphas = [0.9, 0.9, 0.9, 0.9, 0.45]
    xs = np.arange(len(lad))
    ax.bar(xs, lad["diff"], width=0.62, color=[
        (*mpl.colors.to_rgb(GENOTYPE_COLOURS["NLGF"]), a) for a in alphas])
    for x, r in zip(xs, lad.itertuples()):
        ax.text(x, r.diff / 2, f"p = {r.p:.3f}", ha="center", va="center",
                fontsize=8, color="white", weight="semibold", rotation=90)
        ax.text(x, -0.004, f"{r.n_pairs:.0f}", ha="center", va="top", fontsize=7.5,
                color="#868E96")
    ax.axhline(0, color="#495057", lw=1.1)
    ax.set_ylim(lad["diff"].min() * 1.3, 0.03)
    ax.set_xticks(xs)
    ax.set_xticklabels(lad.control, fontsize=6.8)
    ax.set_ylabel("NLGF − WT field stability ratio")
    ax.set_title("Effect holds under gap control\n(pairs contributing, below bars)",
                 fontsize=10.5)

    # D: the dissociation, on the fully matched subset
    ax = axes[3]
    from viral.nlgf.matching import gap_adjusted, matched_subset

    m = matched_subset(d[d.genotype.isin(GENOTYPE_ORDER)], ["affine_rms_px"],
                       "NLGF", "WT")
    rng = np.random.default_rng(0)
    for i, (col, label) in enumerate([("field_corr_ratio", "individual\nfields"),
                                      ("pv_corr_ratio", "population\nvector")]):
        pm, res = compare(m, col)
        for g in GENOTYPE_ORDER:
            v = m.loc[m.genotype == g].groupby("mouse")[col].median().to_numpy()
            off = 0.17 if g == "NLGF" else -0.17
            ax.scatter(i + off + rng.uniform(-0.05, 0.05, v.size), v, s=58,
                       color=GENOTYPE_COLOURS[g], alpha=0.9, edgecolor="white",
                       linewidth=1.1, zorder=3, label=g if i == 0 else None)
            ax.hlines(np.median(v), i + off - 0.13, i + off + 0.13,
                      color=GENOTYPE_COLOURS[g], lw=2.4, zorder=4)
        ax.text(i, 1.16, f"p = {res['p']:.3f}", ha="center", fontsize=9.5)
    ax.axhline(1.0, color="#495057", lw=1.0, ls=":", zorder=0)
    ax.text(1.62, 1.005, "no drift", fontsize=8, color="#868E96", va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["individual\nfields", "population\nvector"])
    ax.set_xlim(-0.5, 1.75)
    ax.set_ylim(0.55, 1.25)
    ax.set_ylabel("Stability ratio (cross-day / within-day)")
    ax.set_title("Cells drift, the population\ncode does not", fontsize=10.5)
    ax.legend(fontsize=9, loc="lower left")

    fig.suptitle(
        "Place fields are less stable across days in NLGF, while the population code "
        "is preserved  ·  registered cells, matched laps",
        y=1.03, fontsize=12.5, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig19_stability.png")
    plt.close(fig)
    for r in lad.itertuples():
        tail = f"  {r.frac_sig:.0%} of draws p<0.05" if np.isfinite(r.frac_sig) else ""
        print(f"  {r.control.replace(chr(10), ' '):36} diff {r.diff:+.3f}  "
              f"p {r.p:.4f}  ({r.n_pairs:.0f} pairs){tail}")


if __name__ == "__main__":
    set_style()
    figure()
    print(f"wrote figure to {PLOTS}")
