"""Figure: landmark anchoring, without the machinery.

The top row is the whole argument and needs no explanation beyond its axes: line the
data up on the landmarks, and see whether firing rate does anything there. The grey
line is the same operation on positions exactly out of phase with the landmarks, where
there is nothing to see - it is the control that turns "there is a bump" into "there is
a bump BECAUSE of the landmarks", since a regular 45 cm comb would find periodic
structure in the corridor whether or not landmarks caused it.

The middle panel is the reason this was not simply plotted in the first place: animals
slow down at landmarks, and deconvolved rate rises with speed. The bottom row repeats
the rate modulation after regressing rate on speed within each session, so the reader
can see how much of the bump survives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, p_bracket, set_style, superplot

PSEUDO_C = "#ADB5BD"


def _curves(npz, df, key):
    out = {}
    for g in GENOTYPE_ORDER:
        stack = []
        for r in df[df.genotype == g].itertuples():
            name = f"{r.mouse}_{r.date}_{key}"
            if name in npz:
                stack.append(npz[name])
        out[g] = np.vstack(stack) if stack else None
    return out


def _band(ax, x, arr, colour, label):
    if arr is None or arr.shape[0] < 2:
        return
    m = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0) / np.sqrt(np.sum(np.isfinite(arr), axis=0))
    ax.plot(x, m, color=colour, lw=2.0, label=label)
    ax.fill_between(x, m - se, m + se, color=colour, alpha=0.18, lw=0)


def main() -> None:
    set_style()
    df = pd.read_csv(RESULTS / "landmark_profile.csv")
    npz = np.load(RESULTS / "landmark_profiles.npz")
    df = df[df.genotype.isin(GENOTYPE_ORDER)]
    x = npz[f"{df.iloc[0].mouse}_{df.iloc[0].date}_offsets"]

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))

    # A: the plot itself
    ax = axes[0, 0]
    for g in GENOTYPE_ORDER:
        _band(ax, x, _curves(npz, df, "rate_real")[g], GENOTYPE_COLOURS[g], g)
    _band(ax, x, np.vstack([v for v in _curves(npz, df, "rate_pseudo").values()
                            if v is not None]), PSEUDO_C, "away from landmarks")
    ax.axvline(0, color="#495057", lw=1.0, ls=":")
    ax.set_xlabel("Distance from landmark (cm)")
    ax.set_ylabel("Population firing rate\n(normalised)")
    ax.set_title("A  Rate around a landmark", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    # B: the confound
    ax = axes[0, 1]
    for g in GENOTYPE_ORDER:
        _band(ax, x, _curves(npz, df, "speed_real")[g], GENOTYPE_COLOURS[g], g)
    _band(ax, x, np.vstack([v for v in _curves(npz, df, "speed_pseudo").values()
                            if v is not None]), PSEUDO_C, "away from landmarks")
    ax.axvline(0, color="#495057", lw=1.0, ls=":")
    ax.set_xlabel("Distance from landmark (cm)")
    ax.set_ylabel("Running speed (cm/s)")
    ax.set_title("B  Animals slow at landmarks", loc="left", fontweight="bold")

    # C: rate after removing the speed relationship
    ax = axes[0, 2]
    for g in GENOTYPE_ORDER:
        _band(ax, x, _curves(npz, df, "resid_real")[g], GENOTYPE_COLOURS[g], g)
    ax.axvline(0, color="#495057", lw=1.0, ls=":")
    ax.axhline(0, color="#CED4DA", lw=0.9)
    ax.set_xlabel("Distance from landmark (cm)")
    ax.set_ylabel("Rate, speed removed")
    ax.set_title("C  What survives speed correction", loc="left", fontweight="bold")

    # D-F: the modulation indices
    for ax, (col, label) in zip(axes[1], [
            ("rate_mod", "Rate modulation\n(near − far)/(near + far)"),
            ("speed_mod", "Speed modulation"),
            ("resid_mod", "Rate modulation,\nspeed removed")]):
        s = df.dropna(subset=[col])
        superplot(ax, s, col, genotypes=GENOTYPE_ORDER)
        pm, res = compare(s, col)
        p_bracket(ax, res["p"])
        pseudo = df[f"{col}_pseudo"].median()
        ax.axhline(pseudo, color=PSEUDO_C, lw=1.4, ls="--", zorder=0)
        ax.text(0.02, 0.02, "grey line: away from landmarks", transform=ax.transAxes,
                fontsize=7.5, color="#868E96")
        ax.set_ylabel(label)
        ax.axhline(0, color="#CED4DA", lw=0.8, zorder=0)

    fig.suptitle("Landmark anchoring, measured as firing rate around the landmarks",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = PLOTS / "fig27_landmark_profile.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    for col in ["rate_mod", "speed_mod", "resid_mod"]:
        s = df.dropna(subset=[col])
        _, r = compare(s, col)
        print(f"  {col:11s} NLGF {r['mean_a']:+.4f}  WT {r['mean_b']:+.4f}  "
              f"diff {r['diff']:+.4f}  p = {r['p']:.3f}   "
              f"[away from landmarks: {df[col + '_pseudo'].median():+.4f}]")


if __name__ == "__main__":
    main()
