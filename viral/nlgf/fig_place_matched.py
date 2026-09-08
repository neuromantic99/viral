"""Figure 10: holding lap count fixed removes the place-cell confound.

The detection method is unchanged - same shuffle, same 99th percentile, same
five-consecutive-bins rule, same additional check. The only difference is that every
session is scored on its first 30 laps and the threshold is regenerated from those same
30 laps, so every session is tested at equal statistical power.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, set_style, superplot

METRICS = [
    ("frac_place_cells", "Fraction place cells"),
    ("field_width_cm", "Field width (cm)"),
    ("spatial_info_pc", "Spatial info (bits/event)"),
    ("reliability_pc", "Odd–even correlation"),
]


def figure() -> None:
    old = pd.read_csv(RESULTS / "place_sessions_rewarded_None.csv")
    old = old[old.genotype.isin(GENOTYPE_ORDER)]
    new = pd.read_csv(RESULTS / "place_lap_matched.csv")
    merged = old.merge(new[["mouse", "date", "frac_place_cells"]],
                       on=["mouse", "date"], suffixes=("_un", "_lm"))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))

    # Top left: the confound, before and after
    for ax, col, label in zip(
        axes[0][:2],
        ["frac_place_cells_un", "frac_place_cells_lm"],
        ["All laps (as cached)", "First 30 laps, threshold regenerated"],
    ):
        for g in GENOTYPE_ORDER:
            s = merged[merged.genotype == g]
            ax.scatter(s.n_trials, s[col], s=26, color=GENOTYPE_COLOURS[g],
                       alpha=0.7, edgecolor="none", label=g)
        ok = merged.dropna(subset=["n_trials", col])
        rho, p = spearmanr(ok.n_trials, ok[col])
        b = np.polyfit(ok.n_trials, ok[col], 1)
        xs = np.linspace(ok.n_trials.min(), ok.n_trials.max(), 20)
        ax.plot(xs, np.polyval(b, xs), color="#495057", lw=1.5, ls="--", zorder=3)
        ax.set_xlabel("Trials available in session")
        ax.set_ylabel("Fraction place cells")
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}\nrho = {rho:+.2f}, p = {p:.1g}", fontsize=10.5)
    axes[0][0].legend(loc="lower right")

    # Top right: yield distribution collapses onto one value
    ax = axes[0][2]
    for i, (col, label) in enumerate([("frac_place_cells_un", "all laps"),
                                      ("frac_place_cells_lm", "30 laps")]):
        for g in GENOTYPE_ORDER:
            v = merged.loc[merged.genotype == g, col].dropna()
            x = i + (0.18 if g == "NLGF" else -0.18)
            rng = np.random.default_rng(0)
            ax.scatter(x + rng.uniform(-0.06, 0.06, v.size), v, s=20,
                       color=GENOTYPE_COLOURS[g], alpha=0.6, edgecolor="none")
            ax.hlines(v.median(), x - 0.13, x + 0.13,
                      color=GENOTYPE_COLOURS[g], lw=2.4, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["all laps", "30 laps"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction place cells")
    ax.set_title("Spread collapses onto ~50%", fontsize=10.5)

    # Bottom: the four lap-matched genotype comparisons
    for ax, (value, label) in zip(axes[1], METRICS[:3]):
        per_mouse, res = compare(new, value, how="median")
        superplot(ax, new, value, legend=(value == METRICS[0][0]),
                  p_value=res["p"])
        ax.set_ylabel(label)
        ax.set_title(label, fontsize=10.5)

    n_n = new[new.genotype == "NLGF"].mouse.nunique()
    n_w = new[new.genotype == "WT"].mouse.nunique()
    fig.suptitle(
        f"Same detection method, equal laps: the place-cell difference disappears  ·  "
        f"{len(new)} sessions, {n_n} NLGF / {n_w} WT mice",
        y=1.01, fontsize=12.5, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig10_place_lap_matched.png")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    figure()
    print(f"wrote figure to {PLOTS}")
