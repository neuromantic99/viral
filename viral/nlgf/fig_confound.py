"""Figure 6: the Tier 2 result is a data-quantity artefact.

Place cell yield, field width and reliability all rise with the number of laps in the
session, and decoding error falls with laps and with cell count. NLGF run more trials
(65.5 vs 46.6) and have more cells (556 vs 497), so all four apparent genotype effects
are what more data predicts.

The mechanism is in the place-field definition itself. place_threshold is the 99th
percentile of 2000 per-lap circularly-shuffled rate maps. Each shuffle averages the
session's laps with an independent random rotation per lap, so the shuffled mean map
flattens and its variance falls as 1/n_laps. With more laps the threshold drops toward
the cell's own mean rate while the real map keeps its structure, and more cells clear
it - measured here as the ratio of the median threshold to the median rate map:

    32 laps  ratio 1.61   51% of cells pass
    54 laps  ratio 1.72   51%
    91 laps  ratio 1.39   92%
   103 laps  ratio 1.46   82%

Two further reasons the criterion is permissive on this data. The 7.5 cm smoothing
runs over 2 cm bins, so sigma is 3.75 bins and "5 consecutive supra-threshold bins" is
roughly one independent test, not five. And filter_additional_check asks for
within-field above out-of-field firing on at least max(3, 15% of laps), which is far
below the 50% expected by chance, so it removes 1-10% of candidates rather than acting
as a real control.
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
from viral.nlgf.stats import compare, minimum_detectable_difference
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, set_style, strip_with_mice


def figure() -> None:
    place = pd.read_csv(RESULTS / "place_sessions_rewarded_None.csv")
    place = place[place.genotype.isin(GENOTYPE_ORDER)]
    matched = pd.read_csv(RESULTS / "decoding_matched.csv")
    matched = matched[matched.genotype.isin(GENOTYPE_ORDER)]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))

    # A, B: the confound itself
    from scipy.stats import spearmanr
    for ax, value, label in zip(
        axes[:2],
        ["frac_place_cells", "field_width_cm"],
        ["Fraction place cells", "Field width (cm)"],
    ):
        for g in GENOTYPE_ORDER:
            s = place[place.genotype == g]
            ax.scatter(s.n_trials, s[value], s=26, color=GENOTYPE_COLOURS[g],
                       alpha=0.65, edgecolor="none", label=g)
        ok = place.dropna(subset=["n_trials", value])
        rho, p = spearmanr(ok.n_trials, ok[value])
        b = np.polyfit(ok.n_trials, ok[value], 1)
        xs = np.linspace(ok.n_trials.min(), ok.n_trials.max(), 20)
        ax.plot(xs, np.polyval(b, xs), color="#495057", lw=1.4, ls="--", zorder=3)
        ax.set_xlabel("Trials in session")
        ax.set_ylabel(label)
        ax.set_title(f"{label} tracks lap count\nSpearman rho = {rho:+.2f}, p = {p:.1g}",
                     fontsize=10.5)
    axes[0].legend(loc="lower right")

    # C: unmatched versus matched decoding, per mouse
    ax = axes[2]
    un = place.groupby(["genotype", "mouse"])["decoding_error_cm"].median()
    ma = matched.groupby(["genotype", "mouse"])["decoding_error_matched"].median()
    both = pd.concat([un.rename("unmatched"), ma.rename("matched")], axis=1).dropna()
    both = both.reset_index()
    for g in GENOTYPE_ORDER:
        s = both[both.genotype == g]
        ax.plot([0, 1], [s.unmatched, s.matched], color=GENOTYPE_COLOURS[g],
                alpha=0.45, lw=1.2, zorder=1)
        ax.scatter([0] * len(s), s.unmatched, s=52, color=GENOTYPE_COLOURS[g],
                   edgecolor="white", linewidth=1.1, zorder=3, label=g)
        ax.scatter([1] * len(s), s.matched, s=52, color=GENOTYPE_COLOURS[g],
                   edgecolor="white", linewidth=1.1, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["unmatched", "trial- and\ncell-matched"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Decoding error (cm)")
    ax.set_title("Matching removes the separation", fontsize=10.5)

    # D: the matched comparison, which is the answer
    ax = axes[3]
    pm, res = compare(matched, "decoding_error_matched", how="median")
    mde = minimum_detectable_difference(pm, "decoding_error_matched")
    strip_with_mice(ax, pm, "decoding_error_matched")
    ax.set_ylabel("Decoding error (cm)")
    ax.set_title(f"Matched: 24 trials, 100 cells\ndiff {res['diff']:+.2f} cm, "
                 f"p = {res['p']:.3f}", fontsize=10.5)
    ax.text(0.5, -0.21,
            f"observed {100*res['diff']/res['mean_b']:+.0f}%  ·  "
            f"detectable {100*mde/res['mean_b']:.0f}%",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#868E96")

    fig.suptitle(
        "Spatial coding differences in NLGF are explained by data quantity, not coding",
        y=1.04, fontsize=13, weight="semibold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS / "fig6_confound.png")
    plt.close(fig)
    print(f"matched: diff {res['diff']:+.2f} cm, p={res['p']:.4f}, mde {mde:.2f}")


if __name__ == "__main__":
    set_style()
    figure()
