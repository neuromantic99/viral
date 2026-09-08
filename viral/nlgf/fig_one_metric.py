"""One metric, one panel: the same measurement made within a session and across days.

Field correlation is a single quantity - how well one cell's spatial map matches
another copy of itself. Computed between two halves of ONE session it measures how
reproducible the map is when nothing has had a chance to change; computed between two
sessions it measures whether the map came back. Same cells, same metric, same pipeline;
only the interval differs.

That is what makes this the right single slide. A null on its own invites "your assay is
blunt". Here the assay demonstrably works - it separates the genotypes on the right-hand
pair - so the flat left-hand pair means the within-session map really is normal, not
that the measurement could not tell.

Colour is reserved for genotype. The brackets and p-values are a single neutral ink, so
the only thing in the panel carrying meaning by hue is WT versus NLGF - at talk size, a
second colour on the annotation layer reads as a second variable from the back of a room.

The cross-day p is gap-adjusted (NLGF pairs average a slightly longer interval and field
correlation decays with it); the within-session p is not, because a split-half inside one
session cannot depend on the interval between sessions. Raw and adjusted barely differ
here anyway - p = 0.019 and 0.022 - so nothing rests on the choice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.matching import gap_adjusted
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import exact_permutation
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, set_style

# One ink colour for every non-data mark, so nothing competes with the two
# genotype colours - the only thing in the panel that should carry meaning.
BRACKET = "#212529"

POS = {("within", "WT"): 0.0, ("within", "NLGF"): 0.62,
       ("across", "WT"): 1.85, ("across", "NLGF"): 2.47}


def main() -> None:
    set_style()
    d = pd.read_csv(RESULTS / "spatial_stability.csv")
    d = d[d.genotype.isin(GENOTYPE_ORDER)]

    fig, ax = plt.subplots(figsize=(10.5, 7.6))
    rng = np.random.default_rng(0)

    for cond, col in [("within", "field_corr_within"), ("across", "field_corr_cross")]:
        for g in GENOTYPE_ORDER:
            s = d[d.genotype == g].dropna(subset=[col])
            x = POS[(cond, g)]
            colour = GENOTYPE_COLOURS[g]
            per_mouse = s.groupby("mouse")[col].median()

            bp = ax.boxplot([per_mouse.to_numpy()], positions=[x], widths=0.36,
                            showfliers=False, patch_artist=True, zorder=1)
            for patch in bp["boxes"]:
                patch.set(facecolor=colour, alpha=0.15, edgecolor=colour, linewidth=2.0)
            for part in ("whiskers", "caps", "medians"):
                for artist in bp[part]:
                    artist.set(color=colour, linewidth=2.4)

            ax.scatter(x - 0.26 + rng.uniform(-0.05, 0.05, len(s)), s[col], s=26,
                       color=colour, alpha=0.32, lw=0, zorder=2)
            ax.scatter(x + 0.26 + rng.uniform(-0.04, 0.04, per_mouse.size),
                       per_mouse.to_numpy(), s=150, color=colour, alpha=0.95,
                       edgecolor="white", linewidth=1.8, zorder=4)

    # p-values: within-session raw, cross-day gap-adjusted
    pm_w = d.groupby(["mouse", "genotype"], as_index=False).field_corr_within.median()
    p_w = exact_permutation(pm_w, "field_corr_within")["p"]
    adj, _ = gap_adjusted(d, "field_corr_cross", "gap_days")
    pm_a = adj.groupby(["mouse", "genotype"], as_index=False).adj.median()
    p_a = exact_permutation(pm_a, "adj")["p"]

    top = d[["field_corr_within", "field_corr_cross"]].max().max()
    for cond, p_ in [("within", p_w), ("across", p_a)]:
        x0, x1 = POS[(cond, "WT")], POS[(cond, "NLGF")]
        y = top + 0.05
        ax.plot([x0, x0, x1, x1], [y, y + 0.022, y + 0.022, y], lw=1.8,
                color=BRACKET)
        ax.text((x0 + x1) / 2, y + 0.038, f"p = {p_:.2f}", ha="center", va="bottom",
                fontsize=22, color=BRACKET)

    ax.set_xticks([(POS[("within", "WT")] + POS[("within", "NLGF")]) / 2,
                   (POS[("across", "WT")] + POS[("across", "NLGF")]) / 2])
    ax.set_xticklabels(["Within one session\n(two halves)",
                        "Across days\n(same cells)"], fontsize=24)
    ax.set_xlim(-0.55, 3.0)
    ax.set_ylim(None, top + 0.16)
    ax.set_ylabel("Place field correlation", fontsize=24)
    ax.tick_params(axis="y", labelsize=20)
    ax.tick_params(axis="x", length=0, pad=12)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [plt.Line2D([], [], marker="o", ls="none", ms=15,
                          mfc=GENOTYPE_COLOURS[g], mec="white", label=g)
               for g in GENOTYPE_ORDER]
    ax.legend(handles=handles, frameon=False, fontsize=22, loc="lower left",
              handletextpad=0.4, labelspacing=0.3)

    # lay the axes out first, then place the two header lines into the space left for
    # them - suptitle and a figure-level subtitle fight each other if tight_layout runs
    # afterwards, and at talk size the collision is a whole line of text
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.suptitle("The map RSC builds is normal. Keeping it is not.",
                 fontsize=27, fontweight="bold", y=0.995)
    fig.text(0.5, 0.895, "7 NLGF vs 8 WT mice  ·  small dots sessions, large dots mice",
             ha="center", fontsize=16, color="#868E96")
    out = PLOTS / "fig26_one_metric.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  within session: p = {p_w:.4f}")
    print(f"  across days   : p = {p_a:.4f} (gap-adjusted)")


if __name__ == "__main__":
    main()
