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
from viral.nlgf.style import (GENOTYPE_COLOURS, GENOTYPE_ORDER, SHORT_NAME,
                              set_style)

# One ink colour for every non-data mark, so nothing competes with the two
# genotype colours - the only thing in the panel that should carry meaning.
BRACKET = "#212529"

def _positions(genotypes):
    """Two condition blocks, one slot per genotype in each, with a gap between blocks."""
    step, gap = 0.62, 1.25
    span = step * (len(genotypes) - 1)
    return {(cond, g): c * (span + gap) + i * step
            for c, cond in enumerate(("within", "across"))
            for i, g in enumerate(genotypes)}


def figure(genotypes, out_name: str, brackets) -> None:
    set_style()
    POS = _positions(genotypes)
    d = pd.read_csv(RESULTS / "spatial_stability.csv")
    d = d[d.genotype.isin(genotypes)]

    fig, ax = plt.subplots(figsize=(12.5, 9.0))
    rng = np.random.default_rng(0)

    for cond, col in [("within", "field_corr_within"), ("across", "field_corr_cross")]:
        for g in genotypes:
            s = d[d.genotype == g].dropna(subset=[col])
            x = POS[(cond, g)]
            colour = GENOTYPE_COLOURS[g]
            per_mouse = s.groupby("mouse")[col].median()

            bp = ax.boxplot([per_mouse.to_numpy()], positions=[x], widths=0.36,
                            showfliers=False, patch_artist=True, zorder=1)
            for patch in bp["boxes"]:
                patch.set(facecolor=colour, alpha=0.15, edgecolor=colour, linewidth=2.6)
            for part in ("whiskers", "caps", "medians"):
                for artist in bp[part]:
                    artist.set(color=colour, linewidth=3.0)

            ax.scatter(x - 0.26 + rng.uniform(-0.05, 0.05, len(s)), s[col], s=38,
                       color=colour, alpha=0.32, lw=0, zorder=2)
            ax.scatter(x + 0.26 + rng.uniform(-0.04, 0.04, per_mouse.size),
                       per_mouse.to_numpy(), s=230, color=colour, alpha=0.95,
                       edgecolor="white", linewidth=2.2, zorder=4)

    # p-values: within-session raw, cross-day gap-adjusted. A split-half inside one
    # session cannot depend on the interval between sessions, so only the cross-day
    # side is adjusted.
    pm_w = d.groupby(["mouse", "genotype"], as_index=False).field_corr_within.median()
    adj, _ = gap_adjusted(d, "field_corr_cross", "gap_days")
    pm_a = adj.groupby(["mouse", "genotype"], as_index=False).adj.median()

    top = d[["field_corr_within", "field_corr_cross"]].max().max()
    for cond in ("within", "across"):
        for level, (a, b) in enumerate(brackets):
            if a not in genotypes or b not in genotypes:
                continue
            if cond == "within":
                p_ = exact_permutation(pm_w, "field_corr_within",
                                       group_a=a, group_b=b)["p"]
            else:
                p_ = exact_permutation(pm_a, "adj", group_a=a, group_b=b)["p"]
            x0, x1 = POS[(cond, b)], POS[(cond, a)]
            x0, x1 = min(x0, x1), max(x0, x1)
            y = top + 0.05 + level * 0.105
            ax.plot([x0, x0, x1, x1], [y, y + 0.024, y + 0.024, y], lw=2.4,
                    color=BRACKET)
            ax.text((x0 + x1) / 2, y + 0.042, f"p = {p_:.2f}", ha="center",
                    va="bottom", fontsize=30 if len(brackets) == 1 else 24,
                    color=BRACKET)

    ax.set_xticks([np.mean([POS[("within", g)] for g in genotypes]),
                   np.mean([POS[("across", g)] for g in genotypes])])
    ax.set_xticklabels(["Within one session\n(two halves)",
                        "Across days\n(same cells)"], fontsize=32)
    xs = list(POS.values())
    ax.set_xlim(min(xs) - 0.55, max(xs) + 0.55)
    ax.set_ylim(None, top + 0.19 + 0.105 * (len(brackets) - 1))
    ax.set_ylabel("Place field correlation", fontsize=32)
    ax.tick_params(axis="y", labelsize=27)
    ax.tick_params(axis="x", length=0, pad=16)
    ax.spines[["top", "right"]].set_visible(False)

    # with three groups there is no empty corner left inside the axes, so the legend
    # goes above them, laid out in one row
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=20,
                          mfc=GENOTYPE_COLOURS[g], mec="white",
                          label=SHORT_NAME.get(g, g))
               for g in genotypes]
    if len(genotypes) > 2:
        ax.legend(handles=handles, frameon=False, fontsize=27, ncol=len(genotypes),
                  loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  handletextpad=0.4, columnspacing=1.6)
    else:
        ax.legend(handles=handles, frameon=False, fontsize=29, loc="lower left",
                  handletextpad=0.4, labelspacing=0.25)

    # lay the axes out first, then place the two header lines into the space left for
    # them - suptitle and a figure-level subtitle fight each other if tight_layout runs
    # afterwards, and at talk size the collision is a whole line of text
    fig.tight_layout(rect=(0, 0, 1, 0.855 if len(genotypes) == 2 else 0.80))
    fig.suptitle("The map RSC builds is normal. Keeping it is not.",
                 fontsize=34, fontweight="bold",
                 y=0.995 if len(genotypes) == 2 else 1.0)
    counts = "  ·  ".join(f"{d[d.genotype == g].mouse.nunique()} {SHORT_NAME.get(g, g)}"
                          for g in genotypes)
    fig.text(0.5, 0.895 if len(genotypes) == 2 else 0.925, f"{counts} mice  ·  small dots sessions, large dots mice",
             ha="center", fontsize=20, color="#868E96")
    out = PLOTS / out_name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"wrote {out}")
    for cond in ("within", "across"):
        for a, b in brackets:
            if a not in genotypes or b not in genotypes:
                continue
            p_ = (exact_permutation(pm_w, "field_corr_within", group_a=a, group_b=b)["p"]
                  if cond == "within"
                  else exact_permutation(pm_a, "adj", group_a=a, group_b=b)["p"])
            print(f"  {cond:6s} {a:15s} vs {b:15s} p = {p_:.4f}")


if __name__ == "__main__":
    from viral.nlgf.oligo import OLIGO

    figure(GENOTYPE_ORDER, "fig26_one_metric.png", [("NLGF", "WT")])
    print()
    figure(["WT", "NLGF", OLIGO], "fig29_one_metric_oligo.png",
           [("NLGF", "WT"), (OLIGO, "NLGF")])
