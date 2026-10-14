"""Within-session spatial coding is intact in NLGF - with a positive control.

A null is only worth showing if the same figure demonstrates the method could have
found something. Five within-session measures of spatial coding are null; the sixth
panel is the cross-day measure computed from the SAME cells in the SAME sessions by the
same pipeline, and it is not null. So the flat panels are not flatness of the analysis.

That is also the biological claim in one picture. The map RSC builds inside a session is
normal in every way measured - how many cells are spatially tuned, how sharp their
fields are, how much information they carry, how reproducibly they repeat within the
session, and how well position can be read out of them. What is impaired is keeping
that map across days.

Every panel is matched by construction on the covariate that would otherwise carry it:
laps for place-cell detection (each session scored on its first 30 laps with the
threshold regenerated from those laps), cells for decoding, and session gap for the
cross-day panel. Those matchings are what removed the apparent effects in Tiers 2 and 5,
so a null here is a null after the confound is gone, not before.
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
from viral.nlgf.stats import compare, exact_permutation
from viral.nlgf.style import GENOTYPE_ORDER, p_bracket, set_style, superplot

WITHIN = [
    ("place_lap_matched", "frac_place_cells", "Place cells\n(fraction, 30 laps)"),
    ("place_lap_matched", "field_width_cm", "Field width (cm)"),
    ("place_lap_matched", "spatial_info_pc", "Spatial information\n(bits/event)"),
    ("spatial_stability", "field_corr_within", "Within-session field\nreproducibility"),
    ("decoding_matched", "decoding_error_matched", "Position decoding\nerror (cm)"),
]
ACROSS = ("spatial_stability", "field_corr_cross", "Cross-day field\ncorrelation")


def _load(stem: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"{stem}.csv")
    return df[df.genotype.isin(GENOTYPE_ORDER)]


def main() -> None:
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.4))
    axes = axes.ravel()

    for ax, (stem, col, label) in zip(axes, WITHIN):
        d = _load(stem).dropna(subset=[col])
        superplot(ax, d, col, genotypes=GENOTYPE_ORDER)
        pm, res = compare(d, col)
        p_bracket(ax, res["p"])
        ax.set_ylabel(label)

    # the positive control, visibly set apart
    ax = axes[5]
    stem, col, label = ACROSS
    d = _load(stem).dropna(subset=[col])
    adj, _ = gap_adjusted(d, col, "gap_days")
    superplot(ax, adj.assign(**{col: adj["adj"]}), col, genotypes=GENOTYPE_ORDER)
    pm = adj.groupby(["mouse", "genotype"], as_index=False).adj.median()
    res = exact_permutation(pm, "adj", group_a="NLGF", group_b="WT")
    p_bracket(ax, res["p"])
    ax.set_ylabel(label + "\n(gap-adjusted)")
    ax.set_facecolor("#FFF4E6")
    for spine in ax.spines.values():
        spine.set_edgecolor("#E8590C")
        spine.set_linewidth(1.4)
    ax.set_title("positive control: the same pipeline,\nthe same cells, across days",
                 fontsize=9, color="#E8590C", loc="left")

    fig.suptitle(
        "Within-session spatial coding is intact in NLGF; only its maintenance across "
        "days is not\n"
        "five within-session measures, all matched by construction, all null  ·  "
        "the orange panel shows the method is not simply insensitive",
        fontsize=12.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = PLOTS / "fig24_within_session.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    for stem, col, label in WITHIN + [ACROSS]:
        d = _load(stem).dropna(subset=[col])
        if col == ACROSS[1]:
            adj, _ = gap_adjusted(d, col, "gap_days")
            pm = adj.groupby(["mouse", "genotype"], as_index=False).adj.median()
            r = exact_permutation(pm, "adj", group_a="NLGF", group_b="WT")
        else:
            _, r = compare(d, col)
        print(f"  {label.replace(chr(10), ' '):40s} NLGF {r['mean_a']:8.3f}  "
              f"WT {r['mean_b']:8.3f}  diff {r['diff']:+.3f}  p = {r['p']:.3f}")


if __name__ == "__main__":
    main()
