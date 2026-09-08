"""One panel, ten seconds: within-session spatial coding is normal, maintenance is not.

Every measure is put on a single axis as a standardised effect size (Hedges' g, NLGF
minus WT, computed over mouse means), so quantities in centimetres, bits and fractions
can sit together. Each is SIGNED so that negative always means NLGF impaired - decoding
error and centroid shift are inverted, since a smaller value is better on those. Without
that, two measures pointing opposite ways would mean the same biology and the panel
would be unreadable at a glance.

The reading is meant to be immediate: everything measured within a session sits on the
zero line, and the one measure that spans days does not. The within-session rows are
grey and the across-day row is coloured, so the contrast survives being seen from the
back of a lecture theatre.

The bars are 95% confidence intervals on g, which assume normality and are there to show
precision. The p-values printed on the right are the exact permutation tests over mice
used everywhere else in this package - those are the inference, the bars are the
illustration.
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

# (table, column, label, higher_is_better). The flag orients the axis so that negative
# is always "NLGF worse", whatever the raw units do.
WITHIN = [
    ("place_lap_matched", "frac_place_cells", "Place cells", True),
    ("place_lap_matched", "field_width_cm", "Field width", True),
    ("place_lap_matched", "spatial_info_pc", "Spatial information", True),
    ("spatial_stability", "field_corr_within", "Within-session reproducibility", True),
    ("decoding_matched", "decoding_error_matched", "Position decoding", False),
]
ACROSS = ("spatial_stability", "field_corr_cross", "Cross-day field correlation", True)

GREY, HOT = "#495057", "#E8590C"


def _hedges_g(a: np.ndarray, b: np.ndarray) -> tuple:
    na, nb = a.size, b.size
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan, np.nan, np.nan
    d = (a.mean() - b.mean()) / sp
    g = d * (1 - 3 / (4 * (na + nb) - 9))          # small-sample correction
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb - 2)))
    return g, g - 1.96 * se, g + 1.96 * se


def _row(stem: str, col: str, higher_better: bool, adjust_gap: bool = False) -> dict:
    df = pd.read_csv(RESULTS / f"{stem}.csv")
    df = df[df.genotype.isin(GENOTYPE_ORDER)].dropna(subset=[col])
    if adjust_gap:
        df, _ = gap_adjusted(df, col, "gap_days")
        col = "adj"
    pm = df.groupby(["mouse", "genotype"], as_index=False)[col].median()
    res = exact_permutation(pm, col, group_a="NLGF", group_b="WT")
    a = pm.loc[pm.genotype == "NLGF", col].to_numpy()
    b = pm.loc[pm.genotype == "WT", col].to_numpy()
    g, lo, hi = _hedges_g(a, b)
    # inverting the sign reverses the interval, so the bounds are reordered after the
    # flip - otherwise the "lower" bound plots to the right of the "upper" one
    flip = 1.0 if higher_better else -1.0
    lo_f, hi_f = sorted((lo * flip, hi * flip))
    return dict(g=g * flip, lo=lo_f, hi=hi_f, p=res["p"],
                n_a=res["n_a"], n_b=res["n_b"])


def main() -> None:
    set_style()
    rows = [dict(_row(s, c, hb), label=lab, kind="within")
            for s, c, lab, hb in WITHIN]
    s, c, lab, hb = ACROSS
    rows.append(dict(_row(s, c, hb, adjust_gap=True), label=lab, kind="across"))
    d = pd.DataFrame(rows)[::-1].reset_index(drop=True)   # across-day at the top

    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    ax.axvline(0, color="#212529", lw=1.4, zorder=2)

    n_across = int((d.kind == "across").sum())
    for i, r in d.iterrows():
        colour = HOT if r.kind == "across" else GREY
        ax.plot([r.lo, r.hi], [i, i], color=colour, lw=2.8, solid_capstyle="round",
                alpha=0.9, zorder=3)
        ax.scatter(r.g, i, s=165 if r.kind == "across" else 100, color=colour,
                   edgecolor="white", linewidth=1.6, zorder=4)
        ax.text(2.68, i, f"p = {r.p:.2f}", va="center", ha="left",
                fontsize=11, color=colour,
                fontweight="bold" if r.p < 0.05 else "normal")

    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.label, fontsize=11.5)
    for tick, kind in zip(ax.get_yticklabels(), d.kind):
        tick.set_color(HOT if kind == "across" else "#212529")
        if kind == "across":
            tick.set_fontweight("bold")

    # the across-day rows sit at the bottom (index 0 upward), so the divider goes just
    # above them and each group label sits over its own block
    ax.axhline(n_across - 0.5, color="#CED4DA", lw=1.2, ls="--", zorder=1)
    ax.text(-3.45, len(d) - 0.62, "WITHIN SESSION", fontsize=10, color=GREY,
            fontweight="bold", va="center", ha="left")
    ax.text(-3.45, n_across - 0.85, "ACROSS DAYS", fontsize=10, color=HOT,
            fontweight="bold", va="center", ha="left")

    ax.set_xlim(-3.5, 3.35)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_ylim(-0.55, len(d) - 0.3)
    ax.set_xlabel("NLGF vs WT   (Hedges' g,   negative = NLGF impaired)", fontsize=11.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.suptitle("The map RSC builds is normal. Keeping it is not.",
                 fontsize=16, fontweight="bold", y=1.06)
    fig.text(0.5, 0.985, f"{d.n_a.iloc[0]} NLGF vs {d.n_b.iloc[0]} WT mice   ·   "
             "matched by construction on laps, cells and session gap",
             ha="center", fontsize=10, color="#868E96")
    fig.tight_layout()
    out = PLOTS / "fig25_forest.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"wrote {out}\n")
    print(d[["label", "kind", "g", "lo", "hi", "p"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
