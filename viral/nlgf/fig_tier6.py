"""The finding: RSC builds a normal map under amyloid but does not hold onto it.

Panel A is the phenomenon in one animal. The same registered cells are shown on two
days: ordered by where they peak on day 1, then the SAME ordering applied to day 2. A
map that came back intact would show the same diagonal twice. The second panel is
visibly looser - individual fields have moved.

Panels B and C are the argument, and they have to be read together. Fields are less
correlated across days in NLGF (B), but within a single session they are just as
reproducible (C). That pairing is what rules out the simple explanation: if NLGF maps
were merely noisier, the within-session panel would show it too. It does not, so what
differs is maintenance, not construction.

Panel D is the part that constrains the mechanism. The POPULATION vector correlation is
unaffected. Individual cells lose their fields while the geometry of the code as a whole
survives, so this is not the map degrading - it is the cells carrying it turning over.

Cross-day quantities are gap-adjusted: field correlation falls about 0.06 per day and
NLGF pairs average a slightly longer interval, so the raw contrast is part genotype and
part calendar. Within-session reproducibility needs no such adjustment - a split-half
inside one session cannot depend on the gap to the next.
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
from viral.nlgf.spatial_stability import _lap_maps, _smooth
from viral.nlgf.stats import compare, exact_permutation
from viral.nlgf.style import (GENOTYPE_ORDER, LANDMARKS_CM, p_bracket, set_style,
                              superplot)

DEMO = ("JB026", "2024-12-15", "2024-12-16")

# Candidates for the demo, from the best-registered short-gap pairs in each genotype.
# Printed with their own cross-day and within-session correlations so a chosen example
# can be described honestly rather than implied to be typical.
CANDIDATES = [
    ("JB030", "2025-03-13", "2025-03-14", "WT"),
    ("JB030", "2025-03-10", "2025-03-11", "WT"),
    ("JB026", "2024-12-15", "2024-12-16", "WT"),
    ("JB033", "2025-03-18", "2025-03-19", "WT"),
    ("J035", "2026-06-11", "2026-06-12", "WT"),
]
# JB031 2025-04-02 -> 2025-04-03 is deliberately absent: 25 registered cells and both
# correlations come back NaN. It survives into spatial_stability.csv but is a failed
# registration, not a candidate.


def _normalise(m: np.ndarray) -> np.ndarray:
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    return (m - lo) / np.where(hi - lo > 0, hi - lo, 1.0)


def demo_panels(ax_a, ax_b) -> None:
    mouse, d1, d2 = DEMO
    matches = pd.read_csv(RESULTS / f"matches_{mouse}_{d1}_{d2}.csv")
    m1 = _smooth(_lap_maps(mouse, d1, matches.row_a.to_numpy()))
    m2 = _smooth(_lap_maps(mouse, d2, matches.row_b.to_numpy()))
    keep = np.isfinite(m1).all(axis=1) & np.isfinite(m2).all(axis=1)
    m1, m2 = m1[keep], m2[keep]
    order = np.argsort(np.argmax(m1, axis=1))       # ordered on day 1 only
    x = np.linspace(20, 160, m1.shape[1])
    for ax, m, label in ((ax_a, m1, "Day 1"), (ax_b, m2, "Day 2")):
        ax.imshow(_normalise(m)[order], aspect="auto", cmap="magma", vmin=0, vmax=1,
                  extent=[x[0], x[-1], m.shape[0], 0], interpolation="nearest")
        for lm in LANDMARKS_CM:
            ax.axvline(lm, color="#8ED1FC", lw=1.4, ls="--", alpha=0.8)
        ax.set_xlabel("Position (cm)")
        ax.set_title(label, fontsize=26)
    ax_a.set_ylabel("Registered cell\n(ordered by day 1)")
    ax_b.set_yticklabels([])
    print(f"  demo: {mouse} {d1} -> {d2}, {int(keep.sum())} registered cells")


def candidates() -> None:
    """Every candidate pair drawn the way panel A draws it, for picking an example."""
    set_style()
    d = pd.read_csv(RESULTS / "spatial_stability.csv")
    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    for i, (mouse, d1, d2, g) in enumerate(CANDIDATES):
        row, col = divmod(i, 2)
        ax_a, ax_b = axes[row, col * 2], axes[row, col * 2 + 1]
        global DEMO
        DEMO = (mouse, d1, d2)
        demo_panels(ax_a, ax_b)
        r = d[(d.mouse == mouse) & (d.date_a == d1) & (d.date_b == d2)]
        cross = float(r.field_corr_cross.iloc[0]) if len(r) else float("nan")
        within = float(r.field_corr_within.iloc[0]) if len(r) else float("nan")
        ax_a.set_title(f"{g}  {mouse}  day 1", fontsize=14)
        ax_b.set_title(f"day 2  ·  cross {cross:.2f} / within {within:.2f}",
                       fontsize=14)
        ax_a.set_ylabel("Cell (ordered by day 1)" if col == 0 else "")
    fig.suptitle("Candidate pairs for panel A  ·  drawn exactly as the talk figure "
                 "draws them", fontsize=19, weight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = PLOTS / "fig40_stability_candidates.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    set_style(talk=True)
    d = pd.read_csv(RESULTS / "spatial_stability.csv")
    d = d[d.genotype.isin(GENOTYPE_ORDER)]

    fig = plt.figure(figsize=(26, 13.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.6], wspace=0.20,
                             left=0.05, right=0.985, top=0.84, bottom=0.10)
    left = outer[0].subgridspec(1, 2, wspace=0.06)
    right = outer[1].subgridspec(1, 3, wspace=0.42)
    demo_panels(fig.add_subplot(left[0]), fig.add_subplot(left[1]))

    adj, _ = gap_adjusted(d, "field_corr_cross", "gap_days")
    # Ordered so the finding lands last: the map is built normally (within-session),
    # its population geometry survives the night (PV), and only then the thing that
    # does not - the fields themselves.
    panels = [
        ("within", "field_corr_within", "Within-session field\nreproducibility"),
        ("across", "pv_corr_cross", "Population vector\ncorrelation, cross-day"),
        ("across", "field_corr_cross", "Cross-day field\ncorrelation"),
    ]
    for i, (kind, col, label) in enumerate(panels):
        ax = fig.add_subplot(right[i])
        if kind == "across":
            frame, _ = gap_adjusted(d, col, "gap_days")
            superplot(ax, frame.assign(**{col: frame["adj"]}), col,
                      genotypes=GENOTYPE_ORDER, legend=(i == 0))
            pm = frame.groupby(["mouse", "genotype"], as_index=False).adj.median()
            res = exact_permutation(pm, "adj")
        else:
            superplot(ax, d.dropna(subset=[col]), col, genotypes=GENOTYPE_ORDER)
            _, res = compare(d.dropna(subset=[col]), col)
        p_bracket(ax, res["p"], fontsize=26)
        ax.set_ylabel(label)
        print(f"  {label.replace(chr(10), ' '):36s} NLGF {res['mean_a']:.3f}  "
              f"WT {res['mean_b']:.3f}  p = {res['p']:.3f}")

    fig.suptitle("NLGF mice build a normal map, but the cells carrying it turn over",
                 fontsize=34, weight="semibold", y=0.94)
    out = PLOTS / "fig39_tier6_stability.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    if "--candidates" in sys.argv:
        candidates()
    else:
        main()
