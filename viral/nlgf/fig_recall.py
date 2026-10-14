"""Figure 11: stability across the two-month gap, and why it cannot be answered.

The measures work and the geometry is strikingly preserved, but only 3 NLGF and 4 WT
mice have usable imaging on both sides of the gap. The detectable effect is six to
twenty times the observed one, so these are uninformative panels rather than negative
ones - and the plot says so, by drawing the detectable band behind the data.
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
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, set_style

METRICS = [
    ("pv_lag1", "Lap-to-lap PV correlation", "Within-session stability"),
    ("geom_at_20cm", "Geometry similarity at 20 cm", "Representational geometry"),
]


def figure() -> None:
    d = pd.read_csv(RESULTS / "recall_stability.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # Within-mouse trajectories
    for ax, (value, label, title) in zip(axes[:2], METRICS):
        piv = (d.groupby(["genotype", "mouse", "epoch"])[value].median()
               .unstack("epoch").dropna())
        for g in GENOTYPE_ORDER:
            s = piv.loc[g] if g in piv.index.get_level_values(0) else None
            if s is None or len(s) == 0:
                continue
            ax.plot([0, 1], [s["pre-gap"], s["post-gap"]],
                    color=GENOTYPE_COLOURS[g], alpha=0.6, lw=1.5, zorder=1)
            ax.scatter([0] * len(s), s["pre-gap"], s=62, color=GENOTYPE_COLOURS[g],
                       edgecolor="white", linewidth=1.2, zorder=3, label=g)
            ax.scatter([1] * len(s), s["post-gap"], s=62, color=GENOTYPE_COLOURS[g],
                       edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["before", "after\n(63–71 days)"])
        ax.set_xlim(-0.35, 1.35)
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=10.5)
    axes[0].legend(loc="upper right")

    # Observed change against what the design could resolve
    ax = axes[2]
    piv = d.groupby(["genotype", "mouse", "epoch"])[
        [m for m, _, _ in METRICS]].median()
    rows = []
    for value, label, _ in METRICS:
        p = piv[value].unstack("epoch").dropna()
        delta = (p["post-gap"] - p["pre-gap"]).rename(value).reset_index()
        pm, res = compare(delta, value)
        mde = minimum_detectable_difference(pm, value)
        rows.append((label, res["diff"], mde))

    ys = np.arange(len(rows))[::-1]
    for y, (label, diff, mde) in zip(ys, rows):
        scale = abs(mde) if mde else 1.0
        ax.barh(y, abs(mde) / scale, height=0.42, color="#DEE2E6",
                edgecolor="none", zorder=1)
        ax.barh(y, abs(diff) / scale, height=0.42,
                color=GENOTYPE_COLOURS["NLGF"], zorder=2)
        ax.text(abs(diff) / scale + 0.03, y,
                f"{abs(mde/diff):.0f}× too small to resolve",
                va="center", fontsize=9, color="#495057")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 1.55)
    ax.set_xlabel("Observed genotype difference in change\n(grey = detectable at this n)")
    ax.set_title("Underpowered, not negative", fontsize=10.5)

    fig.suptitle(
        "Long-term stability: the measures work, the sample does not  ·  "
        "3 NLGF / 4 WT mice imaged on both sides of the gap",
        y=1.04, fontsize=12.5, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig11_recall_stability.png")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    figure()
    print(f"wrote figure to {PLOTS}")
