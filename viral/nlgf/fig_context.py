"""Figure 9: contextual updating across the reversal.

The headline is not the genotype comparison, which is null. It is that reversal does
not disturb the representation in EITHER genotype: accuracy on the first reversal
sessions sits at each mouse's own learning baseline. The textures are physically
unchanged and only their meaning flips, so a representation indifferent to the flip is
coding texture identity, not reward value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.context import CELL_CURVE
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, set_style, superplot


def figure() -> None:
    d = pd.read_csv(RESULTS / "context_decoding.csv")
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.3))

    # Neurometric curve
    ax = axes[0]
    for g in GENOTYPE_ORDER:
        sub = d[d.genotype == g]
        per_mouse = sub.groupby("mouse")[[f"curve_{n}" for n in CELL_CURVE]].median()
        m = per_mouse.mean()
        sem = per_mouse.std(ddof=1) / np.sqrt(len(per_mouse))
        ax.plot(CELL_CURVE, m, color=GENOTYPE_COLOURS[g], lw=2, marker="o", ms=5, label=g)
        ax.fill_between(CELL_CURVE, m - sem, m + sem,
                        color=GENOTYPE_COLOURS[g], alpha=0.18, lw=0)
    ax.axhline(0.5, color="#CED4DA", lw=0.9, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(CELL_CURVE)
    ax.set_xticklabels(CELL_CURVE)
    ax.set_xlabel("Cells in decoder")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Texture decoding saturates\nby ~80 cells in both", fontsize=10.5)
    ax.legend(loc="lower right")

    # Level at the operating point
    ax = axes[1]
    per_mouse, res = compare(d, "acc_early")
    superplot(ax, d, "acc_early", legend=True)
    ax.axhline(0.5, color="#CED4DA", lw=0.9, ls="--", zorder=0)
    ax.set_ylabel("Balanced accuracy, 20 cells")
    ax.set_title(f"Early corridor (20–100 cm)\ndiff {res['diff']:+.3f}, "
                 f"p = {res['p']:.3f}", fontsize=10.5)

    # Per-mouse learning -> reversal
    ax = axes[2]
    pm = (d.groupby(["genotype", "mouse", "stage"])["acc_early"].median()
          .unstack("stage").dropna().reset_index())
    for g in GENOTYPE_ORDER:
        s = pm[pm.genotype == g]
        ax.plot([0, 1], [s.learning, s.reversal], color=GENOTYPE_COLOURS[g],
                alpha=0.5, lw=1.3, zorder=1)
        ax.scatter([0] * len(s), s.learning, s=55, color=GENOTYPE_COLOURS[g],
                   edgecolor="white", linewidth=1.1, zorder=3, label=g)
        ax.scatter([1] * len(s), s.reversal, s=55, color=GENOTYPE_COLOURS[g],
                   edgecolor="white", linewidth=1.1, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["learning", "reversal"])
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylabel("Balanced accuracy, 20 cells")
    pms = pm.copy()
    pms["shift"] = pms.reversal - pms.learning
    r = compare(pms, "shift")[1]
    ax.set_title(f"Reversal does not disturb it\nshift diff {r['diff']:+.3f}, "
                 f"p = {r['p']:.3f}", fontsize=10.5)

    # Early reversal against each mouse's own learning baseline
    ax = axes[3]
    base = d[d.stage == "learning"].groupby("mouse")["acc_early"].median()
    rev = d[(d.stage == "reversal") & (d.day <= 2)].copy()
    rev["delta"] = rev.acc_early - rev.mouse.map(base)
    rev = rev.dropna(subset=["delta"])
    superplot(ax, rev, "delta")
    ax.axhline(0, color="#495057", lw=1.1, zorder=2)
    ax.set_ylabel("Reversal d1–2 − own learning baseline")
    ax.set_title("Sitting on baseline, both\ngenotypes", fontsize=10.5)

    n_n = d[d.genotype == "NLGF"].mouse.nunique()
    n_w = d[d.genotype == "WT"].mouse.nunique()
    fig.suptitle(
        f"RSC codes texture identity, not reward value  ·  {len(d)} sessions, "
        f"{n_n} NLGF / {n_w} WT mice  ·  learning and reversal only",
        y=1.04, fontsize=12.5, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig9_context.png")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    figure()
    print(f"wrote figure to {PLOTS}")
