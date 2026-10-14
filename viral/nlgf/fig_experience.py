"""Figure 16: matched decoding accuracy against accumulated experience.

Three views of the same data, because pooling naively gives the wrong answer here.

Pooling raw sessions mixes two relationships that point in OPPOSITE directions:

    within mouse    decoding improves with experience   (WT slope -3.17)
    between mice    animals sampled at higher average experience decode WORSE
                    (WT +2.11, NLGF +4.29, p = 0.045)

so the pooled slope is a weighted average of the two and largely cancels (WT -1.62,
NLGF +0.23). That is Simpson's paradox, and it means a naive pooled scatter would hide
a real within-animal effect.

Centring x and y on each mouse's own mean removes the between-mouse component entirely,
leaving only the within-animal question - which is the one "does experience refine the
map" actually asks.

The significance quoted is still the per-mouse slope test (sign-flip over animals), not
the within-centred regression: centring removes the confound but sessions within a
mouse are still not independent, so the centred p is anticonservative.

Accuracy is matched at 24 trials and 100 cells, so any improvement is coding refinement
rather than accumulating data.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

import viral.dec as dec
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, set_style

AXIS = "cum_trials"
MIN_SESSIONS = 3


def _signflip(v: np.ndarray):
    v = np.asarray(v, float)
    null = np.array([np.mean(v * np.array(s)) for s in product([-1, 1], repeat=v.size)])
    return v.mean(), (np.sum(np.abs(null) >= abs(v.mean())) + 1) / (null.size + 1)


def figure() -> None:
    d = pd.read_csv(RESULTS / "experience.csv")
    p = dec.prepare(d.assign(cum_trials=d[AXIS]), logit_y=False)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # A: raw, with per-mouse lines so the nesting is visible
    ax = axes[0]
    for g in GENOTYPE_ORDER:
        s = p[p.genotype == g]
        ax.scatter(s.cum_trials, s.y, s=22, color=GENOTYPE_COLOURS[g], alpha=0.45,
                   edgecolor="none", label=g)
        for _, gg in s.groupby("mouse"):
            if len(gg) >= 2:
                o = gg.sort_values("cum_trials")
                ax.plot(o.cum_trials, o.y, color=GENOTYPE_COLOURS[g], alpha=0.35, lw=1)
    ax.set_xlabel("Cumulative trials (all task sessions)")
    ax.set_ylabel("Matched decoding error (cm)")
    ax.set_title("Raw, lines join sessions of a mouse", fontsize=10.5)
    ax.legend(fontsize=9)

    # B: within-mouse centred, which is the question
    ax = axes[1]
    for g in GENOTYPE_ORDER:
        s = p[p.genotype == g].copy()
        s["x_c"] = s.exper - s.groupby("mouse").exper.transform("mean")
        s["y_c"] = s.y - s.groupby("mouse").y.transform("mean")
        ax.scatter(s.x_c, s.y_c, s=26, color=GENOTYPE_COLOURS[g], alpha=0.6,
                   edgecolor="none", label=g)
        fit = linregress(s.x_c, s.y_c)
        xs = np.linspace(s.x_c.min(), s.x_c.max(), 20)
        ax.plot(xs, fit.intercept + fit.slope * xs, color=GENOTYPE_COLOURS[g], lw=2)
    ax.axhline(0, color="#CED4DA", lw=0.9, zorder=0)
    ax.axvline(0, color="#CED4DA", lw=0.9, zorder=0)
    ax.set_xlabel("Experience, centred within mouse (SD)")
    ax.set_ylabel("Decoding error, centred within mouse (cm)")
    ax.set_title("Within-mouse: the effect the raw pool hides", fontsize=10.5)
    ax.legend(fontsize=9)

    # C: the per-mouse slopes, which is what the test uses
    ax = axes[2]
    rng = np.random.default_rng(0)
    for i, g in enumerate(GENOTYPE_ORDER):
        slopes = np.array([np.polyfit(gg.exper, gg.y, 1)[0]
                           for _, gg in p[p.genotype == g].groupby("mouse")
                           if len(gg) >= MIN_SESSIONS])
        m, pv = _signflip(slopes)
        c = GENOTYPE_COLOURS[g]
        ax.scatter(i + rng.uniform(-0.09, 0.09, slopes.size), slopes, s=62, color=c,
                   alpha=0.9, edgecolor="white", linewidth=1.2, zorder=3)
        ax.hlines(m, i - 0.25, i + 0.25, color=c, lw=2.6, zorder=4)
        ax.text(i, ax.get_ylim()[1], f"p = {pv:.3f}", ha="center", va="bottom",
                fontsize=9, color=c)
    ax.axhline(0, color="#495057", lw=1.1, zorder=2)
    ax.set_xticks(range(len(GENOTYPE_ORDER)))
    ax.set_xticklabels(GENOTYPE_ORDER)
    ax.set_xlim(-0.6, len(GENOTYPE_ORDER) - 0.4)
    ax.set_ylabel("Slope per mouse (cm per SD)")
    ax.set_title("One slope per mouse\n(negative = improving)", fontsize=10.5)

    fig.suptitle(
        "Decoding improves with experience within animals, but the raw pooled slope "
        "cancels it against an opposite between-animal trend",
        y=1.03, fontsize=12.5, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig16_experience.png")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    figure()
    print(f"wrote figure to {PLOTS}")
