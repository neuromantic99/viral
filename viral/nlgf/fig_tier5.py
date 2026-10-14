"""Tier 5 for the talk: offline synchronous events, and what is normal about them in NLGF.

During in-task immobility the population produces brief bursts of coordinated firing -
the cortical analogue of the events that carry hippocampal replay. Shipley et al. (2026),
in this same App-NL-G-F line, found reactivation RATE intact in hippocampus but replay
STRUCTURE degraded, so both have to be measured before the comparison means anything.

Panel A is the design itself, in one continuous window: the animal runs, stops, and the
ICA ensembles defined from the running frames are then looked for in the immobility that
follows. Cells are ordered by ensemble membership, and the ordering comes from the
running epoch, so any structure visible after the animal stops was not put there by the
sort. It replaces an earlier unsorted raster that could not show what an event was - at
~12% participation these events are a threshold crossing of the smoothed population
rate, not a visible burst.

Panel B is the group-level version of what panel A illustrates: how strongly the
running-defined ensembles recur during immobility, measured against a shuffled template.

Panel C is the prior question for panel D, and the one an earlier version of this
analysis got wrong. Are the events structured at all? Scored against a null that CIRCULARLY SHIFTS
each cell's trace - preserving its rate and autocorrelation exactly and destroying only
cross-cell timing - the answer is emphatically yes. Scored against the margin-preserving
shuffle used first, which fixes every event's size and every cell's rate, the same data
looked like noise: that null left the observed matrix nowhere to move. A measure with no
dynamic range cannot support a claim about genotype either way, which is why this panel
comes before the next one.

Panel D tests assembly structure by genotype. Both genotype tests are null.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.fig_ensemble_raster import ensemble_panels
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import (GENOTYPE_COLOURS, GENOTYPE_ORDER, p_bracket, set_style,
                              superplot)

DEMO = ("JB030", "2025-03-11", "WT")
ZOOM_SECONDS = 7.0


def main() -> None:
    set_style(talk=True)
    fig = plt.figure(figsize=(25, 19))
    outer = fig.add_gridspec(2, 1, hspace=0.24, height_ratios=[1.55, 1],
                             left=0.06, right=0.985, top=0.91, bottom=0.06)
    top = outer[0].subgridspec(3, 1, hspace=0.09, height_ratios=[1, 1, 3])
    bot = outer[1].subgridspec(1, 3, wspace=0.34)

    # A: the design - run, stop, and the running-defined ensembles looked for at rest
    ensemble_panels([fig.add_subplot(top[i]) for i in range(3)])

    # B: is there any structure to degrade?
    sce = pd.read_csv(RESULTS / "sce_structure.csv")
    sce = sce[sce.genotype.isin(GENOTYPE_ORDER)].dropna(subset=["cell_coparticipation"])
    ax = fig.add_subplot(bot[1])
    # A scatter of observed against null is the wrong plot here: every shuffled value is
    # ~0, so the points stack on the y-axis and the identity line runs off through empty
    # space - the strongest result on the figure ends up looking like a broken render.
    # The z distribution says the same thing directly.
    z = sce.cell_coparticipation_z
    ax.hist(z, bins=18, color="#495057", alpha=0.85)
    ax.axvline(0, color="#E8590C", lw=3.4)
    ax.text(0.02, 0.96, "shuffle", transform=ax.transAxes, va="top", fontsize=21,
            color="#E8590C")
    ax.text(0.97, 0.96,
            f"observed {sce.cell_coparticipation.median():.3f}\n"
            f"chance {sce.cell_coparticipation_null.median():.3f} "
            f"± {sce.cell_coparticipation_null_sd.median():.3f}\n"
            f"{(z > 2).mean():.0%} of sessions z > 2",
            transform=ax.transAxes, ha="right", va="top", fontsize=21)
    ax.set_xlabel("Co-participation vs shuffle (z)")
    ax.set_ylabel("Sessions")

    # C: structure, by genotype - on the EXCESS, not the z.
    #
    # The circular-shift null is extremely tight: chance co-participation is
    # 0.00005 +/- 0.0013 against an observed 0.140, so the z runs past 100. Dividing by
    # a session-specific SD that itself varies with event count and cell activity makes
    # the z a poor between-group quantity - the same reason reactivation.py compares
    # excess_r rather than mean_rz. observed - null is the same contrast without the
    # unstable denominator.
    ax = fig.add_subplot(bot[2])
    superplot(ax, sce, "cell_coparticipation_excess", genotypes=GENOTYPE_ORDER)
    _, res = compare(sce, "cell_coparticipation_excess")
    p_bracket(ax, res["p"], fontsize=26)
    ax.set_ylabel("Assembly structure\n(observed − shuffle)")
    print(f"  structure: NLGF {res['mean_a']:.3f} vs WT {res['mean_b']:.3f}, "
          f"p = {res['p']:.3f}")

    # D: reactivation strength, by genotype
    ax = fig.add_subplot(bot[0])
    rea = pd.read_csv(RESULTS / "reactivation_freeze_poolmatched.csv")
    rea = rea[rea.genotype.isin(GENOTYPE_ORDER)].dropna(subset=["excess_r"])
    superplot(ax, rea, "excess_r", genotypes=GENOTYPE_ORDER, legend=True)
    _, res2 = compare(rea, "excess_r")
    p_bracket(ax, res2["p"], fontsize=26)
    ax.set_ylabel("Reactivation strength\n(excess over shuffle)")
    print(f"  strength:  NLGF {res2['mean_a']:.3f} vs WT {res2['mean_b']:.3f}, "
          f"p = {res2['p']:.3f}")

    rate = pd.read_csv(RESULTS / "activity_sessions.csv")
    rate = rate[rate.genotype.isin(GENOTYPE_ORDER)].still_sce_rate_hz.median()
    fig.suptitle("Offline ensemble reactivation is structured, and unchanged in NLGF",
                 fontsize=34, weight="semibold", y=0.975)
    out = PLOTS / "fig36_tier5_reactivation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
