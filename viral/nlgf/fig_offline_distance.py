"""Resting correlations as a function of place field distance (Grosmark fig. 4 e/f).

The claim this figure supports is narrow and, for once, positive: what the RSC
population does during immobility is not spatially blind. Cells whose fields sit close
together on the track are more correlated at rest than cells whose fields sit far apart,
in both genotypes, in both freeze epochs.

It is worth separating from the reactivation strength measure because it is immune to
the thing that made that measure so hard to interpret. excess_r tracks how many cells
were recorded, which is why reactivation.py has to match the pool. A distance-dependent
SLOPE cannot be manufactured that way: adding cells, or raising every cell's coupling,
lifts the whole curve and leaves the near-minus-far difference where it was. The
genotype contrast here is therefore on a quantity that needed no matching to be fair.

Panels A and B are the curves themselves, one genotype each, pre and post overlaid. The
grey band is the null that matters: the measured correlation matrix with the field PEAKS
permuted across cells, so every correlation and the whole distribution of them survive
and only the pairing with distance is destroyed. A curve that sits above its own band at
short distances is the result.

Panel C puts the slope into the genotype test. Panel D asks the prior question one
session at a time - how far above the permuted null each session's slope falls - because
a group mean built from sessions that individually show nothing would mean something
different from one built from sessions that individually show the effect.

Oligo-BACE1-KO is excluded: it is an NLGF-background rescue arm and there is no deficit
here to rescue.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.fig_freeze_prepost import sign_flip_p
from viral.nlgf.offline_distance import FAR_CM, NEAR_CM, curve_of, distance_bins
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import (GENOTYPE_COLOURS, GENOTYPE_ORDER, p_bracket, set_style,
                              superplot)

MAX_CM = 110.0     # bins beyond this hold few pairs: fields live in 20-160 cm
EPOCH_STYLE = {"pre": dict(ls="--", alpha=0.75), "post": dict(ls="-", alpha=1.0)}


def _mouse_mean(sub: pd.DataFrame, column: str) -> np.ndarray:
    """Curves averaged within mouse first, then across mice.

    Sessions within an animal are not independent and the animals contribute between 1
    and 8 sessions each, so a straight session mean would weight the figure towards
    whichever mouse was recorded most - and that mouse is not the same one in each
    genotype.
    """
    per_mouse = [
        np.nanmean([curve_of(r, column) for _, r in g.iterrows()], axis=0)
        for _, g in sub.groupby("mouse")
    ]
    return np.array(per_mouse)


def curve_panel(ax, d: pd.DataFrame, genotype: str,
                ylabel: bool = True) -> None:
    """Correlation in EXCESS of the session's own peak-permuted curve.

    Plotting the raw correlation instead needs one null band per epoch, because the
    permuted curve is flat at that epoch's grand mean and the two epochs do not share a
    mean - so the two bands would sit at different heights and a curve crossing its own
    null would read as the curve falling below the other epoch's. Subtracting each
    session's own null puts every curve on one axis whose zero means the same thing, and
    it subtracts exactly the quantity that is NOT the claim here: the overall level of
    coupling. The levels are printed in the panel instead.
    """
    x = distance_bins() + 10.0   # bins are 20 cm wide; plot at the centre
    keep = x <= MAX_CM
    sub_g = d[d.genotype == genotype]

    for epoch, style in EPOCH_STYLE.items():
        sub = sub_g[sub_g.epoch == epoch]
        if not len(sub):
            continue
        curves = _mouse_mean(sub, "curve") - _mouse_mean(sub, "perm_curve")
        m = np.nanmean(curves, axis=0)[keep]
        sem = np.nanstd(curves, axis=0)[keep] / np.sqrt(max(curves.shape[0], 1))
        ax.plot(x[keep], m, color=GENOTYPE_COLOURS[genotype], lw=4.2,
                label=f"{epoch} freeze", zorder=4, **style)
        ax.fill_between(x[keep], m - sem, m + sem, color=GENOTYPE_COLOURS[genotype],
                        alpha=0.18, lw=0, zorder=3)

    ax.axhline(0, color="#495057", lw=2.4, zorder=2)
    ax.axvspan(x[0], NEAR_CM + 10, color=GENOTYPE_COLOURS[genotype], alpha=0.07, lw=0,
               zorder=0)
    ax.axvspan(FAR_CM + 10, MAX_CM, color="#495057", alpha=0.05, lw=0, zorder=0)
    ax.set_xlim(x[keep][0], MAX_CM)
    ax.set_xlabel("Distance between place field peaks (cm)")
    if ylabel:
        ax.set_ylabel("Resting correlation\nabove permuted peaks")
    ax.legend(frameon=False, fontsize=20, loc="upper right")
    n_mice = sub_g.mouse.nunique()
    ax.set_title(f"{genotype}  ·  {n_mice} mice, "
                 f"{sub_g.groupby(['mouse','date']).ngroups} sessions",
                 fontsize=24, color=GENOTYPE_COLOURS[genotype])
    ax.text(0.015, 0.04, f"mean resting correlation {sub_g.far.mean():.3f}",
            transform=ax.transAxes, fontsize=18, color="#868E96")


def slope_panel(ax, d: pd.DataFrame) -> None:
    """Near minus far, one number per mouse, both freeze epochs pooled.

    Pooled rather than post-only because the epochs are shown separately in A and B and
    agree there, and because the test that matters most on this panel - does the slope
    exist at all - is not a genotype test and has no reason to spend half the data. The
    genotype contrast is still run on the same per-mouse numbers.

    The existence claim is made across ALL TEN mice together. That is legitimate here
    and would not be if the claim were about genotype: an exact sign-flip over 6 WT
    cannot return a p below 0.031 and over 4 NLGF cannot go below 0.125, so testing the
    genotypes separately asks two underpowered questions instead of one answerable one.
    """
    superplot(ax, d, "near_minus_far", genotypes=GENOTYPE_ORDER, legend=True)
    ax.axhline(0, color="#495057", lw=1.6, ls="--", zorder=1)
    per_mouse, res = compare(d, "near_minus_far")
    p_bracket(ax, res["p"], fontsize=26)
    v = per_mouse.near_minus_far.to_numpy()
    ax.text(0.5, 0.94, f"all {v.size} mice:  {int((v > 0).sum())}/{v.size} positive,  "
                       f"p = {sign_flip_p(v):.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=19,
            color="#212529")
    ax.set_ylabel(f"Near − far correlation\n(<{NEAR_CM:.0f} cm vs >{FAR_CM:.0f} cm)")
    print(f"  slope: NLGF {res['mean_a']:+.4f} vs WT {res['mean_b']:+.4f}, "
          f"genotype p = {res['p']:.3f}")
    print(f"  slope > 0 across all {v.size} mice: {v.mean():+.4f}, "
          f"{int((v > 0).sum())}/{v.size} positive, sign-flip p = {sign_flip_p(v):.4f}")


def z_panel(ax, d: pd.DataFrame) -> None:
    z = d.slope_z.dropna()
    bins = np.histogram_bin_edges(z, bins=16)
    for g in GENOTYPE_ORDER:
        ax.hist(d[d.genotype == g].slope_z.dropna(), bins=bins, alpha=0.7,
                color=GENOTYPE_COLOURS[g], label=g)
    ax.axvline(0, color="#495057", lw=2.2)
    ax.axvline(2, color="#212529", lw=2.6, ls="--")
    ax.text(2, 0.62, " z = 2", transform=ax.get_xaxis_transform(), color="#212529",
            fontsize=19, va="top")
    ax.set_xlabel("Session slope vs permuted peaks (z)")
    ax.set_ylabel("Session-epochs")
    ax.legend(frameon=False, fontsize=20, loc="upper right")
    ax.text(0.02, 0.86, f"{(z > 2).mean():.0%} of session-epochs z > 2\n"
                        f"median z = {z.median():.1f}",
            transform=ax.transAxes, va="top", fontsize=19, color="#495057")
    print(f"  slope_z: median {z.median():.2f}, {(z > 2).mean():.0%} above 2")


def main() -> None:
    set_style(talk=True)
    d = pd.read_csv(RESULTS / "offline_distance.csv")
    d = d[d.genotype.isin(GENOTYPE_ORDER)]
    fig, axes = plt.subplots(1, 4, figsize=(33, 8.4))
    fig.subplots_adjust(left=0.048, right=0.99, top=0.80, bottom=0.15, wspace=0.31)
    curve_panel(axes[0], d, "WT")
    curve_panel(axes[1], d, "NLGF", ylabel=False)
    lo = min(ax.get_ylim()[0] for ax in axes[:2])
    hi = max(ax.get_ylim()[1] for ax in axes[:2])
    for ax in axes[:2]:
        ax.set_ylim(lo, hi)
    slope_panel(axes[2], d)
    z_panel(axes[3], d)
    for ax, letter in zip(axes, "ABCD"):
        ax.text(-0.17, 1.13, letter, transform=ax.transAxes, fontsize=36,
                weight="bold", va="top")
    fig.suptitle("Resting correlations follow the map: cells with nearby fields stay "
                 "coupled offline, in both genotypes",
                 fontsize=33, weight="semibold", y=0.975)
    out = PLOTS / "fig44_offline_distance.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
