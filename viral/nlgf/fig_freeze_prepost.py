"""Pre versus post freeze: does a session of running change what recurs afterwards?

The wheel freeze brackets the task with two matched epochs of enforced immobility, so
the same animal provides a before and an after under identical conditions. If running
the corridor drove consolidation of the run ensembles, reactivation should be stronger
in the post epoch than the pre epoch, and the DIFFERENCE is within-session - it needs no
matching for cell count, epoch length or field of view, because both epochs come from
one recording of one population.

That within-session design is the reason this comparison is worth showing even though
it is null. The genotype contrast on the LEVEL of reactivation needed three attempts to
control (see fig_reactivation), because excess_r tracks how many cells were recorded.
The pre-to-post delta cancels all of that by construction.

The result is that the delta is flat, in both genotypes, on both epochs' terms. Two
readings are available and the figure does not adjudicate between them:

  the track is familiar    these are not first exposures. Grosmark's post-over-pre
                           increase is a learning effect, and by the sessions analysed
                           here the map already exists, so there may be nothing left to
                           consolidate. The one first-exposure session in the set cannot
                           settle this.
  RSC is not hippocampus   the cortical end of the loop may hold its ensembles at a
                           steady level and be read out by hippocampus rather than
                           re-trained by it.

Panel C is the prior question, and the one that keeps the null honest: reactivation is
present in BOTH epochs, above the shuffled template, so a flat delta is two real
measurements that agree and not two failures to measure anything.

Genotype is tested on the delta by exact permutation over mice; the level in each epoch
is tested against zero by exact sign-flip over mice. Both are at the animal level.
Oligo-BACE1-KO is excluded throughout - it is an NLGF-background rescue arm, and there
is no pre/post deficit here for it to rescue.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import collapse_to_mice, compare
from viral.nlgf.style import (GENOTYPE_COLOURS, GENOTYPE_ORDER, p_bracket, set_style,
                              superplot)

VALUE = "excess_r"
EPOCHS = ["pre", "post"]


def sign_flip_p(v: np.ndarray) -> float:
    """Exact sign-flip test over animals, as elsewhere in this package."""
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    stats = [np.mean(v * np.array(s)) for s in product([1, -1], repeat=len(v))]
    return float(np.mean(np.abs(stats) >= abs(v.mean()) - 1e-12))


def load() -> pd.DataFrame:
    d = pd.read_csv(RESULTS / "reactivation_freeze_poolmatched.csv")
    d = d[d.genotype.isin(GENOTYPE_ORDER)].dropna(subset=[VALUE])
    # Only sessions that produced BOTH epochs can contribute a delta, and the level
    # panel has to be read on the same sessions as the delta panel or the two disagree
    # for reasons that have nothing to do with the epochs
    both = (d.groupby(["mouse", "date"])["epoch"].nunique() == 2)
    keep = both[both].index
    return d.set_index(["mouse", "date"]).loc[keep].reset_index()


def paired_panel(ax, d: pd.DataFrame) -> None:
    wide = d.pivot_table(index=["genotype", "mouse", "date"], columns="epoch",
                         values=VALUE).reset_index()
    for i, g in enumerate(GENOTYPE_ORDER):
        sub = wide[wide.genotype == g]
        xs = np.array([0, 1]) + i * 2.4
        for r in sub.itertuples():
            ax.plot(xs, [r.pre, r.post], color=GENOTYPE_COLOURS[g], alpha=0.3,
                    lw=2.0, zorder=2)
            ax.scatter(xs, [r.pre, r.post], s=38, color=GENOTYPE_COLOURS[g],
                       alpha=0.45, zorder=3, edgecolors="none")
        per_mouse = sub.groupby("mouse")[EPOCHS].median()
        for r in per_mouse.itertuples():
            ax.plot(xs, [r.pre, r.post], color=GENOTYPE_COLOURS[g], lw=4.6, zorder=5,
                    solid_capstyle="round")
            ax.scatter(xs, [r.pre, r.post], s=210, color=GENOTYPE_COLOURS[g],
                       zorder=6, edgecolors="white", linewidths=2.0)
        ax.text(np.mean(xs), 1.02, f"{g}\n{len(per_mouse)} mice, {len(sub)} sessions",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=20, color=GENOTYPE_COLOURS[g])
    ax.axhline(0, color="#CED4DA", lw=1.4, ls="--", zorder=0)
    ax.set_xticks([0, 1, 2.4, 3.4])
    ax.set_xticklabels(["pre", "post"] * 2)
    ax.set_xlim(-0.55, 3.95)
    ax.set_ylabel("Reactivation strength\n(excess over shuffle)")
    ax.set_xlabel("Freeze epoch")


def delta_panel(ax, d: pd.DataFrame) -> None:
    wide = d.pivot_table(index=["genotype", "mouse", "date"], columns="epoch",
                         values=VALUE).reset_index()
    wide["delta"] = wide.post - wide.pre
    superplot(ax, wide, "delta", genotypes=GENOTYPE_ORDER, legend=True)
    ax.axhline(0, color="#495057", lw=1.6, ls="--", zorder=1)
    per_mouse, res = compare(wide, "delta")
    for i, g in enumerate(GENOTYPE_ORDER):
        v = per_mouse[per_mouse.genotype == g].delta.to_numpy()
        ax.text(i, 0.86, f"vs 0:  p = {sign_flip_p(v):.2f}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=18, color="#495057")
    p_bracket(ax, res["p"], fontsize=26)
    ax.set_ylabel("Post − pre\n(reactivation strength)")
    print(f"  delta: NLGF {res['mean_a']:+.3f} vs WT {res['mean_b']:+.3f}, "
          f"genotype p = {res['p']:.3f}")
    for g in GENOTYPE_ORDER:
        v = per_mouse[per_mouse.genotype == g].delta.to_numpy()
        print(f"    {g} delta {v.mean():+.3f}, sign-flip p = {sign_flip_p(v):.3f} "
              f"({v.size} mice)")


def level_panel(ax, d: pd.DataFrame) -> None:
    """Is reactivation above the shuffled template in each epoch separately?"""
    rng = np.random.default_rng(0)
    positions, labels = [], []
    for i, g in enumerate(GENOTYPE_ORDER):
        for j, epoch in enumerate(EPOCHS):
            x = i * 2.4 + j
            sub = d[(d.genotype == g) & (d.epoch == epoch)]
            ax.scatter(x + rng.uniform(-0.12, 0.12, len(sub)), sub[VALUE], s=42,
                       color=GENOTYPE_COLOURS[g], alpha=0.4, edgecolors="none",
                       zorder=3)
            per_mouse = collapse_to_mice(sub, VALUE)
            ax.scatter(np.full(len(per_mouse), x), per_mouse[VALUE], s=210,
                       color=GENOTYPE_COLOURS[g], zorder=5, edgecolors="white",
                       linewidths=2.0)
            v = per_mouse[VALUE].to_numpy()
            ax.plot([x - 0.3, x + 0.3], [v.mean()] * 2, color="#212529", lw=3.4,
                    zorder=6)
            p = sign_flip_p(v)
            ax.text(x, 1.01, f"p = {p:.2f}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=18, color="#495057")
            positions.append(x)
            labels.append(f"{g}\n{epoch}")
            print(f"  level {g} {epoch}: {v.mean():+.3f}, sign-flip p = {p:.3f}")
    ax.axhline(0, color="#495057", lw=1.8, ls="--", zorder=1)
    ax.text(0.015, 0.03, "0 = shuffled template", transform=ax.transAxes,
            fontsize=18, color="#868E96")
    # A sign-flip test over n mice cannot return a p below 2 / 2**n. At 4 NLGF mice
    # that floor IS 0.125, so the NLGF columns are at the smallest p the test can
    # produce - they are not a weaker result than WT's 0.031, they are a shorter ruler
    ax.text(0.985, 0.10, "sign-flip floor: 0.03 at 6 mice, 0.13 at 4",
            transform=ax.transAxes, ha="right", fontsize=17, color="#868E96")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reactivation strength\n(excess over shuffle)")


def main() -> None:
    set_style(talk=True)
    d = load()
    fig, axes = plt.subplots(1, 3, figsize=(26, 8.6))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.14, wspace=0.30)
    paired_panel(axes[0], d)
    delta_panel(axes[1], d)
    level_panel(axes[2], d)
    for ax, letter in zip(axes, "ABC"):
        ax.text(-0.13, 1.14, letter, transform=ax.transAxes, fontsize=36,
                weight="bold", va="top")
    fig.suptitle("Run ensembles reactivate in both freeze epochs, and the task between "
                 "them does not change how strongly",
                 fontsize=33, weight="semibold", y=0.975)
    out = PLOTS / "fig43_prepost_freeze.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
