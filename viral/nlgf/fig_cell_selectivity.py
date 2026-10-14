"""Does an individual RSC cell code the texture, or what the texture is worth?

The population decoder transferred across the reversal, which reads as "texture, not
value". But a decoder reports whatever dominates the population: if RSC holds a texture
population and a smaller value population side by side - which is what you would expect
of an integrator - the decoder locks onto the larger one and the smaller is invisible.
This asks the question one cell at a time.

Each matched cell gets a texture selectivity in the pre-reversal session and again in
the post-reversal session, both referenced to the SAME physical texture. Textures do not
change at a reversal; only which one is rewarded does. So a texture cell keeps its sign
and a value cell flips it.

Sign flips happen for boring reasons too, so the measurement sits on a ladder of three
controls, each removing one of them:

  within-session, odd vs even trials   the contingency cannot have changed, so this is
                                       pure measurement noise
  across days, NO reversal             adds ordinary representational drift, at the
                                       same day gaps and registration quality
  across days, reversal                adds the contingency change

Only the step from the second rung to the third can be about value, and the day-gap
difference between the rungs is regressed out on top.
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

from viral.nlgf.cell_selectivity import summarise
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import exact_permutation
from viral.nlgf.style import (GENOTYPE_COLOURS, GENOTYPE_ORDER, p_bracket, set_style,
                              superplot)

TEXTURE_C, VALUE_C = "#5F3DC4", "#E8590C"


def _sign_flip_p(d: np.ndarray) -> float:
    """Exact sign-flip test: every assignment of +/- to the per-mouse values."""
    stats = [np.mean(d * np.array(s)) for s in product([1, -1], repeat=len(d))]
    return float(np.mean(np.abs(stats) >= abs(d.mean()) - 1e-12))


def _dbic(x: np.ndarray) -> float:
    from sklearn.mixture import GaussianMixture

    v = x.reshape(-1, 1)
    return float(np.diff([GaussianMixture(k, random_state=0, n_init=5).fit(v).bic(v)
                          for k in (1, 2)])[0])


def load():
    df = pd.read_csv(RESULTS / "cell_selectivity.csv")
    fr = pd.read_csv(RESULTS / "cell_flip_rates.csv")
    fr["gap"] = (pd.to_datetime(fr.date_b) - pd.to_datetime(fr.date_a)).dt.days
    return df, fr


def main() -> None:
    set_style()
    df, fr = load()
    rev_cells = df[df.crosses_reversal]
    both = summarise(rev_cells)
    nr, rev = fr[~fr.crosses_reversal], fr[fr.crosses_reversal]

    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34)

    # ---- A: the sel_A / sel_B plane ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    lim = 0.52
    ax.axhline(0, color="#CED4DA", lw=0.8)
    ax.axvline(0, color="#CED4DA", lw=0.8)
    ax.plot([-lim, lim], [-lim, lim], color=TEXTURE_C, lw=1.2, ls="--", zorder=1)
    ax.plot([-lim, lim], [lim, -lim], color=VALUE_C, lw=1.2, ls="--", zorder=1)
    for g in GENOTYPE_ORDER:
        s = both[both.genotype == g]
        ax.scatter(s.sel_a, s.sel_b, s=13, alpha=0.5, lw=0,
                   color=GENOTYPE_COLOURS[g], label=f"{g} (n={len(s)})")
    ax.text(0.34, 0.44, "texture", color=TEXTURE_C, fontsize=8, ha="center")
    ax.text(-0.34, 0.44, "value", color=VALUE_C, fontsize=8, ha="center")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("texture selectivity, pre-reversal")
    ax.set_ylabel("texture selectivity, post-reversal")
    ax.set_title("A  Cells selective on both days", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    # ---- B: is it one population or two? -------------------------------------------
    # Deliberately NOT the doubly-selective cells used in A: requiring significance in
    # BOTH sessions excludes near-zero selectivity by construction, which digs a hole
    # at zero and makes any distribution look bimodal. Conditioning only on session A
    # leaves the post-reversal axis free to take any value, so a second mode below zero
    # would mean something.
    ax = fig.add_subplot(gs[0, 1])
    open_set = rev_cells[rev_cells.sig_a]
    folded_open = (np.sign(open_set.sel_a) * open_set.sel_b).to_numpy()
    folded = (np.sign(both.sel_a) * both.sel_b).to_numpy()
    ax.hist(folded_open, bins=40, color="#495057", alpha=0.85)
    ax.axvline(0, color="k", lw=0.9)
    d_bic = _dbic(folded_open)
    ax.set_xlabel("post-reversal selectivity, signed by pre-reversal preference")
    ax.set_ylabel("cells")
    ax.set_title("B  One population, shifted", loc="left", fontweight="bold")
    ax.text(0.02, 0.96,
            f"selective pre-reversal only (n={len(folded_open)})\n"
            f"{'2 components favoured' if d_bic < -10 else '1 component sufficient'}, "
            f"$\\Delta$BIC = {d_bic:.0f}\n"
            f"{(folded_open < 0).mean():.0%} of the mass below zero",
            transform=ax.transAxes, va="top", fontsize=8)

    # ---- C: the control ladder -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    rungs = [("within session\n(odd vs even trials)", fr.flip_within.to_numpy(), "#ADB5BD"),
             ("across days\nNO reversal", nr.flip_across.to_numpy(), "#4C6EF5"),
             ("across days\nREVERSAL", rev.flip_across.to_numpy(), VALUE_C)]
    for i, (label, vals, colour) in enumerate(rungs):
        ax.bar(i, vals.mean(), 0.6, color=colour, alpha=0.75)
        ax.scatter(np.full(len(vals), i) + np.random.default_rng(0)
                   .normal(0, 0.07, len(vals)), vals, s=16, color="#212529",
                   alpha=0.5, lw=0, zorder=3)
        ax.text(i, vals.mean(), f" {vals.mean():.1%}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([r[0] for r in rungs], fontsize=8)
    ax.set_ylabel("fraction of selective cells that flip sign")
    ax.set_title("C  Flipping needs a reversal, not just a day", loc="left",
                 fontweight="bold")

    # ---- D: excess over the drift predicted by each pair's own gap ------------------
    ax = fig.add_subplot(gs[1, 0])
    b, a = np.polyfit(nr.gap, nr.flip_across, 1)
    ax.scatter(nr.gap, nr.flip_across, s=26, color="#4C6EF5", alpha=0.6, lw=0,
               label=f"no reversal (n={len(nr)})")
    xs = np.linspace(0, nr.gap.max(), 50)
    ax.plot(xs, a + b * xs, color="#4C6EF5", lw=1.4, ls="--", label="drift model")
    ax.scatter(rev.gap, rev.flip_across, s=52, color=VALUE_C, alpha=0.9, lw=0,
               label=f"reversal (n={len(rev)})")
    ax.set_xlabel("gap between sessions (days)")
    ax.set_ylabel("fraction flipping")
    ax.set_title("D  Reversal pairs sit above the drift line", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=7)

    # ---- E: per-mouse excess, the actual test --------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    excess = (rev.assign(excess=rev.flip_across - (a + b * rev.gap))
              .groupby(["mouse", "genotype"], as_index=False).excess.mean())
    order = excess.sort_values("excess")
    ax.barh(np.arange(len(order)), order.excess,
            color=[GENOTYPE_COLOURS[g] for g in order.genotype])
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order.mouse, fontsize=8)
    ax.set_xlabel("excess flipping over gap-matched drift")
    p = _sign_flip_p(excess.excess.to_numpy())
    ax.set_title(f"E  {excess.excess.mean():+.1%} per mouse, p = {p:.3f}",
                 loc="left", fontweight="bold")

    # ---- F: does it depend on whether the animal re-evaluated? ----------------------
    ax = fig.add_subplot(gs[1, 2])
    for g, colour in [(True, "#212529"), (False, "#ADB5BD")]:
        s = rev[rev.reversed_behaviourally == g]
        if s.empty:
            continue
        ax.scatter(s.licking_dprime, s.flip_across, s=48, lw=0, alpha=0.85,
                   color=colour,
                   label=f"{'learned' if g else 'not learned'} (n={len(s)})")
    ax.axvline(1.0, color="#868E96", ls=":", lw=1)
    ax.axhline(a + b * rev.gap.mean(), color="#4C6EF5", ls="--", lw=1.2,
               label="drift expectation")
    ax.set_xlabel("post-reversal licking d'")
    ax.set_ylabel("fraction flipping")
    ax.set_title("F  Flipping does not wait for the animal", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Per-cell texture vs value coding across a reversal", fontsize=14,
                 fontweight="bold", y=0.975)
    out = PLOTS / "fig20_cell_selectivity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    print(f"\n{len(both)} doubly-selective cells over {len(rev)} reversal pairs; "
          f"{(folded > 0).mean():.1%} keep sign")
    print(f"unconditioned on session B: n={len(folded_open)}, "
          f"{(folded_open < 0).mean():.1%} below zero, dBIC(2-1) = {d_bic:.0f}")
    for label, vals, _ in rungs:
        print(f"  {label.replace(chr(10),' '):38s} {vals.mean():.3f}  (n={len(vals)})")
    print(f"excess over gap-matched drift: {excess.excess.mean():+.3f}, "
          f"exact sign-flip p = {p:.4f} over {len(excess)} mice")
    pm = (rev.assign(excess=rev.flip_across - (a + b * rev.gap))
          .groupby(["mouse", "genotype"], as_index=False).excess.mean())
    res = exact_permutation(pm, "excess")
    print(f"genotype difference in excess flipping: {res['diff']:+.3f}, p = {res['p']:.3f}")


if __name__ == "__main__":
    main()
