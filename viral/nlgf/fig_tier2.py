"""Tier 2 for the talk: RSC codes space, and that code is unchanged in NLGF.

Panel A is the phenomenon, shown in one session rather than asserted: place cells sorted
by where they peak, tiling the corridor. It is drawn on ALL laps - cells are ordered and
displayed on the same data - which is the conventional presentation and the crisper one.

Note what that panel can and cannot support. Part of its diagonal is built by the sort:
order any matrix by its row peaks and a diagonal appears, even in noise
(fig15_split_half_demo.png shows exactly that on this data). So the panel illustrates
WHERE the fields sit, not that they are reliable. The claim that these are place cells
does not rest on it - cells are selected by a per-lap circular-shuffle test at the 99th
percentile with five consecutive supra-threshold bins - and the split-half version, in
which cells are ordered on odd laps and the even laps plotted, is in fig13 for anyone
who asks whether the fields repeat.

Panels B and C are the genotype comparison, on the two measures an audience actually
asks about: how many cells are spatially tuned, and how well position can be read out of
them. Both are matched by construction - every session scored on its first 30 laps with
the shuffle threshold regenerated from those laps, and decoding on equal cell counts -
because unmatched versions of both track how much data a session happened to contain
rather than genotype.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.fig_examples import (CANDIDATES, CONFIG, EXAMPLES, _normalise,
                                     session_maps)
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import (GENOTYPE_COLOURS, GENOTYPE_ORDER, LANDMARKS_CM,
                              p_bracket, set_style, superplot)

# The talk example, picked from the candidate grid (fig32) on how clearly it reads.
# An example session, not a typical one - 92nd percentile for lap-to-lap reliability
# within NLGF. It is deliberately an NLGF session: showing the phenomenon in a diseased
# animal makes the "coding is intact" point before the group panels are reached.
SNAKE = ("J030", "2026-05-19")
SPLIT_HALF = False   # True orders on odd laps and plots even laps (see fig13)

# Panel D. A mouse needs this many learning sessions, spanning this many distinct days,
# before a within-animal trend can be estimated from it at all.
MIN_SESSIONS, MIN_DAYS = 4, 3

# No panel subheadings on the talk version - the axis labels have to carry the
# identity on their own, so they say what the quantity is rather than its units.
PANELS = [
    ("place_lap_matched", "frac_place_cells", "Fraction place cells", ""),
    ("decoding_matched", "decoding_error_matched", "Decoding error (cm)", ""),
]

# Chance for a decoder guessing uniformly at random on the 20-160 cm stretch: for two
# independent uniform draws on a track of length L, the expected absolute difference is
# L/3, so 140/3 = 46.7 cm.
#
# Worth knowing that this is not the only "chance" one could draw. A degenerate decoder
# that always answers the middle of the track scores L/4 = 35 cm, better than random
# guessing, so a session approaching 35 cm is not decoding so much as refusing to
# commit. The worst session in this dataset sits at 43.5 cm.
CHANCE = {"decoding_error_matched": (20.0, 160.0)}


def _load(stem: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"{stem}.csv")
    return df[df.genotype.isin(GENOTYPE_ORDER)]


def _snake(ax, d, fontsize: int = 26, title: str | None = None):
    """Draw one session's sorted field map. Shared by panel A and the candidate grid,
    so what you choose from is exactly what you get."""
    odd, even, pcs = d["odd"][d["pcs"]], d["even"][d["pcs"]], d["pcs"]
    if SPLIT_HALF:
        sort_on, show = odd, even
    else:
        sort_on = show = np.nanmean([odd, even], axis=0)
    order = np.argsort(np.argmax(sort_on, axis=1))
    img = _normalise(show)[order]
    x = CONFIG.start + np.arange(d["odd"].shape[1]) * CONFIG.bin_size
    im = ax.imshow(img, aspect="auto", cmap="magma", vmin=0, vmax=1,
                   extent=[x[0], x[-1], img.shape[0], 0], interpolation="nearest")
    for lm in LANDMARKS_CM:
        ax.axvline(lm, color="#8ED1FC", lw=1.6, ls="--", alpha=0.85)
    ax.set_xlabel("Position (cm)")
    if title:
        ax.set_title(title, fontsize=fontsize)
    return im, int(pcs.sum())


def candidates() -> None:
    """Every candidate session drawn in the panel-A style, for picking a talk example."""
    set_style()
    items = [(g, m, dt, pct) for g, lst in CANDIDATES.items() for m, dt, pct in lst]
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    for ax, (g, m, dt, pct) in zip(axes.ravel(), items):
        d = session_maps(m, dt)
        if d is None:
            ax.axis("off")
            continue
        n = int(d["pcs"].sum())
        _snake(ax, d, fontsize=15,
               title=f"{g}  {m} {dt}\n{n}/{d['n_cells']} place cells"
                     f"  ·  reliability {pct}th pct")
        ax.set_ylabel("Place cell")
        print(f"  {g:5s} {m} {dt}  {n}/{d['n_cells']} place cells  ({pct}th pct)",
              flush=True)
    fig.suptitle("Candidate sessions for panel A  ·  drawn exactly as the talk figure "
                 "draws it (all laps, sorted by peak)",
                 fontsize=20, weight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = PLOTS / "fig32_tier2_snake_candidates.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")


def learning_panel(ax) -> None:
    """Spatial information against learning day, one line per mouse.

    Drawn WITHIN mouse because that is where the effect lives. Pooling sessions across
    animals gives rho = +0.35, which understates it: mice differ a lot in baseline
    spatial information, and that between-animal spread dilutes the within-animal climb.
    Estimated per mouse and tested with an exact sign-flip over animals, it is
    rho = +0.66 with all 7 mice positive.

    Note what does NOT change: field width does not narrow (+0.35, p = 0.30) and
    lap-to-lap reliability does not improve (-0.07, p = 0.81). So the information gain
    is not fields becoming tighter or steadier - "tuning sharpens" is true only in the
    sense of information content.
    """
    from itertools import product
    from scipy.stats import spearmanr

    d = _load("place_lap_matched")
    d = d[d.stage == "learning"].dropna(subset=["day", "spatial_info_pc"])
    rhos = []
    for mouse, s in d.groupby("mouse"):
        if len(s) < MIN_SESSIONS or s.day.nunique() < MIN_DAYS:
            continue
        s = s.sort_values("day")
        # x is the ORDINAL position of each imaged learning session, not the training-day
        # label. Spearman is rank-based, so the statistic is identical either way, but
        # one mouse was still learning at day 17 and its raw-day line would stretch the
        # axis and squash every other animal into the left quarter.
        order = np.arange(1, len(s) + 1)
        ax.plot(order, s.spatial_info_pc, marker="o", ms=9, lw=2.4, alpha=0.9,
                color=GENOTYPE_COLOURS[s.genotype.iloc[0]])
        rhos.append(spearmanr(s.day, s.spatial_info_pc)[0])

    r = np.array([v for v in rhos if np.isfinite(v)])
    stats = [np.mean(r * np.array(sg)) for sg in product([1, -1], repeat=len(r))]
    pv = float(np.mean(np.abs(stats) >= abs(r.mean()) - 1e-12))
    ax.set_xlabel("Learning session (imaged)")
    ax.set_xticks(range(1, 6))
    ax.text(0.03, 0.97, f"{(r > 0).sum()}/{len(r)} mice increase\np = {pv:.3f}",
            transform=ax.transAxes, va="top", fontsize=24)
    print(f"  spatial info vs day: mean within-mouse rho = {r.mean():+.3f}, "
          f"{(r > 0).sum()}/{len(r)} positive, p = {pv:.4f}")


def centred_panel(ax) -> None:
    """Decoding error against experience, both centred within each mouse.

    Centring removes every between-animal difference, leaving only the within-animal
    question. It has to be shown next to the per-mouse slopes, not instead of them: the
    fitted lines here are anticonservative, because centring removes the between-mouse
    confound but sessions within a mouse are still not independent, so a regression over
    ~93 centred points behaves as if there were 93 animals rather than 14. The panel
    beside it carries the honest test.
    """
    from viral import dec

    d = pd.read_csv(RESULTS / "experience.csv")
    d = d[d.genotype.isin(GENOTYPE_ORDER)].dropna(subset=["cum_trials", "acc"])
    f = dec.prepare(d, logit_y=False)
    for g in GENOTYPE_ORDER:
        s = f[f.genotype == g].copy()
        s["x_c"] = s.exper - s.groupby("mouse").exper.transform("mean")
        s["y_c"] = s.y - s.groupby("mouse").y.transform("mean")
        ax.scatter(s.x_c, s.y_c, s=46, alpha=0.55, lw=0,
                   color=GENOTYPE_COLOURS[g], label=g)
        b, a = np.polyfit(s.x_c, s.y_c, 1)
        xs = np.linspace(s.x_c.min(), s.x_c.max(), 20)
        ax.plot(xs, a + b * xs, color=GENOTYPE_COLOURS[g], lw=3.4)
    ax.axhline(0, color="#CED4DA", lw=1.2, zorder=0)
    ax.axvline(0, color="#CED4DA", lw=1.2, zorder=0)
    ax.set_xlabel("Experience (centred, SD)")
    ax.set_ylabel("Decoding error (centred, cm)")
    ax.legend(frameon=False, fontsize=21, loc="upper right")


def experience_panel(ax) -> None:
    """One dot per mouse: the slope of its decoding error against its own experience.

    This is the unit the test actually uses. The alternative presentations are worse for
    opposite reasons - a raw pooled scatter mixes a within-animal improvement with an
    opposite between-animal trend and shows nothing (rho = -0.05, p = 0.65), while a
    within-mouse centred regression over all sessions looks convincing but counts ~93
    observations from 14 animals. A slope per mouse has neither problem: each animal
    contributes exactly one number, and the p is a sign-flip over those numbers.

    Negative means the map decodes better as that animal gains experience. Most mice are
    below zero and neither genotype reaches significance on its own.
    """
    from itertools import product

    from viral import dec

    d = pd.read_csv(RESULTS / "experience.csv")
    d = d[d.genotype.isin(GENOTYPE_ORDER)].dropna(subset=["cum_trials", "acc"])
    f = dec.prepare(d, logit_y=False)
    rng = np.random.default_rng(0)
    for i, g in enumerate(GENOTYPE_ORDER):
        slopes = np.array([np.polyfit(gg.exper, gg.y, 1)[0]
                           for _, gg in f[f.genotype == g].groupby("mouse")
                           if len(gg) >= MIN_SESSIONS])
        colour = GENOTYPE_COLOURS[g]
        ax.scatter(i + rng.uniform(-0.11, 0.11, slopes.size), slopes, s=190,
                   color=colour, alpha=0.9, edgecolor="white", linewidth=2.0, zorder=3)
        ax.hlines(slopes.mean(), i - 0.26, i + 0.26, color=colour, lw=4.0, zorder=4)
        stats = [np.mean(slopes * np.array(sg))
                 for sg in product([1, -1], repeat=slopes.size)]
        pv = float(np.mean(np.abs(stats) >= abs(slopes.mean()) - 1e-12))
        ax.text(i, ax.get_ylim()[1], f"p = {pv:.2f}", ha="center", va="bottom",
                fontsize=22, color=colour)
        print(f"  {g:5s} slope per mouse: mean {slopes.mean():+.2f} cm/SD, "
              f"{(slopes < 0).sum()}/{slopes.size} improving, p = {pv:.3f}")
    ax.axhline(0, color="#495057", lw=1.6, zorder=0)
    ax.set_xticks(range(len(GENOTYPE_ORDER)))
    ax.set_xticklabels(GENOTYPE_ORDER)
    ax.set_xlim(-0.55, len(GENOTYPE_ORDER) - 0.45)
    ax.set_ylabel("Decoding error vs experience\n(slope per mouse, cm/SD)")
    ax.tick_params(axis="x", length=0)


def main() -> None:
    set_style(talk=True)
    mouse, date = SNAKE
    d = session_maps(mouse, date)
    if d is None:
        raise SystemExit(f"no maps for {mouse} {date}")

    # explicit margins rather than tight_layout: a colorbar makes tight_layout
    # unreliable, and it was pulling panel A's title into the suptitle
    # snake spans both rows on the left; the four summary panels fill a 2x2 on the
    # right, so five panels still fit a 16:9 slide
    fig = plt.figure(figsize=(27, 13.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.15], wspace=0.20,
                             left=0.045, right=0.985, top=0.86, bottom=0.09)
    right = outer[0, 1].subgridspec(2, 1, hspace=0.42)
    top = right[0].subgridspec(1, 2, wspace=0.36)      # genotype comparisons
    bot = right[1].subgridspec(1, 2, wspace=0.40)      # experience

    # ---- A: the phenomenon, one session ------------------------------------------
    ax = fig.add_subplot(outer[0, 0])
    _snake(ax, d)
    im = ax.images[0]
    ax.set_ylabel("Place cell")
    cb = plt.colorbar(im, ax=ax, fraction=0.034, pad=0.09)
    cb.set_label("Normalised rate", fontsize=26)
    cb.ax.tick_params(labelsize=21)

    # ---- B, C: the genotype comparison -------------------------------------------
    for i, (stem, col, label, title) in enumerate(PANELS, start=1):
        ax = fig.add_subplot(top[0, i - 1])
        s = _load(stem).dropna(subset=[col])
        superplot(ax, s, col, genotypes=GENOTYPE_ORDER, legend=(i == 1))
        _, res = compare(s, col)
        if col in CHANCE:
            lo, hi = CHANCE[col]
            ax.axhline((hi - lo) / 3, color="#868E96", lw=2.4, ls="--", zorder=0)
            ax.text(0.98, (hi - lo) / 3, "chance", ha="right", va="bottom",
                    fontsize=21, color="#868E96", transform=ax.get_yaxis_transform())
        p_bracket(ax, res["p"], fontsize=26)
        ax.set_ylabel(label)
        if title:
            ax.set_title(title, fontsize=26)
        print(f"  {title:20s} NLGF {res['mean_a']:.3f}  WT {res['mean_b']:.3f}  "
              f"p = {res['p']:.3f}  ({res['n_a']}v{res['n_b']} mice)")

    centred_panel(fig.add_subplot(bot[0, 0]))
    experience_panel(fig.add_subplot(bot[0, 1]))

    fig.suptitle("RSC represents space, and that representation is intact in NLGF",
                 fontsize=34, weight="semibold", y=0.97)
    out = PLOTS / "fig31_tier2_place_coding.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    if "--candidates" in sys.argv:
        candidates()
    else:
        main()
