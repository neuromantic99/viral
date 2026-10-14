"""Tier 4 for the talk: what happens to the texture code when the texture changes meaning.

The corridor carries two tactile textures. At a reversal the textures are physically
unchanged and only the reward mapping swaps, so the design separates a sensory code from
a value code cleanly: a sensory representation should be unaffected, a value
representation should reorganise.

Panel A is the manipulation, in one animal. Licking is plotted per PHYSICAL texture, so
the crossing lines are the animal discovering the swap - the textures did not change,
what they were worth did.

Panel B is what individual cells do, and it needs no decoder. Each matched cell's
texture preference is scored on both days; the bars are how often that preference
reverses. The first bar is the noise floor (two halves of ONE session, where the
contingency cannot have changed), the second is ordinary across-day drift at matched
gaps, the third adds the reversal. Flipping needs a reversal, not just a day.

Panel C is the population summary of the same thing. A decoder trained to report texture
on one session is tested on another; pairs that straddle a reversal are compared with
that same animal's pairs that do not, so each mouse is its own control for drift,
registration and cell count. Transfer is high either way - this is a dominantly sensory
code - but it drops across a reversal. Cell responses come first because the decoder is
a readout of them, not an independent line of evidence.

Panel D is the genotype test, and it is null.

Panel E is why the figure claims what it claims and nothing more. Both the pool size and
the trials-per-texture the decoder uses are matched by construction, and neither has a
principled value - 12 trials rather than 9 is a convention. Sweeping it shows the two
quantities behave completely differently. The transfer drop keeps its sign, size and
significance across the whole range (p between 0.008 and 0.016). A genotype difference,
by contrast, wanders: its p runs 0.043, 0.075, 0.354, 0.213, 0.055 as the parameter
moves, crossing 0.05 three times and lowest at BOTH ends of the range. That is a
quantity being resampled, not measured, so no genotype p is claimed here - the grey line
in panel E is drawn without significance marks deliberately.

Both claimed results also survive leave-one-mouse-out: the transfer drop reaches at
worst p = 0.016 with any single animal removed, and the per-cell excess p = 0.008.
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

from viral.nlgf.load import load_digest
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare, exact_permutation
from viral.nlgf.style import (GENOTYPE_COLOURS, GENOTYPE_ORDER, p_bracket, set_style,
                              superplot)
from viral.nlgf.textures import texture_labels

DEMO = ("JB026", ["2024-12-12", "2024-12-13", "2024-12-14", "2024-12-15",
                  "2024-12-16", "2024-12-17"])
TEX_COLOURS = ("#5F3DC4", "#E8590C")
T = "transfer_texture"


def _sign_flip_p(v: np.ndarray) -> float:
    stats = [np.mean(v * np.array(s)) for s in product([1, -1], repeat=len(v))]
    return float(np.mean(np.abs(stats) >= abs(v.mean()) - 1e-12))


def demo_panel(ax) -> None:
    mouse, dates = DEMO
    lick = {0: [], 1: []}
    reversal_at = None
    for i, date in enumerate(dates):
        idx, code, meta = texture_labels(mouse, date)
        d = load_digest(mouse, date)
        m = dict(zip(idx.tolist(), code.tolist()))
        t = d.trials[d.trials.idx.isin(m)].copy()
        t["tex"] = t.idx.map(m)
        g = t.groupby("tex").licks_AZ.mean()
        for k in (0, 1):
            lick[k].append(g.get(k, np.nan))
        if reversal_at is None and i > 0 and meta["rewarded_code"] != first_code:
            reversal_at = i - 0.5
        if i == 0:
            first_code = meta["rewarded_code"]

    x = np.arange(len(dates))
    for k in (0, 1):
        ax.plot(x, lick[k], marker="o", ms=13, lw=3.4, color=TEX_COLOURS[k],
                label=f"Texture {chr(65 + k)}")
    if reversal_at is not None:
        ax.axvline(reversal_at, color="#212529", lw=2.6, ls="--")
        ax.text(reversal_at, ax.get_ylim()[1], " reward swaps", ha="left", va="top",
                fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x])
    ax.set_xlabel("Session")
    ax.set_ylabel("Anticipatory licks")
    ax.legend(frameon=False, fontsize=22, loc="center left")


def transfer_panel(ax) -> None:
    d = pd.read_csv(RESULTS / "cross_session_transfer.csv")
    d = d[d.genotype.isin(GENOTYPE_ORDER)]
    rev, nr = d[d.crosses_reversal], d[~d.crosses_reversal]
    both = sorted(set(rev.mouse) & set(nr.mouse))
    a = np.array([rev[rev.mouse == m][T].mean() for m in both])
    b = np.array([nr[nr.mouse == m][T].mean() for m in both])
    for m, x0, x1 in zip(both, b, a):
        colour = GENOTYPE_COLOURS[d[d.mouse == m].genotype.iloc[0]]
        ax.plot([0, 1], [x0, x1], color=colour, lw=2.6, alpha=0.65, marker="o", ms=13)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Same\ncontingency", "Across\nreversal"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Texture decoder transfer")
    p = _sign_flip_p(a - b)
    p_bracket(ax, p, x0=0, x1=1, fontsize=26)
    ax.text(0.03, 0.03, f"{(a < b).sum()}/{len(a)} mice drop", transform=ax.transAxes,
            va="bottom", fontsize=22)
    print(f"  transfer: {b.mean():.3f} -> {a.mean():.3f}, "
          f"{(a < b).sum()}/{len(a)} drop, p = {p:.4f}")


def flip_panel(ax) -> None:
    """The three bars, plus the number that matters: how much of the reversal bar is
    reversal.

    15% is not the size of a value population. It contains the 4% that ordinary
    across-day drift produces anyway, and the reversal pairs sit at slightly longer gaps
    than the controls, so the honest quantity is the excess over the drift predicted at
    each pair's OWN gap. That is +5.9 points, not 15.
    """
    fr = pd.read_csv(RESULTS / "cell_flip_rates.csv")
    # the table carries every registered genotype, Oligo-BACE1-KO included; this is a
    # WT-vs-NLGF figure and pooling a third arm into the bars would be wrong
    fr = fr[fr.genotype.isin(GENOTYPE_ORDER)]
    fr["gap"] = (pd.to_datetime(fr.date_b) - pd.to_datetime(fr.date_a)).dt.days
    rev, nr = fr[fr.crosses_reversal], fr[~fr.crosses_reversal]
    bars = [("Within one\nsession", fr.flip_within.to_numpy(), "#ADB5BD"),
            ("Across days\nno reversal", nr.flip_across.to_numpy(), "#4C6EF5"),
            ("Across days\nreversal", rev.flip_across.to_numpy(), "#E8590C")]
    rng = np.random.default_rng(0)
    for i, (label, v, colour) in enumerate(bars):
        ax.bar(i, v.mean(), 0.62, color=colour, alpha=0.8)
        ax.scatter(i + rng.normal(0, 0.07, len(v)), v, s=44, color="#212529",
                   alpha=0.5, lw=0, zorder=3)
        ax.text(i, v.mean(), f" {v.mean():.0%}", ha="center", va="bottom",
                fontsize=23, fontweight="bold")
    # drift expected at each reversal pair's own gap, fitted on the non-reversal pairs
    b_, a_ = np.polyfit(nr.gap, nr.flip_across, 1)
    excess = (rev.assign(e=rev.flip_across - (a_ + b_ * rev.gap))
              .groupby("mouse").e.mean().to_numpy())
    ax.annotate("", xy=(2, bars[2][1].mean()), xytext=(2, a_ + b_ * rev.gap.mean()),
                arrowprops=dict(arrowstyle="<->", lw=2.4, color="#212529"))
    ax.text(2.42, (bars[2][1].mean() + a_ + b_ * rev.gap.mean()) / 2,
            f"{excess.mean():+.0%}\nover drift\np = {_sign_flip_p(excess):.3f}",
            va="center", ha="center", fontsize=21)
    ax.set_xticks(range(3))
    ax.set_xticklabels([b[0] for b in bars], fontsize=21)
    ax.set_xlim(-0.6, 2.95)
    ax.set_ylabel("Cells reversing\ntexture preference")
    ax.tick_params(axis="x", length=0)
    print(f"  flips: {bars[0][1].mean():.3f} / {bars[1][1].mean():.3f} / "
          f"{bars[2][1].mean():.3f}; excess over gap-matched drift "
          f"{excess.mean():+.3f}, p = {_sign_flip_p(excess):.4f}")


def genotype_panel(ax) -> None:
    d = pd.read_csv(RESULTS / "cross_session_transfer.csv")
    d = d[d.genotype.isin(GENOTYPE_ORDER) & d.crosses_reversal]
    superplot(ax, d, T, genotypes=GENOTYPE_ORDER, legend=True)
    _, res = compare(d, T)
    p_bracket(ax, res["p"], fontsize=26)
    ax.set_ylabel("Transfer across reversal")
    print(f"  genotype: NLGF {res['mean_a']:.3f} vs WT {res['mean_b']:.3f}, "
          f"p = {res['p']:.3f}")


def sweep_panel(ax) -> None:
    """The result under the matching parameter it is most exposed to."""
    files = {9: "cross_session_transfer_npc9.csv", 10: "cross_session_transfer_npc10.csv",
             12: "cross_session_transfer.csv", 14: "cross_session_transfer_npc14.csv",
             16: "cross_session_transfer_npc16.csv"}
    npc, drop, dp, geno = [], [], [], []
    for n, f in files.items():
        path = RESULTS / f
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d = d[d.genotype.isin(GENOTYPE_ORDER)]
        rev, nr = d[d.crosses_reversal], d[~d.crosses_reversal]
        both = sorted(set(rev.mouse) & set(nr.mouse))
        a = np.array([rev[rev.mouse == m][T].mean() for m in both])
        b = np.array([nr[nr.mouse == m][T].mean() for m in both])
        g = exact_permutation(rev.groupby(["mouse", "genotype"], as_index=False)[T].mean(), T)
        npc.append(n); drop.append((a - b).mean()); dp.append(_sign_flip_p(a - b))
        geno.append(g["diff"])
    ax.axhline(0, color="#ADB5BD", lw=1.6)
    ax.plot(npc, drop, marker="o", ms=13, lw=3.2, color="#212529",
            label="transfer drop (claimed)")
    for x, y, pv in zip(npc, drop, dp):
        ax.annotate(f"p={pv:.3f}", (x, y), textcoords="offset points", xytext=(0, -34),
                    ha="center", fontsize=18, color="#212529")
    # drawn WITHOUT p-values: the sweep is the evidence that this one is not stable
    ax.plot(npc, geno, marker="s", ms=11, lw=2.6, color="#ADB5BD", ls="--",
            label="genotype difference (not claimed)")
    ax.set_xlabel("Trials per texture (matching parameter)")
    ax.set_ylabel("Effect size")
    ax.set_xticks(npc)
    ax.legend(frameon=False, fontsize=19, loc="lower right")


def main() -> None:
    set_style(talk=True)
    fig = plt.figure(figsize=(20, 14.5))
    # The sweep lives in fig38, not here: it needs more than ten seconds to read, and
    # its job is to answer a challenge rather than to carry the story. sweep_panel() is
    # kept below so it can be dropped into a backup slide.
    axes = fig.subplots(2, 2)
    # Top row is what individual cells do, bottom row is what a decoder reads out of
    # the population: responses first, then the population summary of them.
    demo_panel(axes[0, 0])
    flip_panel(axes[0, 1])
    transfer_panel(axes[1, 0])
    genotype_panel(axes[1, 1])
    fig.suptitle("When a texture changes meaning, RSC partly re-codes it - "
                 "equally in both genotypes",
                 fontsize=34, weight="semibold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = PLOTS / "fig34_tier4_reversal.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
