"""Figure 18: what the registered cells say.

Two questions, and they differ sharply in how well the data answer them.

TEXTURE OR VALUE. Across a reversal the textures are physically unchanged and only
their meaning flips, so a decoder trained on texture identity before the switch and
tested after it separates a sensory code from a value code in one step. A value code
would INVERT - transfer below chance - because the labels have swapped. This is the
question registration was done for, and it comes out cleanly.

DAY-TO-DAY STABILITY. Non-reversal pairs measure whether the same cells carry the same
code a day or two later. NLGF transfer worse, with identical within-session decoding,
which is the right signature for a stability deficit. But transfer correlates with
registration quality and NLGF register worse, so the panel also shows that relationship
and the quality-matched subset, rather than asserting the effect.

The quality-matched comparison restricts to pairs inside the affine-residual range both
genotypes occupy. Matching by construction rather than adjusting, following Tier 5,
where a covariate with no common support produced a spurious significant result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, set_style, superplot


def load() -> pd.DataFrame:
    d = pd.read_csv(RESULTS / "cross_session_transfer.csv")
    qc = pd.read_csv(RESULTS / "registration_qc.csv")
    return d.merge(qc[["mouse", "date_a", "date_b", "affine_rms_px", "outlier_frac"]],
                   on=["mouse", "date_a", "date_b"], how="left")


def figure() -> None:
    d = load()
    rev, nonrev = d[d.crosses_reversal], d[~d.crosses_reversal]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    # A: the texture-vs-value test
    ax = axes[0]
    within = rev[["within_a", "within_b"]].mean(axis=1)
    rng = np.random.default_rng(0)
    for i, (vals, lab) in enumerate([(within, "within\nsession"),
                                     (rev.transfer_texture, "across\nreversal")]):
        for g in GENOTYPE_ORDER:
            m = (rev.genotype == g).to_numpy()
            v = np.asarray(vals)[m]
            ax.scatter(i + rng.uniform(-0.08, 0.08, v.size), v, s=62,
                       color=GENOTYPE_COLOURS[g], alpha=0.9, edgecolor="white",
                       linewidth=1.2, zorder=3, label=g if i == 0 else None)
        ax.hlines(np.median(vals), i - 0.24, i + 0.24, color="#495057", lw=2.4, zorder=4)
    for j in range(len(rev)):
        ax.plot([0, 1], [within.iloc[j], rev.transfer_texture.iloc[j]],
                color="#ADB5BD", lw=1, alpha=0.6, zorder=1)
    ax.axhline(0.5, color="#E8590C", lw=1.2, ls="--", zorder=0)
    ax.text(1.45, 0.505, "chance", fontsize=8.5, color="#E8590C", va="bottom")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["within\nsession", "across\nreversal"])
    ax.set_xlim(-0.45, 1.6)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("Texture decoding accuracy")
    ax.set_title(f"Texture code survives the reversal\n{len(rev)} pairs, all above chance",
                 fontsize=10.5)
    ax.legend(fontsize=9, loc="lower left")

    # B: day-to-day transfer by genotype
    ax = axes[1]
    pm, res = compare(nonrev, "transfer_texture")
    superplot(ax, nonrev, "transfer_texture", p_value=res["p"], legend=True)
    ax.axhline(0.5, color="#E8590C", lw=1.1, ls="--", zorder=0)
    ax.set_ylabel("Transfer accuracy, day to day")
    ax.set_title("Day-to-day stability", fontsize=10.5)

    # C: the confound
    ax = axes[2]
    for g in GENOTYPE_ORDER:
        s = nonrev[nonrev.genotype == g]
        ax.scatter(s.affine_rms_px, s.transfer_texture, s=30,
                   color=GENOTYPE_COLOURS[g], alpha=0.65, edgecolor="none", label=g)
    ok = nonrev.dropna(subset=["affine_rms_px", "transfer_texture"])
    rho, p = spearmanr(ok.affine_rms_px, ok.transfer_texture)
    b = np.polyfit(ok.affine_rms_px, ok.transfer_texture, 1)
    xs = np.linspace(ok.affine_rms_px.min(), ok.affine_rms_px.max(), 20)
    ax.plot(xs, np.polyval(b, xs), color="#495057", lw=1.4, ls="--", zorder=3)
    ax.set_xlabel("Registration residual (px)")
    ax.set_ylabel("Transfer accuracy")
    ax.set_title(f"Worse registration, worse transfer\nrho = {rho:+.2f}, p = {p:.1g}",
                 fontsize=10.5)
    ax.legend(fontsize=9)

    # D: quality-matched subset
    ax = axes[3]
    a = nonrev.loc[nonrev.genotype == "NLGF", "affine_rms_px"]
    b_ = nonrev.loc[nonrev.genotype == "WT", "affine_rms_px"]
    lo, hi = max(a.min(), b_.min()), min(a.max(), b_.max())
    matched = nonrev[(nonrev.affine_rms_px >= lo) & (nonrev.affine_rms_px <= hi)]
    pm2, res2 = compare(matched, "transfer_texture")
    superplot(ax, matched, "transfer_texture", p_value=res2["p"])
    ax.axhline(0.5, color="#E8590C", lw=1.1, ls="--", zorder=0)
    ax.set_ylabel("Transfer accuracy")
    ax.set_title(f"Matched on registration quality\n{lo:.1f}–{hi:.1f} px, "
                 f"{len(matched)} of {len(nonrev)} pairs", fontsize=10.5)

    fig.suptitle(
        "Registered cells: the texture code survives a reversal, and day-to-day "
        "transfer is confounded with registration quality",
        y=1.03, fontsize=12.5, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig18_cross_session.png")
    plt.close(fig)
    return res, res2, lo, hi, len(matched)


if __name__ == "__main__":
    set_style()
    res, res2, lo, hi, n = figure()
    print(f"all non-reversal pairs : diff {res['diff']:+.3f}, p = {res['p']:.3f}")
    print(f"quality-matched ({lo:.1f}-{hi:.1f} px, n={n}): "
          f"diff {res2['diff']:+.3f}, p = {res2['p']:.3f}")
    print(f"wrote figure to {PLOTS}")
