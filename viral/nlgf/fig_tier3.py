"""Tier 3 for the talk: RSC firing is anchored to the landmarks, equally in both genotypes.

Panel A is the measurement with nothing between the reader and the data: take every
landmark crossing, line them up, plot population rate against distance from the
landmark. The grey line is the identical operation on positions exactly out of phase
with the landmarks, where there is nothing - and it is what turns "there is a bump" into
"there is a bump BECAUSE of the landmarks". The corridor's landmarks form a regular
45 cm comb, so any periodic structure at all would produce a peak at zero; the
out-of-phase control is the only thing that rules that out.

Panel B is the objection this analysis has to answer. Animals slow at landmarks, and
deconvolved rate rises with speed, so a rate bump can be a speed bump wearing a hat.

Panel C answers it. Rate is regressed on speed across bins within each session and the
RESIDUAL profile is plotted: what is left of the landmark peak once the speed
relationship is removed. Note this is the residual CURVE, not the residual ratio - the
(near - far)/(near + far) index becomes unusable on residuals, because residuals are
centred near zero and the denominator can vanish (one ordinary session came out at +40).
The curve has no such problem, and it is the honest way to show a correction whose
summary statistic is ill-conditioned.

Panel D is the functional version of the same question, and the strongest result here.
Position is decoded from the population per spatial bin, and the decoding error is
regressed on distance to the nearest landmark: if landmarks anchor the code, error
should be LOWER near them. That slope is expressed as a z against the SAME phase-shifted
comb null, so it asks whether the code is tied to these landmarks specifically rather
than to any 45 cm spacing. Every one of the 15 mice is above zero (WT p = 0.008,
NLGF p = 0.016, pooled p = 0.0001), and the genotypes do not differ (p = 0.52).

Panel C is the genotype test on the rate profile: (near - far) / (near + far), zero
meaning flat, one sentence to explain and bounded in [-1, +1] by construction.

It is deliberately the UNCORRECTED index. The speed-corrected version divides the same
difference by a sum of rate RESIDUALS, which are centred near zero, so the ratio
explodes whenever a session's near and far residual means happen to cancel - one
ordinary session (JB027 2024-12-11, uncorrected +0.059, bang on the median) came out at
+40 and dominated the axis. That is arithmetic, not biology, and the same
divide-by-a-vanishing-denominator failure as the rescue index in fig23. Panel B carries
the speed story instead; the speed-corrected profile is in fig27.
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
from viral.nlgf.stats import compare
from viral.nlgf.style import (GENOTYPE_COLOURS, GENOTYPE_ORDER, p_bracket, set_style,
                              superplot)

PSEUDO_C = "#ADB5BD"


def _stack(npz, df, key, genotype=None):
    rows = df if genotype is None else df[df.genotype == genotype]
    out = [npz[f"{r.mouse}_{r.date}_{key}"] for r in rows.itertuples()
           if f"{r.mouse}_{r.date}_{key}" in npz]
    return np.vstack(out) if out else None


def _band(ax, x, arr, colour, label, lw=3.0):
    if arr is None or arr.shape[0] < 2:
        return
    m = np.nanmean(arr, axis=0)
    se = np.nanstd(arr, axis=0) / np.sqrt(np.sum(np.isfinite(arr), axis=0))
    ax.plot(x, m, color=colour, lw=lw, label=label)
    ax.fill_between(x, m - se, m + se, color=colour, alpha=0.18, lw=0)


def main() -> None:
    set_style(talk=True)
    df = pd.read_csv(RESULTS / "landmark_profile.csv")
    df = df[df.genotype.isin(GENOTYPE_ORDER)]
    npz = np.load(RESULTS / "landmark_profiles.npz")
    x = npz[f"{df.iloc[0].mouse}_{df.iloc[0].date}_offsets"]

    # three profiles on top, two genotype tests beneath: the curves carry the argument
    # and the boxplots carry the statistics
    fig = plt.figure(figsize=(25, 14.5))
    outer = fig.add_gridspec(2, 1, hspace=0.34, left=0.05, right=0.985,
                             top=0.90, bottom=0.08)
    top = outer[0].subgridspec(1, 3, wspace=0.30)
    bot = outer[1].subgridspec(1, 2, wspace=0.26)
    axes = [fig.add_subplot(top[0, i]) for i in range(3)] + \
           [fig.add_subplot(bot[0, i]) for i in range(2)]

    # A: the phenomenon, with the out-of-phase control
    ax = axes[0]
    for g in GENOTYPE_ORDER:
        _band(ax, x, _stack(npz, df, "rate_real", g), GENOTYPE_COLOURS[g], g)
    _band(ax, x, _stack(npz, df, "rate_pseudo"), PSEUDO_C, "away from landmarks", lw=2.4)
    ax.axvline(0, color="#495057", lw=1.6, ls=":")
    ax.set_xlabel("Distance from landmark (cm)")
    ax.set_ylabel("Population rate (normalised)")
    ax.legend(frameon=False, fontsize=22, loc="upper left")

    # B: the confound
    ax = axes[1]
    for g in GENOTYPE_ORDER:
        _band(ax, x, _stack(npz, df, "speed_real", g), GENOTYPE_COLOURS[g], g)
    ax.axvline(0, color="#495057", lw=1.6, ls=":")
    ax.set_xlabel("Distance from landmark (cm)")
    ax.set_ylabel("Running speed (cm/s)")

    # C: the same profile after removing the speed relationship
    ax = axes[2]
    for g in GENOTYPE_ORDER:
        _band(ax, x, _stack(npz, df, "resid_real", g), GENOTYPE_COLOURS[g], g)
    _band(ax, x, _stack(npz, df, "resid_pseudo"), PSEUDO_C, "away from landmarks",
          lw=2.4)
    ax.axvline(0, color="#495057", lw=1.6, ls=":")
    ax.axhline(0, color="#CED4DA", lw=1.2, zorder=0)
    ax.set_xlabel("Distance from landmark (cm)")
    ax.set_ylabel("Rate, speed removed")
    peak = np.nanmean(_stack(npz, df, "resid_real")[:, np.abs(x) <= 2])
    ax.text(0.03, 0.97, f"peak survives\n({peak:+.2f} at the landmark)",
            transform=ax.transAxes, va="top", fontsize=22)

    # D: the genotype test on the rate profile
    ax = axes[3]
    s = df.dropna(subset=["rate_mod"])
    superplot(ax, s, "rate_mod", genotypes=GENOTYPE_ORDER, legend=True)
    _, res = compare(s, "rate_mod")
    p_bracket(ax, res["p"], fontsize=26)
    ax.axhline(0, color="#CED4DA", lw=1.2, zorder=0)
    ax.set_ylabel("Landmark modulation\n(near - far)/(near + far)")

    # E: the decoder - position is read out better near landmarks
    ax = axes[4]
    from itertools import product

    lm = pd.read_csv(RESULTS / "landmarks.csv")
    lm = lm[lm.genotype.isin(GENOTYPE_ORDER)].dropna(subset=["error_landmark_z"])
    superplot(ax, lm, "error_landmark_z", genotypes=GENOTYPE_ORDER)
    _, res_d = compare(lm, "error_landmark_z")
    p_bracket(ax, res_d["p"], fontsize=26)
    ax.axhline(0, color="#495057", lw=1.8, zorder=0)
    pm = lm.groupby(["mouse", "genotype"], as_index=False).error_landmark_z.median()
    v = pm.error_landmark_z.to_numpy()
    stats = [np.mean(v * np.array(sg)) for sg in product([1, -1], repeat=len(v))]
    p_all = float(np.mean(np.abs(stats) >= abs(v.mean()) - 1e-12))
    ax.text(0.03, 0.03, f"{(v > 0).sum()}/{len(v)} mice above zero\np = {p_all:.4f}",
            transform=ax.transAxes, va="bottom", fontsize=23)
    ax.set_ylabel("Decoding sharper near landmarks\n(z vs shifted-comb null)")

    fig.suptitle("RSC activity is anchored to the landmarks, equally in both genotypes",
                 fontsize=34, weight="semibold", y=0.975)
    out = PLOTS / "fig33_tier3_landmarks.png"
    fig.savefig(out)
    plt.close(fig)
    for col in ["rate_mod", "resid_mod"]:
        _, r = compare(df.dropna(subset=[col]), col)
        print(f"  {col:10s} NLGF {r['mean_a']:+.3f}  WT {r['mean_b']:+.3f}  "
              f"p = {r['p']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
