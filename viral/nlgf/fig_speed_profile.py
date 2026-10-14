"""Figure 12: the animals slow at the landmarks, and what that costs Tier 3."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.chang import ChangConfig
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.style import (GENOTYPE_COLOURS, GENOTYPE_ORDER, LANDMARKS_CM,
                              set_style, strip_with_mice)

CHANG = ChangConfig()


def figure() -> None:
    s = pd.read_csv(RESULTS / "speed_profile.csv")
    z = np.load(RESULTS / "speed_profiles.npz")
    l = pd.read_csv(RESULTS / "landmarks.csv")
    l = l[l.genotype.isin(GENOTYPE_ORDER)]
    geno = dict(zip(s.mouse + "_" + s.date, s.genotype))
    mouse_of = dict(zip(s.mouse + "_" + s.date, s.mouse))
    x = CHANG.bin_centres_cm

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # Speed against position, averaged within mouse then across mice
    ax = axes[0]
    for g in GENOTYPE_ORDER:
        keys = [k for k in z.files if geno.get(k) == g]
        by_mouse = {}
        for k in keys:
            by_mouse.setdefault(mouse_of[k], []).append(z[k][0])
        stack = np.vstack([np.nanmean(np.vstack(v), axis=0) for v in by_mouse.values()])
        m = np.nanmean(stack, axis=0)
        sem = np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
        ax.plot(x, m, color=GENOTYPE_COLOURS[g], lw=2, label=f"{g} (n={stack.shape[0]})")
        ax.fill_between(x, m - sem, m + sem, color=GENOTYPE_COLOURS[g], alpha=0.18, lw=0)
    for lm in LANDMARKS_CM:
        ax.axvline(lm, color="#495057", lw=1.1, ls="--", zorder=0)
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("Running speed (cm/s)")
    ax.set_title("Speed dips at every landmark\n(dashed: 45, 90, 135 cm)", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=9)

    # The phase-shifted z, against zero
    ax = axes[1]
    per = s.groupby(["genotype", "mouse"])["speed_landmark_z"].median().reset_index()
    strip_with_mice(ax, per, "speed_landmark_z")
    ax.axhline(0, color="#495057", lw=1.1, zorder=2)
    ax.set_ylabel("Speed vs landmark distance (z)")
    ax.set_title("Slowing is landmark-locked\nWT p = 0.020, NLGF p = 0.163", fontsize=10.5)
    ax.text(0.5, -0.28, "positive = slower near landmarks", transform=ax.transAxes,
            ha="center", fontsize=8.5, color="#868E96")

    # What it costs the coding measure
    ax = axes[2]
    m = l.merge(s[["mouse", "date", "speed_landmark_z"]], on=["mouse", "date"])
    pm = m.groupby(["genotype", "mouse"])[
        ["error_landmark_z", "speed_landmark_z"]].median().reset_index()
    for g in GENOTYPE_ORDER:
        sub = pm[pm.genotype == g]
        ax.scatter(sub.speed_landmark_z, sub.error_landmark_z, s=72,
                   color=GENOTYPE_COLOURS[g], alpha=0.85, edgecolor="white",
                   linewidth=1.2, label=g)
    fit = smf.ols("error_landmark_z ~ speed_landmark_z", data=pm).fit()
    xs = np.linspace(pm.speed_landmark_z.min(), pm.speed_landmark_z.max(), 20)
    ax.plot(xs, fit.params["Intercept"] + fit.params["speed_landmark_z"] * xs,
            color="#868E96", lw=1.4, ls="--", zorder=0)
    ax.axhline(0, color="#CED4DA", lw=0.9, zorder=0)
    ax.set_xlabel("Speed vs landmark distance (z)")
    ax.set_ylabel("Decoding error vs landmark distance (z)")
    ax.set_title(f"Coding effect survives, reduced\n"
                 f"intercept {fit.params['Intercept']:+.2f} "
                 f"(was {pm.error_landmark_z.mean():+.2f})", fontsize=10.5)
    ax.legend(fontsize=9)

    fig.suptitle(
        "Animals slow at the landmarks — a dwell-time route to lower decoding error "
        "that the phase-shifted null cannot catch",
        y=1.03, fontsize=12.5, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig12_speed_landmarks.png")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    figure()
    print(f"wrote figure to {PLOTS}")
