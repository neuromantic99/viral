"""Figure 1: behaviour. Is the task performed equivalently by genotype?

Two claims, deliberately kept apart because they point opposite ways:
  - discrimination (what the animal knows) is indistinguishable
  - locomotion (what the animal does) is not

The second matters mainly as a covariate. Place coding, synchronous event rate and
offline reactivation all depend on running speed, so a genotype that runs faster needs
speed controlled before any of those differences can be read as coding differences.

All panels are restricted to the 2P rig. Speeds are in cm/s derived from the rig's
wheel circumference (2P 34.7, box 53.4) and rig is not balanced across genotype.
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
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, set_style, strip_with_mice

STAGES = [("learning", "Learning"), ("reversal", "Reversal"),
          ("recall", "Recall"), ("recall_reversal", "Recall reversal")]


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "behaviour_full.csv")
    return df[df.genotype.isin(GENOTYPE_ORDER) & (df.rig == "2P")]


def _curve(ax, df, value, stage, max_session=8, min_mice=3):
    """Curves are truncated at the last session index where BOTH genotypes still have
    min_mice contributing. Beyond that the two lines are drawn from different animals
    and an apparent divergence is a change in who is left, not a change in behaviour."""
    d = df[(df.stage == stage) & (df.session_in_stage < max_session)]

    counts = (d.groupby(["genotype", "session_in_stage"])["mouse"].nunique()
              .unstack("genotype").reindex(columns=GENOTYPE_ORDER).fillna(0))
    ok = counts[(counts >= min_mice).all(axis=1)].index
    if len(ok) == 0:
        return
    last = int(ok.max())

    for g in GENOTYPE_ORDER:
        sub = d[(d.genotype == g) & (d.session_in_stage <= last)]
        # Mean over mice at each session index, so a mouse with many sessions does not
        # dominate the curve
        per_mouse = sub.groupby(["session_in_stage", "mouse"])[value].mean().reset_index()
        stat = per_mouse.groupby("session_in_stage")[value].agg(["mean", "sem", "size"])
        if stat.empty:
            continue
        x = stat.index.to_numpy()
        ax.plot(x, stat["mean"], color=GENOTYPE_COLOURS[g], lw=2, marker="o",
                ms=4.5, label=f"{g}", zorder=3)
        ax.fill_between(x, stat["mean"] - stat["sem"], stat["mean"] + stat["sem"],
                        color=GENOTYPE_COLOURS[g], alpha=0.18, lw=0)
    ax.axhline(0, color="#CED4DA", lw=0.9, ls="--", zorder=0)
    ax.set_xlabel("Session within stage")
    ax.set_xticks(range(last + 1))
    n_txt = "  ".join(
        f"{int(counts.loc[i, GENOTYPE_ORDER[0]])}/{int(counts.loc[i, GENOTYPE_ORDER[1]])}"
        for i in range(last + 1)
    )
    ax.text(0.5, -0.30, f"mice WT/NLGF   {n_txt}", transform=ax.transAxes,
            ha="center", fontsize=8, color="#868E96")


def figure_discrimination(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))

    for ax, (stage, title) in zip(axes[0], STAGES[:3]):
        _curve(ax, df, "licking_dprime", stage)
        ax.set_title(title)
        ax.set_ylabel("Licking d'" if ax is axes[0][0] else "")
    axes[0][0].legend(loc="lower right")

    for ax, value, label in zip(
        axes[1],
        ["licking_dprime", "speed_dprime", "lick_rate_unrewarded"],
        ["Licking d'", "Speed d'", "P(lick | unrewarded)"],
    ):
        per_mouse, res = compare(df, value, how="median")
        strip_with_mice(ax, per_mouse, value)
        ax.set_ylabel(label)
        ax.set_title(f"{label}\ndiff {res['diff']:+.2f},  p = {res['p']:.3f}",
                     fontsize=10.5)
        if value != "lick_rate_unrewarded":
            ax.axhline(0, color="#CED4DA", lw=0.9, ls="--", zorder=0)

    fig.suptitle(
        "Discrimination learning is intact in NLGF  ·  2P rig only  ·  "
        "sessions faint, mouse means solid; exact permutation over mice",
        y=1.005, fontsize=12.5, weight="semibold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS / "fig1_behaviour_discrimination.png")
    plt.close(fig)


def figure_locomotion(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))

    for ax, value, label in zip(
        axes[:3],
        ["mean_trial_speed", "median_trial_time", "n_trials"],
        ["Mean trial speed (cm/s)", "Median trial time (s)", "Trials per session"],
    ):
        per_mouse, res = compare(df, value, how="median")
        strip_with_mice(ax, per_mouse, value)
        ax.set_ylabel(label)
        ax.set_title(f"{label}\ndiff {res['diff']:+.2f},  p = {res['p']:.3f}",
                     fontsize=10.5)

    # Speed by stage, where the effect is clearest
    ax = axes[3]
    width = 0.36
    for i, g in enumerate(GENOTYPE_ORDER):
        means, sems, xs = [], [], []
        for j, (stage, _) in enumerate(STAGES):
            pm = (df[(df.stage == stage) & (df.genotype == g)]
                  .groupby("mouse")["mean_trial_speed"].median())
            if pm.size < 3:
                continue
            means.append(pm.mean())
            sems.append(pm.std(ddof=1) / np.sqrt(pm.size))
            xs.append(j + (i - 0.5) * width)
        ax.bar(xs, means, width=width, yerr=sems, capsize=3,
               color=GENOTYPE_COLOURS[g], alpha=0.9, label=g,
               error_kw=dict(lw=1.2, ecolor="#495057"))
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([t for _, t in STAGES], rotation=25, ha="right")
    ax.set_ylabel("Mean trial speed (cm/s)")
    ax.set_title("Speed by stage", fontsize=10.5)
    ax.legend()

    fig.suptitle(
        "NLGF run faster at matched discrimination: consistent in direction across "
        "all four stages, marginal on any single test",
        y=1.04, fontsize=12.5, weight="semibold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS / "fig2_behaviour_locomotion.png")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    df = load()
    figure_discrimination(df)
    figure_locomotion(df)
    print(f"wrote figures to {PLOTS}")
