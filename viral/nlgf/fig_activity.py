"""Figure 3: excitability and synchrony.

Layout follows the logic of the test rather than the data:
  row 1  firing rate, in quiescence (clean) and running (speed-confounded)
  row 2  synchrony, on absolute-scale measures, with Chang's SCE rate last and
         labelled as self-normalising
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
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, set_style, superplot

PANELS_RATE = [
    ("still_rate_median", "Event rate (Hz)", "Firing rate (rest)"),
    ("still_rate_p90", "Event rate (Hz)", "Firing rate (rest) — 90th Percentile"),
    ("still_frac_silent", "Fraction silent cells", "Silent population"),
    ("run_rate_median", "Event rate (Hz)", "Firing rate (running)"),
]
PANELS_SYNC = [
    ("still_mean_pairwise_r", "Mean pairwise r", "Population coupling"),
    ("still_mua_fano", "MUA Fano factor", "Population co-fluctuation"),
    (
        "still_frac_frames_5pct_coactive",
        "P(>5% cells co-active)",
        "Absolute co-activation",
    ),
    ("still_sce_rate_hz", "SCE rate (Hz)", "Chang SCE — self-normalising"),
]


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "activity_sessions.csv")
    return df[df.genotype.isin(GENOTYPE_ORDER)]


def _panel(ax, df, value, label, title, legend: bool = False) -> dict:
    per_mouse, res = compare(df, value, how="median")
    superplot(ax, df, value, legend=legend, p_value=res["p"])
    ax.set_ylabel(label)
    ax.set_title(title, fontsize=10.5)
    return dict(metric=value, **res)


def figure(df: pd.DataFrame) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    results = []
    for i, (ax, (value, label, title)) in enumerate(zip(axes[0], PANELS_RATE)):
        if value in df.columns:
            results.append(_panel(ax, df, value, label, title, legend=(i == 0)))
    for ax, (value, label, title) in zip(axes[1], PANELS_SYNC):
        if value in df.columns:
            results.append(_panel(ax, df, value, label, title))

    n_nlgf = df[df.genotype == "NLGF"].mouse.nunique()
    n_wt = df[df.genotype == "WT"].mouse.nunique()
    fig.suptitle(
        f"RSC excitability and synchrony  ·  {len(df)} sessions, "
        f"{n_nlgf} NLGF / {n_wt} WT mice  ·  one point per mouse, "
        f"exact permutation over mice",
        y=1.02,
        fontsize=12.5,
        weight="semibold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS / "fig3_activity_synchrony.png")
    # plt.close(fig)
    return pd.DataFrame(results)


def figure_rate_distribution(df: pd.DataFrame) -> None:
    """The distribution itself, not just its summary.

    A hyperactive subpopulation and a uniform rate increase give the same median but
    different distributions, and only the first is the amyloid phenotype.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, epoch in zip(axes, ["still", "run"]):
        for g in GENOTYPE_ORDER:
            sub = df[df.genotype == g]
            per_mouse = sub.groupby("mouse")[
                [f"{epoch}_rate_median", f"{epoch}_rate_p90", f"{epoch}_frac_silent"]
            ].median()
            ax.scatter(
                per_mouse[f"{epoch}_rate_median"],
                per_mouse[f"{epoch}_rate_p90"],
                s=70,
                color=GENOTYPE_COLOURS[g],
                alpha=0.85,
                edgecolor="white",
                linewidth=1.2,
                label=g,
            )
        lims = ax.get_xlim()
        ax.plot(lims, lims, color="#CED4DA", lw=0.9, ls="--", zorder=0)
        ax.set_xlabel(f"{epoch} rate, median cell (Hz)")
        ax.set_ylabel(f"{epoch} rate, 90th percentile cell (Hz)")
        ax.set_title(f"{epoch}: tail versus centre", fontsize=10.5)
    axes[0].legend()
    fig.suptitle(
        "Is any rate difference a shift of the whole distribution, or a heavy tail?",
        y=1.03,
        fontsize=12,
        weight="semibold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS / "fig4_rate_distribution.png")
    # plt.close(fig)


if __name__ == "__main__":
    set_style()
    df = load()
    res = figure(df)
    figure_rate_distribution(df)
    1 / 0
    print(res.to_string(index=False))
    print(f"\nwrote figures to {PLOTS}")
