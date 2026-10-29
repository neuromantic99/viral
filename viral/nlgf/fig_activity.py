"""Figure 3: excitability and synchrony.

Layout follows the logic of the test rather than the data:
  row 1  firing rate, in quiescence (clean) and running (speed-confounded)
  row 2  synchrony, on absolute-scale measures, with the synchronous-event rate last and
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
from viral.nlgf.style import (
    GENOTYPE_COLOURS,
    GENOTYPE_ORDER,
    set_style,
    superplot,
    p_bracket,
)

# Fixed-sequence (gatekeeping) testing. The rescue contrast is only meaningful where
# there is a deficit to rescue, so it is tested at ALPHA only if the deficit is
# significant at ALPHA. This holds the familywise error at ALPHA for the pair without
# inflating either p, which Holm would (it doubles the smaller p at m = 2).
ALPHA = 0.05

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
    ("still_sce_rate_hz", "Synchronous event rate (Hz)", "Synchronous events"),
]


def load() -> pd.DataFrame:
    """Tier 1, with the graded co-fluctuation measure merged in.

    `still_graded_cofluct` comes from activity_graded.py: the same population
    co-fluctuation quantity, but scored on the denoised trace rather than on spks > 0.
    Binarising costs roughly a factor of three in sensitivity here, so the graded column
    is the measure and the binary one (still_mua_fano) is kept only for the supplement.
    """
    df = pd.read_csv(RESULTS / "activity_sessions.csv")
    graded = RESULTS / "activity_graded.csv"
    if graded.exists():
        g = pd.read_csv(graded)[
            [
                "mouse",
                "date",
                "still_graded_cofluct",
                "still_graded_mean_r",
                "run_graded_mean_r",
            ]
            # 29/10/2026: seed of the random cell subsample, shown in the figure title;
            # absent in CSVs made before the seed fix in activity_graded.py
            + (["subsample_seed"] if "subsample_seed" in pd.read_csv(graded, nrows=0).columns else [])
        ]
        df = df.merge(g, on=["mouse", "date"], how="left")
    return df[df.genotype.isin(GENOTYPE_ORDER)]


# 29/10/2026 JB: outdated
# Four panels for the talk: is the cell firing normally, is it firing normally while
# running, are synchronous events as frequent - and are cells still fluctuating together.
# Three panels. Synchronous-event rate is dropped: explaining what an SCE is costs more
# talk time than the null is worth. Mean pairwise correlation replaces the
# co-fluctuation index - they are the SAME quantity (index = 1 + (n-1)r, and both give
# p = 0.034), so showing both would present one result twice under two names. The
# correlation is the more familiar of the two.
# Ordered running-then-rest within each measure, so the two pairs line up and the slide
# ends on the one panel that differs. The rest/running contrast is the argument: during
# running, shared sensory and motor drive imposes correlation from outside and would
# mask any intrinsic coordination deficit; at rest, correlation is generated internally.
# An effect that respects that boundary is hard to explain as an artefact.
PANELS_TALK = [
    ("run_rate_median", "Event rate (Hz)", "Firing rate, running"),
    ("still_rate_median", "Event rate (Hz)", "Firing rate, rest"),
    ("run_graded_mean_r", "Mean pairwise correlation", "Population coupling, running"),
    ("still_graded_mean_r", "Mean pairwise correlation", "Population coupling, rest"),
]
# Panels that must share a y-axis, so the reader can compare them directly rather than
# across two silently different scales.
SHARED_Y = [
    ("run_rate_median", "still_rate_median"),
    ("run_graded_mean_r", "still_graded_mean_r"),
]
# Panels whose session distribution is heavy-tailed enough to flatten every box on a
# linear axis. A log axis shows the spread without dropping any point; the test runs on
# per-mouse medians and is unaffected by the scale.
LOG_PANELS: set = set()


def _panel(ax, df, value, label, title, legend: bool = False) -> dict:
    # deficit is NLGF vs WT
    _, deficit = compare(df, value, how="median", group_a="NLGF", group_b="WT")
    # rescue is Oligo-BACE1-KO vs NLGF
    _, rescue = compare(
        df, value, how="median", group_a="Oligo-BACE1-KO", group_b="NLGF"
    )
    rescue_tested = deficit["p"] < ALPHA
    # the scale must be set before superplot draws, so the bracket is placed in the
    # coordinates it will actually be shown in
    if value in LOG_PANELS:
        ax.set_yscale("log")
    superplot(ax, df, value, legend=legend)

    p_bracket(ax=ax, p_value=deficit["p"], x0=0, x1=1)
    p_bracket(
        ax=ax,
        p_value=rescue["p"],
        x0=1,
        x1=2,
        text=None if rescue_tested else "not tested (no deficit)",
    )

    ax.set_ylabel(label)
    ax.set_title(title)
    return dict(
        metric=value,
        wt=deficit["mean_b"],  # mean of WT per-mouse medians
        nlgf=deficit["mean_a"],  # mean of NLGF per-mouse medians
        oligo=rescue["mean_a"],  # mean of Oligo per-mouse medians
        deficit_diff=deficit["diff"],  # NLGF − WT
        deficit_p=deficit["p"],
        rescue_diff=rescue["diff"],  # Oligo − NLGF
        rescue_p=rescue["p"],  # reported, but only interpretable where rescue_tested
        rescue_tested=rescue_tested,
        n_wt=deficit["n_b"],
        n_nlgf=deficit["n_a"],
        n_oligo=rescue["n_a"],
    )


def figure(
    df: pd.DataFrame,
    panels=None,
    out: str = "fig3_rsc_activity_synchrony.png",
    title: str | None = None,
    shape=None,
) -> pd.DataFrame:
    panels = panels if panels is not None else PANELS_TALK
    rows, cols = shape if shape else (1, len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 7.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    results = []
    for i, (ax, (value, label, panel_title)) in enumerate(zip(axes, panels)):
        if value in df.columns:
            results.append(
                _panel(
                    ax,
                    df.dropna(subset=[value]),
                    value,
                    label,
                    panel_title,
                    legend=(i == 0),
                )
            )
        else:
            ax.axis("off")

    # equalise the y-axis within each measure pair
    by_value = {v: ax for ax, (v, _, _) in zip(axes, panels)}
    for group in SHARED_Y:
        present = [by_value[v] for v in group if v in by_value]
        if len(present) > 1:
            lo = min(a.get_ylim()[0] for a in present)
            hi = max(a.get_ylim()[1] for a in present)
            for a in present:
                a.set_ylim(lo, hi)

    n_nlgf = df[df.genotype == "NLGF"].mouse.nunique()
    n_wt = df[df.genotype == "WT"].mouse.nunique()
    n_oligo = df[df.genotype == "Oligo-BACE1-KO"].mouse.nunique()
    # 29/10/2026 (Claude Opus 5.5, medium effort, at JB's request): show the random
    # subsample in the title. The seed is read from activity_graded.csv itself, so the
    # title cannot claim a seed the coupling values were not made with; a CSV from before
    # the seed fix has no `subsample_seed` column and gets a warning instead.
    n_cells = int(df["n_cells"].max()) if "n_cells" in df.columns else "?"
    if "subsample_seed" in df.columns and df["subsample_seed"].notna().any():
        seeds = sorted(int(x) for x in df["subsample_seed"].dropna().unique())
        if seeds == [0]:
            seed_line = (f"\n one random subsample of {n_cells} cells per session"
                         "\n fixed seed 0, same cells in all panels")
        else:
            seed_line = (f"\n one random subsample of {n_cells} cells per session"
                         f"\n seed {seeds} (coupling) / 0 (rate)")
    else:
        seed_line = ("\n WARNING: activity_graded.csv predates the seed fix"
                     "\n (29/10/2026): coupling not reproducible, re-run it")
    fig.suptitle(
        # (title or "RSC fires normally in NLGF, but coordinates less during rest")
        # + f"\n{n_nlgf} NLGF / {n_wt} WT mice  ·  "
        #   "small dots sessions, large dots mice",
        "p: exact permutation over mice, uncorrected across measures"
        + f"\n Oligo vs NLGF tested only if deficit p < 0.05"
        + f"\n {n_nlgf} NLGF / {n_wt} WT / {n_oligo} Oligo-BACE1-KO mice"
        + f"\n small dots sessions, large dots mice"
        + seed_line,
        y=1.04,
        fontsize=34,
        weight="semibold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS / out)
    plt.close(fig)
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
    plt.close(fig)


if __name__ == "__main__":
    set_style(talk=True)
    df = load()
    # 2x2: rate on the top row, coupling on the bottom, running then rest in each
    res = figure(df, shape=(2, 2))
    # the full eight-panel version stays available for the supplement
    figure(
        df,
        panels=PANELS_RATE + PANELS_SYNC,
        shape=(2, 4),
        out="figS1_activity_synchrony_full.png",
        title="RSC excitability and synchrony, all measures",
    )
    figure_rate_distribution(df)
    print(res.to_string(index=False))
    print(f"\nwrote figures to {PLOTS}")
