"""Shared plot styling, so every figure in this package reads as one set."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

# Genotype is the only categorical axis that appears in every figure, so it gets a
# fixed mapping. WT is the neutral reference; NLGF is the one the eye should find.
GENOTYPE_COLOURS = {
    "WT": "#4C6EF5",
    "NLGF": "#E8590C",
    "Oligo-BACE1-KO": "#0CA678",
    "Neuronal-BACE1-KO": "#AE3EC9",
}
GENOTYPE_ORDER = ["WT", "NLGF"]

# Axis labels only. The full names are too wide to sit side by side under a two-box
# plot; the legend and titles keep the full name.
SHORT_NAME = {"Oligo-BACE1-KO": "Oligo-KO", "Neuronal-BACE1-KO": "Neuronal-KO"}

# Text drawn by hand (p-brackets, the SuperPlot legend) sits outside rcParams, so it
# would stay small when set_style(talk=True) scales everything else. set_style records
# the scale here and those helpers multiply by it.
_SCALE = 1.0


def text_scale() -> float:
    return _SCALE

REWARDED_COLOUR = "#1C7ED6"
UNREWARDED_COLOUR = "#F03E3E"

# Corridor geometry
LANDMARKS_CM = (45.0, 90.0, 135.0)
AZ_START_CM = 150.0
CORRIDOR_END_CM = 180.0


def set_style(talk: bool = False) -> None:
    """Shared rcParams. `talk=True` scales every text element for projection.

    One switch rather than per-figure font sizes, so a talk figure and its paper
    counterpart cannot drift apart. Marker and line sizes scale too: type alone growing
    turns a figure into text with a small chart in it.
    """
    global _SCALE
    scale = 1.7 if talk else 1.0
    _SCALE = scale
    mpl.rcParams.update(
        {
            "lines.linewidth": 1.5 * scale,
            "lines.markersize": 6 * scale,
            "patch.linewidth": 1.0 * scale,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0 * scale,
            "axes.labelsize": 16 * scale,
            "axes.titlesize": 16 * scale,
            "axes.titleweight": "semibold",
            "axes.titlepad": 10 * scale,
            "xtick.labelsize": 16 * scale,
            "ytick.labelsize": 16 * scale,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 16 * scale,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def landmark_lines(ax, colour: str = "#ADB5BD", label_az: bool = True) -> None:
    """Mark the three tactile landmarks and the reward zone on a position axis."""
    for x in LANDMARKS_CM:
        ax.axvline(x, color=colour, lw=0.8, ls=":", zorder=0)
    ax.axvspan(
        AZ_START_CM, CORRIDOR_END_CM, color="#FFF3BF", alpha=0.55, lw=0, zorder=0
    )
    if label_az:
        ax.text(
            (AZ_START_CM + CORRIDOR_END_CM) / 2,
            ax.get_ylim()[1],
            "AZ",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#868E96",
        )


def p_bracket(
    ax,
    p_value: float,
    x0: float = 0,
    x1: float = 1,
    pad_frac: float = 0.06,
    height_frac: float = 0.035,
    label: str | None = None,
    fontsize: float | None = None,
) -> None:
    """Significance bracket spanning two groups, with the p above it.

    Drawn above everything already plotted, and the y limit is extended to make room,
    so it never lands on a data point however the values happen to be distributed.
    Calling it twice on one axis stacks the second bracket above the first, which is
    what three-group panels need: two brackets at the same height would collide on the
    middle group they share.

    `label` prefixes the p, for panels carrying more than one contrast.
    """
    # On a log axis, evenly spaced positions have to be computed in log space or the
    # bracket lands far above the data (or below the axis) - the arithmetic that spaces
    # it is linear in whatever coordinates it is given.
    log = ax.get_yscale() == "log"
    lo, hi = ax.get_ylim()
    if log:
        llo, lhi = np.log10(max(lo, 1e-12)), np.log10(hi)
        lspan = lhi - llo
        ly = lhi + pad_frac * lspan
        lh = height_frac * lspan
        y, h_top = 10 ** ly, 10 ** (ly + lh)
        ax.plot([x0, x0, x1, x1], [y, h_top, h_top, y], lw=1.1 * _SCALE,
                color="#495057", clip_on=False, zorder=6)
        text_y = 10 ** (ly + lh * 1.15)
        top = 10 ** (ly + lh * 3.2)
    else:
        span = hi - lo
        y = hi + pad_frac * span
        h = height_frac * span
        ax.plot([x0, x0, x1, x1], [y, y + h, y + h, y], lw=1.1 * _SCALE,
                color="#495057", clip_on=False, zorder=6)
        text_y = y + h * 1.15
        top = y + h * 3.2

    text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.2f}"
    if label:
        text = f"{label}  {text}"
    ax.text((x0 + x1) / 2, text_y, text, ha="center", va="bottom",
            fontsize=fontsize if fontsize else (9.5 if not label else 8.5) * _SCALE,
            color="#212529", clip_on=False, zorder=6)
    ax.set_ylim(lo, top)


def superplot(
    ax,
    sessions,
    value: str,
    p_value: float | None = None,
    genotypes=GENOTYPE_ORDER,
    show_box: bool = True,
    legend: bool = False,
) -> None:
    """Sessions as faint dots, animals as solid markers, box over the animal means.

    The SuperPlot convention (Lord et al. 2020, J Cell Biol) for nested data: the
    replicate level is visible so the reader can see the spread and the within-animal
    consistency, but the ANIMAL level is what is drawn boldly and what the statistics
    are computed on. Session dots are coloured by mouse so a reader can follow one
    animal's sessions.

    The distinction matters: 104 sessions come from 15 animals, and every p-value in
    this package is an exact permutation over those 15. Drawing sessions as if they
    were the unit would imply an n seven times larger than the one being tested, which
    is why the animal markers are large and opaque and the session dots are not.

    `sessions` must be session-level rows with columns genotype, mouse and `value`.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    for i, g in enumerate(genotypes):
        sub = sessions[(sessions.genotype == g)].dropna(subset=[value])
        if sub.empty:
            continue
        colour = GENOTYPE_COLOURS[g]
        per_mouse = sub.groupby("mouse")[value].median()

        if show_box and per_mouse.size > 2:
            bp = ax.boxplot(
                [per_mouse.to_numpy()],
                positions=[i],
                widths=0.30,
                showfliers=False,
                patch_artist=True,
                zorder=1,
            )
            for patch in bp["boxes"]:
                patch.set(facecolor=colour, alpha=0.13, edgecolor=colour, linewidth=1.2)
            for part in ("whiskers", "caps"):
                for artist in bp[part]:
                    artist.set(color=colour, linewidth=1.1)
            for med in bp["medians"]:
                med.set(color=colour, linewidth=2.2)

        # Sessions: small, faint, offset so they do not sit under the animal markers
        for j, (mouse, rows) in enumerate(sub.groupby("mouse")):
            v = rows[value].to_numpy(dtype=float)
            ax.scatter(
                i - 0.30 + rng.uniform(-0.055, 0.055, v.size),
                v,
                s=13,
                color=colour,
                alpha=0.30,
                edgecolor="none",
                zorder=2,
            )

        # Animals: large, opaque, the unit of analysis
        ax.scatter(
            i + 0.24 + rng.uniform(-0.05, 0.05, per_mouse.size),
            per_mouse.to_numpy(),
            s=58,
            color=colour,
            alpha=0.95,
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )

    ax.set_xticks(range(len(genotypes)))
    ax.set_xticklabels(
        [
            f"{SHORT_NAME.get(g, g)}\n"
            # f"{sessions[sessions.genotype == g].mouse.nunique()} mice\n"
            # f"{int((sessions.genotype == g).sum())} sessions"
            for g in genotypes
        ]
    )
    ax.set_xlim(-0.6, len(genotypes) - 0.4)

    if p_value is not None and len(genotypes) == 2:
        p_bracket(ax, p_value)

    if legend:
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=4 * _SCALE,
                markerfacecolor="#868E96",
                markeredgecolor="none",
                alpha=0.45,
                label="session",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=8 * _SCALE,
                markerfacecolor="#495057",
                markeredgecolor="white",
                markeredgewidth=1.1,
                label="mouse (unit of test)",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="best",
            fontsize=8.5 * _SCALE,
            handletextpad=0.4,
            borderpad=0.4,
            labelspacing=0.35,
        )


def strip_with_mice(ax, per_mouse, value: str, genotypes=GENOTYPE_ORDER) -> None:
    """One dot per mouse, with the group mean and a 95% interval behind it.

    Kept for panels whose input is already collapsed to animals. Prefer superplot()
    where session-level rows are available.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    for i, g in enumerate(genotypes):
        v = per_mouse.loc[per_mouse.genotype == g, value].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        colour = GENOTYPE_COLOURS[g]
        ax.scatter(
            i + rng.uniform(-0.09, 0.09, v.size),
            v,
            s=46,
            color=colour,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.1,
            zorder=3,
        )
        ax.hlines(v.mean(), i - 0.26, i + 0.26, color=colour, lw=2.4, zorder=4)
        if v.size > 1:
            sem = v.std(ddof=1) / np.sqrt(v.size)
            ax.vlines(
                i,
                v.mean() - 1.96 * sem,
                v.mean() + 1.96 * sem,
                color=colour,
                lw=1.4,
                zorder=2,
            )
    ax.set_xticks(range(len(genotypes)))
    ax.set_xticklabels(
        [f"{g}\n(n={int((per_mouse.genotype==g).sum())})" for g in genotypes]
    )
    ax.set_xlim(-0.6, len(genotypes) - 0.4)
