"""Which place fields come back the next day, cell by cell, in the Ziv-style layout.

The reference day is sorted by field position, then the SAME cell ordering is applied to
the second day and each cell is coloured by whether its field came back near where it
was. A cell counts as stable if its peak moved less than STABLE_CM; the threshold is the
convention from that literature, not something tuned here, and the panel is descriptive
- the statistics live in fig39.

Two choices worth knowing about.

WHICH CELLS. Only cells with a reliable field on the REFERENCE day are shown. Reliability
is judged by splitting that day's laps odd/even and requiring the two halves to
correlate above MIN_RELIABILITY: a cell with no consistent field on day 1 cannot be said
to have lost it by day 2, and including such cells would fill the panel with blue for
reasons that have nothing to do with stability. The test uses only day-1 data, so it
cannot bias the day-2 comparison.
WHAT RED MEANS ON THE REFERENCE. Every cell is red on day 1 by construction - it is the
day the stability is measured against. The colour only carries information in the second
column.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.spatial_stability import CONFIG, _lap_maps, _rowwise_corr, _smooth
from viral.nlgf.style import set_style

STABLE_CM = 20.0
MIN_RELIABILITY = 0.3
MAX_CELLS = 300  # keep the panels readable; drawn as an evenly spaced subset
# RSC fields are broad - about 16 cm on a 140 cm track - so a gain that suits sharp
# hippocampal fields buries the diagonal here. Fewer rows and a lower gain keep it.
# down the colormap so only real fields stay bright
NORM = "zscore"  # "minmax" or "zscore" - see _norm
# minmax needs VMAX > 1 to push its manufactured background peaks down the colormap;
# zscore suppresses them at source, so the ceiling is a real number of SDs instead.
# z-scored maps are negative for roughly half their bins, so a floor at 0 clips all of
# them to black and the panel binarises. Median peak is ~2.1 SD, so -1 to 3 spans the
# real range.
VMIN, VMAX = (0.5, 1.6) if NORM == "zscore" else (0.0, 0.5)
CACHE = Path.home() / ".cache" / "viral_scratch" / "stable_unstable.npz"

PAIRS = [
    ("JB026", "2024-12-15", "2024-12-16", "WT"),
    ("JB030", "2025-03-13", "2025-03-14", "WT"),
    ("JB033", "2025-03-18", "2025-03-19", "WT"),
    ("J030", "2026-05-18", "2026-05-19", "NLGF"),
    ("J030", "2026-05-11", "2026-05-13", "NLGF"),
    ("J031", "2026-05-21", "2026-05-22", "NLGF"),
]
RED = LinearSegmentedColormap.from_list("red", ["white", "#C92A2A"])
BLUE = LinearSegmentedColormap.from_list("blue", ["white", "#3B5BDB"])


def _norm(m: np.ndarray, how: str = "minmax") -> np.ndarray:
    """Per-cell normalisation.

    minmax  every cell's peak becomes 1 and its trough 0, whatever its tuning strength.
            A cell with no field still shows a bright "peak", which is why the
            background looks noisy and why VMAX has to be raised to suppress it.
    zscore  each bin in SDs above that cell's own mean, so a sharply tuned cell is
            bright and a weakly tuned one stays dim. Suppresses the same noise honestly
            rather than by rescaling the colormap, at the cost of brightness now partly
            encoding tuning strength rather than position alone.

    Either way the stability numbers are unchanged: both are affine per cell, so the
    argmax the peak shift is computed from is identical.
    """
    if how == "zscore":
        mu = m.mean(axis=1, keepdims=True)
        sd = m.std(axis=1, keepdims=True)
        return np.divide(m - mu, sd, out=np.zeros_like(m), where=sd > 0)
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    return np.divide(m - lo, hi - lo, out=np.zeros_like(m), where=(hi - lo) > 0)


def pair_maps(mouse: str, d1: str, d2: str):
    """Rate maps for a registered pair, cached - each call reads two sessions of spks.

    The cache holds the FULL sorted set of RAW maps. Caching after the MAX_CELLS
    subsample froze that parameter; caching after normalisation would freeze NORM the
    same way. Anything tunable is applied at draw time.
    """
    key = f"{mouse}_{d1}_{d2}"
    if CACHE.exists():
        z = np.load(CACHE)
        if f"{key}_m1" in z:
            return z[f"{key}_m1"], z[f"{key}_m2"], z[f"{key}_s"].astype(bool)
    matches = pd.read_csv(RESULTS / f"matches_{mouse}_{d1}_{d2}.csv")
    laps1 = _lap_maps(mouse, d1, matches.row_a.to_numpy())
    laps2 = _lap_maps(mouse, d2, matches.row_b.to_numpy())
    if laps1 is None or laps2 is None:
        return None
    # reliability from day 1 only, so the filter cannot touch the day-2 comparison
    odd, even = _smooth(laps1[0::2]), _smooth(laps1[1::2])
    reliable = _rowwise_corr(odd, even) > MIN_RELIABILITY
    m1, m2 = _smooth(laps1), _smooth(laps2)
    keep = reliable & np.isfinite(m1).all(axis=1) & np.isfinite(m2).all(axis=1)
    m1, m2 = m1[keep], m2[keep]
    order = np.argsort(np.argmax(m1, axis=1))
    m1, m2 = m1[order], m2[order]
    shift = np.abs(np.argmax(m1, axis=1) - np.argmax(m2, axis=1)) * CONFIG.bin_size
    out = (m1, m2, shift < STABLE_CM)  # raw; normalised at draw time
    store = dict(np.load(CACHE)) if CACHE.exists() else {}
    store.update({f"{key}_m1": out[0], f"{key}_m2": out[1], f"{key}_s": out[2]})
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **store)
    return out


def _draw(ax, m: np.ndarray, stable: np.ndarray, title: str) -> None:
    """One row per cell, single colormap, no stable/unstable split.

    Two alternatives were tried and rejected. Filled tuning curves, as the hippocampal
    literature draws them, need sharp fields: RSC fields are about 16 cm on a 140 cm
    track and often have secondary bumps, so at any gain that makes the peaks visible
    the curves merge and the sorted diagonal disappears. Colouring each row red or blue
    by whether its field returned was legible but busy - the eye reads the hue before
    the position, which is the wrong way round when the panel's job is to show WHERE
    fields sit and how that changes. The fraction returning is quoted in the title
    instead, where it does not compete with the image.
    """
    ax.imshow(
        m,
        aspect="auto",
        cmap="magma",
        vmin=VMIN,
        vmax=VMAX,
        extent=[CONFIG.start, CONFIG.end, m.shape[0], 0],
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Position (cm)")


# The clearest pair in each genotype, for a talk slide. Deliberately NOT representative
# - these are the extremes of a distribution whose medians are 0.538 and 0.433, so the
# gap shown is about four times the group difference. Fine as an illustration when it is
# described as one; the group figure is fig39.
# Both 1-day gaps, so the contrast is not interval. Chosen on the metric the panel
# actually shows - the fraction of fields returning - rather than on cross-day
# correlation: 62% against 37%, the widest gap-matched pair available.
EXTREMES = [
    ("JB030", "2025-03-24", "2025-03-25", "WT"),
    ("JB036", "2025-07-07", "2025-07-08", "NLGF"),
]


def extremes() -> None:
    set_style(talk=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 20))
    for i, (mouse, d1, d2, g) in enumerate(EXTREMES):
        out = pair_maps(mouse, d1, d2)
        m1, m2, stable = out
        m1, m2 = _norm(m1, NORM), _norm(m2, NORM)
        if len(m1) > MAX_CELLS:
            take = np.linspace(0, len(m1) - 1, MAX_CELLS).astype(int)
            m1, m2, stable = m1[take], m2[take], stable[take]
        _draw(axes[i, 0], m1, stable, f"{g}   Day 1")
        _draw(axes[i, 1], m2, stable, f"Day 2  ·  {stable.mean():.0%} of fields return")
        axes[i, 0].set_ylabel("Place cell")
        axes[i, 1].set_yticklabels([])
        for ax in axes[i]:
            ax.title.set_fontsize(26)
        print(f"  {g:5s} {mouse} {d1}->{d2}: {stable.mean():.0%} stable")
    fig.suptitle(
        "Place fields return the next day in wild-type, less so under amyloid",
        fontsize=30,
        weight="semibold",
        y=0.965,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = PLOTS / "fig42_stability_extremes.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    set_style()
    fig, axes = plt.subplots(2, 6, figsize=(21, 10))
    for i, (mouse, d1, d2, g) in enumerate(PAIRS):
        out = pair_maps(mouse, d1, d2)
        row, col = divmod(i, 3)
        ax_a, ax_b = axes[row, col * 2], axes[row, col * 2 + 1]
        if out is None:
            ax_a.axis("off")
            ax_b.axis("off")
            continue
        m1, m2, stable = out
        m1, m2 = _norm(m1, NORM), _norm(m2, NORM)
        if len(m1) > MAX_CELLS:  # evenly spaced through the sorted order
            take = np.linspace(0, len(m1) - 1, MAX_CELLS).astype(int)
            m1, m2, stable = m1[take], m2[take], stable[take]
        _draw(ax_a, m1, np.ones(len(m1), bool), f"{g}  {mouse}\nDay 1 (reference)")
        _draw(ax_b, m2, stable, f"Day 2  ·  {stable.mean():.0%} stable")
        ax_a.set_ylabel("Place cell" if col == 0 else "")
        ax_b.set_yticklabels([])
        print(
            f"  {g:5s} {mouse} {d1}->{d2}: {len(m1)} reliable cells, "
            f"{stable.mean():.0%} stable"
        )

    fig.suptitle(
        "Place fields on two consecutive days  ·  same registered cells, "
        f"ordered by day 1  ·  'stable' = peak moved < {STABLE_CM:.0f} cm",
        fontsize=17,
        weight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = PLOTS / "fig41_stable_unstable.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    extremes()
    #     extremes()
    # if "--extremes" in sys.argv:
    #     extremes()
    # else:
    #     main()
