"""Example raster: population activity during in-task immobility.

The synchrony measures come from viral.nlgf.immobility - stillness ANYWHERE in the
session, not the wheel-freeze epochs. That matters for how this is drawn: in-task
immobility is fragmented, scattered between and within trials, so concatenating the
retained frames would manufacture adjacencies that never happened. Instead a real
contiguous stretch of the session is shown, with the immobility periods shaded, which
also puts the events in behavioural context - running, then stillness, then events.

Two things this panel got wrong on the first attempt, both worth keeping written down.

The window was the longest single immobility bout, which contained no events at all -
correctly, since events occur at ~0.89/min and the longest bout is 10-30 s, so the
expected count is well under one. The window is now chosen to MAXIMISE the number of
detected events, spanning several bouts if need be.

Cells were ordered by time of first spike within the bout, which manufactures a
descending diagonal out of noise for exactly the reason the snake plots are sorted
split-half. Cells are now left in their arbitrary suite2p order: a real synchronous
event is a VERTICAL stripe and needs no sorting to be visible, whereas any ordering
chosen from the displayed data invents horizontal structure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.chang import ChangConfig, detect_sce, smoothed_rates
from viral.nlgf.immobility import immobility_mask
from viral.nlgf.paths import PLOTS, spks_path
from viral.nlgf.style import GENOTYPE_COLOURS, set_style

CHANG = ChangConfig()
FS = 30
PAD_S = 6.0
N_CELLS = 250

EXAMPLES = {"WT": ("J035", "2026-06-18"), "NLGF": ("JB036", "2025-07-08")}


WINDOW_S = 100.0


def best_window(mask: np.ndarray, events_abs: np.ndarray,
                n_frames: int) -> Tuple[int, int]:
    """Window of WINDOW_S containing the most detected events."""
    width = int(WINDOW_S * FS)
    if events_abs.size == 0:
        s = int(np.argmax(np.convolve(mask.astype(float), np.ones(width), "valid")))
        return s, min(s + width, n_frames)
    starts = np.arange(0, max(1, n_frames - width), FS)
    counts = [np.sum((events_abs >= s) & (events_abs < s + width)) for s in starts]
    s = int(starts[int(np.argmax(counts))])
    return s, min(s + width, n_frames)


def panel(ax_raster, ax_mua, mouse: str, date: str, genotype: str) -> None:
    spks_full = np.asarray(np.load(spks_path(mouse, date), mmap_mode="r"),
                           dtype=np.float32)
    rng = np.random.default_rng(0)
    cells = np.sort(rng.choice(spks_full.shape[0], min(N_CELLS, spks_full.shape[0]),
                               replace=False))
    spks = spks_full[cells]
    mask, _ = immobility_mask(mouse, date)
    if mask.size != spks.shape[1]:
        mask = mask[: spks.shape[1]]

    # Smooth on the CONTINUOUS recording, then restrict - same as the analysis
    rates = smoothed_rates(spks.astype(np.float32), CHANG)
    idx = np.flatnonzero(mask)
    from viral.nlgf.activity import detect_sce_contiguous
    ev = detect_sce_contiguous(rates, mask)
    events_abs = np.array([a for a, _ in ev], dtype=int)
    events_abs_end = np.array([b for _, b in ev], dtype=int)

    lo, hi = best_window(mask, events_abs, spks.shape[1])
    win = slice(lo, hi)
    t = np.arange(lo, hi) / FS
    in_win = (events_abs >= lo) & (events_abs < hi)

    # Cells UNSORTED: a synchronous event is a vertical stripe and any ordering taken
    # from the displayed data would invent horizontal structure
    sub = spks[:, win] > 0
    y, x = np.nonzero(sub)
    # shade every immobility bout inside the window
    m = mask[win].astype(np.int8)
    for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, m, 0]))[i::2] for i in (0, 1)]):
        ax_raster.axvspan(t[a], t[min(b, t.size - 1)],
                          color=GENOTYPE_COLOURS[genotype], alpha=0.10, lw=0, zorder=0)
    for a, b in zip(events_abs[in_win], events_abs_end[in_win]):
        ax_raster.axvspan(a / FS, max(b / FS, a / FS + 0.1), color="#E8590C",
                          alpha=0.30, lw=0, zorder=1)
    ax_raster.scatter(t[x], y, s=1.1, c="#212529", marker="|", linewidths=0.5, zorder=3)
    ax_raster.set_ylim(sub.shape[0], 0)
    ax_raster.set_xlim(t[0], t[-1])
    ax_raster.set_ylabel("Cell (unsorted)")
    ax_raster.set_title(
        f"{genotype}   {mouse} {date}   ·   {mask[win].sum() / FS:.0f} s immobile in "
        f"this window, {int(in_win.sum())} synchronous events", fontsize=10.5)
    ax_raster.set_xticklabels([])

    # MUA z-scored over the immobility frames, which is what the detector thresholds
    mua_all = rates.mean(axis=0)
    mu, sd = mua_all[mask].mean(), mua_all[mask].std()
    ax_mua.plot(t, (mua_all[win] - mu) / sd, color="#495057", lw=0.9)
    ax_mua.axhline(CHANG.sce_threshold_sd, color="#E8590C", lw=1.1, ls="--",
                   label=f"{CHANG.sce_threshold_sd:.0f} SD threshold")
    for a, b in zip(events_abs[in_win], events_abs_end[in_win]):
        ax_mua.axvspan(a / FS, max(b / FS, a / FS + 0.1), color="#E8590C",
                       alpha=0.30, lw=0)
    ax_mua.set_xlim(t[0], t[-1])
    ax_mua.set_ylabel("Population\nactivity (SD)")
    ax_mua.set_xlabel("Time in session (s)")
    ax_mua.legend(fontsize=8, loc="upper right")


def figure() -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 9.5),
                             gridspec_kw={"height_ratios": [3, 1, 3, 1]})
    for i, (g, (mouse, date)) in enumerate(EXAMPLES.items()):
        print(f"  {g}: {mouse} {date}", flush=True)
        panel(axes[2 * i], axes[2 * i + 1], mouse, date, g)
    fig.suptitle(
        "Synchronous events during in-task immobility  ·  shaded band is the retained "
        "immobility bout, orange marks detected events",
        y=1.0, fontsize=12.5, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig17_example_raster.png")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    figure()
    print(f"wrote figure to {PLOTS}")
