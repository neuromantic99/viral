"""Single-session example panels.

Everything else in this set is a summary statistic. These show the data the statistics
were computed on, so a reader can see that place coding exists and that the two
genotypes look alike, rather than taking p = 0.74 on trust.

Example sessions are chosen on a stated criterion, not by eye: the session closest to
its own genotype's MEDIAN on both place cell yield and decoding error. Picking the
prettiest session would make the panel an advert rather than an example.

    WT    J035 2026-06-18   yield 0.500 (median 0.520), error 24.3 cm (median 23.9)
    NLGF  JB036 2025-07-08  yield 0.492 (median 0.516), error 19.4 cm (median 20.1)

The snake plot is SPLIT-HALF. Cells are ordered by their peak on odd laps and the even
laps are displayed. Sorting and displaying the same data manufactures a clean diagonal
out of pure noise - it is the standard way to make an uninformative population look
beautifully tuned - so the ordering and the picture come from disjoint trials.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.constants import grosmark_config
from viral.grosmark_analysis import filter_additional_check
from viral.imaging_utils import activity_trial_position, trial_is_imaged
from viral.models import Cached2pSession
from viral.nlgf.paths import CACHED_2P, PLOTS, spks_path
from viral.nlgf.place_threshold import shuffle_threshold
from viral.nlgf.style import LANDMARKS_CM, set_style
from viral.utils import get_wheel_circumference_from_rig, has_n_consecutive_trues

CONFIG = grosmark_config
WHEEL = get_wheel_circumference_from_rig("2P")
N_LAPS = 30
N_CELLS = 250

# Representative pair: closest to their genotype's MEDIAN on both yield and error
EXAMPLES = {
    "WT": ("J035", "2026-06-18"),
    "NLGF": ("JB036", "2025-07-08"),
}

# Candidates for a talk slide, ranked by lap-to-lap reliability - the property that
# makes a snake plot look crisp. Percentiles are within genotype, and are printed on
# each panel so a chosen session can be described honestly rather than implied to be
# typical.
CANDIDATES = {
    "WT": [("JB030", "2025-03-11", 100), ("JB026", "2024-12-17", 98),
           ("JB026", "2024-12-15", 96), ("JB030", "2025-03-14", 94)],
    "NLGF": [("J030", "2026-05-13", 100), ("JB021", "2024-12-04", 97),
             ("JB021", "2024-12-06", 95), ("J030", "2026-05-19", 92)],
}


def session_maps(mouse: str, date: str) -> Optional[Dict]:
    """all_trials, place cell mask and the odd/even split-half maps."""
    session = Cached2pSession.model_validate_json(
        (CACHED_2P / f"{mouse}_{date}.json").read_text())
    trials = []
    for t in session.trials:
        try:
            if trial_is_imaged(t):
                trials.append(t)
        except (AssertionError, IndexError):
            continue
    if len(trials) < N_LAPS:
        return None
    trials = trials[:N_LAPS]

    spks_full = np.asarray(np.load(spks_path(mouse, date), mmap_mode="r"),
                           dtype=np.float32)
    rng = np.random.default_rng(0)
    pool = np.sort(rng.choice(spks_full.shape[0], N_CELLS, replace=False))
    spks = spks_full[pool]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_trials = np.array([
            activity_trial_position(
                trial=t, flu=spks, wheel_circumference=WHEEL,
                bin_size=CONFIG.bin_size, start=CONFIG.start, max_position=CONFIG.end,
                verbose=False, do_shuffle=False, threshold_speed=True)
            for t in trials], dtype=np.float32)
        smoothed = gaussian_filter1d(np.nanmean(all_trials, 0),
                                     sigma=7.5 / CONFIG.bin_size, axis=1)
        odd = gaussian_filter1d(np.nanmean(all_trials[1::2], 0),
                                sigma=7.5 / CONFIG.bin_size, axis=1)
        even = gaussian_filter1d(np.nanmean(all_trials[0::2], 0),
                                 sigma=7.5 / CONFIG.bin_size, axis=1)

    threshold = shuffle_threshold(all_trials, n_shuffles=2000)
    n = int((2 / CONFIG.bin_size) * 5)
    if n % 2 == 0:
        n += 1
    supra = smoothed > threshold
    pcs = has_n_consecutive_trues(supra, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extra = filter_additional_check(
            all_trials=all_trials[:, pcs, :], place_threshold=threshold[pcs, :],
            smoothed_matrix=smoothed[pcs, :], n_consecutive_trues=n)
    combined = pcs.copy()
    combined[pcs] = extra
    return dict(mouse=mouse, date=date, odd=odd, even=even, pcs=combined,
                n_cells=N_CELLS)


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.nan_to_num(m)
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    return (m - lo) / np.where(hi - lo > 0, hi - lo, 1.0)


def figure() -> None:
    data = {}
    for g, (mouse, date) in EXAMPLES.items():
        print(f"  {g}: {mouse} {date}", flush=True)
        out = session_maps(mouse, date)
        if out is not None:
            data[g] = out

    fig, axes = plt.subplots(1, len(data), figsize=(5.4 * len(data), 5.6))
    if len(data) == 1:
        axes = [axes]
    x = CONFIG.start + np.arange(data[list(data)[0]]["odd"].shape[1]) * CONFIG.bin_size

    for ax, (g, d) in zip(axes, data.items()):
        odd, even, pcs = d["odd"][d["pcs"]], d["even"][d["pcs"]], d["pcs"]
        order = np.argsort(np.argmax(odd, axis=1))       # sort on ODD
        img = _normalise(even)[order]                     # display EVEN
        im = ax.imshow(img, aspect="auto", cmap="magma", vmin=0, vmax=1,
                       extent=[x[0], x[-1], img.shape[0], 0], interpolation="nearest")
        for lm in LANDMARKS_CM:
            ax.axvline(lm, color="#8ED1FC", lw=1.1, ls="--", alpha=0.85)
        ax.set_xlabel("Position (cm)")
        ax.set_ylabel("Place cell (sorted by odd-lap peak)")
        ax.set_title(f"{g}   {d['mouse']} {d['date']}\n"
                     f"{int(pcs.sum())}/{d['n_cells']} place cells, 30 laps",
                     fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02,
                     label="normalised rate (even laps)")

    fig.suptitle(
        "Place coding in single sessions  ·  sorted on odd laps, plotted on even laps, "
        "so the diagonal is not built by the sort",
        y=1.0, fontsize=12.5, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig13_example_snake.png")
    plt.close(fig)


def figure_candidates() -> None:
    """A grid of candidate sessions, so a talk slide can be chosen from real options."""
    n = max(len(v) for v in CANDIDATES.values())
    fig, axes = plt.subplots(2, n, figsize=(4.3 * n, 9.0))
    for row, (g, cands) in enumerate(CANDIDATES.items()):
        for col, (mouse, date, pct) in enumerate(cands):
            ax = axes[row][col]
            d = session_maps(mouse, date)
            if d is None:
                ax.axis("off")
                continue
            odd, even = d["odd"][d["pcs"]], d["even"][d["pcs"]]
            order = np.argsort(np.argmax(odd, axis=1))
            x = CONFIG.start + np.arange(odd.shape[1]) * CONFIG.bin_size
            ax.imshow(_normalise(even)[order], aspect="auto", cmap="magma",
                      vmin=0, vmax=1, extent=[x[0], x[-1], odd.shape[0], 0],
                      interpolation="nearest")
            for lm in LANDMARKS_CM:
                ax.axvline(lm, color="#8ED1FC", lw=1.0, ls="--", alpha=0.8)
            ax.set_title(f"{g}  {mouse} {date}\n{int(d['pcs'].sum())} place cells  ·  "
                         f"{pct}th pct reliability", fontsize=9.5)
            ax.set_xlabel("Position (cm)")
            if col == 0:
                ax.set_ylabel("Place cell")
    fig.suptitle("Candidate sessions for a talk slide  ·  ranked by lap-to-lap "
                 "reliability within genotype  ·  split-half throughout",
                 y=1.0, fontsize=12.5, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig14_snake_candidates.png")
    plt.close(fig)


def figure_split_demo() -> None:
    """Why the split-half sort matters: the same session, sorted both ways."""
    mouse, date = EXAMPLES["WT"]
    d = session_maps(mouse, date)
    odd, even = d["odd"][d["pcs"]], d["even"][d["pcs"]]
    x = CONFIG.start + np.arange(odd.shape[1]) * CONFIG.bin_size

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, (title, order, img) in zip(axes, [
        ("Sorted AND plotted on the same laps\n"
         "(the diagonal is partly built by the sort)",
         np.argsort(np.argmax(even, axis=1)), even),
        ("Sorted on odd laps, plotted on even laps\n"
         "(the diagonal is real tuning)",
         np.argsort(np.argmax(odd, axis=1)), even),
    ]):
        ax.imshow(_normalise(img)[order], aspect="auto", cmap="magma", vmin=0, vmax=1,
                  extent=[x[0], x[-1], img.shape[0], 0], interpolation="nearest")
        for lm in LANDMARKS_CM:
            ax.axvline(lm, color="#8ED1FC", lw=1.0, ls="--", alpha=0.8)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Position (cm)")
    axes[0].set_ylabel("Place cell")
    fig.suptitle(f"Why snake plots must be split-half  ·  {mouse} {date}, "
                 f"identical data, two sort orders", y=1.0, fontsize=12.5,
                 weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig15_split_half_demo.png")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "examples"):
        figure()
    if which in ("all", "demo"):
        figure_split_demo()
    if which in ("all", "candidates"):
        figure_candidates()
    print(f"wrote figures to {PLOTS}")
