"""A Grosmark-style ensemble reactivation panel.

The plain raster in fig36 could not show what a synchronous event is: with 250 sparse
cells in unsorted order, an event is a faint vertical smudge. The fix used in the replay
literature is to sort cells by their weight in an ICA ensemble TEMPLATE, and the reason
that is legitimate - unlike sorting a snake plot on the data it displays - is that the
template comes from a different epoch entirely. Here it is built from RUNNING activity
and applied to OFFLINE activity, so any vertical structure that appears offline was not
put there by the sort.

Four elements, following Grosmark:

  (i)   reactivation strength over time for the leading ensembles, R(t) = z(t)' P z(t)
        with P = ww' and its diagonal zeroed, so a cell cannot co-activate with itself
  (ii)  the raster, cells ordered by ensemble weight, members' spikes coloured
NO SYNCHRONOUS-EVENT SHADING. An earlier version marked detected SCEs as Grosmark marks
SWRs, and it was misleading twice over. The events are detected on the 250-cell POOL
while the raster shows the 60 place cells drawn from it, so the band referred to a
population the reader could not see. And checking the one event in this window, 6 of the
60 displayed cells fired during it - exactly the baseline for a random 0.33 s immobility
window (6.0/60). Even in the pool it was 31 of 250. These events are a statistical
threshold crossing of a smoothed population rate, not a visible burst, and shading them
invites the eye to find a column that is not there. The panel's claim rests on the
reactivation traces and the coloured blocks instead.

The speed trace is what makes the design readable without narration. One continuous
window spans the animal running and then stopping, so the template epoch and the test
epoch are the SAME stretch of time rather than two panels the reader has to relate.
Ensembles are defined from running frames across the whole session; what the shaded
immobility band shows is the template being applied where it was never fitted. Any
structure inside that band was not put there by the analysis.

Grosmark also plots the template weights alongside. They are omitted here: with 60 cells
the weight panel needed its own explanation of what an ICA weight is and what a negative
one means, which is more machinery than the point requires. Membership is carried by the
colour of the spikes and the block boundaries instead.

The point of the panel is the alignment between (i) and (iv): reactivation peaks should
land on the events, not between them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.constants import SERVER_PATH  # noqa: F401  (import parity with the package)
from viral.ensemble_reactivation import build_run_ensembles
from viral.models import Cached2pSession
from viral.nlgf.activity import CONFIG as CHANG_CFG
from viral.nlgf.activity import detect_sce_contiguous
from viral.nlgf.immobility import immobility_mask
from viral.nlgf.paths import CACHED_2P, PLOTS, spks_path
from viral.nlgf.reactivation import N_PLACE_CELLS, N_TOTAL_CELLS, local_place_cells
from viral.nlgf.style import set_style
from viral.chang import smoothed_rates

# a session the reactivation pipeline actually succeeded on: 230 place cells
# available, 5 ICA components, the strongest excess_r in the cohort
DEMO = ("JB030", "2025-03-25")
COMBINED = True            # one continuous window spanning running -> stopping -> rest
WINDOW_COMBINED_S = 30.0
WINDOW_S = 8.0
OFFLINE_SIGMA = 3.0
N_SHOW = 2                 # ensembles to plot
MEMBER_FRAC = 0.20         # top fraction of |weight| counted as ensemble members
COLOURS = ("#3B5BDB", "#C92A2A")


def reactivation_timecourse(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """R(t) = z(t)' (ww' - diag(ww')) z(t), vectorised over time."""
    proj = w @ z                                  # (T,)
    correction = (w ** 2) @ (z ** 2)              # remove each cell's self term
    return proj ** 2 - correction


def ensemble_panels(axes=None) -> None:
    """Draw the run-to-rest illustration, into `axes` if given or its own figure if not.

    Split out so fig36 can show the same panels as its demonstration rather than a
    second, separately computed version of them.
    """
    set_style(talk=True)
    standalone = axes is None
    mouse, date = DEMO
    spks_full = np.asarray(np.load(spks_path(mouse, date), mmap_mode="r"),
                           dtype=np.float64)
    rng = np.random.default_rng(0)
    pool = np.sort(rng.choice(spks_full.shape[0], N_TOTAL_CELLS, replace=False))
    spks = spks_full[pool]
    session = Cached2pSession.model_validate_json(
        (CACHED_2P / f"{mouse}_{date}.json").read_text())

    pcs = local_place_cells(session, spks, pool=pool)
    idx = np.flatnonzero(pcs)
    chosen = np.sort(rng.choice(idx, min(N_PLACE_CELLS, idx.size), replace=False))
    place = spks[chosen]
    ens = build_run_ensembles(session, place, rewarded=None,
                              n_component_method="circular_shift")
    print(f"  {ens.shape[1]} ensembles from {place.shape[0]} place cells")
    if ens.shape[1] == 0:
        raise SystemExit("no ensembles")

    from viral.nlgf.load import load_digest

    digest = load_digest(mouse, date)
    if COMBINED:
        # The wheel-freeze epochs sit OUTSIDE the task, where the animal is stationary
        # throughout - a window there has no running in it and the speed trace is flat
        # at zero. In-task immobility alternates with running lap by lap, which is what
        # makes a run -> stop transition visible in one continuous window.
        mask, _ = immobility_mask(mouse, date)
        mask = mask[: place.shape[1]]
    else:
        masks = digest.immobility_masks(n_frames=place.shape[1])
        mask = masks.get("post", masks.get("pre")) if masks else None
        if mask is None:
            mask, _ = immobility_mask(mouse, date)
            mask = mask[: place.shape[1]]
    # Choose the window by EVENT CONTENT, not by bout length: the point of the panel is
    # whether reactivation peaks land on the events, and the longest quiet bout shows
    # neither. Events are detected first, then the window is placed over the densest
    # cluster inside a single contiguous bout (never spanning a gap, or the time axis
    # would be a splice).
    # cast AFTER thresholding: gaussian_filter1d preserves dtype, so smoothing a bool
    # array returns all-False and every event count comes out zero
    rates = smoothed_rates((spks > 0).astype(np.float32), CHANG_CFG)
    all_events = detect_sce_contiguous(rates, mask)
    ev_starts = np.array([a for a, _ in all_events], dtype=int)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    bouts = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= 5 * 30]
    width = int(WINDOW_S * 30)
    # Centre on an EVENT, inside its own bout. The panel exists to show whether
    # reactivation peaks land on events, so an event at the edge of the window defeats
    # it; and the window must not cross a bout boundary or the time axis is a splice.
    width = int(WINDOW_S * 30)
    lo = hi = None
    for a, b in sorted(bouts, key=lambda x: x[0] - x[1]):
        inside = ev_starts[(ev_starts >= a) & (ev_starts < b)]
        if inside.size == 0:
            continue
        centre = int(inside[inside.size // 2])
        lo = max(a, min(centre - width // 2, b - width))
        hi = min(b, lo + width)
        break
    if lo is None:
        lo, hi = bouts[0][0], min(bouts[0][1], bouts[0][0] + width)
    print(f"  window {lo / 30:.0f}-{hi / 30:.0f}s "
          f"({(hi - lo) / 30:.0f}s), {len(all_events)} events in the epoch")

    seg = gaussian_filter1d(place[:, lo:hi], sigma=OFFLINE_SIGMA, axis=1)
    z = (seg - seg.mean(axis=1, keepdims=True)) / (seg.std(axis=1, keepdims=True) + 1e-9)
    # Group by membership rather than sorting on one ensemble's signed weight. With a
    # signed sort, A's members land at BOTH ends of the raster and B's are scattered
    # through the middle, so a reactivation peak in B has no visible counterpart in the
    # spikes - which is exactly the mismatch this panel is meant to make visible.
    thr = [np.quantile(np.abs(ens[:, k]), 1 - MEMBER_FRAC)
           for k in range(min(N_SHOW, ens.shape[1]))]
    is_a = np.abs(ens[:, 0]) > thr[0]
    is_b = (np.abs(ens[:, 1]) > thr[1]) & ~is_a if len(thr) > 1 else np.zeros_like(is_a)
    rank = np.where(is_a, 0, np.where(is_b, 1, 2))
    # Members are ordered by their own ensemble's |weight|; NON-members keep their
    # original cell order. Sorting the remainder by weight too drew a smooth gradient
    # across the bottom two thirds of the weight panel that looked like structure and
    # was entirely the sort.
    cell_id = np.arange(ens.shape[0], dtype=float)
    within = np.where(is_a, -np.abs(ens[:, 0]),
                      np.where(is_b, -np.abs(ens[:, 1]), cell_id))
    order_by = np.lexsort((within, rank))
    member = rank          # per-cell group, indexed by order_by where drawn
    strengths = [reactivation_timecourse(z, ens[:, k]) for k in range(min(N_SHOW, ens.shape[1]))]

    events = [(a, b) for a, b in all_events if lo <= a < hi]

    def draw(ax_top, ax_ras, frames, evs, title):
        tt = frames / 30
        segw = gaussian_filter1d(place[:, frames], sigma=OFFLINE_SIGMA, axis=1)
        zz = (segw - segw.mean(axis=1, keepdims=True)) / (
            segw.std(axis=1, keepdims=True) + 1e-9)
        for k in range(min(N_SHOW, ens.shape[1])):
            ax_top.plot(tt, reactivation_timecourse(zz, ens[:, k]), color=COLOURS[k],
                        lw=2.6, label=f"Ensemble {chr(65 + k)}")
        for a, b in evs:
            for axis in (ax_top, ax_ras):
                axis.axvspan(a / 30, max(b / 30, a / 30 + 0.15), color="#FFD43B",
                             alpha=0.55, lw=0, zorder=0)
        ax_top.set_xlim(tt[0], tt[-1]); ax_top.set_xticklabels([])
        ax_top.set_title(title, fontsize=26)

        raster = (place[:, frames] > 0)[order_by]
        for row, (spk, grp) in enumerate(zip(raster, member[order_by])):
            x = tt[np.flatnonzero(spk)]
            ax_ras.scatter(x, np.full(x.size, row), s=95, marker="|", linewidths=2.6,
                           c=COLOURS[grp] if grp < 2 else "#212529", zorder=3)
        for boundary in (np.sum(member == 0), np.sum(member < 2)):
            ax_ras.axhline(boundary - 0.5, color="#495057", lw=1.6, ls=":", zorder=4)
        ax_ras.set_xlim(tt[0], tt[-1]); ax_ras.set_ylim(raster.shape[0], -1)
        ax_ras.set_xlabel("Time (s)")

    rest_frames = np.arange(lo, hi)

    if COMBINED:
        # instantaneous speed on every frame, from the UNTHRESHOLDED position trace so
        # the stationary stretches are represented rather than dropped
        pos, frm = digest.run_position_cm_all, digest.run_frame_all.astype(int)
        d_pos, d_frm = np.diff(pos), np.diff(frm).astype(float)
        ok = (d_frm > 0) & (np.abs(d_pos) < 40)
        v = np.zeros(place.shape[1])
        v[np.clip(frm[:-1][ok], 0, place.shape[1] - 1)] = d_pos[ok] / d_frm[ok] * 30
        speed = gaussian_filter1d(np.clip(v, 0, 120), sigma=6)

        # a window containing a run -> stop transition AND an event in the stopped part
        width = int(WINDOW_COMBINED_S * 30)
        best, best_score = None, -1
        for a, _b in bouts:
            start = max(0, a - width // 2)
            stop = min(place.shape[1], start + width)
            ran = speed[start:a].mean() if a > start else 0
            n_ev = int(((ev_starts >= start) & (ev_starts < stop)).sum())
            score = n_ev * 100 + ran
            if score > best_score:
                best, best_score = (start, stop), score
        w0, w1 = best
        frames = np.arange(w0, w1)
        tt = frames / 30
        print(f"  combined window {w0 / 30:.0f}-{w1 / 30:.0f}s")

        if axes is None:
            fig = plt.figure(figsize=(20, 13))
            gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 3], hspace=0.09,
                                  left=0.115, right=0.985, top=0.87, bottom=0.09)
            axes = [fig.add_subplot(gs[i]) for i in range(3)]
        ax_v, ax_top, ax_ras = axes

        segw = gaussian_filter1d(place[:, frames], sigma=OFFLINE_SIGMA, axis=1)
        zz = (segw - segw.mean(axis=1, keepdims=True)) / (
            segw.std(axis=1, keepdims=True) + 1e-9)
        still_here = mask[frames]
        for axis in (ax_v, ax_top, ax_ras):
            m8 = still_here.astype(np.int8)
            for a, b in zip(*[np.flatnonzero(np.diff(np.r_[0, m8, 0]))[i::2]
                              for i in (0, 1)]):
                axis.axvspan(tt[a], tt[min(b, tt.size - 1)], color="#4C6EF5",
                             alpha=0.09, lw=0, zorder=0)
            axis.set_xlim(tt[0], tt[-1])

        ax_v.plot(tt, speed[frames], color="#212529", lw=3.0)
        ax_v.set_ylabel("Speed\n(cm/s)"); ax_v.set_xticklabels([])
        ax_v.text(0.005, 0.9, "shaded = immobile", transform=ax_v.transAxes,
                  fontsize=19, color="#4C6EF5", va="top")
        for k in range(min(N_SHOW, ens.shape[1])):
            ax_top.plot(tt, reactivation_timecourse(zz, ens[:, k]), color=COLOURS[k],
                        lw=2.6, label=f"Ensemble {chr(65 + k)}")
        ax_top.set_ylabel("Reactivation\nstrength"); ax_top.set_xticklabels([])
        ax_top.legend(frameon=False, fontsize=20, ncol=2, loc="upper left")

        raster = (place[:, frames] > 0)[order_by]
        for row, (spk, grp) in enumerate(zip(raster, member[order_by])):
            x = tt[np.flatnonzero(spk)]
            ax_ras.scatter(x, np.full(x.size, row), s=95, marker="|", linewidths=2.6,
                           c=COLOURS[grp] if grp < 2 else "#212529", zorder=3)
        for boundary in (np.sum(member == 0), np.sum(member < 2)):
            ax_ras.axhline(boundary - 0.5, color="#495057", lw=1.6, ls=":", zorder=4)
        ax_ras.set_ylim(raster.shape[0], -1)
        ax_ras.set_ylabel("Cell (grouped by ensemble)")
        ax_ras.set_xlabel("Time in session (s)")

    if standalone:
        fig = ax_ras.figure
        fig.suptitle("Ensembles are defined while the animal runs, then looked for "
                     "when it stops\n"
                     f"{mouse} {date}  ·  {int((member == 0).sum())} cells in ensemble "
                     f"A, {int((member == 1).sum())} in B",
                     fontsize=29, weight="semibold", y=0.985)
        out = PLOTS / "fig37_ensemble_raster.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")


def main() -> None:
    ensemble_panels()


if __name__ == "__main__":
    main()
