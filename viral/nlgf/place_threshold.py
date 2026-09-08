"""Generate the missing place_threshold caches locally.

J031 and J032 have almost no cached thresholds (2 of 7 and 0 of 9), which removes them
from every place-cell-dependent analysis and left Tier 5 with two NLGF mice. The
threshold is a Monte Carlo quantity, so it can be regenerated without the server.

The computation is identical to get_place_cells' shuffle loop, but restructured to be
tractable. That loop calls activity_trial_position(do_shuffle=True) once per trial per
shuffle - 2000 x n_trials calls - and each call bins the whole trial by position before
throwing the binning away and circularly permuting the RESULT:

    activity_trial_position(..., do_shuffle=True)
        -> bins the trial into (n_cells, n_bins)
        -> returns circularly_permute_rows(that)

Since the permutation acts on the binned map, the binning does not depend on the
shuffle and only has to happen once. Here all_trials is computed a single time and the
2000 shuffles are gathers over it, which turns hours per session into seconds. The
resulting distribution is the same one, drawn the same way.

Written into the same filename the server cache uses, so viral.nlgf.place picks these
up with no change.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.constants import grosmark_config
from viral.imaging_utils import activity_trial_position, trial_is_imaged
from viral.models import Cached2pSession
from viral.nlgf.paths import CACHED_2P, spks_path
from viral.nlgf.place import threshold_path
from viral.utils import get_wheel_circumference_from_rig

CONFIG = grosmark_config
WHEEL = get_wheel_circumference_from_rig("2P")
N_SHUFFLES = 2000
SIGMA_BINS = 7.5 / CONFIG.bin_size


def build_all_trials(session: Cached2pSession, spks: np.ndarray,
                     rewarded: Optional[bool] = None) -> Optional[np.ndarray]:
    trials = []
    for t in session.trials:
        try:
            if trial_is_imaged(t) and (rewarded is None or t.texture_rewarded == rewarded):
                trials.append(t)
        except (AssertionError, IndexError):
            continue
    if len(trials) < 10:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array([
            activity_trial_position(
                trial=t, flu=spks, wheel_circumference=WHEEL, bin_size=CONFIG.bin_size,
                start=CONFIG.start, max_position=CONFIG.end, verbose=False,
                do_shuffle=False, threshold_speed=True)
            for t in trials
        ], dtype=np.float32)


def shuffle_threshold(all_trials: np.ndarray, n_shuffles: int = N_SHUFFLES,
                      seed: int = 0) -> np.ndarray:
    """99th percentile of the per-lap circularly permuted, trial-averaged, smoothed map."""
    n_trials, n_cells, n_bins = all_trials.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n_shuffles, n_cells, n_bins), dtype=np.float32)
    base = np.arange(n_bins)
    trial_idx = np.arange(n_trials)[:, None, None]
    cell_idx = np.arange(n_cells)[None, :, None]

    for s in range(n_shuffles):
        # Shift of 0 excluded, matching circularly_permute_rows
        shifts = rng.integers(1, n_bins, size=(n_trials, n_cells))
        idx = (base[None, None, :] - shifts[:, :, None]) % n_bins
        permuted = all_trials[trial_idx, cell_idx, idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean = np.nanmean(permuted, axis=0)
        out[s] = gaussian_filter1d(mean, sigma=SIGMA_BINS, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanpercentile(out, 99, axis=0)


def generate(mouse: str, date: str, rewarded: Optional[bool] = None,
             overwrite: bool = False) -> bool:
    target = threshold_path(mouse, date, rewarded)
    if target.exists() and not overwrite:
        return False
    spath = spks_path(mouse, date)
    jpath = CACHED_2P / f"{mouse}_{date}.json"
    if not spath.exists() or not jpath.exists():
        return False
    session = Cached2pSession.model_validate_json(jpath.read_text())
    spks = np.asarray(np.load(spath, mmap_mode="r"), dtype=np.float32)
    all_trials = build_all_trials(session, spks, rewarded)
    if all_trials is None:
        return False
    np.save(target, shuffle_threshold(all_trials))
    return True


def build_missing() -> None:
    from viral.nlgf.load import ledger

    led = ledger()
    led = led[led.has_spks]
    todo = [r for r in led.itertuples()
            if not threshold_path(r.mouse, r.date, None).exists()]
    print(f"{len(todo)} sessions missing a place_threshold", flush=True)
    ok = fail = 0
    for i, r in enumerate(todo, 1):
        try:
            ok += bool(generate(r.mouse, r.date))
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        print(f"  [{i}/{len(todo)}] {r.mouse} {r.date} ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    build_missing()
