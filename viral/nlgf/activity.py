"""Tier 1: single-cell excitability and population synchrony.

The most direct test of the amyloid phenotype, and the cheapest: it needs the spike
matrix and a set of frames, no place fields and no decoder. Neuronal hyperactivity
near plaques is this lab's own signature finding in APP mice, and network
hypersynchrony is its population-level counterpart.

Two epochs are scored separately and they answer different questions:

  RUN   frames the animal was running (>5 cm/s sustained 3 s, from the digest). NLGF
        run faster (see fig_behaviour), and firing rate rises with speed, so a
        difference here is confounded by definition and speed must be carried as a
        covariate.

  STILL in-task immobility, from viral.nlgf.immobility: speed below 3 cm/s for at
        least 3 s, anywhere in the session, minus a 3 s settling period at the start of
        each bout and minus lick-adjacent frames. Speed is near zero in both genotypes
        by construction, so this is the clean comparison, and it is where a rate
        difference means excitability rather than behaviour.

        Deliberately NOT get_iti_still_frames, which scores stillness only inside the
        ITI and so needs the encoder sampled there. That sampling (trigger_panda_ITI)
        only exists from the 2025 cohort on, and its absence in 2024 reads as zero
        stillness rather than as unmeasurable - which, since the old cohort is mostly
        NLGF, would have silently dropped most NLGF mice from this comparison.

On the synchrony measures. Chang's SCE rate z-scores the MUA within the epoch being
scored, so an epoch with globally more synchrony raises its own detection threshold -
it is close to self-normalising and is the wrong primary statistic for a group
comparison. It is reported, but the headline measures are ones with an absolute
scale: mean pairwise correlation, and the Fano factor of the population signal.

Cells are subsampled to a fixed count as well as frames. Cell yield varies threefold
between preparations for reasons of injection quality, and it was the covariate that
removed the apparent genotype effects in Tiers 2 and 5; rate percentiles and pairwise
correlations both depend on how many cells are in the pool.

Frame counts are equalised across sessions before any correlation is computed.
Correlation estimates are biased by sample size, and the genotypes do not contribute
equal amounts of quiescence, so unequalised correlations would differ for that reason
alone.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.chang import ChangConfig, detect_sce, smoothed_rates
from viral.nlgf.immobility import immobility_mask
from viral.nlgf.cohort import MIN_CELLS, describe, sessions
from viral.nlgf.load import load_digest
from viral.nlgf.paths import RESULTS, spks_path

FS = 30
CONFIG = ChangConfig()

# Correlation and Fano are computed on this many frames, sampled without replacement,
# so that every session contributes the same amount of data
N_FRAMES_EQUALISED = 6000  # 200 s of quiescence
MIN_FRAMES = 3000
N_CELLS = MIN_CELLS  # fixed subsample, from the cohort floor


def detect_sce_contiguous(rates: np.ndarray, mask: np.ndarray,
                          min_bout_frames: int = 3 * FS):
    """SCEs detected within each contiguous immobility bout, in absolute frames.

    detect_sce must not be given a concatenated mask. It merges events less than 250 ms
    apart, but adjacent columns of a masked array can be minutes apart in real time, so
    separate events get welded together across the gaps. Measured on one session: run
    on the concatenated mask the median event lasted 14.9 s and the longest 65 s, with
    24 of 27 spanning over a second; run per contiguous bout the median is 0.30 s and
    the longest 1.0 s, which is what a synchronous event actually looks like.

    That inflates event duration enormously and hence participation, since a 15 s
    window catches almost every cell at least once. Chang et al. scored contiguous rest
    epochs, where the problem cannot arise; it appears only when the method is
    generalised to fragmented in-task immobility.

    Returns (start, end) pairs in absolute session frames.
    """
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    events = []
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start < min_bout_frames:
            continue
        for a, b in detect_sce(rates[:, start:end], CONFIG):
            events.append((int(start + a), int(start + b)))
    return events


def event_onsets(spks: np.ndarray) -> np.ndarray:
    """Event onsets from the binarised spike matrix.

    spks is already 0/1 (thresholded OASIS), but a burst appears as a run of ones, so
    counting ones conflates event number with burst length. The codebase's own
    convention for this is remove_consecutive_ones; the same thing is done here with a
    diff, which is far faster than apply_along_axis over thousands of cells.
    """
    b = spks > 0
    onsets = np.zeros_like(b)
    onsets[:, 0] = b[:, 0]
    onsets[:, 1:] = b[:, 1:] & ~b[:, :-1]
    return onsets


def _rates_hz(onsets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = int(mask.sum())
    if n == 0:
        return np.full(onsets.shape[0], np.nan)
    return onsets[:, mask].sum(axis=1) / (n / FS)


def _synchrony(
    spks_epoch: np.ndarray, rates_epoch: np.ndarray, rng: np.random.Generator,
    spks_full: Optional[np.ndarray] = None, rates_full: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> Dict:
    """Correlation, Fano and SCE measures on one epoch's frames.

    rates_epoch must be smoothed_rates computed on the CONTINUOUS session and then
    indexed by the epoch mask, not smoothed after masking. The ITI mask is heavily
    fragmented, so smoothing after selection would convolve across gaps that are
    minutes apart in real time.
    """
    n_cells, n_frames = spks_epoch.shape
    out: Dict[str, float] = {}

    if n_frames >= MIN_FRAMES:
        take = rng.choice(n_frames, min(N_FRAMES_EQUALISED, n_frames), replace=False)
        take.sort()
        sub = spks_epoch[:, take]

        # Mean pairwise correlation: absolute scale, unlike the SCE threshold
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(sub)
        c = c[np.triu_indices_from(c, k=1)]
        c = c[np.isfinite(c)]
        out["mean_pairwise_r"] = float(np.mean(c)) if c.size else np.nan
        out["frac_pairs_r_above_0.1"] = float(np.mean(c > 0.1)) if c.size else np.nan

        # Population co-fluctuation. Fano > 1 means cells fire together more than
        # independent Poisson units would.
        mua = sub.sum(axis=0).astype(float)
        out["mua_fano"] = float(mua.var() / mua.mean()) if mua.mean() > 0 else np.nan

        # Absolute co-activation: fraction of frames in which more than 5% of the
        # population is simultaneously active. Unlike the SCE threshold this does not
        # rescale itself to the epoch.
        out["frac_frames_5pct_coactive"] = float(np.mean(mua > 0.05 * n_cells))
        out["n_frames_synchrony"] = float(sub.shape[1])
    else:
        for k in ("mean_pairwise_r", "frac_pairs_r_above_0.1", "mua_fano",
                  "frac_frames_5pct_coactive", "n_frames_synchrony"):
            out[k] = np.nan

    # Chang's SCEs, detected WITHIN each contiguous bout - see detect_sce_contiguous
    if n_frames >= MIN_FRAMES and mask is not None:
        sce = detect_sce_contiguous(rates_full, mask)
        out["sce_rate_hz"] = len(sce) / (n_frames / FS)
        if sce:
            part = [float((spks_full[:, s:e] > 0).any(axis=1).mean()) for s, e in sce]
            out["sce_participation"] = float(np.mean(part))
            out["sce_duration_s"] = float(np.median([(e - s) / FS for s, e in sce]))
        else:
            out["sce_participation"] = np.nan
            out["sce_duration_s"] = np.nan
    else:
        out["sce_rate_hz"] = np.nan
        out["sce_participation"] = np.nan
        out["sce_duration_s"] = np.nan

    return out


def session_activity(mouse: str, date: str) -> Optional[Dict]:
    path = spks_path(mouse, date)
    if not path.exists():
        return None

    digest = load_digest(mouse, date)
    spks = np.load(path, mmap_mode="r")
    n_cells_total, n_frames = spks.shape
    if n_cells_total < N_CELLS:
        return None

    rng_cells = np.random.default_rng(0)
    keep_cells = np.sort(rng_cells.choice(n_cells_total, N_CELLS, replace=False))
    spks = np.asarray(spks)[keep_cells] > 0
    n_cells = N_CELLS
    onsets = event_onsets(spks)
    # float32, not bool: gaussian_filter1d preserves the input dtype, so smoothing a
    # boolean array returns all-False and every downstream SCE count is zero
    rates_full = smoothed_rates(spks.astype(np.float32), CONFIG)

    run_mask = np.zeros(n_frames, dtype=bool)
    run_frames = digest.run_frame[digest.run_frame < n_frames].astype(int)
    run_mask[run_frames] = True

    try:
        still_mask, _ = immobility_mask(mouse, date)
    except Exception:
        still_mask = np.zeros(n_frames, dtype=bool)
    if still_mask.size != n_frames:
        still_mask = np.zeros(n_frames, dtype=bool)

    rng = np.random.default_rng(0)
    row: Dict = dict(
        mouse=mouse, date=date, genotype=digest.genotype,
        session_type=digest.meta["session_type"],
        n_cells=n_cells, n_cells_total=n_cells_total, n_frames=n_frames,
        run_seconds=float(run_mask.sum() / FS),
        still_seconds=float(still_mask.sum() / FS),
        frac_session_still=float(still_mask.mean()),
    )

    for name, mask in (("run", run_mask), ("still", still_mask)):
        rates = _rates_hz(onsets, mask)
        finite = rates[np.isfinite(rates)]
        row[f"{name}_rate_median"] = float(np.median(finite)) if finite.size else np.nan
        row[f"{name}_rate_mean"] = float(np.mean(finite)) if finite.size else np.nan
        # Distribution shape: hyperactivity in APP mice is a heavy upper tail plus a
        # silent population, not a shift of the whole distribution, so the tails are
        # reported alongside the centre
        row[f"{name}_rate_p90"] = float(np.percentile(finite, 90)) if finite.size else np.nan
        row[f"{name}_frac_silent"] = (
            float(np.mean(finite < 0.01)) if finite.size else np.nan
        )
        if mask.sum() >= MIN_FRAMES:
            row.update({f"{name}_{k}": v for k, v in
                        _synchrony(spks[:, mask], rates_full[:, mask], rng,
                                   spks_full=spks, rates_full=rates_full,
                                   mask=mask).items()})

    return row


def activity_table() -> pd.DataFrame:
    led = sessions()
    print(f"cohort: {describe(led)}", flush=True)
    rows = []
    for i, r in enumerate(led.itertuples(), 1):
        try:
            row = session_activity(r.mouse, r.date)
            if row is not None:
                row.update(stage=r.stage, day=r.day)
                rows.append(row)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        if i % 10 == 0:
            print(f"  [{i}/{len(led)}] kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "activity_sessions.csv", index=False)
    return df


if __name__ == "__main__":
    df = activity_table()
    print(f"\n{len(df)} sessions, {df.mouse.nunique()} mice")
    print(df.groupby("genotype").agg(
        mice=("mouse", "nunique"), sessions=("date", "size"),
        still_sec=("still_seconds", "median"),
        still_rate=("still_rate_median", "median"),
        run_rate=("run_rate_median", "median"),
        pairwise_r=("still_mean_pairwise_r", "median"),
    ).to_string())
