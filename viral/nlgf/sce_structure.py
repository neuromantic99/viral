"""Is the STRUCTURE of synchronous events degraded in NLGF, with their rate preserved?

Shipley et al. (2026, Curr Biol), on this same App-NL-G-F line in hippocampus, found
reactivation RATE intact and replay STRUCTURE degraded - which cells were recruited and
how they co-fired. Tier 5 here measured only rate-like quantities (excess_r, mean_rz)
and found nothing, so it is silent on their actual claim rather than contradicting it.
This measures the structure.

It also fixes Tier 5's other weakness. That tier needed a wheel-freeze epoch and so ran
on 4 NLGF versus 6 WT mice - the smallest cohort in the study, and a weak null. Working
from in-task immobility instead uses every session, which is the same move that rescued
the synchrony analysis earlier.

Three structural measures, all computed on the participation matrix (cells x events,
1 if a cell fired during that event):

  recruitment concentration  Gini coefficient over per-cell participation counts. High
                             means a committed subset carries most events; low means
                             every event draws a fresh random crowd.
  cell co-participation      mean pairwise correlation between cells' participation
                             vectors. Cells that belong to the same assembly should
                             appear together.
  event similarity           mean pairwise cosine similarity between events. High means
                             events repeat the same cell set, i.e. a stereotyped
                             pattern rather than noise.

TWO CHOICES THAT DECIDED THE ANSWER, both got wrong on the first pass.

  THE NULL. The original version shuffled which cells joined each event while fixing
  every event's size AND every cell's rate. That is so constrained the observed matrix
  has almost nowhere to move, and every measure came back sitting on its null - which
  was read as "there is no assembly structure here". It was a property of the null. The
  null used now independently CIRCULARLY SHIFTS each cell's trace: rate and
  autocorrelation are preserved exactly, and only cross-cell timing is destroyed, which
  is where assembly structure actually lives. Against it the same data give z of +15 to
  +73. The structure was there the whole time.

  THE REPRESENTATION. Participation was binarised - 1 if a cell fired at all during the
  event - which discards amplitude. That is inconsistent on its face, since SCE
  DETECTION uses the graded smoothed rate; the amplitude was carefully kept and then
  thrown away one step later. Scoring participation as the mean denoised z over the
  event roughly quintuples the co-participation z (15.6 -> 73.0). Binary also inflates
  its own null (0.049 vs 0.000), because events differ in size and every cell then
  co-varies with event size; the graded null is clean.

Cells are subsampled to N_CELLS and events to N_EVENTS before anything is computed,
because all three measures are biased by the size of the matrix they are estimated from.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.chang import smoothed_rates
from viral.nlgf.activity import CONFIG, FS, detect_sce_contiguous, event_onsets
from viral.nlgf.cohort import sessions
from viral.nlgf.immobility import immobility_mask
from viral.nlgf.load import load_digest
from viral.nlgf.paths import DFF, RESULTS, spks_path

N_CELLS = 250
N_EVENTS = 20          # matched across sessions; pairwise measures are biased by count
N_SHUFFLES = 100
GRADED = True          # score participation from the denoised trace, not spks > 0


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _measures(P: np.ndarray) -> Dict[str, float]:
    """Structure of one participation matrix (cells x events). Works graded or binary."""
    counts = P.sum(axis=1)
    out = dict(recruitment_gini=_gini(np.clip(counts, 0, None)))

    A = P - P.mean(axis=1, keepdims=True)
    keep = A.std(axis=1) > 0
    if keep.sum() > 2:
        C = np.corrcoef(A[keep])
        iu = np.triu_indices_from(C, k=1)
        out["cell_coparticipation"] = float(np.nanmean(C[iu]))
    else:
        out["cell_coparticipation"] = np.nan

    norms = np.linalg.norm(P, axis=0)
    ok = norms > 0
    if ok.sum() > 2:
        E = P[:, ok] / norms[ok]
        S = E.T @ E
        iu = np.triu_indices_from(S, k=1)
        out["event_similarity"] = float(np.nanmean(S[iu]))
    else:
        out["event_similarity"] = np.nan
    return out


def _participation(trace: np.ndarray, events, graded: bool) -> np.ndarray:
    P = np.zeros((trace.shape[0], len(events)), dtype=np.float32)
    for j, (a, b) in enumerate(events):
        seg = trace[:, a:b + 1]
        P[:, j] = seg.mean(axis=1) if graded else (seg > 0).any(axis=1)
    return P


def _null(trace: np.ndarray, events, graded: bool,
          rng: np.random.Generator) -> Dict[str, float]:
    """Circular-shift null: each cell's trace rolled independently.

    Every cell keeps its rate and its autocorrelation exactly; only the timing BETWEEN
    cells is destroyed. That is the minimal null for a co-activation claim. The earlier
    margin-preserving shuffle also fixed each event's size, which removed the variance
    the signal is carried in and made real structure look absent.
    """
    n_frames = trace.shape[1]
    acc: Dict[str, list] = {}
    for _ in range(N_SHUFFLES):
        shifts = rng.integers(0, n_frames, size=trace.shape[0])
        shifted = np.stack([np.roll(trace[i], int(k)) for i, k in enumerate(shifts)])
        for key, val in _measures(_participation(shifted, events, graded)).items():
            acc.setdefault(key, []).append(val)
    out = {}
    for k, v in acc.items():
        out[k] = float(np.nanmean(v))
        out[f"{k}__sd"] = float(np.nanstd(v))
    return out


def session_sce_structure(mouse: str, date: str) -> Optional[Dict]:
    path = spks_path(mouse, date)
    if not path.exists():
        return None
    spks_all = np.load(path, mmap_mode="r")
    n_total, n_frames = spks_all.shape
    if n_total < N_CELLS:
        return None

    rng = np.random.default_rng(abs(hash((mouse, date))) % (2**32))
    keep = np.sort(rng.choice(n_total, N_CELLS, replace=False))
    spks = np.asarray(spks_all)[keep] > 0
    rates = smoothed_rates(spks.astype(np.float32), CONFIG)

    # events are always detected from the deconvolved MUA, exactly as Chang et al. do;
    # only how a cell's PARTICIPATION in them is scored changes with GRADED
    trace = spks.astype(np.float32)
    if GRADED:
        dn = DFF / f"{mouse}_{date}_denoised.npy"
        if not dn.exists():
            return None
        d_arr = np.asarray(np.load(dn, mmap_mode="r"))[keep].astype(np.float32)
        if d_arr.shape[1] != spks.shape[1]:
            return None
        trace = (d_arr - d_arr.mean(axis=1, keepdims=True)) / (
            d_arr.std(axis=1, keepdims=True) + 1e-9)

    try:
        still, _ = immobility_mask(mouse, date)
    except Exception:  # noqa: BLE001
        return None
    if still.size != n_frames or still.sum() < 10 * FS:
        return None

    events = detect_sce_contiguous(rates, still)
    if len(events) < N_EVENTS:
        return None
    chosen = sorted(rng.choice(len(events), N_EVENTS, replace=False))
    events = [events[i] for i in chosen]
    P = _participation(trace, events, GRADED)

    obs = _measures(P)
    null = _null(trace, events, GRADED, rng)
    digest = load_digest(mouse, date)
    row = dict(mouse=mouse, date=date, genotype=digest.genotype,
               session_type=digest.meta["session_type"],
               n_events_total=len(events), n_cells=N_CELLS, graded=GRADED,
               still_seconds=float(still.sum() / FS),
               mean_event_size=float((P > 0).sum(axis=0).mean()))
    for k in obs:
        mu, sd = null.get(k, np.nan), null.get(f"{k}__sd", np.nan)
        row[k] = obs[k]
        row[f"{k}_null"] = mu
        row[f"{k}_null_sd"] = sd
        row[f"{k}_excess"] = obs[k] - mu
        # z of the observed value against its own session's null distribution
        row[f"{k}_z"] = (obs[k] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    return row


def structure_table() -> pd.DataFrame:
    led = sessions()
    rows = []
    for i, r in enumerate(led.itertuples(), 1):
        try:
            out = session_sce_structure(r.mouse, r.date)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
            continue
        if out is not None:
            out.update(stage=r.stage)
            rows.append(out)
        if i % 10 == 0:
            print(f"  [{i}/{len(led)}] kept {len(rows)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "sce_structure.csv", index=False)
    print(f"wrote {len(df)} sessions")
    return df


if __name__ == "__main__":
    df = structure_table()
    if df.empty:
        sys.exit()
    from viral.nlgf.stats import exact_permutation

    for col in ["recruitment_gini_excess", "cell_coparticipation_excess",
                "event_similarity_excess"]:
        s = df[df.genotype.isin(["WT", "NLGF"])].dropna(subset=[col])
        pm = s.groupby(["mouse", "genotype"], as_index=False)[col].median()
        if pm.groupby("genotype").size().min() < 3:
            continue
        r = exact_permutation(pm, col)
        print(f"  {col:32s} NLGF {r['mean_a']:+.4f}  WT {r['mean_b']:+.4f}  "
              f"diff {r['diff']:+.4f}  p = {r['p']:.3f}  ({r['n_a']}v{r['n_b']} mice)")
