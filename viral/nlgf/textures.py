"""Per-trial texture identity, cached.

The digests store texture_rewarded, which is what every within-session analysis needs.
Across a reversal it is the wrong label: the reward contingency flips while the physical
texture does not, so testing whether a representation is sensory or value-based requires
the texture ITSELF. That lives only in the session JSON, so it is pulled out once per
session and cached as a small npz.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.imaging_utils import trial_is_imaged
from viral.models import Cached2pSession
from viral.nlgf.paths import CACHED_2P, DERIVED

TEXTURE_CACHE = DERIVED / "textures"
TEXTURE_CACHE.mkdir(parents=True, exist_ok=True)


def texture_labels(mouse: str, date: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """(trial_index, texture_code, meta) for every trial in the session.

    texture_code is 0/1 assigned by sorting the texture names, so the SAME physical
    texture gets the same code in every session of every mouse - which is the whole
    point, since the mapping from texture to reward is what the reversal changes.
    """
    cache = TEXTURE_CACHE / f"{mouse}_{date}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["idx"], z["code"], dict(
            names=list(z["names"]), rewarded_code=int(z["rewarded_code"]))

    session = Cached2pSession.model_validate_json(
        (CACHED_2P / f"{mouse}_{date}.json").read_text())
    names = sorted({t.texture for t in session.trials})
    lookup = {n: i for i, n in enumerate(names)}

    idx, code, rewarded = [], [], []
    for i, t in enumerate(session.trials):
        try:
            if not trial_is_imaged(t):
                continue
        except (AssertionError, IndexError):
            continue
        idx.append(i)
        code.append(lookup[t.texture])
        rewarded.append(int(t.texture_rewarded))

    idx = np.asarray(idx, dtype=int)
    code = np.asarray(code, dtype=int)
    rewarded = np.asarray(rewarded, dtype=int)
    # Which texture code is the rewarded one in THIS session. Comparing this between
    # two sessions is how a reversal is detected from the data rather than from the
    # session_type string.
    rewarded_code = int(code[rewarded == 1][0]) if (rewarded == 1).any() else -1

    np.savez_compressed(cache, idx=idx, code=code, rewarded_code=rewarded_code,
                        names=np.array(names))
    return idx, code, dict(names=names, rewarded_code=rewarded_code)


if __name__ == "__main__":
    from viral.nlgf.cohort import sessions

    coh = sessions(min_cells=250, stages=["learning", "reversal"])
    for i, r in enumerate(coh.itertuples(), 1):
        try:
            _, _, meta = texture_labels(r.mouse, r.date)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        if i % 20 == 0:
            print(f"  [{i}/{len(coh)}]", flush=True)
    print("done", flush=True)
