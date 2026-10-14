"""ITI quiescence masks, cached locally.

get_iti_still_frames carries the reasoning that matters here - the 3 s calcium-bleed
exclusion, the lick padding, Grosmark's stillness definition - so it is reused rather
than reimplemented. Its only server dependency is get_session_n_frames, which reads
the frame count from the suite2p .npy header on the share; that one function is
redirected at the local spks file, which has the same one-column-per-frame shape.

The redirection is done by rebinding the name inside viral.multiple_sessions rather
than editing it, and is scoped to the call.

Why the ITI at all: the freeze epochs are the cleaner offline data, but they exist for
only a subset of sessions and none of the older NLGF cohort. The ITI is 20 s per trial,
so a 60 trial session carries ~20 min of in-task quiescence, and it is available for
every session with imaging. That is what makes an offline comparison possible at
n = 6 NLGF against 8 WT rather than n = 1.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

import viral.multiple_sessions as ms
from viral.models import Cached2pSession
from viral.nlgf.paths import CACHED_2P, DERIVED, spks_path
from viral.utils import read_npy_shape

ITI_MASKS = DERIVED / "iti_masks"
ITI_MASKS.mkdir(parents=True, exist_ok=True)


def local_n_frames(mouse: str, date: str) -> int:
    p = spks_path(mouse, date)
    if not p.exists():
        raise FileNotFoundError(f"No local spks for {mouse} {date}")
    return int(read_npy_shape(p)[1])


@contextlib.contextmanager
def _local_frame_count():
    original = ms.get_session_n_frames
    ms.get_session_n_frames = local_n_frames
    try:
        yield
    finally:
        ms.get_session_n_frames = original


def iti_still_mask(mouse: str, date: str, use_cache: bool = True
                   ) -> Tuple[np.ndarray, pd.DataFrame]:
    cache = ITI_MASKS / f"{mouse}_{date}.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["mask"], pd.DataFrame(
            z["records"], columns=list(z["record_columns"])
        )

    session = Cached2pSession.model_validate_json(
        (CACHED_2P / f"{mouse}_{date}.json").read_text()
    )
    with _local_frame_count():
        mask, per_trial = ms.get_iti_still_frames(session)

    # 'frames' holds a variable-length array per trial and is dropped from the cache;
    # the session-wide mask carries the same information
    cols = [c for c in per_trial.columns if c != "frames"]
    np.savez_compressed(
        cache,
        mask=mask,
        records=per_trial[cols].to_numpy(dtype=float) if len(per_trial) else np.zeros((0, len(cols))),
        record_columns=np.array(cols),
    )
    return mask, per_trial[cols] if len(per_trial) else pd.DataFrame(columns=cols)


def build_all(overwrite: bool = False) -> None:
    from viral.nlgf.load import ledger

    led = ledger()
    led = led[led.has_spks]
    ok = fail = skip = 0
    for i, r in enumerate(led.itertuples(), 1):
        if (ITI_MASKS / f"{r.mouse}_{r.date}.npz").exists() and not overwrite:
            skip += 1
            continue
        try:
            mask, _ = iti_still_mask(r.mouse, r.date, use_cache=False)
            ok += 1
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"FAIL {r.mouse} {r.date}: {type(e).__name__}: {e}", flush=True)
        if i % 20 == 0:
            print(f"  [{i}/{len(led)}] ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"done: ok={ok} skipped={skip} failed={fail}", flush=True)


if __name__ == "__main__":
    build_all(overwrite="--overwrite" in sys.argv)
