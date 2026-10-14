"""Run the per-session tier measures over the Oligo-BACE1-KO sessions.

Deliberately NOT done by running each tier's own table builder under a genotype arm:
those builders write to fixed filenames, so an Oligo run would overwrite the NLGF/WT
tables that every existing figure reads. Instead the per-session functions are called
directly and the results land in parallel `*_oligo.csv` files, which the figures then
concatenate with the existing tables. Nothing already computed is touched.

The measures themselves are genotype-blind - the arm only ever decided which sessions
were included - so a row computed here is identical to one computed in the original
run for the same session.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.cohort import arm, sessions
from viral.nlgf.oligo import OLIGO
from viral.nlgf.paths import RESULTS


def _run(name: str, fn: Callable, led: pd.DataFrame, many: bool = False) -> pd.DataFrame:
    """Apply `fn(mouse, date)` across sessions, keeping failures visible but non-fatal."""
    rows: List[Dict] = []
    for i, r in enumerate(led.itertuples(), 1):
        try:
            got = fn(r.mouse, r.date)
            if got is None:
                continue
            recs = got if many else [got]
            for rec in recs:
                rec.update(stage=r.stage, day=getattr(r, "day", np.nan))
                rows.append(rec)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {name} {r.mouse} {r.date}: {type(e).__name__}: {e}",
                  flush=True)
            traceback.print_exc()
        if i % 5 == 0:
            print(f"  [{name} {i}/{len(led)}] {len(rows)} rows", flush=True)
    df = pd.DataFrame(rows)
    out = RESULTS / f"{name}_oligo.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)", flush=True)
    return df


def main() -> None:
    with arm([OLIGO]):
        led = sessions()
    print(f"{len(led)} eligible Oligo sessions, {led.mouse.nunique()} mice", flush=True)
    print(led.groupby(["mouse", "stage"]).size().to_string(), flush=True)

    from viral.nlgf.activity import session_activity
    from viral.nlgf.decoding_matched import session_matched_decoding
    from viral.nlgf.place import session_place
    from viral.nlgf.place_matched import session_place_matched
    from viral.nlgf.reactivation import session_reactivation

    print("\n--- Tier 1: activity and synchrony ---", flush=True)
    _run("activity_sessions", session_activity, led)

    print("\n--- Tier 2: place coding (all trials) ---", flush=True)
    _run("place_sessions_rewarded_None", lambda m, d: session_place(m, d, None), led)

    print("\n--- Tier 2: place coding, lap-matched ---", flush=True)
    _run("place_lap_matched", session_place_matched, led)

    print("\n--- Tier 2: position decoding, cell-matched ---", flush=True)
    _run("decoding_matched", session_matched_decoding, led)

    print("\n--- Tier 5: offline reactivation ---", flush=True)
    have_freeze = led[led.has_freeze & led.freeze_usable] if "has_freeze" in led else led
    print(f"{len(have_freeze)} sessions with a usable freeze epoch", flush=True)
    _run("reactivation_freeze_poolmatched", session_reactivation, have_freeze, many=True)


if __name__ == "__main__":
    main()
