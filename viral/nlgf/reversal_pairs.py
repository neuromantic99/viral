"""Pair list for registration: pre-reversal against a session the animal has ACTUALLY reversed.

The consecutive-pair registration produced last-learning -> FIRST-reversal pairs, which
is the worst possible choice for a texture-versus-value question: on reversal day 1 the
animal usually has not re-evaluated, so nothing has changed for it and a code that does
not change proves nothing. Measured across 22 mice, only 3 reach licking d' > 1.0 on
day 1; the median is day 2, and three mice never reach it at all.

The consequence showed up directly: across the 7 day-1 pairs, texture transfer tracked
how much the animal had learned (Spearman rho = -0.72, exact p = 0.077). The two
animals that had not switched behaviourally showed the HIGHEST transfer.

So this generates the pair that actually asks the question: each mouse's last learning
session against its earliest reversal session where behaviour shows the contingency has
been re-learned. Both sessions must have imaging and the suite2p files registration
needs.

CRITERION is licking d' >= D_PRIME_CRITERION on the reversal session. Chosen over "a
fraction of the mouse's own learning asymptote" because that version fails for animals
with a very high learning d' - JB017 reaches 2.21 on reversal day 2, plainly reversed,
but that is only 69% of its 3.18 learning best.

The gap is reported per pair and is longer than the consecutive pairs, since the
criterion session may be several days after the last learning session. Registration
quality falls with gap, so those columns should be checked before the pairs are used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.cohort import sessions
from viral.nlgf.paths import DFF, RESULTS

D_PRIME_CRITERION = 1.0
SUITE2P_FILES = ("stat", "ops", "iscell")


def _has_files(mouse: str, date: str) -> bool:
    return all((DFF / f"{mouse}_{date}_{k}.npy").exists() for k in SUITE2P_FILES)


def build() -> pd.DataFrame:
    coh = sessions(min_cells=250, stages=["learning", "reversal"]).copy()
    coh["when"] = pd.to_datetime(coh.date)
    beh = pd.read_csv(RESULTS / "behaviour_full.csv")
    beh["key"] = beh.mouse + "|" + beh.session_name.str.lower().str.strip()
    coh["key"] = coh.mouse + "|" + coh.session_type.str.lower().str.strip()
    coh = coh.merge(beh[["key", "licking_dprime", "session_in_stage"]].drop_duplicates("key"),
                    on="key", how="left")

    rows = []
    for mouse, g in coh.sort_values("when").groupby("mouse"):
        learning = g[g.stage == "learning"]
        reversal = g[(g.stage == "reversal") &
                     (g.licking_dprime >= D_PRIME_CRITERION)]
        if learning.empty or reversal.empty:
            rows.append(dict(mouse=mouse, genotype=g.genotype.iloc[0], pre=None,
                             post=None, reason="no imaged session meets criterion"))
            continue
        pre, post = learning.iloc[-1], reversal.iloc[0]
        rows.append(dict(
            mouse=mouse, genotype=pre["genotype"], pre=pre["date"], post=post["date"],
            gap=(post["when"] - pre["when"]).days,
            pre_type=pre["session_type"], post_type=post["session_type"],
            post_dprime=round(float(post["licking_dprime"]), 2),
            files_ok=_has_files(mouse, pre["date"]) and _has_files(mouse, post["date"]),
            reason=""))
    df = pd.DataFrame(rows)
    usable = df[df.pre.notna() & df.files_ok.fillna(False)].copy()
    usable = usable[["mouse", "genotype", "pre", "post", "gap", "post_dprime"]]
    usable.to_csv(RESULTS / "reversal_pairs_to_register.csv", index=False)
    return df, usable


if __name__ == "__main__":
    df, usable = build()
    print("All mice:")
    print(df.to_string(index=False))
    print(f"\n{len(usable)} pairs ready to register "
          f"(NLGF {(usable.genotype=='NLGF').sum()}, WT {(usable.genotype=='WT').sum()})")
    print(f"  gap: median {usable.gap.median():.0f} d, range {usable.gap.min()}-{usable.gap.max()} d")
    already = sum((RESULTS / f"matches_{r.mouse}_{r.pre}_{r.post}.csv").exists()
                  for r in usable.itertuples())
    print(f"  already registered from the consecutive run: {already}")
    print(f"\nwritten to {RESULTS / 'reversal_pairs_to_register.csv'}")
