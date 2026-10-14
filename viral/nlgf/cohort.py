"""Inclusion criteria, in one place.

Every tier selects sessions through here rather than filtering ad hoc, so what is in
an analysis is visible in a single file.

STAGES. Learning and reversal only. Recall and recall-reversal ran roughly two months
after the initial sessions and the imaging is not of comparable quality, so they are
excluded rather than pooled. Unsupervised sessions are excluded too: they carry no
reward contingency, which makes them meaningless for Tier 4 and not comparable for the
rest.

CELL FLOOR. Cell yield varies with injection quality, not genotype - it is not
significantly different between groups (643 vs 525, p = 0.35) but ranges from 133 to
958 across mice, with only 1.2-3.2x variation within a mouse. That makes it a nuisance
variable of exactly the kind that has to be matched by construction, since it was the
covariate that removed the apparent genotype effects in both Tier 2 and Tier 5.

Sessions below MIN_CELLS are dropped so that the matched subsample can be larger:
the floor and the subsample size are the same number, so every included session
contributes an equally sized, randomly drawn population. At 250 this keeps 104 of 114
sessions and costs one NLGF mouse (JB019, 119-161 cells). Set MIN_CELLS to 150 for a
sensitivity check that keeps all 8.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List

import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.load import ledger
from viral.nlgf.paths import spks_path
from viral.utils import read_npy_shape

STAGES: List[str] = ["learning", "reversal"]
MIN_CELLS = 250
GENOTYPES = ["WT", "NLGF"]


def n_cells(mouse: str, date: str) -> int:
    """Cell count from the .npy header, without loading the array."""
    p = spks_path(mouse, date)
    if not p.exists():
        return 0
    try:
        return int(read_npy_shape(p)[0])
    except Exception:  # noqa: BLE001
        return 0


def sessions(min_cells: int = MIN_CELLS, stages: List[str] | None = None,
             genotypes: List[str] | None = None) -> pd.DataFrame:
    d = ledger()
    d = d[d.has_spks]
    d = d[d.stage.isin(stages if stages is not None else STAGES)]
    d = d[d.genotype.isin(genotypes if genotypes is not None else GENOTYPES)]
    d = d.copy()
    d["n_cells"] = [n_cells(r.mouse, r.date) for r in d.itertuples()]
    d = d[d.n_cells >= min_cells]
    # Reversal day number, the experience axis for Tier 4
    d["day"] = d.session_type.str.extract(r"day\s*(\d+)", expand=False).astype(float)
    return d.reset_index(drop=True)


def describe(d: pd.DataFrame) -> str:
    g = d.groupby("genotype").agg(mice=("mouse", "nunique"), sessions=("date", "size"))
    return "  ".join(f"{k} {v.mice} mice/{v.sessions} sessions" for k, v in g.iterrows())


if __name__ == "__main__":
    for floor in (150, 250):
        d = sessions(min_cells=floor)
        print(f"min_cells={floor}: {describe(d)}")
        print(pd.crosstab(d.genotype, d.stage).to_string(), "\n")


@contextmanager
def arm(genotypes: List[str]):
    """Temporarily change which genotypes `sessions()` returns.

    The tier tables all call `sessions()` with no arguments, so they inherit the
    module-level GENOTYPES. Rebinding it here lets the same tier code run for a
    different genotype contrast without editing any of them, and guarantees the default
    is restored even if the tier raises - so an Oligo run cannot silently change what a
    later NLGF run includes.
    """
    global GENOTYPES
    previous = GENOTYPES
    GENOTYPES = list(genotypes)
    try:
        yield GENOTYPES
    finally:
        GENOTYPES = previous
