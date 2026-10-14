"""The Oligo-BACE1-KO arm: what data exists, and how it can and cannot be compared.

THE TIMING PROBLEM, which governs everything else in this module.

The five Oligo-BACE1-KO mice (JB014, JB015, JB018, JB020, JB022) were run between
2024-10-24 and 2024-12-12. The WT mice were run from 2024-12-10 onwards, all but two of
them in 2025 or 2026. The two cohorts therefore barely overlap in time at all: nearly
every Oligo session precedes nearly every WT session, so an Oligo-vs-WT difference and
a "recorded earlier" difference are the same contrast and cannot be told apart. Batch,
rig drift, surgical practice and the experimenter's own experience all move with date.

The NLGF mice, by contrast, ARE contemporaneous - JB011, JB016, JB019 and JB021 span
2024-10-22 to 2025-02-19, straddling the Oligo window. So:

    Oligo vs NLGF   time-matched, interpretable
    Oligo vs WT     confounded with date; reported, but as a descriptive reference only

Which of these is the biologically meaningful contrast depends on the genetic
background of the Oligo-BACE1-KO line, which is not recorded anywhere in this repo. If
these animals are on an App-NL-G-F background then Oligo-vs-NLGF is the rescue
experiment and is both the right comparison and the well-controlled one. If they are on
a wild-type background then the comparison the biology wants is Oligo-vs-WT, which is
exactly the one the timing has confounded, and the honest answer is that this design
cannot deliver it from these data alone. Both are computed here; the interpretation
needs the background.

The date confound is quantified rather than asserted: `date_overlap()` reports it, and
every genotype figure carries it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.cohort import MIN_CELLS, STAGES, arm, sessions
from viral.nlgf.load import ledger
from viral.nlgf.paths import RESULTS, spks_path

OLIGO = "Oligo-BACE1-KO"
CONTRASTS: List[List[str]] = [["NLGF", OLIGO], ["WT", OLIGO]]


def date_overlap(genotypes: List[str] = (OLIGO, "WT", "NLGF")) -> pd.DataFrame:
    """Date range per genotype, and the fraction of each pairing's sessions that
    fall inside the other's window. This is the number that decides whether a
    genotype contrast is a genotype contrast."""
    led = ledger()
    led = led[led.stage.isin(STAGES)]
    d = pd.to_datetime(led.date)
    rows = []
    for g in genotypes:
        for h in genotypes:
            if g == h:
                continue
            other = d[led.genotype == h]
            if other.empty:
                continue
            mine = d[led.genotype == g]
            inside = ((mine >= other.min()) & (mine <= other.max())).mean()
            rows.append(dict(genotype=g, versus=h, first=str(mine.min().date()),
                             last=str(mine.max().date()), n=len(mine),
                             frac_inside_other_window=round(float(inside), 3)))
    return pd.DataFrame(rows)


def inventory() -> pd.DataFrame:
    """Every Oligo session, with whether it has imaging and whether it is analysable."""
    led = ledger()
    led = led[led.genotype == OLIGO].copy()
    led["n_cells"] = [
        int(np.load(spks_path(r.mouse, r.date), mmap_mode="r").shape[0])
        if spks_path(r.mouse, r.date).exists() else -1
        for r in led.itertuples()]
    led["has_imaging"] = led.n_cells >= 0
    led["eligible"] = led.has_imaging & led.stage.isin(STAGES) & (led.n_cells >= MIN_CELLS)
    return led[["mouse", "date", "session_type", "stage", "n_cells",
                "has_imaging", "eligible"]].sort_values(["mouse", "date"])


def missing_spks() -> pd.DataFrame:
    """Oligo sessions in a learning/reversal stage whose spks.npy is not on the drive.

    These are the files to copy if the arm is to be usable; they are not missing at
    random with respect to stage, which matters most for anything reversal-based.
    """
    inv = inventory()
    return inv[(~inv.has_imaging) & inv.stage.isin(STAGES)]


def cohort_summary() -> pd.DataFrame:
    out = []
    for contrast in CONTRASTS:
        with arm(contrast):
            d = sessions()
        g = d.groupby("genotype").agg(mice=("mouse", "nunique"), sessions=("date", "size"))
        for genotype, row in g.iterrows():
            out.append(dict(contrast=" vs ".join(contrast), genotype=genotype,
                            mice=int(row.mice), sessions=int(row.sessions)))
    return pd.DataFrame(out).drop_duplicates()


def registration_pairs(min_cells: int = MIN_CELLS) -> pd.DataFrame:
    """Consecutive eligible imaging sessions per Oligo mouse, for ROICaT.

    Consecutive rather than all-pairs: the cross-day analyses compare a session with
    the next one, and registering every pair costs quadratically for sessions no
    analysis uses.
    """
    with arm([OLIGO]):
        d = sessions(min_cells=min_cells)
    rows = []
    for mouse, g in d.sort_values("date").groupby("mouse"):
        dates = g.date.tolist()
        for a, b in zip(dates[:-1], dates[1:]):
            gap = (pd.to_datetime(b) - pd.to_datetime(a)).days
            rows.append(dict(mouse=mouse, genotype=OLIGO, pre=a, post=b, gap=gap,
                             stage_a=g[g.date == a].stage.iloc[0],
                             stage_b=g[g.date == b].stage.iloc[0]))
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    # ROICaT needs stat/ops/iscell alongside spks; a pair missing any of them kills the
    # run mid-way, so they are checked here rather than discovered overnight.
    from viral.nlgf.register_roicat import DFF, SUITE2P_FILES

    pairs["files_ready"] = [
        all((DFF / f"{r.mouse}_{dt}_{k}.npy").exists()
            for dt in (r.pre, r.post) for k in SUITE2P_FILES)
        for r in pairs.itertuples()]
    return pairs


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    inv = inventory()
    print("=== Oligo-BACE1-KO inventory ===")
    print(f"{len(inv)} digested sessions, {inv.mouse.nunique()} mice; "
          f"{int(inv.has_imaging.sum())} with imaging; "
          f"{int(inv.eligible.sum())} analysable "
          f"(learning/reversal, >= {MIN_CELLS} cells)")
    print(pd.crosstab(inv.mouse, inv.stage).to_string())
    print("\neligible sessions per mouse and stage:")
    print(pd.crosstab(inv[inv.eligible].mouse, inv[inv.eligible].stage).to_string())

    miss = missing_spks()
    print(f"\n=== missing spks.npy in learning/reversal: {len(miss)} ===")
    if len(miss):
        print(miss[["mouse", "date", "session_type", "stage"]].to_string(index=False))

    print("\n=== date overlap (the confound) ===")
    print(date_overlap().to_string(index=False))

    print("\n=== cohort sizes per contrast ===")
    print(cohort_summary().to_string(index=False))

    pairs = registration_pairs()
    pairs.to_csv(RESULTS / "oligo_pairs_to_register.csv", index=False)
    ready = int(pairs.files_ready.sum()) if len(pairs) else 0
    print(f"\n=== {len(pairs)} ROICaT pairs ({ready} with stat/ops/iscell present) "
          f"written to oligo_pairs_to_register.csv ===")
    print(pairs.to_string(index=False))

    inv.to_csv(RESULTS / "oligo_inventory.csv", index=False)
