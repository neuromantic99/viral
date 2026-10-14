"""Putting a number on the batch confound that separates the Oligo and WT cohorts.

The Oligo-BACE1-KO mice were recorded Oct-Dec 2024 and the WT mice from Dec 2024
onwards, so "Oligo vs WT" and "earlier vs later" are the same contrast and no
statistical adjustment can separate them - there is no common support to adjust over.
That much is unfixable.

What is still available is a bound. NLGF is the only genotype recorded in BOTH eras
(51% of its sessions fall in the Oligo window, 49% in the WT window), so the size of a
pure era effect can be measured inside NLGF, where genotype is held constant. For each
measure:

    era_effect = mean(NLGF recorded in the Oligo era) - mean(NLGF recorded in the WT era)

That is not a correction. It is a yardstick: an Oligo-vs-WT difference smaller than the
NLGF era effect on the same measure is fully consistent with being nothing but batch,
and should not be reported as a genotype effect at all. One several times larger is at
least not explained by the era effect this cohort exhibits - though it is still not
cleanly attributable, since the eras differ for the Oligo mice in ways they may not for
the NLGF mice.

The era effect is itself a between-mouse comparison (different NLGF animals were run in
the two windows), which is the right construction here: that is exactly the kind of
difference the Oligo-vs-WT contrast is exposed to.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.load import ledger
from viral.nlgf.paths import RESULTS
from viral.nlgf.stats import exact_permutation

# First WT session. Everything before it is the Oligo era, everything from it the WT era.
SPLIT_DATE = "2024-12-10"


def label_era(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = df.copy()
    d["era"] = np.where(pd.to_datetime(d[date_col]) < pd.Timestamp(SPLIT_DATE),
                        "early (Oligo era)", "late (WT era)")
    return d


def with_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attach recording date if the table carries only mouse/session identifiers."""
    if "date" in df.columns:
        return df
    if "session_type" not in df.columns:
        raise KeyError("no date and no session_type to merge one onto")
    led = ledger()[["mouse", "date", "session_type"]].copy()
    # behaviour_full spells session names slightly differently from the digest meta
    led["key"] = led.mouse + "|" + led.session_type.str.lower().str.strip()
    d = df.copy()
    d["key"] = d.mouse + "|" + d.session_type.str.lower().str.strip()
    d = d.merge(led[["key", "date"]].drop_duplicates("key"), on="key", how="left")
    # a session with no digest has no date and cannot be placed in an era
    return d.dropna(subset=["date"])


def era_effect(df: pd.DataFrame, value: str, genotype: str = "NLGF") -> dict:
    """Early-vs-late difference within one genotype, as an exact permutation over mice."""
    d = label_era(with_dates(df))
    d = d[(d.genotype == genotype)].dropna(subset=[value])
    if d.empty:
        return dict(diff=np.nan, p=np.nan, n_a=0, n_b=0)
    per_mouse = (d.groupby(["mouse", "era"], as_index=False)[value].median()
                 .rename(columns={"era": "genotype"}))
    # a mouse recorded either side of the split would appear twice; none are, but guard
    per_mouse = per_mouse.drop_duplicates(subset=["mouse"], keep="first")
    return exact_permutation(per_mouse, value,
                             group_a="early (Oligo era)", group_b="late (WT era)")


def confound_scale(df: pd.DataFrame, value: str, group_a: str, group_b: str,
                   anchor: str = "NLGF") -> dict:
    """The genotype difference, the era effect, and the ratio between them.

    ratio < 1  the genotype difference is smaller than a pure batch effect on this
               measure, so it carries no evidence at all
    ratio > 3  larger than the batch effect this cohort shows, which is necessary but
               not sufficient for a genotype reading
    """
    d = with_dates(df).dropna(subset=[value])
    per_mouse = d.groupby(["mouse", "genotype"], as_index=False)[value].median()
    geno = exact_permutation(per_mouse, value, group_a=group_a, group_b=group_b)
    era = era_effect(df, value, genotype=anchor)
    ratio = (abs(geno["diff"]) / abs(era["diff"])
             if np.isfinite(geno.get("diff", np.nan))
             and np.isfinite(era.get("diff", np.nan)) and era["diff"] else np.nan)
    return dict(measure=value, genotype_diff=geno.get("diff", np.nan),
                genotype_p=geno.get("p", np.nan),
                era_diff=era.get("diff", np.nan), era_p=era.get("p", np.nan),
                ratio=ratio,
                verdict=("no evidence beyond batch" if np.isfinite(ratio) and ratio < 1
                         else "comparable to batch" if np.isfinite(ratio) and ratio < 3
                         else "exceeds batch scale" if np.isfinite(ratio) else "n/a"))


def table(df: pd.DataFrame, values: List[str], group_a: str, group_b: str,
          anchor: str = "NLGF") -> pd.DataFrame:
    rows = []
    for v in values:
        if v not in df.columns:
            continue
        try:
            rows.append(confound_scale(df, v, group_a, group_b, anchor))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {v}: {type(e).__name__}: {e}")
    return pd.DataFrame(rows).round(4)


if __name__ == "__main__":
    from viral.nlgf.oligo import OLIGO

    beh = pd.read_csv(RESULTS / "behaviour_full.csv")
    beh = beh[beh.rig == "2P"]
    beh = beh.rename(columns={"session_name": "session_type"})
    values = ["licking_dprime", "n_trials", "mean_trial_speed", "frac_reward_drunk"]
    pd.set_option("display.width", 220)
    print("=== behaviour: Oligo vs WT, against the NLGF era effect ===")
    print(table(beh, values, OLIGO, "WT").to_string(index=False))
    print("\n=== for reference, Oligo vs NLGF (time-matched, no era gap) ===")
    print(table(beh, values, OLIGO, "NLGF").to_string(index=False))
