"""Imaging measures for the Oligo-BACE1-KO arm, against both reference groups.

Each panel is one measure from the tiers that were already built for NLGF vs WT. The
Oligo rows come from `*_oligo.csv` (viral/nlgf/oligo_tiers.py) and are concatenated with
the existing tables; the per-session computation is identical, so a row means the same
thing in both.

Two things are drawn on every measure that the NLGF figures did not need:

  - the contrast is run twice, against NLGF (time-matched) and against WT (not), and
    only the NLGF version supports a genotype reading
  - alongside the WT contrast, the era effect measured inside NLGF, which is how large
    a difference this cohort produces from batch alone (viral/nlgf/era.py). A bar
    shorter than its era marker carries no evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.era import era_effect
from viral.nlgf.oligo import OLIGO
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import SHORT_NAME, p_bracket, set_style, superplot

# (table stem, column, label). Chosen as the headline measure of each tier.
MEASURES: List[Tuple[str, str, str]] = [
    ("activity_sessions", "run_rate_median", "event rate, running (Hz)"),
    ("activity_sessions", "still_rate_median", "event rate, immobile (Hz)"),
    ("activity_sessions", "still_sce_rate_hz", "synchronous event rate (Hz)"),
    ("activity_sessions", "still_sce_participation", "cells per synchronous event"),
    ("place_lap_matched", "frac_place_cells", "place cells (lap-matched)"),
    ("place_lap_matched", "field_width_cm", "field width (cm)"),
    ("place_lap_matched", "reliability_pc", "field reliability"),
    ("decoding_matched", "decoding_error_matched", "decoding error (cm)"),
]


def load(stem: str) -> pd.DataFrame:
    base = RESULTS / f"{stem}.csv"
    extra = RESULTS / f"{stem}_oligo.csv"
    frames = [pd.read_csv(p) for p in (base, extra) if p.exists()]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # session tables key on mouse+date; pair tables (stability) key on both dates
    key = (
        ["mouse", "date"]
        if {"mouse", "date"} <= set(df.columns)
        else (
            ["mouse", "date_a", "date_b"]
            if {"mouse", "date_a", "date_b"} <= set(df.columns)
            else None
        )
    )
    return df.drop_duplicates(subset=key, keep="last") if key else df


def figure(reference: str) -> pd.DataFrame:
    set_style()
    genotypes = [reference, OLIGO]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.2))
    rows = []
    for ax, (stem, col, label) in zip(axes.ravel(), MEASURES):
        df = load(stem)
        if df.empty or col not in df.columns:
            ax.axis("off")
            continue
        sub = df[df.genotype.isin(genotypes)].dropna(subset=[col])
        if sub.groupby("genotype").mouse.nunique().min() < 3:
            ax.axis("off")
            continue
        plt.figure()
        superplot(ax, sub, col, genotypes=genotypes)
        per_mouse, res = compare(sub, col, group_a=OLIGO, group_b=reference)
        p_bracket(ax, res["p"])
        ax.set_ylabel(label)

        era = era_effect(df, col, genotype="NLGF")
        ratio = (
            abs(res["diff"]) / abs(era["diff"])
            if np.isfinite(era.get("diff", np.nan)) and era["diff"]
            else np.nan
        )
        if reference == "WT" and np.isfinite(ratio):
            ax.set_title(
                f"vs batch: {ratio:.1f}x",
                loc="right",
                fontsize=8,
                color="#C92A2A" if ratio < 1 else "#495057",
            )
        rows.append(
            dict(
                reference=reference,
                measure=label,
                column=col,
                oligo=res["mean_a"],
                ref=res["mean_b"],
                diff=res["diff"],
                p=res["p"],
                era_diff=era.get("diff", np.nan),
                ratio=ratio,
                n_mice=int(per_mouse.mouse.nunique()),
            )
        )

    note = (
        "time-matched: every Oligo session falls inside the NLGF window"
        if reference == "NLGF"
        else "NOT time-matched: only 5% of Oligo sessions fall inside the WT window - "
        "'vs batch' is the difference divided by the NLGF era effect"
    )
    fig.suptitle(
        f"Oligo-BACE1-KO imaging vs {SHORT_NAME.get(reference, reference)}" f"\n{note}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = PLOTS / f"fig22_oligo_imaging_{reference.lower()}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tabs = [figure(ref) for ref in ("NLGF", "WT")]
    tab = pd.concat([t for t in tabs if not t.empty], ignore_index=True).round(4)
    tab.to_csv(RESULTS / "oligo_imaging_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + tab.to_string(index=False))
