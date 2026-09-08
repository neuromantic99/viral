"""The Oligo-BACE1-KO arm read as what it is: a rescue experiment.

The Oligo-BACE1-KO mice are on an App-NL-G-F background, so deleting oligodendrocytic
BACE1 is an intervention applied ON TOP of the amyloid model, not a separate genotype.
That fixes the logic of every panel:

    NLGF    the disease baseline, and the control this arm is tested against
    Oligo   the same disease plus the intervention
    WT      the healthy target - how far a rescue would have to go

and it fixes the arithmetic. A rescue is only meaningful where NLGF actually differs
from WT: those are the measures with something to rescue. The rescue index

    (Oligo - NLGF) / (WT - NLGF)

is 0 for no rescue, 1 for a full return to WT, and negative for a worsening. It is
undefined - and deliberately not drawn - where WT and NLGF do not differ, because
dividing by a null denominator manufactures large numbers out of noise.

The Oligo-vs-NLGF contrast is also the only well-controlled one here: every Oligo
session falls inside the NLGF recording window, whereas only 5% fall inside the WT
window (see era.py). So the comparison the biology wants and the comparison the data
supports are, for once, the same one.
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

from viral.nlgf.fig_oligo_imaging import MEASURES as TIER_MEASURES, load
from viral.nlgf.oligo import OLIGO
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.matching import gap_adjusted, histogram_matched
from viral.nlgf.stats import compare, exact_permutation
from viral.nlgf.style import (
    GENOTYPE_COLOURS,
    SHORT_NAME,
    p_bracket,
    set_style,
    superplot,
)

ORDER = ["WT", "NLGF", OLIGO]

# Cross-day stability is pair-level, not session-level, and lives in one table that
# already carries all three genotypes, so it does not need the _oligo concatenation.
STABILITY = [
    ("field_corr_ratio", "field stability\n(cross-day / within-day)"),
    ("field_corr_cross", "cross-day field correlation"),
    ("pv_corr_ratio", "PV correlation ratio"),
    ("centroid_shift_cm", "centroid shift (cm)"),
]
MEASURES = TIER_MEASURES[4:] + [("spatial_stability", c, l) for c, l in STABILITY]
MIN_DISEASE_P = 0.20  # WT-vs-NLGF must be at least this suggestive to call it rescuable

# Which way is healthy. The rescue index is only meaningful when NLGF sits on the
# IMPAIRED side of WT: where NLGF is already better than WT there is no deficit, and
# "moving toward WT" would mean getting worse. Decoding error is the live case here -
# NLGF decodes position better than WT - so labelling it a rescue would invert the
# claim. Measures with no agreed good direction are never labelled.
HIGHER_IS_BETTER = {
    "reliability_pc": True,
    "decoding_error_matched": False,
    "field_corr_ratio": True,
    "field_corr_cross": True,
    "pv_corr_ratio": True,
    "centroid_shift_cm": False,
}


def _contrast(df: pd.DataFrame, col: str, a: str, b: str) -> dict:
    """Gap-matched where the table carries a gap, raw otherwise.

    Cross-day measures decay with the interval between sessions (~0.05-0.08 per day for
    field correlation), and the groups do not have the same gap distribution, so the
    raw contrast on those measures is part genotype and part calendar. Session-level
    tables have no gap and are unaffected.
    """
    sub = df[df.genotype.isin([a, b])]
    if "gap_days" not in sub.columns:
        pm = sub.groupby(["mouse", "genotype"], as_index=False)[col].median()
        return dict(exact_permutation(pm, col, group_a=a, group_b=b), matched=False)
    pm = sub.groupby(["mouse", "genotype"], as_index=False)[col].median()
    raw = exact_permutation(pm, col, group_a=a, group_b=b)
    # headline: adjust the trend out and keep every pair. Histogram matching is carried
    # alongside as a conservative bound - it is unbiased in principle but discards a
    # third of the pairs and destabilises which mice contribute, which at 4-8 animals
    # costs more than the bias it removes.
    adj, _ = gap_adjusted(sub, col, "gap_days")
    pma = adj.groupby(["mouse", "genotype"], as_index=False).adj.median()
    a_res = exact_permutation(pma, "adj", group_a=a, group_b=b)
    m = histogram_matched(sub, col, "gap_days", a, b)
    return dict(
        raw,
        diff=a_res["diff"],
        p=a_res["p"],
        raw_diff=raw["diff"],
        raw_p=raw["p"],
        strict_diff=m["diff"],
        strict_p=m["p"],
        matched=True,
    )


def _stats(df: pd.DataFrame, col: str) -> dict:
    disease = _contrast(df, col, "NLGF", "WT")
    rescue = _contrast(df, col, OLIGO, "NLGF")
    denom = -disease["diff"]  # WT - NLGF
    index = rescue["diff"] / denom if denom else np.nan
    good_high = HIGHER_IS_BETTER.get(col)
    impaired = good_high is not None and (
        (disease["diff"] < 0) if good_high else (disease["diff"] > 0)
    )
    rescuable = np.isfinite(disease["p"]) and disease["p"] <= MIN_DISEASE_P and impaired
    return dict(
        measure=col,
        wt=disease["mean_b"],
        nlgf=disease["mean_a"],
        oligo=rescue["mean_a"],
        disease_diff=disease["diff"],
        disease_p=disease["p"],
        rescue_diff=rescue["diff"],
        rescue_p=rescue["p"],
        gap_matched=rescue["matched"],
        rescue_raw_diff=rescue.get("raw_diff", np.nan),
        rescue_raw_p=rescue.get("raw_p", np.nan),
        disease_strict_p=disease.get("strict_p", np.nan),
        rescue_strict_p=rescue.get("strict_p", np.nan),
        rescue_index=index if rescuable else np.nan,
        rescuable=rescuable,
        nlgf_impaired=impaired,
    )


def main() -> None:
    set_style()
    rows = []
    # fig, axes = plt.subplots(2, 4, figsize=(16, 8.8))
    # for ax, (stem, col, label) in zip(axes.ravel(), MEASURES):
    for stem, col, label in MEASURES:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        df = load(stem)
        if df.empty or col not in df.columns:
            ax.axis("off")
            continue
        sub = df[df.genotype.isin(ORDER)].dropna(subset=[col])
        if sub.groupby("genotype").mouse.nunique().min() < 3:
            ax.axis("off")
            continue
        superplot(ax, sub, col, genotypes=ORDER)
        s = _stats(sub, col)
        rows.append(dict(s, label=label))
        # both planned contrasts: the deficit that establishes there is something to
        # rescue, and the intervention test itself. Oligo-vs-WT is deliberately absent -
        # a non-significant difference from WT is not evidence of rescue (three_group.py)
        # p_bracket(ax, s["disease_p"], x0=0, x1=1, label="deficit")
        # p_bracket(ax, s["rescue_p"], x0=1, x1=2, label="rescue")

        p_bracket(ax, s["disease_p"], x0=0, x1=1, label=None)
        p_bracket(ax, s["rescue_p"], x0=1, x1=2, label=None)
        ax.set_ylabel(label)
        if s["rescuable"]:
            strong = s["rescue_index"] > 0.3 and s["rescue_p"] < 0.05
            ax.set_title(
                f"rescue {s['rescue_index']:+.0%}"
                + (" (gap-adj)" if s["gap_matched"] else ""),
                loc="right",
                fontsize=8,
                color="#2B8A3E" if strong else "#868E96",
            )
        elif (
            s["nlgf_impaired"] is False
            and HIGHER_IS_BETTER.get(col) is not None
            and np.isfinite(s["disease_p"])
            and s["disease_p"] <= MIN_DISEASE_P
        ):
            ax.set_title(
                "NLGF not impaired here", loc="right", fontsize=8, color="#ADB5BD"
            )
        else:
            ax.set_title("no WT/NLGF gap", loc="right", fontsize=8, color="#ADB5BD")
        plt.tight_layout()

    fig.suptitle(
        "Oligo-BACE1-KO on an NLGF background: does deleting oligodendrocytic BACE1 "
        "move NLGF back toward WT?\n"
        "'deficit' = NLGF vs WT (is there anything to rescue), "
        "'rescue' = Oligo vs NLGF (the intervention test). Cross-day measures are "
        "gap-adjusted, all pairs kept.\nOligo vs WT is deliberately not shown: a "
        "non-significant difference from WT is not evidence of rescue.",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = PLOTS / "fig23_rescue.png"
    1 / 0
    # fig.savefig(out, dpi=200, bbox_inches="tight")

    plt.close(fig)
    print(f"wrote {out}")

    tab = pd.DataFrame(rows)
    if tab.empty:
        return
    tab = tab[
        [
            "label",
            "wt",
            "nlgf",
            "oligo",
            "disease_diff",
            "disease_p",
            "disease_strict_p",
            "rescue_raw_diff",
            "rescue_raw_p",
            "rescue_diff",
            "rescue_p",
            "rescue_strict_p",
            "rescue_index",
            "gap_matched",
            "nlgf_impaired",
            "rescuable",
        ]
    ].round(4)
    # tab.to_csv(RESULTS / "rescue_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + tab.to_string(index=False))
    print(
        f"\nmeasures with a WT/NLGF gap to rescue (p <= {MIN_DISEASE_P}): "
        f"{tab.rescuable.sum()} of {len(tab)}"
    )


if __name__ == "__main__":
    main()
