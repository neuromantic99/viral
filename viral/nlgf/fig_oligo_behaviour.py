"""Behaviour of the Oligo-BACE1-KO mice, against both possible reference groups.

Behaviour is the part of this arm that is complete today: it needs only the digests,
which exist for all 45 Oligo sessions, and does not wait on imaging.

The first panel is the date confound, drawn rather than described. It is the reason
the NLGF comparison is the interpretable one and the WT comparison is not: read the
rest of the figure with that panel in view.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.oligo import CONTRASTS, OLIGO
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import (GENOTYPE_COLOURS, SHORT_NAME, p_bracket, set_style,
                              superplot)

MEASURES = [("licking_dprime", "licking d'"),
            ("n_trials", "trials completed"),
            ("mean_trial_speed", "mean trial speed (cm/s)"),
            ("frac_reward_drunk", "fraction of rewards drunk")]


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "behaviour_full.csv")
    return df[df.rig == "2P"].copy()


def _curve(ax, df, value, stage, genotypes, max_session=10, min_mice=3):
    d = df[(df.stage == stage) & (df.session_in_stage < max_session)]
    counts = (d.groupby(["genotype", "session_in_stage"])["mouse"].nunique()
              .unstack("genotype").reindex(columns=genotypes).fillna(0))
    ok = counts[(counts >= min_mice).all(axis=1)].index
    if len(ok) == 0:
        return
    last = int(ok.max())
    for g in genotypes:
        sub = d[(d.genotype == g) & (d.session_in_stage <= last)]
        per_mouse = sub.groupby(["session_in_stage", "mouse"])[value].mean().reset_index()
        stat = per_mouse.groupby("session_in_stage")[value].agg(["mean", "sem"])
        if stat.empty:
            continue
        ax.errorbar(stat.index, stat["mean"], yerr=stat["sem"], marker="o", ms=4,
                    lw=1.6, capsize=2, color=GENOTYPE_COLOURS[g],
                    label=SHORT_NAME.get(g, g))
    ax.set_xlabel(f"{stage} session")
    ax.legend(frameon=False, fontsize=7)


def figure(df: pd.DataFrame, genotypes, tag: str) -> None:
    set_style()
    sub = df[df.genotype.isin(genotypes)]
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.62, wspace=0.4)

    # ---- the confound, first, so it frames everything after it --------------------
    from viral.nlgf.load import ledger

    led = ledger()
    ax = fig.add_subplot(gs[0, :2])
    jitter = np.random.default_rng(0)
    for i, g in enumerate(genotypes):
        s = led[led.genotype == g]
        ax.scatter(pd.to_datetime(s.date), np.full(len(s), i) +
                   jitter.normal(0, 0.06, len(s)), s=16, alpha=0.7,
                   color=GENOTYPE_COLOURS[g], lw=0)
    ax.set_yticks(range(len(genotypes)))
    ax.set_yticklabels(genotypes)
    ax.set_xlabel("recording date")
    overlap = _overlap_fraction(led, genotypes)
    ax.set_title(f"A  Recording dates  -  {overlap:.0%} of Oligo-KO sessions fall "
                 f"inside the {genotypes[0]} window",
                 loc="left", fontweight="bold", fontsize=10)
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")

    ax = fig.add_subplot(gs[0, 2])
    _curve(ax, sub, "licking_dprime", "learning", genotypes)
    ax.axhline(1.0, color="#868E96", ls=":", lw=1)
    ax.set_ylabel("licking d'")
    ax.set_title("B  Learning", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[0, 3])
    _curve(ax, sub, "licking_dprime", "reversal", genotypes)
    ax.axhline(1.0, color="#868E96", ls=":", lw=1)
    ax.set_ylabel("licking d'")
    ax.set_title("C  Reversal", loc="left", fontweight="bold")

    letters = "DEFGHIJK"
    for j, (value, label) in enumerate(MEASURES):
        for k, stage in enumerate(["learning", "reversal"]):
            ax = fig.add_subplot(gs[1 + k, j])
            s = sub[(sub.stage == stage)].dropna(subset=[value])
            if s.mouse.nunique() < 3:
                ax.axis("off"); continue
            superplot(ax, s, value, genotypes=genotypes)
            per_mouse, res = compare(s, value, group_a=OLIGO,
                                     group_b=genotypes[0])
            p_bracket(ax, res["p"])
            ax.set_ylabel(label if k == 0 else "")
            ax.set_title(f"{letters[j + 4 * k]}  {stage}", loc="left",
                         fontweight="bold", fontsize=9)

    fig.suptitle(f"Oligo-BACE1-KO behaviour vs {genotypes[0]}", fontsize=14,
                 fontweight="bold", y=0.995)
    out = PLOTS / f"fig21_oligo_behaviour_{tag}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def _overlap_fraction(led: pd.DataFrame, genotypes) -> float:
    ref = pd.to_datetime(led[led.genotype == genotypes[0]].date)
    mine = pd.to_datetime(led[led.genotype == OLIGO].date)
    if ref.empty or mine.empty:
        return float("nan")
    return float(((mine >= ref.min()) & (mine <= ref.max())).mean())


def stats_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for contrast in CONTRASTS:
        sub = df[df.genotype.isin(contrast)]
        for stage in ["learning", "reversal"]:
            for value, label in MEASURES:
                s = sub[sub.stage == stage].dropna(subset=[value])
                if s.groupby("genotype").mouse.nunique().min() < 3:
                    continue
                per_mouse, res = compare(s, value, group_a=OLIGO,
                                         group_b=contrast[0])
                if not np.isfinite(res.get("diff", np.nan)):
                    continue
                rows.append(dict(contrast=" vs ".join(contrast), stage=stage,
                                 measure=label, oligo=round(res["mean_a"], 3),
                                 reference=round(res["mean_b"], 3),
                                 diff=round(res["diff"], 3), p=round(res["p"], 4),
                                 n_mice=int(per_mouse.mouse.nunique())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load()
    for contrast in CONTRASTS:
        figure(df, contrast, contrast[0].lower().replace("-", ""))
    tab = stats_table(df)
    tab.to_csv(RESULTS / "oligo_behaviour_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + tab.to_string(index=False))
