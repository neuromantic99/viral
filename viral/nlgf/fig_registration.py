"""Technical demo: how many cells survive cross-session registration, and for how long.

No genotype comparison - this is about the method, not the biology. Every cross-day
result in this study rests on ROICaT matching the same neuron in two sessions, so it is
worth showing plainly how well that works and where it stops working.

Two things the panels are meant to settle for a sceptic:

  the yield is high at short gaps   about two thirds of the available cells are matched
                                    across consecutive sessions, which is what makes a
                                    within-cell cross-day comparison possible at all
  and it decays with interval       the fraction falls with the number of days between
                                    sessions, which is why every cross-day analysis in
                                    this package either matches on gap or adjusts for it

"Available" is the smaller of the two sessions' accepted-cell counts - a pair cannot
match more cells than the poorer session contains, so that is the honest denominator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.cohort import n_cells
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.style import set_style

CACHE = Path.home() / ".cache" / "viral_scratch" / "registration_yield.csv"
INK = "#495057"


def load() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE)
    qc = pd.read_csv(RESULTS / "registration_qc.csv")
    qc["gap"] = (pd.to_datetime(qc.date_b) - pd.to_datetime(qc.date_a)).dt.days
    seen: dict = {}

    def nc(m, d):
        if (m, d) not in seen:
            seen[(m, d)] = n_cells(m, d)
        return seen[(m, d)]

    qc["n_avail"] = [min(nc(r.mouse, r.date_a), nc(r.mouse, r.date_b))
                     for r in qc.itertuples()]
    qc["frac"] = qc.n_matched / qc.n_avail
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(CACHE, index=False)
    return qc


def main() -> None:
    set_style(talk=True)
    d = load()
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.6))

    # A: how many cells were matched, against how many were there to match
    ax = axes[0]
    ax.scatter(d.n_avail, d.n_matched, s=90, color=INK, alpha=0.45, lw=0)
    lim = d.n_avail.max() * 1.05
    ax.plot([0, lim], [0, lim], color="#ADB5BD", lw=2.4, ls="--")
    ax.text(lim * 0.97, lim * 0.97, "all cells matched", ha="right", va="bottom",
            fontsize=21, color="#868E96", rotation=38)
    ax.set_xlabel("Cells available (smaller session)")
    ax.set_ylabel("Cells matched")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # B: yield against interval
    ax = axes[1]
    ax.scatter(d.gap + np.random.default_rng(0).uniform(-0.12, 0.12, len(d)),
               d.frac, s=90, color=INK, alpha=0.45, lw=0)
    med = d.groupby("gap").frac.median()
    ax.plot(med.index, med.values, color="#E8590C", lw=3.6, marker="o", ms=13,
            zorder=3, label="median")
    ax.set_xlabel("Days between sessions")
    ax.set_ylabel("Fraction of cells matched")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=22)

    # C: the distribution, as the one number to remember
    ax = axes[2]
    ax.hist(d.frac, bins=18, color=INK, alpha=0.8)
    ax.axvline(d.frac.median(), color="#E8590C", lw=3.6)
    ax.text(d.frac.median(), ax.get_ylim()[1] * 0.96, f" median {d.frac.median():.0%}",
            ha="left", va="top", fontsize=24, color="#E8590C")
    ax.set_xlabel("Fraction of cells matched")
    ax.set_ylabel("Session pairs")

    fig.suptitle(f"Cross-session registration: {len(d)} session pairs from "
                 f"{d.mouse.nunique()} mice", fontsize=34, weight="semibold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = PLOTS / "fig35_registration_yield.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  matched: median {d.n_matched.median():.0f} cells "
          f"({d.frac.median():.0%} of available)")
    print(d.groupby(pd.cut(d.gap, [0, 1, 2, 3, 5, 10, 30]), observed=True)
          .agg(pairs=("frac", "size"), matched=("n_matched", "median"),
               frac=("frac", "median")).round(2).to_string())


if __name__ == "__main__":
    main()
