"""Is there any assembly structure in RSC immobility events to begin with?

Before "NLGF synchronous events are structurally normal" can mean anything, the measure
has to be shown to have dynamic range in this preparation. If WT events are themselves
indistinguishable from a margin-preserving shuffle, then finding NLGF equally
indistinguishable says nothing about NLGF - it says the measure is flat here, and the
comparison with Shipley et al. cannot be made either way.

So the first row asks the prior question, pooling genotypes: does the observed
participation matrix differ from its own null at all? Each session's observed value is
expressed as a z against the spread of ITS OWN shuffle distribution, which is the only
scale on which "close to the null" is meaningful - the raw excesses here are ~0.005,
which sounds negligible but is uninterpretable until compared to a null SD.

Only the second row compares genotypes, and only for measures that cleared the first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, p_bracket, set_style, superplot

MEASURES = [("recruitment_gini", "Recruitment concentration\n(Gini)"),
            ("cell_coparticipation", "Cell co-participation"),
            ("event_similarity", "Event similarity")]


def main() -> None:
    set_style()
    d = pd.read_csv(RESULTS / "sce_structure.csv")
    d = d[d.genotype.isin(GENOTYPE_ORDER)]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.4))

    for j, (col, label) in enumerate(MEASURES):
        # --- row 1: observed against its own null, genotypes pooled ---------------
        ax = axes[0, j]
        s = d.dropna(subset=[col, f"{col}_null"])
        lo = min(s[col].min(), s[f"{col}_null"].min())
        hi = max(s[col].max(), s[f"{col}_null"].max())
        pad = (hi - lo) * 0.08 or 0.01
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#ADB5BD", lw=1.2,
                ls="--", zorder=1)
        for g in GENOTYPE_ORDER:
            t = s[s.genotype == g]
            ax.scatter(t[f"{col}_null"], t[col], s=30, alpha=0.75, lw=0,
                       color=GENOTYPE_COLOURS[g], label=g, zorder=3)
        ax.set_xlabel("shuffled (circular shift)")
        ax.set_ylabel("observed")
        ax.set_aspect("equal", adjustable="datalim")

        z = s[f"{col}_z"].dropna()
        frac = float((z.abs() > 2).mean()) if len(z) else np.nan
        verdict = ("measure has range" if abs(z.median()) > 2
                   else "sits on its null")
        ax.set_title(f"{'ABC'[j]}  {label.splitlines()[0]}\n"
                     f"median z = {z.median():+.1f}  ·  {frac:.0%} of sessions |z|>2  ·  "
                     f"{verdict}",
                     loc="left", fontweight="bold", fontsize=9)
        if j == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower right")

        # --- row 2: genotype, on the z scale --------------------------------------
        ax = axes[1, j]
        t = d.dropna(subset=[f"{col}_z"])
        superplot(ax, t, f"{col}_z", genotypes=GENOTYPE_ORDER)
        _, res = compare(t, f"{col}_z")
        p_bracket(ax, res["p"])
        ax.axhline(0, color="#495057", lw=1.0, ls=":")
        ax.set_ylabel(f"{label}\n(z vs own null)")
        ax.set_title(f"{'DEF'[j]}  Genotype", loc="left", fontweight="bold", fontsize=9)

    fig.suptitle("RSC immobility events carry strong assembly structure, and NLGF "
                 "preserves it\n"
                 "Top: observed vs a circular-shift null, genotypes pooled.  "
                 "Bottom: genotype, on the same z scale.  "
                 "Participation scored from the denoised trace.",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = PLOTS / "fig28_sce_structure.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}\n")

    from itertools import product

    for col, label in MEASURES:
        s = d.dropna(subset=[f"{col}_z"])
        per_mouse = s.groupby(["mouse", "genotype"], as_index=False)[f"{col}_z"].median()
        v = per_mouse[f"{col}_z"].to_numpy()
        stats = [np.mean(v * np.array(sg)) for sg in product([1, -1], repeat=len(v))]
        p_any = float(np.mean(np.abs(stats) >= abs(v.mean()) - 1e-12))
        _, res = compare(s, f"{col}_z")
        print(f"  {label.replace(chr(10), ' '):34s} median z = {s[f'{col}_z'].median():+6.2f}  "
              f"structure vs null p = {p_any:.4f}   |   "
              f"NLGF {res['mean_a']:+.2f} vs WT {res['mean_b']:+.2f}, p = {res['p']:.3f}")


if __name__ == "__main__":
    main()
