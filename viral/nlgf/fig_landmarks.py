"""Figure 8: landmark anchoring of the corridor code."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import compare
from viral.nlgf.style import GENOTYPE_ORDER, set_style, superplot

PANELS = [
    ("error_landmark_z", "Error vs landmark distance (z)",
     "Decoding sharper near landmarks?"),
    ("peak_landmark_z", "Peak density vs distance (z)",
     "Rate map peaks near landmarks?"),
    ("mean_error_cm", "Median decoding error (cm)", "Overall accuracy (matched)"),
]


def figure(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    for ax, (value, label, title) in zip(axes, PANELS):
        per_mouse, res = compare(df, value, how="median")
        superplot(ax, df, value, legend=(value == PANELS[0][0]),
                  p_value=res["p"])
        if value.endswith("_z"):
            ax.axhline(0, color="#CED4DA", lw=0.9, ls="--", zorder=0)
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=10.5)

    n_n = df[df.genotype == "NLGF"].mouse.nunique()
    n_w = df[df.genotype == "WT"].mouse.nunique()
    fig.suptitle(
        f"Landmark anchoring  ·  {len(df)} sessions, {n_n} NLGF / {n_w} WT mice  ·  "
        f"z against phase-shifted landmark combs, 40 trials and 100 cells matched",
        y=1.04, fontsize=12, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig8_landmarks.png")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    df = pd.read_csv(RESULTS / "landmarks.csv")
    figure(df[df.genotype.isin(GENOTYPE_ORDER)])
    print(f"wrote figure to {PLOTS}")
