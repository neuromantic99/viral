"""Figure 5: spatial coding fidelity.

Ordered from the criterion-dependent to the criterion-free. Place cell yield, spatial
information, field width and reliability all depend on the place-field definition,
which on this data is permissive - 90% of cells qualify in some sessions. Decoding
error does not depend on it at all, and is the measure to weight most heavily.
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
from viral.nlgf.stats import compare, minimum_detectable_difference
from viral.nlgf.style import GENOTYPE_COLOURS, GENOTYPE_ORDER, set_style, strip_with_mice

PANELS = [
    ("frac_place_cells", "Fraction place cells", "Place cell yield"),
    ("spatial_info_pc", "Spatial info (bits/event)", "Tuning sharpness"),
    ("field_width_cm", "Field width (cm)", "Field size"),
    ("reliability_pc", "Odd-even map correlation", "Lap-to-lap reliability"),
    ("decoding_error_cm", "Decoding error (cm)", "Population readout"),
]


def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "place_sessions_rewarded_None.csv")
    return df[df.genotype.isin(GENOTYPE_ORDER)]


def figure(df: pd.DataFrame) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2))
    results = []
    for ax, (value, label, title) in zip(axes, PANELS):
        if value not in df.columns:
            continue
        per_mouse, res = compare(df, value, how="median")
        strip_with_mice(ax, per_mouse, value)
        mde = minimum_detectable_difference(per_mouse, value)
        ax.set_ylabel(label)
        stars = " *" if res["p"] < 0.05 else ""
        ax.set_title(f"{title}\ndiff {res['diff']:+.3g},  p = {res['p']:.3f}{stars}",
                     fontsize=10.5)
        ref = abs(res["mean_b"]) or 1.0
        ax.text(0.5, -0.21,
                f"observed {100*res['diff']/ref:+.0f}%  ·  detectable {100*mde/ref:.0f}%",
                transform=ax.transAxes, ha="center", fontsize=8.5, color="#868E96")
        results.append(dict(metric=value, mde=mde, **res))

    n_n = df[df.genotype == "NLGF"].mouse.nunique()
    n_w = df[df.genotype == "WT"].mouse.nunique()
    fig.suptitle(
        f"RSC spatial coding fidelity  ·  {len(df)} sessions, {n_n} NLGF / {n_w} WT "
        f"mice  ·  decoding error is the criterion-free measure",
        y=1.04, fontsize=12.5, weight="semibold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS / "fig5_place_coding.png")
    plt.close(fig)
    return pd.DataFrame(results)


def figure_speed_covariate(df: pd.DataFrame, behaviour: pd.DataFrame) -> None:
    """Decoding error against running speed.

    NLGF run faster, and speed changes both occupancy and how much data a bin gets, so
    a decoding difference has to be checked against speed before it is read as a coding
    difference.
    """
    beh = (behaviour.groupby("mouse")["mean_trial_speed"].median().rename("speed"))
    per_mouse = (df.groupby(["genotype", "mouse"])[
        ["decoding_error_cm", "frac_place_cells", "n_cells"]].median().reset_index()
        .join(beh, on="mouse"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, y, label in zip(
        axes, ["decoding_error_cm", "frac_place_cells"],
        ["Decoding error (cm)", "Fraction place cells"],
    ):
        for g in GENOTYPE_ORDER:
            s = per_mouse[per_mouse.genotype == g]
            ax.scatter(s.speed, s[y], s=80, color=GENOTYPE_COLOURS[g], alpha=0.85,
                       edgecolor="white", linewidth=1.2, label=g)
        ok = per_mouse.dropna(subset=["speed", y])
        if len(ok) > 2:
            b = np.polyfit(ok.speed, ok[y], 1)
            xs = np.linspace(ok.speed.min(), ok.speed.max(), 20)
            ax.plot(xs, np.polyval(b, xs), color="#868E96", lw=1.2, ls="--", zorder=0)
            r = np.corrcoef(ok.speed, ok[y])[0, 1]
            ax.set_title(f"{label} vs speed   (r = {r:+.2f}, n = {len(ok)} mice)",
                         fontsize=10.5)
        ax.set_xlabel("Mean trial speed (cm/s)")
        ax.set_ylabel(label)
    axes[0].legend()
    fig.suptitle("Is coding quality explained by how fast the animal runs?",
                 y=1.03, fontsize=12, weight="semibold")
    fig.tight_layout()
    fig.savefig(PLOTS / "fig6_place_speed_covariate.png")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    df = load()
    res = figure(df)
    beh = pd.read_csv(RESULTS / "behaviour_full.csv")
    figure_speed_covariate(df, beh[beh.rig == "2P"])
    print(res.to_string(index=False))
