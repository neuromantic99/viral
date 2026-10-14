"""Chang et al. (2020) in NLGF and WT: does the CONTENT of resting events change?

The figure our reactivation panels cannot produce. Everywhere else in this package the
offline measure is a STRENGTH - how much of the running correlation structure comes back
- and it is flat from pre to post, in both genotypes. Chang et al. never claimed a
strength difference either. Their claim is that the positions resting synchronous events
DECODE TO move closer to spatial landmarks after the animal has run the track.

Their preparation is ours: dysgranular RSC layers II-III, two-photon, a familiar linear
treadmill carrying tactile landmarks, and the belt clamped for a rest epoch either side
of the run. It is the one published result from a matched design, which makes it the one
worth testing directly rather than adapting.

Panel A is the gate and belongs first. Offline decoded position means nothing unless the
template can decode ONLINE position, so the held-out even-trial error is plotted against
both Chang's reported 11.8 cm and the 46.7 cm chance value. Panels B and C are the
spatial profiles themselves - where decoded events actually land relative to the
landmarks - because the summary statistics below are summarising these, and at our event
counts the profile is the more honest object. Panel D is Chang's own regression of
decoded density on distance to the nearest landmark, where a NEGATIVE slope is the
claim. Panel E is the median landmark distance, the cruder version of the same thing,
against its uniform-decoding chance value. Panel F is explained variance against its
reverse control: EV > REV says REST2 resembles RUN more than REST1 did, EV = REV says the
structure was already there before the animal ran, which is what a familiar track
predicts and what our strength result independently implies.

Sample sizes are small in a way the strength panels are not: synchronous events are rare
in these epochs (single figures per epoch in many sessions), so per-session estimates of
a spatial profile rest on tens of decoded bins. The profiles are therefore pooled across
mice for display and every test is at the animal level.

Oligo-BACE1-KO is excluded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.chang_local import (CONFIG, bin_centres, chance_landmark_distance,
                                    profile_of)
from viral.nlgf.fig_freeze_prepost import sign_flip_p
from viral.nlgf.paths import PLOTS, RESULTS
from viral.nlgf.stats import collapse_to_mice, compare
from viral.nlgf.style import (GENOTYPE_COLOURS, GENOTYPE_ORDER, p_bracket, set_style,
                              superplot)

EPOCHS = ["pre", "post"]
EPOCH_STYLE = {"pre": dict(ls="--", alpha=0.8), "post": dict(ls="-", alpha=1.0)}
CHANG_ERROR_CM = 11.8            # "mean decoding error during running of 11.8 cm"
CHANCE_ERROR_CM = (CONFIG.position_end_cm - CONFIG.position_start_cm) / 3


def load(source: str = "local") -> pd.DataFrame:
    """`local` is chang_local.py's output; `cached` is the WT-only run chang.py left in
    the repo, which predates this package and carries no decoded spatial profile."""
    path = (RESULTS / "chang_local.csv" if source == "local"
            else HERE.parent.parent / "df_summary_chang_WT_denoised.csv")
    d = pd.read_csv(path)
    d = d[d.genotype.isin(GENOTYPE_ORDER)]
    both = d.groupby(["mouse", "date"])["epoch"].nunique() == 2
    return d.set_index(["mouse", "date"]).loc[both[both].index].reset_index()


def present(d: pd.DataFrame) -> list:
    return [g for g in GENOTYPE_ORDER if (d.genotype == g).any()]


def error_panel(ax, d: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    genos = present(d)
    per_session = d.groupby(["genotype", "mouse", "date"]).decoding_error_cm.first(
        ).reset_index()
    for i, g in enumerate(genos):
        sub = per_session[per_session.genotype == g]
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(sub)), sub.decoding_error_cm, s=60,
                   color=GENOTYPE_COLOURS[g], alpha=0.45, edgecolors="none", zorder=3)
        pm = sub.groupby("mouse").decoding_error_cm.median()
        ax.scatter(np.full(len(pm), i), pm, s=230, color=GENOTYPE_COLOURS[g], zorder=5,
                   edgecolors="white", linewidths=2.0)
    ax.axhline(CHANCE_ERROR_CM, color="#E8590C", lw=2.6, ls="--")
    ax.text(0.99, CHANCE_ERROR_CM, "chance ", transform=ax.get_yaxis_transform(),
            color="#E8590C", fontsize=18, va="bottom", ha="right")
    ax.axhline(CHANG_ERROR_CM, color="#495057", lw=2.4, ls=":")
    ax.text(0.99, CHANG_ERROR_CM, "Chang et al. ", transform=ax.get_yaxis_transform(),
            color="#495057", fontsize=18, va="bottom", ha="right")
    ax.set_xticks(range(len(genos)))
    ax.set_xticklabels(genos)
    ax.set_xlim(-0.5, len(genos) - 0.5)
    ax.set_ylim(0, CHANCE_ERROR_CM * 1.12)
    ax.set_ylabel("Online decoding error (cm)\nheld-out trials")
    print("  decoding error: " + ", ".join(
        f"{g} {per_session[per_session.genotype == g].decoding_error_cm.median():.1f}"
        for g in genos))


def profile_panel(ax, d: pd.DataFrame, genotype: str, ylabel: bool = True) -> None:
    x = bin_centres()
    sub_g = d[(d.genotype == genotype) & d.profile.notna()]
    for epoch, style in EPOCH_STYLE.items():
        sub = sub_g[sub_g.epoch == epoch]
        if not len(sub):
            continue
        # Within mouse first: sessions differ enormously in how many events they
        # contribute, and the mice are not equally represented in the two genotypes
        per_mouse = np.array([
            np.nanmean([profile_of(r) for _, r in g.iterrows()], axis=0)
            for _, g in sub.groupby("mouse")])
        m = np.nanmean(per_mouse, axis=0)
        sem = np.nanstd(per_mouse, axis=0) / np.sqrt(max(per_mouse.shape[0], 1))
        ax.plot(x, m, color=GENOTYPE_COLOURS[genotype], lw=3.8,
                label=f"{epoch} freeze", zorder=4, **style)
        ax.fill_between(x, m - sem, m + sem, color=GENOTYPE_COLOURS[genotype],
                        alpha=0.16, lw=0, zorder=3)
    for lm in CONFIG.landmarks_cm:
        ax.axvline(lm, color="#495057", lw=2.0, ls=":", zorder=1)
    ax.axhline(1 / x.size, color="#ADB5BD", lw=2.0, zorder=1)
    ax.text(0.015, 0.03, "dotted = landmarks,  grey = uniform", transform=ax.transAxes,
            fontsize=17, color="#868E96")
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel("Decoded position (cm)")
    if ylabel:
        ax.set_ylabel("Fraction of decoded\nSCE time bins")
    ax.legend(frameon=False, fontsize=19, loc="upper right")
    ax.set_title(f"{genotype}  ·  {sub_g.mouse.nunique()} mice, "
                 f"{sub_g.groupby(['mouse','date']).ngroups} sessions",
                 fontsize=23, color=GENOTYPE_COLOURS[genotype])


def paired_epoch_panel(ax, d: pd.DataFrame, value: str, ylabel: str,
                       reference: float | None = None,
                       reference_label: str = "") -> None:
    genos = present(d)
    wide = d.pivot_table(index=["genotype", "mouse", "date"], columns="epoch",
                         values=value).reset_index().dropna(subset=EPOCHS)
    for i, g in enumerate(genos):
        sub = wide[wide.genotype == g]
        xs = np.array([0, 1]) + i * 2.4
        for r in sub.itertuples():
            ax.plot(xs, [r.pre, r.post], color=GENOTYPE_COLOURS[g], alpha=0.3, lw=2.0,
                    zorder=2)
        pm = sub.groupby("mouse")[EPOCHS].median()
        for r in pm.itertuples():
            ax.plot(xs, [r.pre, r.post], color=GENOTYPE_COLOURS[g], lw=4.4, zorder=5,
                    solid_capstyle="round")
            ax.scatter(xs, [r.pre, r.post], s=200, color=GENOTYPE_COLOURS[g], zorder=6,
                       edgecolors="white", linewidths=2.0)
        if len(genos) > 1:
            ax.text(np.mean(xs), 1.02, g, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=21, color=GENOTYPE_COLOURS[g])
        delta = (sub.post - sub.pre).groupby(sub.mouse).median().to_numpy()
        print(f"  {value} {g}: pre {sub.pre.median():+.5g} post {sub.post.median():+.5g}"
              f", delta sign-flip p = {sign_flip_p(delta):.3f}")
        if reference is not None:
            for j, epoch in enumerate(EPOCHS):
                v = sub.groupby("mouse")[epoch].median().to_numpy() - reference
                ax.text(i * 2.4 + j, 1.10, f"p = {sign_flip_p(v):.2f}",
                        transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=17, color="#495057")
                print(f"    {epoch} vs reference: {int((v > 0).sum())}/{v.size} above,"
                      f" sign-flip p = {sign_flip_p(v):.3f}")
        ax.text(np.mean(xs), 1.19, "vs reference", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=16, color="#868E96")
    if reference is not None:
        ax.axhline(reference, color="#E8590C", lw=2.4, ls="--", zorder=1)
        ax.text(0.015, reference, f" {reference_label}", transform=
                ax.get_yaxis_transform(), color="#E8590C", fontsize=18, va="bottom")
    ax.set_xticks([x for i in range(len(genos)) for x in (i * 2.4, i * 2.4 + 1)])
    ax.set_xticklabels(EPOCHS * len(genos))
    ax.set_xlim(-0.55, (len(genos) - 1) * 2.4 + 1.55)
    ax.set_xlabel("Freeze epoch")
    ax.set_ylabel(ylabel)
    ax.margins(y=0.12)


def ev_panel(ax, d: pd.DataFrame) -> None:
    """EV against its reverse control, one point per session."""
    genos = present(d)
    per_session = d.groupby(["genotype", "mouse", "date"])[
        ["explained_variance", "reverse_explained_variance"]].first().reset_index()
    for i, g in enumerate(genos):
        sub = per_session[per_session.genotype == g]
        xs = np.array([0, 1]) + i * 2.4
        for r in sub.itertuples():
            ax.plot(xs, [r.explained_variance, r.reverse_explained_variance],
                    color=GENOTYPE_COLOURS[g], alpha=0.3, lw=2.0, zorder=2)
        pm = sub.groupby("mouse")[
            ["explained_variance", "reverse_explained_variance"]].median()
        for r in pm.itertuples():
            ax.plot(xs, [r.explained_variance, r.reverse_explained_variance],
                    color=GENOTYPE_COLOURS[g], lw=4.4, zorder=5, solid_capstyle="round")
            ax.scatter(xs, [r.explained_variance, r.reverse_explained_variance], s=200,
                       color=GENOTYPE_COLOURS[g], zorder=6, edgecolors="white",
                       linewidths=2.0)
        if len(genos) > 1:
            ax.text(np.mean(xs), 1.02, g, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=21, color=GENOTYPE_COLOURS[g])
        delta = (sub.explained_variance
                 - sub.reverse_explained_variance).groupby(sub.mouse).median().to_numpy()
        print(f"  EV-REV {g}: {delta.mean():+.4f}, sign-flip p = {sign_flip_p(delta):.3f}")
    ax.set_xticks([x for i in range(len(genos)) for x in (i * 2.4, i * 2.4 + 1)])
    ax.set_xticklabels(["EV", "REV"] * len(genos))
    ax.set_xlim(-0.55, (len(genos) - 1) * 2.4 + 1.55)
    ax.set_ylabel("Explained variance")
    ax.text(0.5, 0.97, "EV > REV is the reactivation claim", transform=ax.transAxes,
            ha="center", va="top", fontsize=18, color="#495057")


def sce_panel(ax, d: pd.DataFrame) -> None:
    """How much data each summary statistic actually rests on.

    Synchronous events are rare in these epochs. A landmark slope fitted to a session
    with three events is not the same measurement as one fitted to twenty, and the
    reader is entitled to see which they are looking at before reading the panels above.
    """
    genos = present(d)
    rng = np.random.default_rng(0)
    positions, labels = [], []
    for i, g in enumerate(genos):
        for j, epoch in enumerate(EPOCHS):
            x = i * 2.4 + j
            sub = d[(d.genotype == g) & (d.epoch == epoch)]
            ax.scatter(x + rng.uniform(-0.13, 0.13, len(sub)), sub.n_sce, s=60,
                       color=GENOTYPE_COLOURS[g], alpha=0.45, edgecolors="none",
                       zorder=3)
            ax.plot([x - 0.3, x + 0.3], [sub.n_sce.median()] * 2, color="#212529",
                    lw=3.2, zorder=5)
            positions.append(x)
            labels.append(f"{g}\n{epoch}")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Synchronous events\nper epoch")
    print(f"  SCEs per epoch: median {d.n_sce.median():.0f}, "
          f"range {d.n_sce.min():.0f}-{d.n_sce.max():.0f}")


def main(source: str = "local") -> None:
    set_style(talk=True)
    d = load(source)
    has_profile = "profile" in d.columns and d.profile.notna().any()
    genos = present(d)

    panels = ["error"] + (["profile"] * len(genos) if has_profile else []) + [
        "slope", "distance", "ev", "sce"]
    n = len(panels)
    rows = 2 if n > 3 else 1
    cols = int(np.ceil(n / rows))
    fig = plt.figure(figsize=(9.0 * cols, 8.3 * rows))
    gs = fig.add_gridspec(rows, cols, hspace=0.45, wspace=0.32, left=0.055,
                          right=0.985, top=0.84 if rows > 1 else 0.76, bottom=0.08)
    axes = [fig.add_subplot(gs[i // cols, i % cols]) for i in range(n)]

    profile_axes = []
    for ax, kind in zip(axes, panels):
        if kind == "error":
            error_panel(ax, d)
        elif kind == "profile":
            g = genos[len(profile_axes)]
            profile_panel(ax, d, g, ylabel=not profile_axes)
            profile_axes.append(ax)
        elif kind == "slope":
            paired_epoch_panel(ax, d, "landmark_slope",
                               "Decoded density vs distance\nto landmark (slope)",
                               reference=0.0, reference_label="no relationship")
        elif kind == "distance":
            paired_epoch_panel(ax, d, "median_landmark_distance",
                               "Median distance from decoded\nposition to landmark (cm)",
                               reference=chance_landmark_distance(),
                               reference_label="uniform decoding")
        elif kind == "ev":
            ev_panel(ax, d)
        elif kind == "sce":
            sce_panel(ax, d)
    if len(profile_axes) == 2:
        lo = min(a.get_ylim()[0] for a in profile_axes)
        hi = max(a.get_ylim()[1] for a in profile_axes)
        for a in profile_axes:
            a.set_ylim(lo, hi)

    for ax, letter in zip(axes, "ABCDEF"):
        ax.text(-0.16, 1.16, letter, transform=ax.transAxes, fontsize=34,
                weight="bold", va="top")

    counts = d.groupby("genotype").mouse.nunique().to_dict()
    who = ", ".join(f"{v} {k}" for k, v in counts.items())
    fig.suptitle("Chang et al. (2020) in this dataset: the decoder works, but resting "
                 "events show no preference for landmarks, and none develops "
                 "across the task\n"
                 f"{who} mice  ·  same region, layer and familiar-track freeze design "
                 "as the original",
                 fontsize=29, weight="semibold", y=0.965 if rows > 1 else 0.95)
    suffix = "" if source == "local" else "_wt"
    out = PLOTS / f"fig45_chang{suffix}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "local")
