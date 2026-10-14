"""Cross-session ROI registration with ROICaT.

MUST be run with the ROICaT virtualenv, not the project one:

    <scratch>/roicat_venv/bin/python -m viral.nlgf.register_roicat --mouse J030 \
        --date-a 2026-05-15 --date-b 2026-05-18

ROICaT is kept in a separate environment on purpose - it pulls torch, jupyter and
~150 other packages, and none of that belongs in the project venv. The two sides
communicate through a CSV of matched ROIs, so the analysis never imports roicat.

Why register at all, and which sessions. The reversal transition is the pair that
matters: the two textures swap meaning while staying physically identical, so a decoder
trained on texture A versus B before the switch and tested after it separates a sensory
code from a value code in one step. Those pairs are 1-8 days apart (median 2), which is
the interval that has been shown to work in this lab. The recall sessions, 63-71 days
later, are not attempted - alignment across that gap has already been tried here and
does not succeed.

Two outputs, and the second matters as much as the first:

    matches      one row per matched ROI pair, with the row index into each session's
                 spks matrix. spks rows correspond to np.flatnonzero(iscell[:, 0]) in
                 order, verified: 868 spks rows against 868 accepted ROIs for
                 J030 2026-05-15.

                 The stat.npy written into the input tree is PRE-FILTERED to accepted
                 cells. The tracking pipeline does not forward iscell to
                 Data_suite2p, so given a full stat.npy it clusters all 1355 ROIs
                 rather than the 868 real cells, and its per-session indices are then
                 into stat rather than into spks - silently off by the rejected ROIs.
                 Filtering up front makes ROICaT's index the spks row by construction,
                 and drops a third of the ROIs it would otherwise embed.
    diagnostics  ROI counts, match count and match rate per pair. If NLGF fields of
                 view register worse than WT, matched-cell count drives every drift
                 number downstream - the same trap as lap count in Tier 2 and
                 field-of-view cell count in Tier 5. Match rate is therefore reported
                 as a result, not a log line, and the analysis equalises matched-cell
                 count across genotypes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HD = Path("/Volumes/hard_drive/viral")
DFF = HD / "dff"
DERIVED = HD / "nlgf_derived"
# Input tree on LOCAL disk, not the exFAT drive. exFAT has no native extended
# attributes, so macOS writes AppleDouble "._name" companion files alongside anything
# carrying metadata - and ROICaT's file discovery picks up "._stat.npy" and tries to
# unpickle it, which fails. Local disk has no AppleDouble, and the tree is only a few
# MB per pair, so this is also faster.
REG_INPUT = Path.home() / ".cache" / "viral_roicat_input"
# ROICaT's own output goes to LOCAL disk too. richfile is a directory format that
# writes thousands of small files, and on exFAT each one spawns an AppleDouble
# companion; on this drive that took 80 minutes for a pair whose analysis finished in
# 10, holding a lock the whole time. Nothing downstream reads it - the match table and
# the QC figure are written separately - so it is scratch.
REG_OUT = Path.home() / ".cache" / "viral_roicat_out"
RESULTS = DERIVED / "results"
QC_DIR = DERIVED / "roicat_qc"
for _p in (REG_INPUT, REG_OUT, RESULTS, QC_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ROICaT discovers suite2p output by directory layout, and the files on the drive are
# flat ({mouse}_{date}_stat.npy), so a tree is built for it. Real copies rather than
# symlinks: ROICaT resolves the stat.npy path and then looks for ops.npy beside the
# RESOLVED file, which with symlinks sends it back to the flat directory looking for a
# bare ops.npy. The three files are a few MB per session.
SUITE2P_FILES = ("stat", "ops", "iscell")

# No pixel scale is stored in ops, so this has to be supplied. It affects the blurring
# kernel and the ROInet crop, so a wrong value degrades matching rather than breaking
# it. 1.0 is ROICaT's own default and a reasonable guess for a 512 px FOV around
# 500 um, but it should be replaced with the real value.
UM_PER_PIXEL = 1.0  # confirmed by JR: ~1 um/pixel for the 512 px FOV


def build_input_tree(mouse: str, dates: List[str]) -> Path:
    root = REG_INPUT / f"{mouse}_{'_'.join(dates)}"
    if root.exists():
        shutil.rmtree(root)
    for date in dates:
        plane = root / date / "suite2p" / "plane0"
        plane.mkdir(parents=True, exist_ok=True)
        stat = np.load(DFF / f"{mouse}_{date}_stat.npy", allow_pickle=True)  # noqa
        iscell = np.load(DFF / f"{mouse}_{date}_iscell.npy")
        accepted = iscell[:, 0].astype(bool)
        if stat.shape[0] != accepted.size:
            raise ValueError(
                f"{mouse} {date}: stat has {stat.shape[0]} ROIs, iscell has "
                f"{accepted.size}")
        np.save(plane / "stat.npy", stat[accepted], allow_pickle=True)
        np.save(plane / "iscell.npy", iscell[accepted])
        # copyfile, not copy2: copy2 carries metadata across and can recreate the
        # AppleDouble companion this move exists to avoid
        shutil.copyfile(DFF / f"{mouse}_{date}_ops.npy", plane / "ops.npy")
    # Belt and braces: nothing beginning with "._" should ever reach the pipeline
    for junk in root.rglob("._*"):
        junk.unlink(missing_ok=True)
    return root


def qc_figure(mouse: str, date_a: str, date_b: str, matches: pd.DataFrame,
              diag: Dict) -> Path:
    """ROI centroids on each day, matched cells highlighted and joined.

    Built from stat.npy and the match table rather than from ROICaT internals, so it
    stays valid if the pipeline version changes. The third panel is the one to read for
    alignment quality: it plots the displacement of each matched pair, which should be a
    tight cloud. A broad or structured cloud means the FOV alignment is fighting the
    match, and the matches should not be trusted.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def centroids(date: str) -> np.ndarray:
        stat = np.load(DFF / f"{mouse}_{date}_stat.npy", allow_pickle=True)  # noqa
        iscell = np.load(DFF / f"{mouse}_{date}_iscell.npy")[:, 0].astype(bool)
        med = np.array([r["med"] for r in stat[iscell]], dtype=float)
        return med[:, ::-1]  # suite2p med is (y, x)

    ca, cb = centroids(date_a), centroids(date_b)
    ma = matches.row_a.to_numpy()
    mb = matches.row_b.to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    for ax, c, m, date, n in ((axes[0], ca, ma, date_a, diag["n_roi_a"]),
                              (axes[1], cb, mb, date_b, diag["n_roi_b"])):
        un = np.setdiff1d(np.arange(len(c)), m)
        ax.scatter(c[un, 0], c[un, 1], s=13, c="#CED4DA", edgecolor="none",
                   label=f"unmatched ({len(un)})")
        ax.scatter(c[m, 0], c[m, 1], s=17, c="#D9480F", edgecolor="none",
                   label=f"matched ({len(m)})")
        ax.set_title(f"{date}   {n} cells", fontsize=11)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.legend(loc="upper right", fontsize=8, markerscale=1.6)
        ax.set_xlabel("x (px)")
    axes[0].set_ylabel("y (px)")

    # Displacement of each matched pair, RELATIVE TO ITS MEDIAN. The median itself is
    # the rigid FOV shift between the two days, which ROICaT corrects and which is not
    # an error. What matters for match quality is the residual scatter around it: a
    # tight isotropic cloud means the matches are mutually consistent, while a broad or
    # structured one means the alignment is fighting the matching and the pairs should
    # not be trusted.
    d = cb[mb] - ca[ma]
    shift = np.median(d, axis=0)
    resid = d - shift
    rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
    ax = axes[2]
    ax.scatter(resid[:, 0], resid[:, 1], s=13, c="#3B5BDB", alpha=0.55,
               edgecolor="none")
    lim = max(6.0, float(np.percentile(np.abs(resid), 99)) * 1.5)
    ax.axhline(0, color="#ADB5BD", lw=0.8)
    ax.axvline(0, color="#ADB5BD", lw=0.8)
    for r in (2, 5):
        ax.add_artist(plt.Circle((0, 0), r, fill=False, color="#ADB5BD",
                                 lw=0.7, ls=":"))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("dx residual (px)")
    ax.set_ylabel("dy residual (px)")
    ax.set_title(f"match consistency\nFOV shift ({shift[0]:+.0f}, {shift[1]:+.0f}) px"
                 f"  ·  residual RMS {rms:.1f} px", fontsize=11)
    diag["fov_shift_px"] = float(np.linalg.norm(shift))
    diag["residual_rms_px"] = rms

    fig.suptitle(
        f"{mouse}   {date_a} -> {date_b}   ({diag.get('gap_days', '?')} d)   "
        f"match rate {diag['match_rate']:.0%}   "
        f"cluster silhouette {diag.get('cluster_silhouette', float('nan')):.2f}",
        y=1.02, fontsize=12.5, weight="semibold")
    fig.tight_layout()
    out = QC_DIR / f"qc_{mouse}_{date_a}_{date_b}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def run_pair(mouse: str, date_a: str, date_b: str,
             um_per_pixel: float = UM_PER_PIXEL,
             use_gpu: bool = False) -> Optional[Dict]:
    from roicat import pipelines, util

    dates = [date_a, date_b]
    root = build_input_tree(mouse, dates)

    params = util.get_default_parameters(pipeline="tracking")
    params["general"]["use_GPU"] = use_gpu
    params["general"]["random_seed"] = 0
    params["data_loading"]["dir_outer"] = str(root)
    params["data_loading"]["common"]["um_per_pixel"] = um_per_pixel
    out_dir = REG_OUT / f"{mouse}_{date_a}_{date_b}"
    out_dir.mkdir(parents=True, exist_ok=True)
    params["results_saving"]["dir_save"] = str(out_dir)
    params["results_saving"]["prefix_name_save"] = f"{mouse}_{date_a}_{date_b}"

    results, run_data, params_used = pipelines.pipeline_tracking(params)

    # The pipeline returns cluster labels (UCIDs) per ROI, split by session
    ucids = results["clusters"]["labels_bySession"]
    if len(ucids) != 2:
        print(f"  unexpected session count: {len(ucids)}")
        return None

    # stat.npy was pre-filtered to accepted cells, so ROICaT's per-session index IS
    # the spks row index. Assert rather than warn: if this ever stops holding, every
    # downstream cell identity is silently wrong.
    n_a = int(np.load(DFF / f"{mouse}_{date_a}_iscell.npy")[:, 0].sum())
    n_b = int(np.load(DFF / f"{mouse}_{date_b}_iscell.npy")[:, 0].sum())
    assert len(ucids[0]) == n_a and len(ucids[1]) == n_b, (
        f"{mouse}: roicat returned {len(ucids[0])}/{len(ucids[1])} ROIs but there are "
        f"{n_a}/{n_b} accepted cells - indices would not map to spks rows")

    ua, ub = np.asarray(ucids[0]), np.asarray(ucids[1])
    # -1 means unclustered
    shared = np.intersect1d(ua[ua >= 0], ub[ub >= 0])
    pairs = []
    for u in shared:
        ia, ib = np.flatnonzero(ua == u), np.flatnonzero(ub == u)
        # A UCID appearing more than once in a session is an ambiguous match; drop it
        if ia.size == 1 and ib.size == 1:
            pairs.append((int(u), int(ia[0]), int(ib[0])))

    matches = pd.DataFrame(pairs, columns=["ucid", "row_a", "row_b"])
    matches.insert(0, "mouse", mouse)
    matches.insert(1, "date_a", date_a)
    matches.insert(2, "date_b", date_b)
    matches.to_csv(RESULTS / f"matches_{mouse}_{date_a}_{date_b}.csv", index=False)

    q = results.get("clusters", {}).get("quality_metrics", {}) or {}
    def _med(key):
        v = np.asarray(q.get(key, []), dtype=float)
        return float(np.nanmedian(v)) if v.size else float("nan")

    diag = dict(mouse=mouse, date_a=date_a, date_b=date_b,
                n_roi_a=int(ua.size), n_roi_b=int(ub.size),
                n_matched=int(len(matches)),
                match_rate=float(len(matches) / min(ua.size, ub.size))
                if min(ua.size, ub.size) else np.nan,
                cluster_silhouette=_med("cluster_silhouette"),
                sample_silhouette=_med("sample_silhouette"),
                cluster_intra_mean=_med("cluster_intra_means"),
                um_per_pixel=um_per_pixel)
    try:
        gap = (pd.to_datetime(date_b) - pd.to_datetime(date_a)).days
        diag["gap_days"] = int(gap)
    except Exception:  # noqa: BLE001
        pass
    try:
        qc_figure(mouse, date_a, date_b, matches, diag)
    except Exception as e:  # noqa: BLE001
        print(f"  QC figure failed: {type(e).__name__}: {e}")
    print("  " + json.dumps(diag))
    return diag


def build_pairs(mode: str) -> pd.DataFrame:
    """Session pairs to register.

    transition   last learning -> first reversal, one per mouse. The pair that
                 separates a sensory code from a value code, since the textures swap
                 meaning while staying physically identical.
    consecutive  every neighbouring pair within learning and reversal. Gives a
                 drift measurement at each mouse's own session spacing.
    all          every pair within a mouse. Gives drift as a function of interval, at
                 the cost of many more runs.
    """
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from viral.nlgf.cohort import sessions

    d = sessions(min_cells=250, stages=["learning", "reversal"]).copy()
    d["when"] = pd.to_datetime(d.date)
    d = d.sort_values(["mouse", "when"])

    rows = []
    for mouse, g in d.groupby("mouse"):
        g = g.reset_index(drop=True)
        if mode == "transition":
            L, R = g[g.stage == "learning"], g[g.stage == "reversal"]
            if L.empty or R.empty:
                continue
            picks = [(L.iloc[-1], R.iloc[0])]
        elif mode == "consecutive":
            picks = [(g.loc[i], g.loc[i + 1]) for i in range(len(g) - 1)]
        elif mode == "all":
            picks = [(g.loc[i], g.loc[j])
                     for i in range(len(g)) for j in range(i + 1, len(g))]
        else:
            raise ValueError(mode)
        for a, b in picks:
            rows.append(dict(mouse=mouse, genotype=a["genotype"],
                             pre=a["date"], post=b["date"],
                             gap=(b["when"] - a["when"]).days))
    pairs = pd.DataFrame(rows)
    # Every file the pipeline needs must be present, or the run dies mid-way overnight
    ok = []
    for r in pairs.itertuples():
        have = all((DFF / f"{r.mouse}_{dt}_{k}.npy").exists()
                   for dt in (r.pre, r.post) for k in SUITE2P_FILES)
        ok.append(have)
    dropped = int((~np.array(ok)).sum())
    if dropped:
        print(f"dropping {dropped} pairs with missing stat/ops/iscell files")
    return pairs[np.array(ok)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cross-session ROI registration with ROICaT. Resumable: a pair "
                    "whose match table already exists is skipped.")
    ap.add_argument("--mouse")
    ap.add_argument("--date-a")
    ap.add_argument("--date-b")
    ap.add_argument("--mode", choices=["transition", "consecutive", "all"],
                    default="consecutive")
    ap.add_argument("--pairs-csv", help="explicit pair list instead of --mode")
    ap.add_argument("--um-per-pixel", type=float, default=UM_PER_PIXEL)
    ap.add_argument("--use-gpu", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo pairs that already have a match table")
    ap.add_argument("--dry-run", action="store_true", help="list the work and stop")
    args = ap.parse_args()

    if args.mouse and args.date_a and args.date_b:
        todo = pd.DataFrame([dict(mouse=args.mouse, pre=args.date_a, post=args.date_b)])
    elif args.pairs_csv:
        todo = pd.read_csv(args.pairs_csv)
    else:
        todo = build_pairs(args.mode)

    if not args.overwrite:
        done = [(RESULTS / f"matches_{r.mouse}_{r.pre}_{r.post}.csv").exists()
                for r in todo.itertuples()]
        n_done = int(np.sum(done))
        if n_done:
            print(f"skipping {n_done} pairs already done (use --overwrite to redo)")
        todo = todo[~np.array(done)].reset_index(drop=True)

    print(f"{len(todo)} pairs to register  (~12 min each, "
          f"~{len(todo) * 12 / 60:.1f} h total)", flush=True)
    if args.dry_run:
        print(todo.to_string(index=False))
        return

    diags = []
    for i, r in enumerate(todo.itertuples(), 1):
        print(f"\n=== [{i}/{len(todo)}] {r.mouse} {r.pre} -> {r.post} ===", flush=True)
        try:
            d = run_pair(r.mouse, r.pre, r.post, args.um_per_pixel, args.use_gpu)
            if d:
                diags.append(d)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        # Written every iteration so an interrupted overnight run keeps its results
        if diags:
            path = RESULTS / "registration_diagnostics.csv"
            new = pd.DataFrame(diags)
            if path.exists() and not args.overwrite:
                old = pd.read_csv(path)
                key = ["mouse", "date_a", "date_b"]
                new = (pd.concat([old, new]).drop_duplicates(subset=key, keep="last"))
            new.to_csv(path, index=False)
    if diags:
        print("\n" + pd.DataFrame(diags).to_string(index=False))


if __name__ == "__main__":
    main()
