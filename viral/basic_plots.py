import sys
from pathlib import Path


HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

import time
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd
from scipy import stats

from matplotlib import pyplot as plt
from tifffile import imwrite

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from matplotlib import pyplot as plt
from scipy.ndimage import percentile_filter
import numpy as np
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from viral.gsheets_importer import gsheet2df
from viral.single_session import load_data, summarise_trial
from viral.constants import (
    BEHAVIOUR_DATA_PATH,
    SERVER_PATH,
    SPREADSHEET_ID,
    TIFF_UMBRELLA,
    CACHE_PATH,
    grosmark_config,
)
from viral.imaging_utils import (
    activity_trial_position,
    get_online_position_and_frames,
    get_resting_position_and_frames,
    subtract_neuropil,
    trial_is_imaged,
)
from viral.learning_stages import get_mouse_sessions
from viral.run_oasis import correct_f
from viral.models import Cached2pSession
from viral.representational_drift import interpolate_nans
from viral.sessions_keep import SESSIONS_KEEP
from viral.utils import (
    boxplot,
    get_genotype,
    get_wheel_circumference_from_rig,
    imshow,
    mixed_effects,
    moving_average,
    upper_triangle_no_diagonal,
    remove_diagonal,
    save_figure,
    upper_triangle_no_diagonal,
)
from viral.multiple_sessions import load_cache, parse_session_number


plt.rcParams["pdf.fonttype"] = 42


def get_firing_rates(
    session: Cached2pSession,
    spks: np.ndarray,
    rewarded: bool | None,
) -> tuple[np.ndarray, np.ndarray]:
    all_resting_frames: List[int] = []
    all_running_frames: List[int] = []
    for trial in session.trials:
        if not trial_is_imaged(trial) or (
            rewarded is not None and trial.texture_rewarded != rewarded
        ):
            continue
        _, resting_frames = get_resting_position_and_frames(
            trial=trial,
            wheel_circumference=get_wheel_circumference_from_rig("2P"),
        )
        _, running_frames = get_online_position_and_frames(
            trial=trial,
            wheel_circumference=get_wheel_circumference_from_rig("2P"),
            threshold_speed=True,
        )

        all_resting_frames.extend(resting_frames)
        all_running_frames.extend(running_frames)

    assert set(all_resting_frames).isdisjoint(
        set(all_running_frames)
    ), "Resting and running frames overlap!"

    spks_resting = spks[:, np.unique(np.array(all_resting_frames))]
    spks_running = spks[:, np.unique(np.array(all_running_frames))]

    return np.sum(spks_resting, 1) / (spks_resting.shape[1] / 30), np.sum(
        spks_running, 1
    ) / (spks_running.shape[1] / 30)


def get_cell_by_cell_correlation(
    session: Cached2pSession,
    spks: np.ndarray,
    rewarded: bool | None,
) -> np.ndarray:
    bin_size = 5
    start = 0
    max_position = 180

    trial_array = np.array(
        [
            activity_trial_position(
                trial=trial,
                flu=spks,
                wheel_circumference=get_wheel_circumference_from_rig("2P"),
                bin_size=bin_size,
                start=start,
                max_position=max_position,
                verbose=False,
                do_shuffle=False,
                threshold_speed=False,
            )
            for trial in session.trials
            if trial_is_imaged(trial)
            and (rewarded is None or trial.texture_rewarded == rewarded)
        ]
    )

    # # Remove trials where every value is NaN
    # trial_array = trial_array[~np.all(np.isnan(trial_array), (1, 2)), :, :]
    # trial_array = interpolate_nans(trial_array)

    trial_averaged = np.nanmean(trial_array, axis=0)
    corr = np.corrcoef(trial_averaged)

    return corr


def firing_rates_plot(rewarded: bool | None) -> None:
    wt = get_firing_rates_df("WT", rewarded)
    nlgf = get_firing_rates_df("NLGF", rewarded)

    all_data = pd.concat([wt, nlgf], ignore_index=True)

    p_values = {}

    for state in ["resting", "running"]:
        for stage in ["Baseline", "Trained"]:
            subset = all_data[
                (all_data["state"] == state) & (all_data["stage"] == stage)
            ]
            assert len(subset) > 100, "make sure nothing weird happend"
            p_value = mixed_effects(
                df=subset,
                dependent_var="firing_rate",
                independent_var="genotype",
                group_name="mouse_id",
            ).filter(like="C(genotype)")
            p_values[f"{state}_{stage}"] = p_value

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharey=True)
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    sns.violinplot(
        data=all_data[all_data["state"] == "resting"],
        x="stage",
        y="firing_rate",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        cut=0,
        ax=axes[0],
    )
    # add p-value annotations above each stage for the resting axis
    stages = ["Baseline", "Trained"]
    # use the axis y-limits (not the raw data max) so extreme outliers don't push the annotation off-screen

    axes[0].set_title("Resting")
    axes[0].set_ylabel("Transients / second")
    axes[0].set_ylim(0, None)

    sns.violinplot(
        data=all_data[all_data["state"] == "running"],
        x="stage",
        y="firing_rate",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        cut=0,
        ax=axes[1],
    )
    sns.despine()
    axes[1].set_title("Running")
    axes[1].set_ylabel("")

    handles, labels = axes[1].get_legend_handles_labels()
    # remove per-axis legends
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    plt.ylim(None, 2.5)

    for idx, state in enumerate(["resting", "running"]):
        ymin_plot, ymax_plot = axes[idx].get_ylim()
        plot_range = ymax_plot - ymin_plot
        text_y = (
            ymax_plot - plot_range * 0.02
        )  # place text just below the top of the axis
        for i, stage in enumerate(stages):
            p_text = f"P = {round(p_values[f"{state}_{stage}"].values[0], 2)}"
            axes[idx].text(i, text_y, p_text, ha="center", va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "firing_rates"
        / f"comparison_firing_rates_rewarded_{rewarded}.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def get_firing_rates_df(genotype: str, rewarded: bool | None) -> pd.DataFrame:

    result = {"unsupervised": [], "learning": [], "learned": []}

    for mouse_name in SESSIONS_KEEP.keys():
        if get_genotype(mouse_name) != genotype:
            continue
        for stage in ["unsupervised", "learning", "learned"]:
            print(f"Doing {mouse_name} at {stage} stage for genotype {genotype}")
            date = SESSIONS_KEEP[mouse_name][stage]
            if date is None:
                continue
            save_path = (
                SERVER_PATH
                / "viral_caches"
                / "firing_rates"
                / f"{mouse_name}_{date}_rewarded_{rewarded}_firing_rates.npy"
            )
            # Weird suffix fiddle because the file is saved as .npy.npz
            load_path = save_path.with_suffix(save_path.suffix + ".npz")
            if load_path.exists():
                result_mouse = np.load(load_path)
                result[stage].append(
                    (
                        mouse_name,
                        result_mouse["resting_rates"],
                        result_mouse["running_rates"],
                    )
                )

    collapsed_result: Dict[str, List] = {
        "stage": [],
        "state": [],
        "firing_rate": [],
        "mouse_id": [],
        "genotype": [],
    }
    for stage in result.keys():
        for idx, mouse_data in enumerate(result[stage]):
            stage_name = "Baseline" if stage == "unsupervised" else "Trained"
            mouse_name, resting_rates, running_rates = mouse_data
            collapsed_result["stage"].extend(
                [stage_name] * (len(resting_rates) + len(running_rates))
            )

            collapsed_result["genotype"].extend(
                [genotype] * (len(resting_rates) + len(running_rates))
            )
            collapsed_result["mouse_id"].extend(
                [mouse_name] * (len(resting_rates) + len(running_rates))
            )
            collapsed_result["state"].extend(["resting"] * len(resting_rates))
            collapsed_result["firing_rate"].extend(resting_rates)

            collapsed_result["state"].extend(["running"] * len(running_rates))
            collapsed_result["firing_rate"].extend(running_rates)

    return pd.DataFrame(collapsed_result)


def save_firing_rates(genotype: str) -> None:

    result: Dict[str, List[tuple[np.ndarray, np.ndarray]]] = {
        "unsupervised": [],
        "learning": [],
        "learned": [],
    }
    for mouse_name in SESSIONS_KEEP.keys():
        if get_genotype(mouse_name) != genotype:
            continue
        for stage in ["unsupervised", "learning", "learned"]:
            print(f"Doing {mouse_name} at {stage} stage")
            date = SESSIONS_KEEP[mouse_name][stage]
            if date is None:
                continue

            spks_path = TIFF_UMBRELLA / date / mouse_name / "suite2p" / "plane0"
            spks_all = np.load(spks_path / "oasis_spikes.npy")
            is_cell = np.load(spks_path / "iscell.npy")[:, 0].astype(bool)
            spks_all = spks_all[is_cell, :]

            session_path = CACHE_PATH / f"{mouse_name}_{date}.json"
            session = Cached2pSession.model_validate_json(session_path.read_text())

            for rewarded in [True, False, None]:

                save_path = (
                    SERVER_PATH
                    / "viral_caches"
                    / "firing_rates"
                    / f"{mouse_name}_{date}_rewarded_{rewarded}_firing_rates_iscell_filtered.npy"
                )
                # Weird suffix fiddle because the file is saved as .npy.npz
                load_path = save_path.with_suffix(save_path.suffix + ".npz")
                if load_path.exists():
                    print(f"aready done {mouse_name}, {date}")
                    result_mouse = np.load(load_path)
                    result[stage].append(
                        (result_mouse["resting_rates"], result_mouse["running_rates"])
                    )
                    continue

                print(f"Processing {mouse_name} on {date} for {stage} stage.")
                mask_files = [
                    file
                    for file in list(
                        (
                            SERVER_PATH
                            / "viral_caches"
                            / "place_cells"
                            / "pcs_combined"
                        ).glob("*.npy")
                    )
                    if mouse_name in file.name
                    and date in file.name
                    and f"rewarded_{rewarded}" in file.name
                    and str(grosmark_config) in file.name
                ]

                assert (
                    len(mask_files) == 1
                ), f"Should find one mask file, found {mask_files}"

                pc_mask = np.load(mask_files[0])
                spks = spks_all[pc_mask, :]
                resting_rates, running_rates = get_firing_rates(
                    session=session,
                    spks=spks,
                    rewarded=rewarded,
                )

                np.savez(
                    save_path,
                    running_rates=running_rates,
                    resting_rates=resting_rates,
                )


def n_responders_session(session: Cached2pSession, rewarded: bool | None) -> float:
    pc_mask = np.load(
        SERVER_PATH
        / "viral_caches"
        / "place_cells"
        / "pcs_combined"
        / f"{session.mouse_name}_{session.date}_rewarded_{rewarded}_{grosmark_config}_pcs_combined_BOD_True.npy"
    )
    return np.sum(pc_mask) / len(pc_mask)


def get_n_responders(genotype: str, rewarded: bool | None) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {"Baseline": [], "Trained": []}
    for mouse_name in SESSIONS_KEEP.keys():
        if get_genotype(mouse_name) != genotype:
            continue
        for stage in ["unsupervised", "learning", "learned"]:
            print(f"Doing {mouse_name} at {stage} stage")
            date = SESSIONS_KEEP[mouse_name][stage]
            if date is None:
                continue
            session_path = CACHE_PATH / f"{mouse_name}_{date}.json"
            session = Cached2pSession.model_validate_json(session_path.read_text())

            n = n_responders_session(session=session, rewarded=rewarded)
            stage_name = "Baseline" if stage == "unsupervised" else "Trained"
            result[stage_name].append(n)

    return result


def n_responders_comparison_plot() -> None:
    result = {"genotype": [], "stage": [], "rewarded": [], "n_responders": []}
    for genotype in ["WT", "NLGF"]:
        for rewarded in [True, False]:
            temp_result = get_n_responders(genotype=genotype, rewarded=rewarded)
            for stage in temp_result.keys():
                result["genotype"].extend([genotype] * len(temp_result[stage]))
                result["stage"].extend([stage] * len(temp_result[stage]))
                result["rewarded"].extend([rewarded] * len(temp_result[stage]))
                result["n_responders"].extend(temp_result[stage])

    df = pd.DataFrame(result)
    df.to_pickle("n_responders_df.pkl")

    df = pd.read_pickle("n_responders_df.pkl")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    sns.boxplot(
        data=df[df["rewarded"] == True],
        x="stage",
        y="n_responders",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        showfliers=False,
        ax=axes[0],
    )
    sns.stripplot(
        data=df[df["rewarded"] == True],
        x="stage",
        y="n_responders",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        dodge=True,
        linewidth=1,
        edgecolor="black",
        ax=axes[0],
    )
    # add p-value annotations above each stage for the resting axis
    stages = ["Baseline", "Trained"]
    # use the axis y-limits (not the raw data max) so extreme outliers don't push the annotation off-screen

    axes[0].set_title("Rewarded")
    axes[0].set_ylabel("Fraction of cells with spatial tuning")

    sns.boxplot(
        data=df[df["rewarded"] == False],
        x="stage",
        y="n_responders",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        showfliers=False,
        ax=axes[1],
    )
    sns.stripplot(
        data=df[df["rewarded"] == False],
        x="stage",
        y="n_responders",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        dodge=True,
        ax=axes[1],
        linewidth=1,
        edgecolor="black",
    )
    sns.despine()

    axes[1].set_title("Unrewarded")
    axes[1].set_ylabel("")

    handles, labels = axes[1].get_legend_handles_labels()
    # remove per-axis legends
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2)

    plt.ylim(0, 1)

    for idx, rewarded in enumerate([True, False]):
        ymin_plot, ymax_plot = axes[idx].get_ylim()
        plot_range = ymax_plot - ymin_plot
        text_y = (
            ymax_plot - plot_range * 0.02
        )  # place text just below the top of the axis
        for i, stage in enumerate(stages):
            p_value = stats.ttest_ind(
                df[
                    (df["stage"] == stage)
                    & (df["rewarded"] == rewarded)
                    & (df["genotype"] == "WT")
                ]["n_responders"],
                df[
                    (df["stage"] == stage)
                    & (df["rewarded"] == rewarded)
                    & (df["genotype"] == "NLGF")
                ]["n_responders"],
            ).pvalue
            print(p_value)
            p_text = f"P = {p_value:.2g}"
            axes[idx].text(i, text_y, p_text, ha="center", va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(
        SERVER_PATH / "viral_plots" / "n_responders" / f"comparison_n_responders.png"
    )


def save_correlations(genotype: str) -> None:

    for mouse_name in tqdm(SESSIONS_KEEP.keys()):
        if get_genotype(mouse_name) != genotype:
            continue
        for stage in ["unsupervised", "learning", "learned"]:
            print(f"Doing {mouse_name} at {stage} stage")
            date = SESSIONS_KEEP[mouse_name][stage]
            if date is None:
                continue

            spks_path = TIFF_UMBRELLA / date / mouse_name / "suite2p" / "plane0"
            spks_all = np.load(spks_path / "oasis_spikes.npy")
            session_path = CACHE_PATH / f"{mouse_name}_{date}.json"
            session = Cached2pSession.model_validate_json(session_path.read_text())

            for rewarded in [True, False, None]:

                save_path = (
                    SERVER_PATH
                    / "viral_caches"
                    / "correlations"
                    / f"{mouse_name}_{date}_rewarded_{rewarded}_correlation.npy"
                )
                if save_path.exists():
                    print(f"aready done {mouse_name}, {date}")
                    continue

                mask_files = [
                    file
                    for file in list(
                        (
                            SERVER_PATH
                            / "viral_caches"
                            / "place_cells"
                            / "pcs_combined"
                        ).glob("*.npy")
                    )
                    if mouse_name in file.name
                    and date in file.name
                    and f"rewarded_{rewarded}" in file.name
                    and str(grosmark_config) in file.name
                ]

                assert (
                    len(mask_files) == 1
                ), f"Should find one mask file, found {mask_files}"

                pc_mask = np.load(mask_files[0])
                spks = spks_all[pc_mask, :]
                corr = get_cell_by_cell_correlation(
                    session=session,
                    spks=spks,
                    rewarded=rewarded,
                )
                np.save(
                    save_path,
                    corr,
                )


def plot_correlations(rewarded: bool | None) -> None:
    wt = get_correlation_df("WT", rewarded)
    nlgf = get_correlation_df("NLGF", rewarded)
    all_data = pd.concat([wt, nlgf], ignore_index=True)

    fig = plt.figure()
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    p_values = {}

    for stage in ["Baseline", "Trained"]:
        subset = all_data[all_data["stage"] == stage]
        assert len(subset) > 100, "make sure nothing weird happend"
        p_value = mixed_effects(
            df=subset,
            dependent_var="correlation",
            independent_var="genotype",
            group_name="mouse_id",
        ).filter(like="C(genotype)")
        p_values[f"{stage}"] = p_value

    sns.boxplot(
        data=all_data,
        x="stage",
        y="correlation",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        palette=palette,
        showfliers=False,
    )

    plt.tight_layout()
    sns.despine()
    plt.ylim(None, 1.49)
    ax = plt.gca()
    ymin_plot, ymax_plot = ax.get_ylim()
    plot_range = ymax_plot - ymin_plot
    text_y = ymax_plot - plot_range * 0.1  # place text just below the top of the axis
    for i, stage in enumerate(["Baseline", "Trained"]):
        p_text = f"P = {round(p_values[stage].values[0], 2)}"
        ax.text(i, text_y, p_text, ha="center", va="top")

    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
        # place legend centered relative to the axes (not the whole figure)
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2)

    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "correlations"
        / f"comparison_correlations_rewarded_{rewarded}.png"
    )


def get_correlation_df(genotype: str, rewarded: bool | None) -> pd.DataFrame:

    result = {"stage": [], "correlation": [], "mouse_id": [], "genotype": []}
    for mouse_name in SESSIONS_KEEP.keys():
        if get_genotype(mouse_name) != genotype:
            continue
        for stage in ["unsupervised", "learning", "learned"]:
            print(f"Doing {mouse_name} at {stage} stage")
            date = SESSIONS_KEEP[mouse_name][stage]
            if date is None:
                continue
            data = np.load(
                SERVER_PATH
                / "viral_caches"
                / "correlations"
                / f"{mouse_name}_{date}_rewarded_{rewarded}_correlation.npy"
            )
            correlations = upper_triangle_no_diagonal(data)
            result["correlation"].extend(correlations.tolist())
            result["mouse_id"].extend([mouse_name] * len(correlations))
            stage_name = "Baseline" if stage == "unsupervised" else "Trained"
            result["stage"].extend([stage_name] * len(correlations))
            result["genotype"].extend([genotype] * len(correlations))

    return pd.DataFrame(result)


def display_fov(stat: np.ndarray, cell_highlight: List[int]) -> None:
    img = np.zeros((512, 512))
    for idx, cell in enumerate(stat):
        ypix = cell["ypix"]
        xpix = cell["xpix"]
        if idx in cell_highlight:
            img[ypix, xpix] = 2
            print(
                f"Highlighting cell {idx} the center is at ({cell['xpix'].mean()}, {cell['ypix'].mean()})"
            )
        else:
            img[ypix, xpix] = 1

    plt.figure()
    plt.imshow(img, cmap="gray")


def example_traces() -> None:

    s2p_path = Path("/Volumes/MarcBusche/Josef/2P/2025-07-04/JB034/suite2p/plane0/")

    stat = np.load(s2p_path / "stat.npy", allow_pickle=True)
    ops = np.load(s2p_path / "ops.npy", allow_pickle=True).item()
    imwrite(SERVER_PATH / "viral_plots" / "example_traces" / "fov.tiff", ops["meanImg"])

    pass
    # np.save(SERVER_PATH / "viral_plots" / "example_traces" / "fov.tiff", stat['me')

    # f_raw = np.load(s2p_path / "F.npy")
    # f_neu = np.load(s2p_path / "Fneu.npy")
    # f_raw = f_raw[:, 10000:17000]
    # f_neu = f_neu[:, 10000:17000]
    # np.save("fraw.npy", f_raw)
    # np.save("fneu.npy", f_neu)
    f_raw = np.load("fraw.npy")
    f_neu = np.load("fneu.npy")

    f = subtract_neuropil(f_raw, f_neu)

    baseline = np.mean(f, axis=1, keepdims=True)
    dff = (f - baseline) / baseline

    colors = sns.color_palette(n_colors=2)

    cells_keep = [5, 6, 9, 19, 26, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 150]
    plt.figure()

    # For grant plot
    cells_keep = cells_keep[:7]
    display_fov(stat, cells_keep)

    # np.random.shuffle(cells_keep)
    n = 0
    for idx in cells_keep:
        data = dff[idx] + n * 1.6
        data = moving_average(data, 5)

        plt.plot(data, color=colors[0], linewidth=1)
        n += 1

    sns.despine(left=True, bottom=True)
    plt.axis("off")

    # --- Add compact bottom-right scalebar (replacing axes) ---
    ax = plt.gca()
    # Sampling rate (Hz). 2P traces elsewhere use 30 FPS
    fs = 30.0
    # Determine trace length in frames from the last plotted line if possible
    # Use dff shape (n_cells, n_frames)
    n_frames = dff.shape[1]
    duration_s = n_frames / fs

    # Choose a time scalebar that's <= 20% of the duration, from nice values
    nice_times = np.array([0.5, 1, 2, 5, 10, 20, 30, 60])
    max_time = max(0.2 * duration_s, 0.5)
    bar_t = (
        float(nice_times[nice_times <= max_time][-1])
        if np.any(nice_times <= max_time)
        else 0.5
    )
    bar_t_frames = bar_t * fs

    # Choose a vertical amplitude scalebar in dF/F units from nice values
    # The traces are offset by 1.4 between cells; choose a modest vertical scalebar
    nice_dff = np.array([0.1, 0.2, 0.5, 1.0])
    # Aim for ~8% of the total y-range
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin if ymax > ymin else max(1.0, len(cells_keep) * 1.4)
    target_dff = 0.08 * y_range
    bar_dff = (
        float(nice_dff[nice_dff <= target_dff][-1])
        if np.any(nice_dff <= target_dff)
        else float(nice_dff[0])
    )

    # Position: bottom-right with small margins (in data coords)
    x0, x1 = ax.get_xlim()
    margin_x = 0.02 * (x1 - x0)
    margin_y = 0.007 * (y_range)

    x_start = x1 - margin_x - bar_t_frames
    y_start = ymin + margin_y

    # Draw horizontal (time) bar
    ax.plot(
        [x_start, x_start + bar_t_frames],
        [y_start, y_start],
        color="black",
        lw=1.5,
        solid_capstyle="butt",
    )
    # Draw vertical (amplitude) bar
    ax.plot(
        [x_start + bar_t_frames, x_start + bar_t_frames],
        [y_start, y_start + bar_dff],
        color="black",
        lw=1.5,
        solid_capstyle="butt",
    )

    # Labels: time (bottom) and dF/F (side)
    # Format helpers
    def _fmt_val(v: float) -> str:
        # Use integer when close, else one decimal
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"

    time_label = f"{_fmt_val(bar_t)} s"
    dff_label = f"{_fmt_val(bar_dff)} ΔF/F"

    # Place time label centered under the horizontal bar (clamp to stay inside axes)
    x_mid = x_start + 0.5 * bar_t_frames
    y_text_bottom = y_start - 0.015 * y_range
    # if y_text_bottom < ymin + 0.005 * y_range:
    #     y_text_bottom = y_start + 0.018 * y_range
    ax.text(
        x_mid,
        y_text_bottom,
        time_label,
        ha="center",
        va="top" if y_text_bottom <= y_start else "bottom",
        fontsize=9,
        color="black",
    )

    # Place dF/F label to the left of the vertical bar, centered vertically
    x_text_side = x_start + bar_t_frames - 0.01 * (x1 - x0)
    y_mid = y_start + 0.5 * bar_dff
    ax.text(
        x_text_side,
        y_mid,
        dff_label,
        ha="right",
        va="center",
        fontsize=9,
        color="black",
    )

    # plt.savefig(SERVER_PATH / "viral_plots" / "example_traces" / f"example_traces.png")
    save_figure(
        SERVER_PATH / "viral_plots" / "example_traces" / "example_traces_grant.pdf"
    )
    plt.show()


def compute_anticipatory_licking() -> Dict[str, List[tuple[float, float]]]:
    result: Dict[str, List[tuple[float, float]]] = {
        "Baseline": [],
        "Trained": [],
    }

    for mouse, sessions in SESSIONS_KEEP.items():
        date_trained = sessions["learned"]

        if date_trained is None:
            continue

        metadata = gsheet2df(SPREADSHEET_ID, mouse, 1)
        first_session_row = metadata[metadata["Type"].str.lower() == "learning day 1"]
        assert len(first_session_row) == 1
        first_session_date = first_session_row["Date"].values[0]

        for date, session_type in zip(
            [first_session_date, date_trained], ["First Session", "Final Session"]
        ):

            session_number = metadata[metadata["Date"] == date][
                "Session Number"
            ].values[0]
            session_path = (
                BEHAVIOUR_DATA_PATH
                / mouse
                / date
                / parse_session_number(session_number)[0]
            )
            trials = load_data(session_path)
            summaries = [
                summarise_trial(trial, get_wheel_circumference_from_rig("2P"))
                for trial in trials
            ]
            rewarded = [trial.licks_AZ > 0 for trial in summaries if trial.rewarded]
            not_rewarded = [
                trial.licks_AZ > 0 for trial in summaries if not trial.rewarded
            ]

            result_tuple = (
                (sum(rewarded) / len(rewarded)) * 100,
                (sum(not_rewarded) / len(not_rewarded)) * 100,
            )

            if session_type == "First Session":
                result["Baseline"].append(result_tuple)
            else:
                result["Trained"].append(result_tuple)
    return result


def basic_anticipatory_licking_plot() -> None:
    if Path("AL_dict.pkl.npy").exists():
        data = np.load("AL_dict.pkl.npy", allow_pickle=True).item()
    else:
        data = compute_anticipatory_licking()
        np.save("AL_dict.pkl.npy", data)

    def make_plot(tuples: List[tuple[float, float]], ax: plt.Axes) -> None:
        means = np.mean(tuples, axis=0)
        for session in tuples:
            ax.plot([1, 0], session, color="gray", alpha=0.3)

        ax.plot([1, 0], means, color="black", linewidth=3, marker="o")

        ax.set_xlim(-0.2, 1.2)
        ax.set_xticks([0, 1], ["Unrewarded\nTexture", "Rewarded\nTexture"])
        arr = np.array(tuples)
        p_value = stats.ttest_rel(arr[:, 0], arr[:, 1]).pvalue
        ax.text(
            0.5,
            80,
            f"P = {p_value:.2g}",
            ha="center",
            va="bottom",
            fontsize=12,
            font="arial",
        )

    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

    make_plot(data["Baseline"], axes[0])
    make_plot(data["Trained"], axes[1])
    axes[0].set_ylabel("Trials with licks in AZ (%)")
    axes[1].set_title("After Learning")
    axes[0].set_title("Before Learning")
    sns.despine()
    plt.tight_layout()

    plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "anticipatory_licking"
        / f"basic_anticipatory_licking_plot.pdf",
        bbox_inches="tight",
        transparent=True,
    )

    1 / 0


if __name__ == "__main__":
    # basic_anticipatory_licking_plot()

    # save_firing_rates("WT")
    # save_firing_rates("NLGF")

    firing_rates_plot(None)
    # for genotype in tqdm(
    #     ["Oligo-BACE1-KO", "NLGF", "WT", "Neuronal-BACE1-KO"], desc="corrleations"
    # ):
    #     # for rewarded in [True, False, None]:

    # example_traces()
    # n_responders_comparison_plot()
    # firing_rates_plot(None)
    # 1 / 0
