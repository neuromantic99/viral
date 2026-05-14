import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pydantic import ValidationError
from scipy import stats

from ensemble_reactivation import main as ensemble_main
from viral.grosmark_analysis import get_place_cells
from viral.sessions_keep import SESSIONS_KEEP
from scipy import stats

from viral.utils import boxplot, degrees_to_cm, get_speed_positions

# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


from viral.models import Cached2pSession, Mouse2pSessions
from viral.cache_2p_sessions import process_session
from viral.constants import (
    BEHAVIOUR_DATA_PATH,
    CACHE_PATH,
    SERVER_PATH,
    SPREADSHEET_ID,
    SYNC_FILE_PATH,
    TIFF_UMBRELLA,
    grosmark_config,
)
from viral.ensemble_reactivation import main as ensemble_main
from viral.grosmark_analysis import get_place_cells, plot_place_cell_heatmap
from viral.gsheets_importer import gsheet2df
from viral.imaging_utils import trial_is_imaged
from viral.models import Cached2pSession, GrosmarkConfig, Mouse2pSessions
from viral.multiple_sessions import parse_session_number
from viral.sessions_keep import SESSIONS_KEEP
from viral.single_session import load_data
from viral.utils import (
    boxplot,
    degrees_to_cm,
    get_genotype,
    get_speed_positions,
    get_wheel_circumference_from_rig,
    shaded_line_plot,
)

CACHE_PATH = HERE.parent / "data" / "cached_2p"

#### PRETTY SURE I MESSED THIS UP IN THE GIT MERGE


def get_session(
    mouse_name: str, date: str, metadata: pd.DataFrame, stage: str
) -> Cached2pSession:
    path = CACHE_PATH / f"{mouse_name}_{date}.json"
    try:
        cached_session = Cached2pSession.model_validate_json(path.read_text())
        print(f"Loaded cached session for {mouse_name} {date} from {path}")
        return cached_session
    except (FileNotFoundError, ValidationError) as e:
        print(f"Cache missing for {mouse_name} {date}. Reprocessing session.")
        row = metadata[metadata["Date"] == date].squeeze(axis=0)
        session_type = row["Type"].lower()
        assert (
            "learning" in session_type if stage == "learned" else stage in session_type
        )
        session_numbers = parse_session_number(row["Session Number"])
        trials = []
        for session_number in session_numbers:
            session_path = (
                BEHAVIOUR_DATA_PATH / mouse_name / row["Date"] / session_number
            )
            trials.extend(load_data(session_path))
        print(f"Got error when loading {mouse_name} {date} from cache. Error is: {e}")

        try:
            wheel_blocked = row["Wheel blocked?"].lower() in {"yes", "true"}
        except KeyError as e:
            print(f"No column 'Wheel blocked?' found: {e}")
            print("Wheel blocked set to None")
            wheel_blocked = False

        process_session(
            trials=trials,
            tiff_directory=TIFF_UMBRELLA / date / mouse_name,
            tdms_path=SYNC_FILE_PATH / Path(row["Sync file"]),
            mouse_name=mouse_name,
            session_type=session_type,
            date=date,
            wheel_blocked=wheel_blocked,
        )

    return Cached2pSession.model_validate_json(path.read_text())


def get_completed_mouse_sessions(mouse_name: str) -> Mouse2pSessions:
    results = [None, None, None]
    for idx, stage in enumerate(["unsupervised", "learning", "learned"]):
        path = CACHE_PATH / f"{mouse_name}_{SESSIONS_KEEP[mouse_name][stage]}.json"
        try:
            results[idx] = Cached2pSession.model_validate_json(path.read_text())
            print(f"Loaded cached session for {mouse_name} {stage} from {path}")
        except (ValidationError, FileNotFoundError) as e:
            print(f"Error retrieving unsupervised session for {mouse_name}: {e}")

    return Mouse2pSessions(
        mouse_name=mouse_name,
        unsupervised=results[0],
        learning=results[1],
        learned=results[2],
    )


def get_completed_mouse_sessions(mouse_name: str) -> Mouse2pSessions:
    results = [None, None, None]
    for idx, stage in enumerate(["unsupervised", "learning", "learned"]):
        path = CACHE_PATH / f"{mouse_name}_{SESSIONS_KEEP[mouse_name][stage]}.json"
        try:
            results[idx] = Cached2pSession.model_validate_json(path.read_text())
            print(f"Loaded cached session for {mouse_name} {stage} from {path}")
        except (ValidationError, FileNotFoundError) as e:
            print(f"Error retrieving unsupervised session for {mouse_name}: {e}")

    return Mouse2pSessions(
        mouse_name=mouse_name,
        unsupervised=results[0],
        learning=results[1],
        learned=results[2],
    )


def get_mouse_sessions(mouse_name: str) -> Mouse2pSessions:
    metadata = gsheet2df(SPREADSHEET_ID, mouse_name, 1)
    stages = ["unsupervised", "learning", "learned"]
    sessions: dict[str, Cached2pSession | None] = {}
    for stage in stages:
        if SESSIONS_KEEP[mouse_name][stage] is None:
            sessions[stage] = None
            continue

        sessions[stage] = get_session(
            mouse_name,
            SESSIONS_KEEP[mouse_name][stage],
            metadata,
            stage=stage,
        )

    return Mouse2pSessions(
        mouse_name=mouse_name,
        unsupervised=sessions["unsupervised"],
        learning=sessions["learning"],
        learned=sessions["learned"],
    )


def store_place_cell_result(mouse_name: str, date: str, config: GrosmarkConfig) -> None:
    print("Processing", mouse_name, date)
    with open(CACHE_PATH / f"{mouse_name}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())
    spks_path = TIFF_UMBRELLA / session.date / session.mouse_name / "suite2p" / "plane0"

    assert (
        spks_path / "full_grosmark_oasis_preprocessed.npy"
    ).exists(), (
        f"File {spks_path / 'full_grosmark_oasis_preprocessed.npy'} does not exist"
    )
    spks = np.load(spks_path / "oasis_spikes.npy")

    for rewarded in [False, True, None]:
        pcs_mask, smoothed_matrix, place_threshold = get_place_cells(
            session=session,
            spks=spks,
            rewarded=rewarded,
            config=config,
            plot=False,
            bin_occupancy_divide=True,
        )


class PlaceCellResults:
    LANDMARK_LOCATIONS = [45, 90, 135]
    LANDMARK_WIDTH = 5

    def __init__(
        self,
        cache_umbrella: Path,
        genotype: str,
        plot_type: Literal["corridor_activity", "tuning"],
        bod: bool = False,
        verbose: bool = False,
    ) -> None:

        filter_files = lambda dir: [
            p
            for p in dir.iterdir()
            if p.is_file()
            and p.suffix == ".npy"
            and "split" not in p.name
            and ("BOD" not in p.name if not bod else "BOD_True" in p.name)
        ]

        self.smoothed_matrix_files = filter_files(cache_umbrella / "smoothed_matrix")
        self.pcs_combined_files = filter_files(cache_umbrella / "pcs_combined")
        self.place_threshold_files = filter_files(cache_umbrella / "place_threshold")

        self.unsupervised: Dict[str, List] = {
            "rewarded": [],
            "unrewarded": [],
            "both": [],
        }
        self.learning: Dict[str, List] = {"rewarded": [], "unrewarded": [], "both": []}
        self.learned: Dict[str, List] = {"rewarded": [], "unrewarded": [], "both": []}

        self.plot_type = plot_type
        self.genotype = genotype
        self.verbose = verbose

    def load_file(
        self, file_list: List[Path], date: str, mouse: str, rewarded: bool | None
    ) -> np.ndarray:
        files_match = [
            file
            for file in file_list
            if f"{mouse}_{date}" in file.name
            and f"rewarded_{rewarded}" in file.name
            and f"{grosmark_config}" in file.name
        ]
        if not files_match:
            raise FileNotFoundError(
                f"No file found for {mouse} {date} rewarded {rewarded}"
            )

        assert (
            len(files_match) == 1
        ), f"more than one file found for {mouse} {date} rewarded {rewarded}"
        if self.verbose:
            print("These files matched is :", files_match[0])
        return np.load(files_match[0])

    def collapsed_matrix_result(
        self, mouse_name: str, stage: str, rewarded: bool | None
    ) -> np.ndarray | float:
        date = SESSIONS_KEEP[mouse_name][stage]
        if self.verbose:
            print(
                f"Loading {mouse_name} stage {stage} rewarded {rewarded}. Date is {date}"
            )

        smoothed_matrix = self.load_file(
            self.smoothed_matrix_files,
            date=date,
            mouse=mouse_name,
            rewarded=rewarded,
        )

        pcs_combined = self.load_file(
            self.pcs_combined_files,
            date=date,
            mouse=mouse_name,
            rewarded=rewarded,
        )
        place_threshold = self.load_file(
            self.place_threshold_files,
            date=date,
            mouse=mouse_name,
            rewarded=rewarded,
        )
        mask = smoothed_matrix[pcs_combined, :] > place_threshold[pcs_combined, :]
        if self.plot_type == "corridor_activity":
            return np.sum(mask, axis=0) / mask.shape[0]
        return self.landmark_tuning(mask)

    def landmark_tuning(self, mask: np.ndarray) -> float:
        n_bins = mask.shape[1]
        assert (
            n_bins
            == (grosmark_config.end - grosmark_config.start) / grosmark_config.bin_size
        )
        bin_to_cm_scaling_factor = (
            grosmark_config.end - grosmark_config.start
        ) / n_bins
        result = []

        for landmark_center in self.LANDMARK_LOCATIONS:
            landmark_bin_center = int(landmark_center / bin_to_cm_scaling_factor)
            start_inside = landmark_bin_center - (5 / bin_to_cm_scaling_factor)
            end_inside = landmark_bin_center + (5 / bin_to_cm_scaling_factor)
            n_cells_in = np.mean(
                np.sum(mask[:, int(start_inside) : int(end_inside)], axis=0)
            )
            start_outside_left = landmark_bin_center - int(
                10 / bin_to_cm_scaling_factor
            )
            end_outside_right = landmark_bin_center + int(10 / bin_to_cm_scaling_factor)
            n_cells_out = np.mean(
                np.concatenate(
                    (
                        np.sum(
                            mask[:, int(start_outside_left) : int(start_inside)], axis=0
                        ),
                        np.sum(
                            mask[:, int(end_inside) : int(end_outside_right)], axis=0
                        ),
                    )
                )
            )
            result.append(n_cells_in / n_cells_out)

        return np.mean(result)

    def driver(self) -> None:
        for stage, store in zip(
            ["unsupervised", "learning", "learned"],
            [self.unsupervised, self.learning, self.learned],
        ):
            for mouse_name in SESSIONS_KEEP.keys():
                if get_genotype(mouse_name) != self.genotype:
                    if self.verbose:
                        print(
                            f"Skipping {mouse_name} as genotype is not {self.genotype}"
                        )
                    continue
                if self.verbose:
                    print(f"Mouse is {self.genotype} genotype, processing {mouse_name}")

                for rewarded in [False, True]:
                    try:
                        result = self.collapsed_matrix_result(
                            mouse_name, stage=stage, rewarded=rewarded
                        )
                    except FileNotFoundError:
                        continue
                    if rewarded is None:
                        store["both"].append(result)
                    else:
                        store["rewarded" if rewarded else "unrewarded"].append(result)

    def plot_result(
        self,
        stage_data: List,
        label: str,
        color: str,
        axis: plt.Axes | None = None,
    ) -> None:
        matrix = np.vstack(stage_data)
        shaded_line_plot(
            arr=matrix,
            x_axis=np.linspace(0, 180, matrix.shape[1]),
            color=color,
            label=label,
            axis=axis,
        )
        for landmark_center in [45, 90, 135]:
            plotter = axis if axis is not None else plt
            plotter.axvspan(
                landmark_center - 2.5, landmark_center + 2.5, color="red", alpha=0.5
            )


def get_speed_summary(
    session: Cached2pSession, rewarded: bool | None, config: GrosmarkConfig
) -> np.ndarray:
    return np.array(
        [
            np.array(
                [
                    speed.speed
                    for speed in get_speed_positions(
                        degrees_to_cm(
                            np.array(trial.rotary_encoder_position),
                            get_wheel_circumference_from_rig("2P"),
                        ),
                        config.start,
                        config.end,
                        config.bin_size,
                        sampling_rate=30,
                    )
                ]
            )
            for trial in session.trials
            if trial_is_imaged(trial)
            and (rewarded is None or trial.texture_rewarded == rewarded)
        ]
    )


def tuning_comparison_plot() -> None:

    1 / 0

    wt = PlaceCellResults(
        SERVER_PATH / "viral_caches" / "place_cells",
        genotype="WT",
        plot_type="corridor_activity",
    )
    wt.driver()
    nlgf = PlaceCellResults(
        SERVER_PATH / "viral_caches" / "place_cells",
        genotype="NLGF",
        plot_type="corridor_activity",
    )
    nlgf.driver()

    colors = sns.color_palette(n_colors=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    wt.plot_result(wt.unsupervised["both"], "WT Unsupervised", colors[0], axis=axes[0])
    nlgf.plot_result(
        nlgf.unsupervised["both"], "NLGF Unsupervised", colors[1], axis=axes[0]
    )

    wt.plot_result(wt.learned["both"], "WT Unsupervised", colors[0], axis=axes[1])
    nlgf.plot_result(nlgf.learned["both"], "NLGF Unsupervised", colors[1], axis=axes[1])

    axes[0].set_ylim(0, 0.5)

    axes[1].set_xlabel("Corridor position (cm)")
    axes[0].set_xlabel("Corridor position (cm)")

    axes[0].set_ylabel("Fraction cells\nsignificantly active")
    axes[0].legend()
    axes[0].set_title("Before learning")
    axes[1].set_title("After learning")
    plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "visual_tuning"
        / f"visual_tuning_genotype_comparison.pdf",
        bbox_inches="tight",
        transparent=True,
    )

    1 / 0


def plot_place_cell_results(
    genotype: str, plot_type: Literal["corridor_activity", "tuning"]
) -> None:

    place_cell_result = PlaceCellResults(
        SERVER_PATH / "viral_caches" / "place_cells",
        genotype=genotype,
        plot_type=plot_type,
        bod=False,
    )
    place_cell_result.driver()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

    for ax, data, name in zip(
        axes,
        [
            place_cell_result.unsupervised,
            # place_cell_result.learning,
            place_cell_result.learned,
        ],
        ["unsupervised", "learned"],
    ):
        plt.sca(ax)
        if plot_type == "tuning":
            sns.boxplot(
                data={
                    "rewarded": data["rewarded"],
                    "unrewarded": data["unrewarded"],
                },
                palette={"rewarded": "blue", "unrewarded": "green"},
                showfliers=False,
            )

            sns.stripplot(
                data={
                    "rewarded": data["rewarded"],
                    "unrewarded": data["unrewarded"],
                },
                palette={"rewarded": "blue", "unrewarded": "green"},
                edgecolor="black",
                linewidth=1,
            )

            plt.axhline(1)
            plt.ylim(0.9, 1.7)
            if ax is axes[0]:
                plt.ylabel("Landmark tuning index")

        if plot_type == "corridor_activity":
            place_cell_result.plot_result(
                # data["unrewarded"],
                data["both"],
                "both",
                "black",
            )
            # place_cell_result.plot_result(data["rewarded"], "rewarded", "blue")
            ax.set_ylim(0, 0.5)
            ax.set_xlabel("Corridor position (cm)")
            if ax is axes[0]:
                ax.set_ylabel("Proportion place cells\nsignificantly active")
                ax.legend()
            ax.set_title(name.capitalize())

    plt.suptitle(genotype)
    plt.tight_layout()
    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "visual_tuning"
        / f"visual_tuning_{genotype}_{plot_type}.png",
        dpi=300,
    )

    # plt.show()


def plot_speed_summary() -> None:
    result = {
        stage: {"rewarded": [], "unrewarded": []}
        for stage in ["unsupervised", "learning", "learned"]
    }

    for mouse_name, dates in SESSIONS_KEEP.items():
        if mouse_name not in {"JB034", "JB035", "JB036"}:
            continue

        for stage, date in dates.items():
            with open(CACHE_PATH / f"{mouse_name}_{date}.json", "r") as f:
                session = Cached2pSession.model_validate_json(f.read())
                for rewarded in [False, True]:
                    speed = get_speed_summary(
                        session, rewarded=rewarded, config=grosmark_config
                    )
                    result[stage]["rewarded" if rewarded else "unrewarded"].append(
                        speed
                    )


def run_ensembles() -> None:
    for mouse in SESSIONS_KEEP.keys():
        if mouse not in {"JB034", "JB035", "JB036"}:
            continue

        for date in SESSIONS_KEEP[mouse].values():
            for rewarded in [True, False, None]:
                print(f"Starting {mouse} {date} rewarded {rewarded}")
                ensemble_main(mouse, date, rewarded=rewarded, plot=False)


def plot_reward_discrimination(genotype: str) -> None:
    place_cell_result = PlaceCellResults(
        SERVER_PATH / "viral_caches" / "place_cells",
        genotype=genotype,
        plot_type="corridor_activity",
    )
    place_cell_result.driver()

    result = {}
    for stage, store in zip(
        ["unsupervised", "learning", "learned"],
        [
            place_cell_result.unsupervised,
            place_cell_result.learning,
            place_cell_result.learned,
        ],
    ):
        rewarded = np.array(store["rewarded"])
        unrewarded = np.array(store["unrewarded"])
        assert rewarded.shape == unrewarded.shape

        reward_zone_size_cm = 10
        n_bins = reward_zone_size_cm // grosmark_config.bin_size
        fraction_rewarded = rewarded[:, -n_bins:].mean(axis=1)
        fraction_unrewarded = unrewarded[:, -n_bins:].mean(axis=1)
        discrimination_index = fraction_rewarded - fraction_unrewarded

        result[stage] = discrimination_index

    collapsed_result = {
        "unsupervised": result["unsupervised"],
        "learned": np.concatenate((result["learning"], result["learned"])),
    }

    plt.figure()
    plt.title(f"Reward discrimination index {genotype}")
    boxplot(collapsed_result)
    plt.ylim(-0.2, 0.3)
    plt.axhline(0, color="grey", linestyle="--")


def reward_discrimination(rewarded: np.ndarray, unrewarded: np.ndarray) -> float:
    reward_zone_size_cm = 10
    n_bins = reward_zone_size_cm // grosmark_config.bin_size
    fraction_rewarded = rewarded[:, -n_bins:].mean(axis=1)
    fraction_unrewarded = unrewarded[:, -n_bins:].mean(axis=1)
    discrimination_index = fraction_rewarded - fraction_unrewarded

    return discrimination_index


def reward_discrimination_comparison_plot() -> None:
    wt = PlaceCellResults(
        SERVER_PATH / "viral_caches" / "place_cells",
        genotype="WT",
        plot_type="corridor_activity",
    )
    wt.driver()
    nlgf = PlaceCellResults(
        SERVER_PATH / "viral_caches" / "place_cells",
        genotype="NLGF",
        plot_type="corridor_activity",
    )
    nlgf.driver()

    result = {"genotype": [], "stage_name": [], "reward_discrimination": []}

    for genotype_name, genotype_data in zip(["WT", "NLGF"], [wt, nlgf]):
        for stage, data in zip(
            ["unsupervised", "learning", "learned"],
            [
                genotype_data.unsupervised,
                genotype_data.learning,
                genotype_data.learned,
            ],
        ):
            rewarded = np.array(data["rewarded"])
            unrewarded = np.array(data["unrewarded"])
            assert rewarded.shape == unrewarded.shape

            reward_zone_size_cm = 10
            n_bins = reward_zone_size_cm // grosmark_config.bin_size
            fraction_rewarded = rewarded[:, -n_bins:].mean(axis=1)
            fraction_unrewarded = unrewarded[:, -n_bins:].mean(axis=1)
            discrimination_index = fraction_rewarded / fraction_unrewarded
            discrimination_index[np.isinf(discrimination_index)] = 1
            stage_name = "Baseline" if stage == "unsupervised" else "Trained"
            result["genotype"].extend([genotype_name] * len(discrimination_index))
            result["stage_name"].extend([stage_name] * len(discrimination_index))
            result["reward_discrimination"].extend(discrimination_index)

    df = pd.DataFrame(result)

    fig = plt.figure()
    colors = sns.color_palette(n_colors=4)
    palette = {"Trained": colors[2], "Baseline": colors[3]}

    p_values = {}

    for stage_name in ["Baseline", "Trained"]:
        subset = df[df["stage_name"] == stage_name]
        assert len(subset) > 2, "make sure nothing weird happend"
        p_values[f"{stage_name}"] = 100

    sns.boxplot(
        data=df,
        x="genotype",
        y="reward_discrimination",
        hue="stage_name",
        hue_order=["Baseline", "Trained"],
        palette=palette,
        showfliers=False,
    )

    sns.stripplot(
        data=df,
        x="genotype",
        y="reward_discrimination",
        hue="stage_name",
        hue_order=["Baseline", "Trained"],
        palette=palette,
        dodge=True,
        linewidth=1,
        edgecolor="black",
    )

    plt.tight_layout()
    sns.despine()
    plt.ylim(None, 3.2)
    ax = plt.gca()
    ymin_plot, ymax_plot = ax.get_ylim()
    plot_range = ymax_plot - ymin_plot
    text_y = ymax_plot - plot_range * 0.1  # place text just below the top of the axis
    for i, genotype in enumerate(["WT", "NLGF"]):
        subset = df[df["genotype"] == genotype]
        p_value = stats.ttest_ind(
            subset[subset["stage_name"] == "Baseline"]["reward_discrimination"],
            subset[subset["stage_name"] == "Trained"]["reward_discrimination"],
        ).pvalue
        p_text = f"P = {round(p_value, 2)}"
        ax.text(i, text_y, p_text, ha="center", va="top")

    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(
        handles[:2], labels[:2], loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2
    )
    plt.xlabel("Genotype")
    plt.ylabel("Reward zone discrimination index")
    plt.axhline(1, color="grey", linestyle="--")
    plt.ylim(None, 3.2)
    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "reward_discrimination"
        / f"reward_discrimination_comparison_plot.png"
    )


def landmark_comparison_plot(bod: bool = False) -> None:

    wt = PlaceCellResults(
        SERVER_PATH / "viral_caches" / "place_cells",
        genotype="WT",
        plot_type="tuning",
        bod=bod,
    )
    wt.driver()
    nlgf = PlaceCellResults(
        SERVER_PATH / "viral_caches" / "place_cells",
        genotype="NLGF",
        plot_type="tuning",
        bod=bod,
    )
    nlgf.driver()

    result = {"genotype": [], "stage_name": [], "landmark_tuning": [], "rewarded": []}

    for genotype_name, genotype_data in zip(["WT", "NLGF"], [wt, nlgf]):
        for stage, data in zip(
            ["unsupervised", "learning", "learned"],
            [
                genotype_data.unsupervised,
                genotype_data.learning,
                genotype_data.learned,
            ],
        ):
            stage_name = "Baseline" if stage == "unsupervised" else "Trained"
            result["genotype"].extend(
                [genotype_name]
                * (len(data["rewarded"]) + len(data["unrewarded"]) + len(data["both"]))
            )
            result["stage_name"].extend(
                [stage_name]
                * (len(data["rewarded"]) + len(data["unrewarded"]) + len(data["both"]))
            )
            result["landmark_tuning"].extend(
                data["rewarded"] + data["unrewarded"] + data["both"]
            )
            result["rewarded"].extend(
                [True] * len(data["rewarded"])
                + [False] * len(data["unrewarded"])
                + ["Both"] * len(data["both"])
            )
    df = pd.DataFrame(result)

    plt.clf()
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    colors = sns.color_palette(n_colors=2)
    palette = {"WT": colors[0], "NLGF": colors[1]}

    for rewarded, ax in zip(["Both", True, False], axes):
        sns.boxplot(
            data=df[df["rewarded"] == rewarded],
            x="stage_name",
            y="landmark_tuning",
            hue="genotype",
            hue_order=["WT", "NLGF"],
            palette=palette,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            data=df[df["rewarded"] == rewarded],
            x="stage_name",
            y="landmark_tuning",
            hue="genotype",
            hue_order=["WT", "NLGF"],
            palette=palette,
            dodge=True,
            linewidth=1,
            edgecolor="black",
            ax=ax,
        )
        ax.set_title(
            "Rewarded"
            if rewarded is True
            else "Unrewarded" if rewarded is False else "Both"
        )
        ax.set_xlabel("Stage")

        if ax is axes[0]:
            ax.set_ylabel("Landmark tuning index")

        ax.axhline(1, color="grey", linestyle="--")

    handles, labels = axes[1].get_legend_handles_labels()
    # remove per-axis legends
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2)
    sns.despine()
    plt.ylim(None, 1.6)

    for idx, rewarded in enumerate([True, False, "Both"]):
        ymin_plot, ymax_plot = axes[idx].get_ylim()
        plot_range = ymax_plot - ymin_plot
        text_y = (
            ymax_plot - plot_range * 0.02
        )  # place text just below the top of the axis

        for i, stage in enumerate(["Baseline", "Trained"]):
            p_value = stats.ttest_ind(
                df[
                    (df["stage_name"] == stage)
                    & (df["genotype"] == "WT")
                    & (df["rewarded"] == rewarded)
                ]["landmark_tuning"],
                df[
                    (df["stage_name"] == stage)
                    & (df["genotype"] == "NLGF")
                    & (df["rewarded"] == rewarded)
                ]["landmark_tuning"],
            ).pvalue
            p_text = f"P = {p_value:.2g}"
            axes[idx].text(i, text_y, p_text, ha="center", va="top")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.rcParams["pdf.fonttype"] = 42
    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "landmark_tuning"
        / f"landmark_tuning_comparison_plot_bod_{bod}.pdf",
        bbox_inches="tight",
        transparent=True,
    )


def plot_place_cell_heatmaps() -> None:

    rewarded = None
    bod = True

    for mouse_name, dates in SESSIONS_KEEP.items():

        for stage, date in dates.items():
            if date is None:
                continue
            smoothed_matrix = np.load(
                SERVER_PATH
                / "viral_caches"
                / "place_cells"
                / "smoothed_matrix"
                / f"{mouse_name}_{date}_rewarded_{rewarded}_{grosmark_config}_smoothed_matrix{"_BOD_True" if bod else ""}.npy"
            )
            pcs_combined = np.load(
                SERVER_PATH
                / "viral_caches"
                / "place_cells"
                / "pcs_combined"
                / f"{mouse_name}_{date}_rewarded_{rewarded}_{grosmark_config}_pcs_combined{"_BOD_True" if bod else ""}.npy"
            )
            smoothed_matrix = smoothed_matrix[pcs_combined, :]

            plot_place_cell_heatmap(
                smoothed_matrix,
                grosmark_config,
            )
            plt.title(f"{mouse_name} {stage} rewarded={rewarded} BOD = {bod}")
            plt.tight_layout()
            plt.savefig(
                SERVER_PATH
                / "viral_plots"
                / "place_cell_heatmaps"
                / f"{mouse_name}_{date}_rewarded_{rewarded}_bod_{bod}.png",
                dpi=300,
            )




if __name__ == "__main__":
    # plot_place_cell_heatmaps()
    # landmark_comparison_plot(bod=False)
    # tuning_comparison_plot()

    # for mouse_name, dates in SESSIONS_KEEP.items():

    #     for stage, date in dates.items():
    #         store_place_cell_result(mouse_name, date, config=grosmark_config)
