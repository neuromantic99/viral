# Allow you to run the file directly, remove if exporting as a proper module
from pathlib import Path
import pickle
import sys
import time

import pandas as pd
from tqdm import tqdm

from scipy.ndimage import gaussian_filter1d


HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.constants import SERVER_PATH
from viral.sessions_keep import SESSIONS_KEEP

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from viral.imaging_utils import (
    activity_trial_position,
    get_online_position_and_frames,
    load_imaging_data,
    trial_is_imaged,
)

from scipy.stats import zscore
from viral.models import Cached2pSession
from viral.utils import get_genotype, get_wheel_circumference_from_rig
from viral.constants import grosmark_config


CACHE_PATH = Path("/Volumes/MarcBusche/Josef/viral_caches/fisher/denoised")

locations = {
    "trial onset": 0,
    "landmark 1": 45,
    "landmark 2": 90,
    "landmark 3": 135,
    "reward": 180,
}


ALL_LOCATIONS = list(locations.values())
LOCATION_NAMES = list(locations.keys())


N_FRAMES_PRE = 30
N_FRAMES_POST = 60


def filter_neurons(dff, session) -> np.ndarray:
    """1) ∆F/F had to exceed three standard deviations of the ROI’s overall activity on at least 25% of all trials; 2)
    the mean ∆F/F across trials had to exceed a peak z-score of 3 at its peak. The z-score for each ROI was determined
    by randomly rotating its ∆F/F time course with respect to its behavior 500 times and the peak value of the
    mean trace was then used to calculate the peak z-score; 3) the minimum of the mean trace amplitude
    (i.e. highest  lowest value) had to exceed 0.2 ∆F/F.
    Skipped number 2 because I don't understand it

    """

    sd_threshold = np.std(dff, axis=1) * 3
    n_over_std_threshold = np.zeros(dff.shape[0])
    n_total_trials = 0
    first_trial = True

    all_trials_dff = []

    for trial in session.trials:

        if not trial_is_imaged(trial):
            continue
        if first_trial:
            # Skip the first trial as we don't have enough data before the trial onset
            first_trial = False
            continue

        n_total_trials += 1
        dff_trial = dff[
            :, trial.trial_start_closest_frame : trial.trial_end_closest_frame
        ]
        all_trials_dff.append(dff_trial)
        mask = np.greater(dff_trial.T, sd_threshold)
        exceeds = np.any(mask, axis=0)
        n_over_std_threshold += exceeds

    mask_25_percent = n_over_std_threshold >= (n_total_trials * 0.25)
    shortest_trial = min([trial_dff.shape[1] for trial_dff in all_trials_dff])
    mean_dff = np.mean(
        np.array([trial_dff[:, :shortest_trial] for trial_dff in all_trials_dff]),
        axis=0,
    )
    amplitude = mean_dff.max(1) - mean_dff.min(1)
    mask_amplitude = amplitude > 0.2
    overall_mask = mask_25_percent & mask_amplitude
    print(f"percentage of neurons that pass the filter: {overall_mask.mean()}")
    return overall_mask


def process_session(session: Cached2pSession, dff: np.ndarray) -> None:

    mask = filter_neurons(dff, session)
    dff = dff[mask, :]
    # Smooth spikes with a 150ms Gaussian kernel, which is about 4.5 frames at 30Hz
    # from scipy.ndimage import gaussian_filter1d

    # t1 = time.time()
    # dff = gaussian_filter1d(dff, sigma=4.5, axis=1)
    # t2 = time.time()
    # print(f"Smoothing took {t2 - t1:.2f} seconds")

    # TODO: might want to increase this and use NaNs for when it goes out of bounds
    result_rewarded: dict[int, list[np.ndarray]] = {
        location: [] for location in ALL_LOCATIONS
    }

    result_unrewarded: dict[int, list[np.ndarray]] = {
        location: [] for location in ALL_LOCATIONS
    }

    first_trial = True

    for trial in session.trials:

        if not trial_is_imaged(trial):
            continue
        if first_trial:
            # Skip the first trial as we don't have enough data before the trial onset
            first_trial = False
            continue

        position, frame_position = get_online_position_and_frames(
            trial,
            get_wheel_circumference_from_rig("2P"),
            threshold_speed=False,
            speed_threshold=None,
        )

        for position_of_interest in ALL_LOCATIONS:
            # idx_crossing = np.where(position > position_of_interest)[0][0]
            idx_crossing = np.argmin(np.abs(position - position_of_interest))
            start_frame = frame_position[idx_crossing] - N_FRAMES_PRE
            end_frame = frame_position[idx_crossing] + N_FRAMES_POST
            assert start_frame >= 0
            assert end_frame < dff.shape[1]

            if trial.texture_rewarded:
                result_rewarded[position_of_interest].append(
                    dff[
                        :,
                        start_frame:end_frame,
                    ]
                )
            else:
                result_unrewarded[position_of_interest].append(
                    dff[
                        :,
                        start_frame:end_frame,
                    ]
                )

    result_all = {
        location: result_rewarded[location] + result_unrewarded[location]
        for location in ALL_LOCATIONS
    }

    peak_rewarded, diffs_rewarded = compute_peak(
        result_rewarded,
        pcs_combined=None,
    )

    np.save(
        CACHE_PATH / f"{session.mouse_name}_{session.date}_peak_rewarded.npy",
        peak_rewarded,
    )
    with open(
        CACHE_PATH / f"{session.mouse_name}_{session.date}_diffs_rewarded.pkl", "wb"
    ) as f:
        pickle.dump(diffs_rewarded, f)

    peak_unrewarded, diffs_unrewarded = compute_peak(
        result_unrewarded,
        pcs_combined=None,
    )

    np.save(
        CACHE_PATH / f"{session.mouse_name}_{session.date}_peak_unrewarded.npy",
        peak_unrewarded,
    )
    with open(
        CACHE_PATH / f"{session.mouse_name}_{session.date}_diffs_unrewarded.pkl", "wb"
    ) as f:
        pickle.dump(diffs_unrewarded, f)

    peak_all, diffs_all = compute_peak(
        result_all,
        pcs_combined=None,
        plot=True,
        plot_name=f"{session.mouse_name}_{session.date}_graph",
    )

    np.save(CACHE_PATH / f"{session.mouse_name}_{session.date}_peak_all.npy", peak_all)
    with open(
        CACHE_PATH / f"{session.mouse_name}_{session.date}_diffs_all.pkl", "wb"
    ) as f:
        pickle.dump(diffs_all, f)

    plot = True
    if plot:
        for title, peak in zip(
            ["Rewarded", "Unrewarded", "All"],
            [peak_rewarded, peak_unrewarded, peak_all],
        ):

            plt.figure()
            plt.bar(
                LOCATION_NAMES,
                [np.sum(peak == i) / peak.shape[0] for i in range(len(ALL_LOCATIONS))],
            )
            plt.ylabel("Fraction of neurons")
            plt.title(title)


def compute_peak(
    result: dict[int, list[np.ndarray]],
    pcs_combined: np.ndarray | None = None,
    plot: bool = False,
    plot_name: str | None = None,
) -> tuple[np.ndarray, dict[int, list[np.ndarray]]]:

    diffs: dict[int, list[np.ndarray]] = {location: [] for location in result.keys()}

    plot = True

    for location, trial_list in result.items():
        # n_trials x n_cells x n_bins
        X = np.array(trial_list)

        trial_averaged = X.mean(0)
        peak = trial_averaged.max(1)
        diffs[location] = peak

        # Trials x cells x bins
        if pcs_combined is not None:
            X = X[:, pcs_combined, :]

        if plot:
            plt.figure()
            x_axis = np.arange(-N_FRAMES_PRE, N_FRAMES_POST) / 30
            for cell in range(X.shape[1]):
                plt.plot(x_axis, np.mean(X[:, cell, :], 0), alpha=0.5, color="gray")

            plt.plot(x_axis, np.mean(X, (0, 1)), color="black", linewidth=2)
            # plt.plot(x_axis, np.mean(X, (0, 1)), color="black", linewidth=2)
            plt.xlabel("Time from landmark (s)")
            plt.ylabel("Z-scored activity")

            plt.title(f"Location: {location}")
            plt.tight_layout()
            plt.savefig(CACHE_PATH / f"{plot_name}_{location}_activity.png")

    # diff_matrix = np.array([diffs[0], diffs[45], diffs[90], diffs[135], diffs[180]])
    diff_matrix = np.array([diffs[location] for location in ALL_LOCATIONS])

    peak = np.argmax(diff_matrix, 0)
    return peak, diffs


def load_data_and_process(mouse: str, date: str) -> None:

    with open(
        SERVER_PATH / "viral_caches" / "cached_2p" / f"{mouse}_{date}.json", "r"
    ) as f:
        session = Cached2pSession.model_validate_json(f.read())

    print(f"Total number of trials: {len(session.trials)}")
    print(
        f"number of trials imaged {len([trial for trial in session.trials if trial_is_imaged(trial)])}"
    )

    if CACHE_PATH / f"{mouse}_{date}_peak_rewarded.npy" in CACHE_PATH.iterdir():
        print(f"Skipping {mouse} at {date} because peak_all.npy already exists")
        return

    # denoised = load_imaging_data(mouse, date)

    # np.save(
    #     Path("/Volumes/hard_drive/viral/dff") / f"{mouse}_{date}_denoised.npy", denoised
    # )
    denoised = np.load(
        Path("/Volumes/hard_drive/viral/dff") / f"{mouse}_{date}_denoised.npy"
    )
    # normalise each trace by the mean
    denoised = denoised / (np.mean(denoised, axis=1, keepdims=True))

    # spks = np.load(Path("/Volumes/hard_drive/viral/dff") / f"{mouse}_{date}_spks.npy")

    process_session(session, denoised)


def process_all() -> None:
    for mouse_name in tqdm(SESSIONS_KEEP.keys()):
        for stage in ["unsupervised", "learning", "learned"]:
            print(f"Doing {mouse_name} at {stage} stage")
            date = SESSIONS_KEEP[mouse_name][stage]
            if date is None:
                print(f"Skipping {mouse_name} at {stage} stage because date is None")
                continue
            load_data_and_process(mouse_name, date)


def plot_results_binary_peak() -> None:
    result: dict[str, list] = {
        "mouse_id": [],
        "genotype": [],
        "stage": [],
        "date": [],
    } | {location: [] for location in LOCATION_NAMES}
    for mouse_name in SESSIONS_KEEP.keys():
        for stage in ["unsupervised", "learning", "learned"]:
            date = SESSIONS_KEEP[mouse_name][stage]
            if date is None:
                continue
            try:
                peak = np.load(CACHE_PATH / f"{mouse_name}_{date}_peak_all.npy")
            except Exception as e:
                print(f"Error loading {mouse_name} at {stage} stage: {e}")
                continue
            result["genotype"].append(get_genotype(mouse_name))
            result["mouse_id"].append(mouse_name)
            result["stage"].append(stage)
            result["date"].append(date)

            for idx, location_name in enumerate(LOCATION_NAMES):
                result[location_name].append(np.sum(peak == idx) / peak.shape[0])

    df = pd.DataFrame(result)
    for genotype in df["genotype"].unique():
        plt.figure()
        stage = "learned"
        df_sub = df[df["genotype"] == genotype]
        df_sub = df_sub[df_sub["stage"] == stage]
        df_sub["landmarks"] = df_sub[["landmark 1", "landmark 2", "landmark 3"]].sum(
            axis=1
        )

        location_names_plot = ["trial onset", "landmarks", "reward"]

        plt.bar(
            location_names_plot,
            [df_sub[location].mean() for location in location_names_plot],
            yerr=[
                df_sub[location].std() / np.sqrt(len(df_sub))
                for location in location_names_plot
            ],
        )
        plt.ylabel("Fraction of neurons")
        plt.title(f"{genotype}")
        plt.ylim(0, 0.6)
        plt.tight_layout()
        plt.savefig(CACHE_PATH / f"{genotype}_{stage}_binary_peak.png")

    1 / 0


def plot_results_analog_peak() -> None:
    result: dict[str, list] = {
        "mouse_id": [],
        "genotype": [],
        "stage": [],
        "date": [],
        "cell_idx": [],
        "location": [],
        "peaks": [],
    }
    for mouse_name in SESSIONS_KEEP.keys():
        if get_genotype(mouse_name) == "Neuronal-BACE1-KO":
            continue
        for stage in ["unsupervised", "learning", "learned"]:
            date = SESSIONS_KEEP[mouse_name][stage]
            if date is None:
                continue
            try:
                with open(CACHE_PATH / f"{mouse_name}_{date}_diffs_all.pkl", "rb") as f:
                    result_dict = pickle.load(f)
            except Exception as e:
                print(f"Error loading {mouse_name} at {stage} stage: {e}")
                continue
            for location_cm, cell_peaks in result_dict.items():
                location_name = LOCATION_NAMES[ALL_LOCATIONS.index(location_cm)]
                n_cells = len(cell_peaks)
                cell_peaks[cell_peaks < 0] = 0
                cell_peaks[cell_peaks > 10] = 10

                result["genotype"].extend([get_genotype(mouse_name)] * n_cells)
                result["mouse_id"].extend([mouse_name] * n_cells)
                result["stage"].extend([stage] * n_cells)
                result["date"].extend([date] * n_cells)
                result["cell_idx"].extend(list(range(n_cells)))
                result["location"].extend([location_name] * n_cells)
                result["peaks"].extend(cell_peaks)

    df = pd.DataFrame(result)

    learned_df = df[df["stage"] == "learning"]

    landmark_locations = {"landmark 1", "landmark 2", "landmark 3"}
    is_landmark = learned_df["location"].isin(landmark_locations)

    learned_landmarks_avg = (
        learned_df.loc[is_landmark]
        .groupby(["mouse_id", "date", "genotype", "stage", "cell_idx"], as_index=False)[
            "peaks"
        ]
        .mean()
        .assign(location="all landmarks")
    )

    learned_plot_df = pd.concat(
        [learned_df.loc[~is_landmark], learned_landmarks_avg],
        ignore_index=True,
    )

    plt.clf()
    sns.violinplot(
        data=learned_plot_df,
        x="location",
        y="peaks",
        hue="genotype",
    )
    # plt.hist(learned_df["peaks"])
    # fix a mixed effects model here with statsmodels or something

    1 / 0


if __name__ == "__main__":
    # process_all()
    plot_results_binary_peak()
