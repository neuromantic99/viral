from collections import defaultdict
from math import ceil, floor
import sys
from pathlib import Path
from typing import Literal

from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from scipy.stats import zscore

from scipy.optimize import curve_fit

from viral.learning_stages import PlaceCellResults

from viral.constants import (
    CACHE_PATH,
    SERVER_PATH,
    TIFF_UMBRELLA,
    grosmark_config,
)
from viral.imaging_utils import activity_trial_position, trial_is_imaged
from viral.models import Cached2pSession
from viral.utils import (
    degrees_to_cm,
    get_genotype,
    get_speed_positions,
    get_wheel_circumference_from_rig,
    round_up_to_base,
    shaded_line_plot,
    basic_normalise,
    imshow,
    exp_model,
    corr_vs_distance,
)
from viral.sessions_keep import SESSIONS_KEEP

from viral.grosmark_analysis import correlations_vs_peak_distance


from sklearn.linear_model import LogisticRegression


import seaborn as sns

sns.set_context("talk")


def do_classify(
    session: Cached2pSession, spks: np.ndarray, plot: bool = False
) -> tuple[np.ndarray, np.ndarray]:

    bin_size = 10
    start = 0
    max_position = 180

    X_list = []
    y = []

    for trial in session.trials:
        if not trial_is_imaged(trial):
            continue

        data = activity_trial_position(
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
        X_list.append(data)
        if trial.texture_rewarded:
            # if random.random() > 0.5:
            y.append(1)
        else:
            y.append(0)

    X = np.array(X_list)

    scores_position = []

    train_size = 0.8

    coefs = []
    for bin in range(X.shape[2]):

        scores = []
        for fold in range(10):
            X_train, X_test, y_train, y_test = train_test_split(
                X[:, :, bin], y, train_size=train_size
            )

            clf = LogisticRegression(penalty="l1", solver="liblinear").fit(
                X_train, y_train
            )

            scores.append(clf.score(X_test, y_test))
            coefs.append(clf.coef_)

        scores_position.append(scores)

    scores_position = np.array(scores_position)
    coefs = np.array(coefs)

    n_trials = X.shape[0]
    split_position = int(n_trials * train_size)

    X_train = X[:split_position, :, :]
    y_train = y[:split_position]

    X_test = X[split_position:, :, :]
    y_test = y[split_position:]

    drift_score = []
    for bin in range(X.shape[2]):

        clf = LogisticRegression(penalty="l1", solver="liblinear").fit(
            X_train[:, bin], y_train
        )

        drift_score.append(clf.score(X_test[:, bin], y_test))

    if plot:
        x_axis = np.arange(start, max_position, bin_size)

        plt.plot(x_axis, drift_score, color="red", label="drift")
        shaded_line_plot(
            arr=scores_position.T,
            x_axis=x_axis,
            color="blue",
            label="not drift",
        )

        plt.axhline(0.5)

        plt.xlabel("Position (cm)")
        plt.ylabel("Classifier accuracy")
        plt.ylim(0, 1)

    return np.array(drift_score), scores_position


def get_landmark_boolean(max_position: int, start: int, bin_size: int) -> np.ndarray:
    n_bins = int((max_position - start) / bin_size)
    bin_to_cm_scaling_factor = (grosmark_config.end - grosmark_config.start) / n_bins
    # bin_centers = np.arange(start, max_position, bin_size) + (bin_size / 2)
    in_landmark = np.array([False] * n_bins)

    for landmark_center in PlaceCellResults.LANDMARK_LOCATIONS:
        landmark_bin_center = landmark_center / bin_to_cm_scaling_factor
        start_inside = floor(landmark_bin_center - (5 / bin_to_cm_scaling_factor))
        end_inside = floor(landmark_bin_center + (5 / bin_to_cm_scaling_factor))
        print(landmark_bin_center, start_inside, end_inside)
        in_landmark[int(start_inside) : int(end_inside)] = True

    return in_landmark


def landmark_drift(
    session: Cached2pSession, spks: np.ndarray, plot: bool = False
) -> tuple[np.ndarray, np.ndarray]:

    bin_size = 2
    start = 0
    max_position = 180
    data_matrix = np.array(
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
        ]
    )

    # For drift need to split this by halves
    X_original = np.nanmean(data_matrix, 1)
    X_original = zscore(X_original, axis=0)

    in_landmark = get_landmark_boolean(max_position, start, bin_size)
    assert len(in_landmark) == X_original.shape[1]
    X_in_landmark = X_original[:, in_landmark]
    y_in_landmark = np.array([True] * X_in_landmark.shape[1])

    X_out_landmark = X_original[:, ~in_landmark]
    y_out_landmark = np.array([False] * X_out_landmark.shape[1])

    X = np.concatenate([X_in_landmark, X_out_landmark], axis=1).T
    y = np.concatenate([y_in_landmark, y_out_landmark], axis=0)

    kf = StratifiedKFold(n_splits=3, shuffle=True)

    scores = []
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        clf = LogisticRegression(penalty="l2", C=1).fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = balanced_accuracy_score(y_test, y_pred)
        scores.append(score)
    assert np.mean(scores) >= 0.6, "fit failed"

    # Prediction_plot
    trial_split_position = X.shape[1] // 2
    clf = LogisticRegression(penalty="l2", C=1).fit(X[:, :trial_split_position], y)

    prob = clf.predict_proba(X[:, trial_split_position:])[:, 1]

    print(f"Landmark drift scores: {scores}")

    return np.array([]), np.array([])


def get_trial_speeds(
    session: Cached2pSession,
    start: int,
    end: int,
    bin_size: int,
    rewarded: bool | None,
) -> np.ndarray:
    trial_speed = np.array(
        [
            np.array(
                [
                    speed.speed
                    for speed in get_speed_positions(
                        degrees_to_cm(
                            np.array(trial.rotary_encoder_position),
                            get_wheel_circumference_from_rig("2P"),
                        ),
                        start,
                        end,
                        bin_size,
                        sampling_rate=30,
                    )
                ]
            )
            for trial in session.trials
            if trial_is_imaged(trial)
            and (rewarded is None or trial.texture_rewarded == rewarded)
        ]
    )
    trial_speed[trial_speed < 5] = np.nan
    trial_speed = interpolate_nans(np.expand_dims(trial_speed, axis=0)).squeeze()
    return trial_speed


def speed_control(
    trial_speed: np.ndarray,
) -> np.ndarray:
    n_trials = trial_speed.shape[0]
    speed_result = np.empty((n_trials, n_trials))

    for i in range(trial_speed.shape[0]):
        for j in range(trial_speed.shape[0]):
            speed_result[i, j] = stats.pearsonr(
                trial_speed[i, :], trial_speed[j, :]
            ).correlation

    return speed_result


def speed_modulation(
    speed_trial: np.ndarray,
    cell_array: np.ndarray,
) -> np.ndarray:

    p_vals = []
    for cell_idx in range(cell_array.shape[1]):
        cell = cell_array[:, cell_idx, :]
        p_vals.append(stats.spearmanr(cell.mean(0), speed_trial.mean(0)).pvalue)
    return np.array(p_vals) < 0.05


def correlation_drift(
    session: Cached2pSession,
    spks: np.ndarray,
    rewarded: bool | None,
    plot: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bin_size = 5
    start = 0
    max_position = 180

    trial_speeds = get_trial_speeds(
        session=session,
        start=start,
        end=max_position,
        bin_size=bin_size,
        rewarded=rewarded,
    )

    speed_correlation = speed_control(
        trial_speed=trial_speeds,
    )

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
                threshold_speed=True,
            )
            for trial in session.trials
            if trial_is_imaged(trial)
            and (rewarded is None or trial.texture_rewarded == rewarded)
        ]
    )

    # Remove trials where every value is NaN
    trial_array = trial_array[~np.all(np.isnan(trial_array), (1, 2)), :, :]

    trial_array = interpolate_nans(trial_array)
    modulated_cells = speed_modulation(trial_speeds, trial_array)

    trial_array = trial_array[:, ~modulated_cells, :]

    population_response = trial_array.mean(1)
    n_trials = trial_array.shape[0]
    n_cells = trial_array.shape[1]

    population_correlation = np.empty((n_trials, n_trials))
    for i in range(trial_array.shape[0]):
        for j in range(trial_array.shape[0]):
            population_correlation[i, j] = stats.pearsonr(
                population_response[i, :], population_response[j, :]
            ).correlation

    cell_by_cell_correlation = np.empty((n_cells, n_trials, n_trials))

    for neuron_idx in range(n_cells):
        for i in range(trial_array.shape[0]):
            for j in range(trial_array.shape[0]):
                cell_by_cell_correlation[neuron_idx, i, j] = stats.pearsonr(
                    trial_array[i, neuron_idx, :], trial_array[j, neuron_idx, :]
                ).correlation

    if plot:
        plt.plot(
            basic_normalise(speed_correlation[trial_group_size:]),
            label="speed",
            color="gray",
        )
        plt.plot(
            basic_normalise(decay[trial_group_size:]),
            label="cell by cell",
            color="blue",
        )
        plt.plot(
            basic_normalise(population_correlation[trial_group_size:]),
            label="population",
            color="red",
        )
        plt.legend()

    return cell_by_cell_correlation, population_correlation, speed_correlation


def interpolate_nans(all_trials_matrix: np.ndarray) -> np.ndarray:
    """This interpolates based on neighbouring values missing data in the trial matrices.
    Need to be very careful with this, as incomparable numbers of NaNs in different genotypes
    could affect genotype comparisons

    """
    for trial_matrix in all_trials_matrix:
        if np.isnan(trial_matrix).any():
            # Replace NaNs with the interpolation of the surrounding values
            for neuron_idx in range(trial_matrix.shape[0]):
                neuron_data = trial_matrix[neuron_idx, :]
                nans = np.isnan(neuron_data)
                not_nans = ~nans
                neuron_data[nans] = np.interp(
                    np.flatnonzero(nans),
                    np.flatnonzero(not_nans),
                    neuron_data[not_nans],
                )

    return all_trials_matrix


def landmark_correlation_drift(
    session: Cached2pSession,
    spks: np.ndarray,
    pcs_mask: np.ndarray,
    rewarded: bool | None,
    save_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    bin_size = 2
    start = 0
    max_position = 180
    data_matrix = np.array(
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

    place_threshold = np.load(save_path)
    place_threshold = place_threshold[pcs_mask, :]
    responding = np.greater(data_matrix, place_threshold)
    in_landmark = get_landmark_boolean(max_position, start, bin_size)
    has_landmark_response = responding[:, :, in_landmark].sum(2) > 4

    trial_times = np.array(
        [
            trial.trial_start_time
            for trial in session.trials
            if trial_is_imaged(trial)
            and (rewarded is None or trial.texture_rewarded == rewarded)
        ]
    )
    trial_times = trial_times - trial_times[0]
    bin_width = 60 * 2
    # Round up to nearest bin_width
    max_time = round_up_to_base(trial_times.max(), bin_width)
    bin_starts = np.arange(0, max_time, bin_width)

    m, (x_corr, y_corr) = correlations_vs_peak_distance(
        np.corrcoef(has_landmark_response.astype(int)),
        trial_times,
        bin_starts=bin_starts,
        plot=False,
    )
    np.savez(
        SERVER_PATH
        / "viral_caches"
        / "landmark_drift_correlation"
        / f"{session.mouse_name}_{session.date}_landmark_drift_correlation_rewarded_{rewarded}.npz",
        x_corr=x_corr,
        y_corr=y_corr,
    )

    return np.array([]), np.array([])


def main() -> None:
    cache_umbrella = SERVER_PATH / "viral_caches" / "drift_correlation"

    for mouse_name in tqdm(SESSIONS_KEEP.keys()):
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
                    / "place_cells"
                    / "place_threshold"
                    / f"{session.mouse_name}_{session.date}_rewarded_{rewarded}_{grosmark_config}_place_threshold.npy"
                )
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

                landmark_correlation_drift(
                    session=session,
                    spks=spks,
                    pcs_mask=pc_mask,
                    rewarded=rewarded,
                    save_path=save_path,
                )

                # whole_trial_correlation(
                #     spks=spks,
                #     session=session,
                #     rewarded=rewarded,
                #     save_path=save_path,
                # )


def whole_trial_correlation(
    spks: np.ndarray,
    session: Cached2pSession,
    rewarded: bool | None,
    save_path: Path,
) -> None:

    cell_wise, population_wise, speed_wise = correlation_drift(
        session, spks, rewarded=rewarded, plot=False
    )

    x_axis = [
        trial.trial_start_time
        for trial in session.trials
        if trial_is_imaged(trial)
        and (rewarded is None or trial.texture_rewarded == rewarded)
    ]

    print("Saving results to ", save_path.name)
    np.savez(
        save_path,
        cell_wise=cell_wise,
        population_wise=population_wise,
        speed_wise=speed_wise,
        x_axis=x_axis,
    )


def plot_overall_scores(genotype: str) -> None:

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for idx, stage in enumerate(["unsupervised", "learning", "learned"]):
        all_data = []
        all_drift = []
        for mouse_name in tqdm(SESSIONS_KEEP.keys()):
            if get_genotype(mouse_name) != genotype:
                continue
            date = SESSIONS_KEEP[mouse_name][stage]
            if date is None:
                continue

            cache_path = (
                SERVER_PATH
                / "viral_caches"
                / "drift"
                / f"{mouse_name}_{date}_drift.npz"
            )

            # if not cache_path.exists():
            #     continue

            data = np.load(cache_path)
            drift_score = data["drift_score"]
            scores = data["scores"]
            all_data.append(scores)
            all_drift.append(drift_score)

        all_data = np.array(all_data)
        all_drift = np.array(all_drift)

        x_axis = np.arange(5, 185, 10)

        shaded_line_plot(
            arr=all_data.mean(2),
            x_axis=x_axis,
            color="blue",
            label=f"Mixed",
            do_moving_average=True,
            axis=axes[idx],
        )

        shaded_line_plot(
            arr=all_drift,
            x_axis=x_axis,
            color="red",
            do_moving_average=True,
            label="Drift",
            axis=axes[idx],
        )
        axes[idx].axhline(0.5, color="grey", linestyle="--")

        axes[idx].set_title(stage)

        axes[idx].set_xticks([0, 90, 180], labels=["0", "90", "180"])
        if idx == 0:
            axes[idx].set_ylabel("Classifier accuracy")
            axes[idx].legend()

        if idx == 1:
            axes[idx].set_xlabel("Corridor position (cm)")

        axes[idx].text(
            180,
            0.45,
            "chance level",
            horizontalalignment="right",
            verticalalignment="center",
            color="gray",
            fontsize=12,
            weight="bold",
            clip_on=True,
        )
    plt.suptitle(genotype)
    plt.tight_layout()
    plt.ylim(0.3, 1)
    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "drift"
        / f"{genotype}_representational_drift.png",
        dpi=300,
    )


def plot_drift_correlation_results(
    genotype: Literal["Oligo-BACE1-KO", "NLGF", "WT", "Neuronal-BACE1-KO"],
    rewarded: bool | None,
) -> None:
    # _, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    result = {"unsupervised": [], "learning": [], "learned": []}
    for idx, stage in enumerate(["unsupervised", "learning", "learned"]):
        all_population = []
        all_cell = []
        all_speed = []
        for mouse_name in tqdm(SESSIONS_KEEP.keys()):
            if get_genotype(mouse_name) != genotype:
                continue
            date = SESSIONS_KEEP[mouse_name][stage]
            if date is None:
                continue

            cache_path = (
                SERVER_PATH
                / "viral_caches"
                / "drift_correlation"
                / f"{mouse_name}_{date}_drift_correlation_rewarded_{rewarded}.npz"
            )
            assert cache_path.exists()
            data = np.load(cache_path)
            cell_wise = data["cell_wise"]
            population_wise = data["population_wise"]
            speed_wise = data["speed_wise"]
            cell_wise = np.nanmean(cell_wise, 0)
            try:
                assert len(cell_wise) == len(population_wise)
            except TypeError:
                continue

            # correlations_vs_peak_distance function has the same logic as a grosmark plot, so
            # hijack. But some of the argument names don't make sense here

            trial_times = data["x_axis"] - data["x_axis"][0]
            # 2 minute bins
            bin_width = 60 * 2
            # Round up to nearest minute
            max_time = round_up_to_base(trial_times.max(), bin_width)
            bin_starts = np.arange(0, max_time, bin_width)

            population_distance_corr, (x_pop, y_pop) = correlations_vs_peak_distance(
                population_wise,
                trial_times,
                bin_starts,
                label=f"{mouse_name}_population",
            )
            cell_distance_corr, (x_cell, y_cell) = correlations_vs_peak_distance(
                cell_wise, trial_times, bin_starts, label=f"{mouse_name}_cell"
            )
            speed_distance_corr, (x_speed, y_speed) = correlations_vs_peak_distance(
                speed_wise, trial_times, bin_starts, label=f"{mouse_name}_speed"
            )
            all_population.append((population_distance_corr, x_pop, y_pop))
            all_cell.append((cell_distance_corr, x_cell, y_cell))
            all_speed.append((speed_distance_corr, x_speed, y_speed))

        result[stage] = all_population

    plt.figure()
    sns.boxplot({k: [x[0] for x in v] for k, v in result.items()}, showfliers=False)
    sns.stripplot(
        {k: [x[0] for x in v] for k, v in result.items()}, color="black", alpha=0.5
    )

    plt.title(f"{genotype}")
    plt.tight_layout()
    plt.ylim(-0.2, 0.2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for idx, stage in enumerate(["unsupervised", "learning", "learned"]):
        for mouse in result[stage]:
            axes[idx].plot(mouse[1], mouse[2])
        axes[idx].set_title(stage)
    plt.ylim(-1, 1)

    plt.tight_layout()
    plt.suptitle(genotype)


def compute_tau(to_fit: np.ndarray, plot: bool = False) -> float:

    # TODO: this should be the time of the trial
    t = np.arange(len(to_fit))
    popt, _ = curve_fit(
        f=exp_model,
        xdata=t,
        ydata=to_fit,
        p0=(to_fit[0], 10, 0),
    )
    if plot:
        plt.figure()
        plt.plot(to_fit, label="data")
        plt.plot(t, exp_model(t, *popt), label="fit")
    return popt[1]


def plot_landmark_drift_correlation_results(
    genotype: Literal["Oligo-BACE1-KO", "NLGF", "WT", "Neuronal-BACE1-KO"],
    rewarded: bool | None,
) -> None:
    # _, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    result = {"unsupervised": [], "learning": [], "learned": []}
    result_mean = {"unsupervised": [], "learning": [], "learned": []}
    for idx, stage in enumerate(["unsupervised", "learning", "learned"]):
        all_results = []
        mean_result = defaultdict(list)
        for mouse_name in SESSIONS_KEEP.keys():
            if get_genotype(mouse_name) != genotype:
                continue
            date = SESSIONS_KEEP[mouse_name][stage]
            if date is None:
                continue

            cache_path = (
                SERVER_PATH
                / "viral_caches"
                / "landmark_drift_correlation"
                / f"{mouse_name}_{date}_landmark_drift_correlation_rewarded_{rewarded}.npz"
            )
            assert cache_path.exists()
            data = np.load(cache_path)
            x_corr = data["x_corr"]
            y_corr = data["y_corr"]
            assert len(x_corr) == len(y_corr)
            all_results.append((x_corr, y_corr))

            for x_val, y_val in zip(x_corr, y_corr):
                mean_result[x_val].append(y_val)

        result[stage] = all_results
        result_mean[stage] = (
            np.array(sorted(mean_result.keys())),
            np.array([np.nanmean(mean_result[k]) for k in sorted(mean_result.keys())]),
        )

    # plt.figure()
    # sns.boxplot({k: [x[0] for x in v] for k, v in result.items()}, showfliers=False)
    # sns.stripplot(
    #     {k: [x[0] for x in v] for k, v in result.items()}, color="black", alpha=0.5
    # )

    # plt.title(f"{genotype}")
    # plt.tight_layout()
    # plt.ylim(-0.2, 0.2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for idx, stage in enumerate(["unsupervised", "learning", "learned"]):
        for mouse in result[stage]:
            axes[idx].plot(mouse[0] / 60, mouse[1], color="lightgray", alpha=0.5)

        axes[idx].plot(
            result_mean[stage][0] / 60,
            result_mean[stage][1],
            color="blue",
            linewidth=2,
            label="Mean",
        )
        axes[idx].set_title(stage)
        axes[idx].set_xlim(0, 50)

    plt.ylim(-0.1, 0.5)

    plt.tight_layout()
    plt.suptitle(genotype)


if __name__ == "__main__":
    # for genotype in ["Oligo-BACE1-KO", "NLGF", "WT", "Neuronal-BACE1-KO"]:
    #     plot_overall_scores(genotype)
    # for genotype in ["Oligo-BACE1-KO", "NLGF", "WT", "Neuronal-BACE1-KO"]:
    #     plot_landmark_drift_correlation_results(genotype, None)
    main()

    1 / 0
