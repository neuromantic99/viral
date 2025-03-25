from pathlib import Path
import random
import sys
from typing import List, Tuple
from scipy.stats import median_abs_deviation, zscore
import seaborn as sns

from scipy.ndimage import gaussian_filter1d

from rastermap import Rastermap

# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


sns.set_theme(context="talk", style="ticks")


# from viral.grosmark_analysis import grosmark_place_field
from viral.grosmark_analysis import grosmark_place_field
from viral.imaging_utils import activity_trial_position, get_ITI_matrix, trial_is_imaged
from viral.rastermap_utils import get_ITI_start_frame
from viral.utils import (
    array_bin_mean,
    degrees_to_cm,
    get_wheel_circumference_from_rig,
    normalize,
    remove_consecutive_ones,
)

from viral.models import Cached2pSession, TrialInfo


# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from matplotlib import pyplot as plt
import numpy as np


from viral.constants import SERVER_PATH, TIFF_UMBRELLA


def compute_dff(f: np.ndarray) -> np.ndarray:
    flu_mean = np.expand_dims(np.mean(f, 1), 1)
    return (f - flu_mean) / flu_mean


def subtract_neuropil(f_raw: np.ndarray, f_neu: np.ndarray) -> np.ndarray:
    return f_raw - f_neu * 0.7


def get_dff(mouse: str, date: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s2p_path = TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0"
    print(f"Suite 2p path is {s2p_path}")
    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)

    spks = np.load(s2p_path / "oasis_spikes.npy")[iscell, :]
    denoised = np.load(s2p_path / "oasis_denoised.npy")[iscell, :]

    f_raw = np.load(s2p_path / "F.npy")[iscell, :]
    f_neu = np.load(s2p_path / "Fneu.npy")[iscell, :]

    dff = compute_dff(subtract_neuropil(f_raw, f_neu))
    return dff, spks, denoised


def remove_landmarks_from_train_matrix(train_matrix: np.ndarray) -> np.ndarray:
    """We seem to get a lot of neurons with a peak at the landmark. This removes the landmark from the
    train matrix so cells do not get sorted by their landmark peak
    TODO: This will not work if bin_size != 1, originally had an integer division which may deal with this
    """

    train_matrix[:, 33:40] = 0
    train_matrix[:, 76:85] = 0
    train_matrix[:, 120:132] = 0
    return train_matrix


def get_position_activity(
    trials: List[TrialInfo],
    dff: np.ndarray,
    wheel_circumference: float,
    bin_size: int,
    start: int,
    max_position: int,
    remove_landmarks: bool,
    ITI_bin_size: int | None,
) -> np.ndarray:
    """If ITI bin_size is None then don't add the ITI on"""

    test_matrices = []
    for _ in range(10):
        train_idx = random.sample(range(len(trials)), len(trials) // 2)
        test_idx = [idx for idx in range(len(trials)) if idx not in train_idx]

        # Find the order in which to sort neurons in a random 50% of the trials
        train_matrix = np.nanmean(
            np.array(
                [
                    activity_trial_position(
                        trials[idx],
                        dff,
                        wheel_circumference,
                        bin_size=bin_size,
                        start=start,
                        max_position=max_position,
                        smoothing_sigma=None,
                    )
                    for idx in train_idx
                ]
            ),
            axis=0,
        )

        if remove_landmarks:
            train_matrix = remove_landmarks_from_train_matrix(train_matrix)

        peak_indices = np.argmax(train_matrix, axis=1)
        sorted_order = np.argsort(peak_indices)

        # model = Rastermap(n_PCs=200, n_clusters=5, locality=1, time_lag_window=0).fit(
        #     train_matrix
        # )
        # sorted_order = model.isort

        # TODO: Review whether this is the best normalisation function
        test_matrix = normalize(
            np.nanmean(
                np.array(
                    [
                        activity_trial_position(
                            trials[idx],
                            dff,
                            wheel_circumference,
                            bin_size=bin_size,
                            start=start,
                            max_position=max_position,
                            smoothing_sigma=None,
                        )
                        for idx in test_idx
                    ]
                ),
                axis=0,
            ),
            axis=1,
        )

        test_matrices.append(test_matrix[sorted_order, :])

    test_matrices_averaged = np.mean(np.array(test_matrices), 0)

    if ITI_bin_size is None:
        return test_matrices_averaged

    # Either return the ITI, or random chunks of rest, remove the comments
    # When decide what to do
    resting = get_resting_chunks(
        trials, dff, chunk_size_frames=15 * 30, speed_threshold=1
    )

    resting = resting[np.argsort(np.argmax(test_matrices_averaged, axis=1)), :]
    return np.hstack((test_matrices_averaged, resting))

    # ITI = get_ITI_matrix(trials, dff, bin_size=ITI_bin_size)
    # ITI = ITI[np.argsort(np.argmax(test_matrices_averaged, axis=1)), :]
    # return np.hstack((test_matrices_averaged, ITI))


def running_during_ITI(trial: TrialInfo) -> bool:

    trigger_states = np.array(
        [
            state.name
            for state in trial.states_info
            if state.name
            in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
        ]
    )

    ITI_positions = degrees_to_cm(
        np.array(trial.rotary_encoder_position)[trigger_states == "trigger_panda_ITI"],
        get_wheel_circumference_from_rig("2P"),
    )
    return max(ITI_positions) - min(ITI_positions) > 20


def place_cells_unsupervised(session: Cached2pSession, dff: np.ndarray) -> None:

    plt.figure()
    plt.title("Unsupervised Session")
    data = get_position_activity(
        [trial for trial in session.trials if trial_is_imaged(trial)],
        dff,
        get_wheel_circumference_from_rig("2P"),
    )
    plt.imshow(data, aspect="auto", cmap="viridis")
    clb = plt.colorbar()
    clb.ax.set_title("Normalised\nactivity", fontsize=12)

    plt.ylabel("cell number")
    plt.xlabel("corrdior position")
    ticks = np.array([50, 100, 150])
    plt.xticks(ticks - 10, ticks)

    plt.title("Unsupervised Session")

    plt.tight_layout()

    plt.savefig(
        HERE.parent
        / "plots"
        / "place_cells"
        / f"place-cells-unsupervised-{session.mouse_name}-{session.date}"
    )


def place_cells(
    session: Cached2pSession, dff: np.ndarray, spks: np.ndarray, denoised: np.ndarray
) -> None:

    # TODO: Don't reuse this function obviously
    spks = binarise_spikes(spks)
    grosmark_place_field(session, spks)
    return

    # _, ax1 = plt.subplots()
    # # ax1.plot(spks[0, :], color="black")
    # # ax1.plot(x[0, :], "o", color="blue", alpha=0.9)
    # # ax1.plot(spks[0, :], ".", color="black", alpha=1)

    # ax2 = ax1.twinx()
    # ax1.plot(dff[0, :], color="red", alpha=0.5)

    start = 30
    max_position = 180
    bin_size = 5
    ITI_bin_size = 1

    vmin = 0
    vmax = 1

    imaging_data = spks

    plt.figure()
    plt.title("Rewarded trials")
    plt.imshow(
        get_position_activity(
            [
                trial
                for trial in session.trials
                if trial_is_imaged(trial) and trial.texture_rewarded
            ],
            imaging_data,
            get_wheel_circumference_from_rig("2P"),
            start=start,
            bin_size=bin_size,
            max_position=max_position,
            remove_landmarks=False,
            ITI_bin_size=ITI_bin_size,
        ),
        aspect="auto",
        cmap="viridis",
        vmax=vmax,
        vmin=vmin,
    )

    clb = plt.colorbar()
    plt.ylabel("cell number")
    plt.xlabel("corrdior position")
    ticks = np.array([50, 100, 150])

    plt.xticks((ticks - start) / bin_size, ticks)
    plt.tight_layout()
    plt.savefig(
        HERE.parent
        / "plots"
        / "place_cells"
        / f"place-cells-rewarded-{session.mouse_name}-{session.date}"
    )

    plt.figure()
    plt.title("Unrewarded trials")
    plt.imshow(
        get_position_activity(
            [
                trial
                for trial in session.trials
                if trial_is_imaged(trial) and not trial.texture_rewarded
            ],
            imaging_data,
            get_wheel_circumference_from_rig("2P"),
            start=start,
            bin_size=bin_size,
            max_position=max_position,
            remove_landmarks=False,
            ITI_bin_size=ITI_bin_size,
        ),
        aspect="auto",
        cmap="viridis",
        vmax=vmax,
        vmin=vmin,
    )

    clb = plt.colorbar()
    clb.ax.set_title("Normalised\nactivity", fontsize=12)
    plt.ylabel("cell number")
    plt.xlabel("corrdior position")
    plt.xticks((ticks - start) / bin_size, ticks)
    plt.tight_layout()
    plt.savefig(
        HERE.parent
        / "plots"
        / "place_cells"
        / f"place-cells-unrewarded-{session.mouse_name}-{session.date}"
    )
    plt.show()


def binarise_spikes(spks: np.ndarray) -> np.ndarray:
    """Implements the calcium imaging preprocessing stepts here:
    https://www.nature.com/articles/s41593-021-00920-7#Sec12

    Though the first steps done in our oasis fork.

    Currently we are not doing wavelet denoising as I've found this makes the fit much worse.
    We have added zhang baseline step. As without this, if our baseline drifts, higher baseline
    periods are considered to have more spikes.

    We are also not normalising by the residual between denoised and actual. It's not clear
    how they do this. What factor are they reducing the residual by? The residual is some
    massive number.

    They threshold based on the MAD. But is it just the MAD or is the MAD deviation from the median?
    I also had to take the MAD of only non-zero periods. As the raw MAD of all cells is 0. This may
    not be true in the hippocampus which is why they may not do this. We're also not currently
    altering the threshold depending or running or not. TOOD: DO THIS


    """

    non_zero_spikes = np.copy(spks)
    non_zero_spikes[non_zero_spikes == 0] = np.nan

    mad = median_abs_deviation(non_zero_spikes, axis=1, nan_policy="omit")

    # Maybe
    # threshold = mad * 1.5

    # Or maybe
    threshold = np.nanmedian(non_zero_spikes, axis=1) + mad * 1.5
    mask = spks - threshold[:, np.newaxis] > 0
    spks[~mask] = 0
    spks[mask] = 1
    return remove_consecutive_ones(spks)


if __name__ == "__main__":

    mouse = "JB027"
    date = "2025-02-26"
    # date = "2024-12-10"

    with open(HERE.parent / "data" / "cached_2p" / f"{mouse}_{date}.json", "r") as f:
        session = Cached2pSession.model_validate_json(f.read())

    print(f"Total number of trials: {len(session.trials)}")
    print(
        f"number of trials imaged {len([trial for trial in session.trials if trial_is_imaged(trial)])}"
    )

    dff, spks, denoised = get_dff(mouse, date)

    print("Got dff")

    assert (
        max(
            trial.states_info[-1].closest_frame_start
            for trial in session.trials
            if trial.states_info[-1].closest_frame_start is not None
        )
        < dff.shape[1]
    ), "Tiff is too short"

    # do_classify(session, spks)

    place_cells(session, dff, spks, denoised)
    # if "unsupervised" in session.session_type.lower():
    #     place_cells_unsupervised(session, dff)
    # else:
    #     place_cells(session, dff, spks, denoised)
