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

from viral.imaging_utils import trial_is_imaged
from viral.rastermap_utils import get_ITI_start_frame
from viral.utils import (
    array_bin_mean,
    degrees_to_cm,
    get_wheel_circumference_from_rig,
    has_five_consecutive_trues,
    normalize,
    remove_consecutive_ones,
    shuffle_rows,
    sort_matrix_peak,
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


def activity_trial_position(
    trial: TrialInfo,
    dff: np.ndarray,
    wheel_circumference: float,
    bin_size: int = 1,
    start: int = 10,
    max_position: int = 170,
    verbose: bool = False,
    do_shuffle: bool = False,
) -> np.ndarray:

    # TODO: remove non running epochs?

    position = degrees_to_cm(
        np.array(trial.rotary_encoder_position), wheel_circumference
    )

    frame_position = np.array(
        [
            state.closest_frame_start
            for state in trial.states_info
            if state.name
            in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
        ]
    )

    assert len(position) == len(frame_position)

    dff_position = []

    for bin_start in range(start, max_position, bin_size):
        frame_idx_bin = np.unique(
            frame_position[
                np.logical_and(position >= bin_start, position < bin_start + bin_size)
            ]
        )
        dff_bin = dff[:, frame_idx_bin]

        if verbose:
            print(f"bin_start: {bin_start}")
            print(f"bin_end: {bin_start + bin_size}")
            print(f"n_frames in bin: {len(frame_idx_bin)}")
        dff_position.append(np.mean(dff_bin, axis=1))

    if do_shuffle:
        return shuffle_rows(np.array(dff_position).T)

    return np.array(dff_position).T


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
    ITI_bin_size: int,
) -> np.ndarray:

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


def get_resting_chunks(
    trials: List[TrialInfo],
    dff: np.ndarray,
    chunk_size_frames: int,
    speed_threshold: float,
) -> np.ndarray:

    all_chunks = []
    for trial in trials:

        lick_frames = []

        for onset, offset in zip(
            [
                event.closest_frame
                for event in trial.events_info
                if event.name == "Port1In"
            ],
            [
                event.closest_frame
                for event in trial.events_info
                if event.name == "Port1Out"
            ],
            strict=True,
        ):
            assert onset is not None
            assert offset is not None
            lick_frames.extend(range(onset, offset + 1))

        position_frames = np.array(
            [
                state.closest_frame_start
                for state in trial.states_info
                if state.name
                in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
            ]
        )

        positions = degrees_to_cm(
            np.array(trial.rotary_encoder_position),
            get_wheel_circumference_from_rig("2P"),
        )

        n_chunks = (position_frames[-1] - position_frames[0]) // chunk_size_frames

        for chunk in range(n_chunks):
            start = chunk * chunk_size_frames
            end = start + chunk_size_frames

            frame_start = position_frames[0] + start
            frame_end = frame_start + chunk_size_frames

            distance_travelled = max(positions[start:end]) - min(positions[start:end])
            speed = distance_travelled / (chunk_size_frames / 30)

            if (
                speed < speed_threshold
                and len(
                    set(lick_frames).intersection(set(range(frame_start, frame_end)))
                )
                == 0
            ):
                all_chunks.append(
                    np.hstack(
                        (
                            array_bin_mean(
                                dff[:, frame_start:frame_end],
                                bin_size=1,
                            ),
                            np.ones((dff.shape[0], 1)) * 100,
                        )
                    )
                )

    return np.hstack(all_chunks)


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


def get_ITI_matrix(
    trials: List[TrialInfo],
    dff: np.ndarray,
    bin_size: int,
    average: bool,
) -> np.ndarray:
    """
    In theory will be 600 frames in an ITI (as is always 20 seconds)
    In practise it's 599 or 598 (or could be something like 597 or 600, fix
    the assertion if so).
    Or could be less if you stop the trial in the ITI.
    So we'll take the first 598 frames of any trial that has 599 or 598
    """

    matrices = []

    for trial in trials:
        assert trial.trial_end_closest_frame is not None

        # This would be good, but in practise it rarely occurs
        # if running_during_ITI(trial):
        #     continue

        chunk = dff[:, get_ITI_start_frame(trial) : int(trial.trial_end_closest_frame)]

        n_frames = chunk.shape[1]
        if n_frames < 550:
            # Imaging stopped in the middle
            continue
        elif n_frames in {598, 599}:
            matrices.append(array_bin_mean(chunk[:, :598], bin_size=bin_size))
        else:
            raise ValueError(f"Chunk with {n_frames} frames not understood")

    print(f"Number of still ITI trials: {len(matrices)}")
    # return normalize(np.mean(np.array(matrices), 0), axis=1)
    print(f"ITI shape {matrices[0].shape}")

    if average:
        return np.mean(
            np.array([normalize(matrix, axis=1) for matrix in matrices]), axis=0
        )

    # Hack, stack a column of 1s so you can distinguish trials. Remove if doing proper analysis
    for idx in range(len(matrices)):
        matrices[idx] = np.hstack([matrices[idx], np.ones((matrices[idx].shape[0], 1))])
    return np.hstack([matrix for matrix in matrices])


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


def grosmark_place_field(session: Cached2pSession, spks: np.ndarray) -> np.ndarray:
    """The position of the animal during online running epochs on the 2-m-long run belts was binned into 100,
      2-cm spatial bins. For each cell, the as within spatial-bin firing rate was calculated across all bins
    based on its sparsified spike estimate vector, Ssp. This firing rate by position vector was subsequently
    smoothed with a 7.5-cm Gaussian kernel leading to the smoothed firing rate by position vector.
    In addition, for each cell, 2,000 shuffled smoothed firing rate by position vectors were computed for each
    cell following the per-lap randomized circular permutation of estimated activity vector, Ssp. For each cell,
    putative PFs were defined as those in which the observed smoothed firing rate by position vectors exceeded the
    99th percentile of their shuffled smoothed firing rate by position vectors as assessed on a per spatial-bin basis for
    at least five consecutive spatial bins
    """

    bin_size = 2
    start = 0
    end = 180
    activity_matrix = np.mean(
        np.array(
            [
                activity_trial_position(
                    trial,
                    spks,
                    get_wheel_circumference_from_rig("2P"),
                    bin_size,
                    start,
                    end,
                )
                for trial in session.trials
                if trial.texture_rewarded and trial_is_imaged(trial)
            ]
        ),
        0,
    )
    sigma_cm = 7.5  # Desired smoothing in cm
    sigma_bins = sigma_cm / bin_size  # Convert to bin units

    # Apply Gaussian smoothing along each row (axis=1)
    smoothed_matrix = gaussian_filter1d(activity_matrix, sigma=sigma_bins, axis=1)

    shuffled_matrices = np.array(
        [
            gaussian_filter1d(
                np.mean(
                    np.array(
                        [
                            activity_trial_position(
                                trial,
                                spks,
                                get_wheel_circumference_from_rig("2P"),
                                bin_size,
                                start,
                                end,
                                do_shuffle=True,
                            )
                            for trial in session.trials
                            if trial.texture_rewarded and trial_is_imaged(trial)
                        ]
                    ),
                    0,
                ),
                sigma=sigma_bins,
                axis=1,
            )
            for _ in range(2000)
        ]
    )

    place_threshold = np.percentile(shuffled_matrices, 99, axis=0)

    shuffled_place_cells = np.array(
        [
            has_five_consecutive_trues(shuffled_matrices[idx, :, :] > place_threshold)
            for idx in range(shuffled_matrices.shape[0])
        ]
    )

    pcs = has_five_consecutive_trues(smoothed_matrix > place_threshold)

    print(
        f"percent place cells shuffed {np.mean(np.sum(shuffled_place_cells, axis=1) / shuffled_place_cells.shape[1])}"
    )

    print(f"percent place cells not shuffled {np.sum(pcs) / len(pcs)}")

    plt.figure()
    plt.imshow(
        zscore(sort_matrix_peak(smoothed_matrix[pcs, :]), axis=1),
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plt.title("real")

    plt.colorbar()
    plt.legend()

    plt.figure()

    plt.imshow(
        zscore(
            sort_matrix_peak(shuffled_matrices[0, shuffled_place_cells[0, :], :]),
            axis=1,
        ),
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    plt.title("shuffled")
    plt.colorbar()
    plt.legend()
    plt.show()

    return np.ndarray([])


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
    non_zero_spikes = np.copy(spks)
    non_zero_spikes[non_zero_spikes == 0] = np.nan
    mad = median_abs_deviation(non_zero_spikes, axis=1, nan_policy="omit")
    mask = spks - mad[:, np.newaxis] * 1.5 > 0
    spks[~mask] = 0
    spks[mask] = 1
    return remove_consecutive_ones(spks)


if __name__ == "__main__":

    mouse = "JB027"
    date = "2025-02-26"

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

    if "unsupervised" in session.session_type.lower():
        place_cells_unsupervised(session, dff)
    else:
        place_cells(session, dff, spks, denoised)
