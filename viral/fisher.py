# Allow you to run the file directly, remove if exporting as a proper module
from pathlib import Path
import sys


HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.constants import CACHE_PATH, SERVER_PATH


import numpy as np
import matplotlib.pyplot as plt

from viral.imaging_utils import (
    activity_trial_position,
    load_imaging_data,
    trial_is_imaged,
)

from scipy.stats import zscore
from viral.models import Cached2pSession
from viral.utils import get_wheel_circumference_from_rig
from viral.constants import grosmark_config


def process_session(session: Cached2pSession, spks: np.ndarray) -> None:
    bin_size = 5
    start = 0
    max_position = 200

    trial_list = []
    rewarded = []

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
            bin_occupancy_divide=True,
        )

        trial_list.append(data)
        if trial.texture_rewarded:
            rewarded.append(1)
        else:
            rewarded.append(0)

    X = np.array(trial_list)
    pcs_combined = np.load(
        CACHE_PATH.parent
        / "place_cells"
        / "pcs_combined"
        / f"{session.mouse_name}_{session.date}_rewarded_{None}_{grosmark_config}_pcs_combined_BOD_True.npy"
    )

    # Trials x cells x bins
    X = X[:, pcs_combined, :]

    landmark_locations = [45, 90, 135]
    n_bins = int(max_position / bin_size)
    bin_to_cm_scaling_factor = (max_position - start) / n_bins

    plt.clf()

    landmark_triggered_average = []
    for landmark_center in landmark_locations:
        landmark_bin_center = int(landmark_center / bin_to_cm_scaling_factor)
        landmark_triggered_average.append(
            zscore(
                np.nanmean(
                    X[:, :, landmark_bin_center - 5 : landmark_bin_center + 5], 0
                ),
                axis=0,
            )
        )

    landmark_triggered_average = np.nanmean(np.array(landmark_triggered_average), 0)

    for cell in range(landmark_triggered_average.shape[0]):
        if (
            landmark_triggered_average[cell][:5].mean()
            > landmark_triggered_average[cell][5:].mean()
        ):
            plt.plot(landmark_triggered_average[cell], alpha=0.5, color="grey")

    plt.axvline(8, color="red", linestyle="--")

    1 / 0


def main() -> None:

    mouse = "JB030"
    date = "2025-03-13"

    with open(
        SERVER_PATH / "viral_caches" / "cached_2p" / f"{mouse}_{date}.json", "r"
    ) as f:
        session = Cached2pSession.model_validate_json(f.read())

    print(f"Total number of trials: {len(session.trials)}")
    print(
        f"number of trials imaged {len([trial for trial in session.trials if trial_is_imaged(trial)])}"
    )

    dff, spks, denoised = load_imaging_data(mouse, date)

    process_session(session, dff)


if __name__ == "__main__":
    main()
