from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd


HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from viral.constants import SERVER_PATH, TIFF_UMBRELLA, CACHE_PATH, grosmark_config
from viral.imaging_utils import (
    activity_trial_position,
    get_online_position_and_frames,
    get_resting_position_and_frames,
    trial_is_imaged,
)
from viral.models import Cached2pSession
from viral.representational_drift import interpolate_nans
from viral.sessions_keep import SESSIONS_KEEP
from viral.utils import (
    boxplot,
    get_genotype,
    get_wheel_circumference_from_rig,
    imshow,
    remove_diagonal,
    upper_triangle_no_diagonal,
)


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


def firing_rates_plot(genotype: str, rewarded: bool | None) -> None:

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
                    (result_mouse["resting_rates"], result_mouse["running_rates"])
                )

    collapsed_result: Dict[str, List] = {
        "stage": [],
        "state": [],
        "firing_rate": [],
        "mouse_id": [],
    }
    for stage in result.keys():
        for idx, mouse_data in enumerate(result[stage]):
            resting_rates, running_rates = mouse_data
            collapsed_result["stage"].extend(
                [stage] * (len(resting_rates) + len(running_rates))
            )
            collapsed_result["mouse_id"].extend(
                [idx] * (len(resting_rates) + len(running_rates))
            )
            collapsed_result["state"].extend(["resting"] * len(resting_rates))
            collapsed_result["firing_rate"].extend(resting_rates)

            collapsed_result["state"].extend(["running"] * len(running_rates))
            collapsed_result["firing_rate"].extend(running_rates)

    plt.figure()
    plt.title(f"{genotype} rewarded={rewarded}")
    sns.boxplot(
        pd.DataFrame(collapsed_result),
        x="stage",
        y="firing_rate",
        hue="state",
        showfliers=False,
    )

    plt.ylim(-0.1, 2.5)
    plt.tight_layout()
    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "firing_rates"
        / f"{genotype}_rewarded_{rewarded}_firing_rates"
    )


def save_firing_rates(genotype: str) -> None:

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
            session_path = CACHE_PATH / f"{mouse_name}_{date}.json"
            session = Cached2pSession.model_validate_json(session_path.read_text())

            for rewarded in [True, False, None]:

                save_path = (
                    SERVER_PATH
                    / "viral_caches"
                    / "firing_rates"
                    / f"{mouse_name}_{date}_rewarded_{rewarded}_firing_rates.npy"
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


def n_responders(session: Cached2pSession, rewarded: bool | None) -> float:
    pc_mask = np.load(
        SERVER_PATH
        / "viral_caches"
        / "place_cells"
        / "pcs_combined"
        / f"{session.mouse_name}_{session.date}_rewarded_{rewarded}_{grosmark_config}_pcs_combined.npy"
    )
    return np.sum(pc_mask) / len(pc_mask)


def n_responders_plot(genotype: str, rewarded: bool | None) -> None:
    result: Dict[str, List[float]] = {"unsupervised": [], "learning": [], "learned": []}
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

            n = n_responders(session=session, rewarded=rewarded)
            result[stage].append(n)

    plt.figure()
    plt.title(genotype)
    boxplot(result)
    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "fraction_place_cells"
        / f"{genotype}_rewarded_{rewarded}"
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


def plot_correlations(genotype: str, rewarded: bool | None) -> None:

    result = {"unsupervised": [], "learning": [], "learned": []}
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
            # result[stage].extend(correlations.tolist())
            result[stage].append(np.median(correlations))

    plt.figure()
    plt.title(genotype)
    colors = ["blue", "orange", "green"]
    boxplot(result)
    plt.ylim(-0.1, 0.2)
    # for idx, stage in enumerate(result.keys()):
    #     plt.hist(
    #         result[stage],
    #         bins=50,
    #         alpha=0.5,
    #         label=stage,
    #         density=True,
    #         color=colors[idx],
    #     )

    # plt.xlim(-1, 1)
    plt.legend()
    plt.savefig(
        SERVER_PATH
        / "viral_plots"
        / "correlations"
        / f"{genotype}_rewarded_{rewarded}_correlations"
    )


if __name__ == "__main__":
    for genotype in tqdm(
        ["Oligo-BACE1-KO", "NLGF", "WT", "Neuronal-BACE1-KO"], desc="corrleations"
    ):
        # for rewarded in [True, False, None]:
        plot_correlations(genotype, False)
    1 / 0
