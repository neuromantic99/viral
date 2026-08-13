from pathlib import Path
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind
from sklearn.decomposition import PCA
from tqdm import tqdm
import seaborn as sns

# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.imaging_utils import activity_trial_position, trial_is_imaged
from viral.constants import SERVER_PATH, TIFF_UMBRELLA, CACHE_PATH, grosmark_config
from viral.models import Cached2pSession
from viral.sessions_keep import SESSIONS_KEEP
from viral.utils import get_genotype, get_wheel_circumference_from_rig


def get_variance_explained(
    mouse_name: str, date: str, rewarded: bool | None = None
) -> float:

    save_path = Path(
        SERVER_PATH
        / "viral_caches"
        / "response_profiles"
        / f"{mouse_name}_{date}_rewarded_{rewarded}.npy",
    )
    signal = np.load(save_path)
    mean = gaussian_filter1d(np.nanmean(signal, axis=0), sigma=1)
    print("Mean shape", mean.shape)
    if np.any(np.isnan(mean)):
        warnings.warn(f"NaNs found in mean response for {mouse_name} {date}")
        print("Mean shape", mean.shape)

    pca = PCA(n_components=50)
    # X of shape (n_samples, n_features)
    # mean is currently (n_cells, n_position_bins)
    # so transpose it
    pca.fit(mean.T)
    ratio = pca.explained_variance_ratio_
    transformed = pca.transform(mean.T).T
    cum = np.cumsum(ratio)

    # Components taken to explain 80% variance
    n_components_80 = np.where(cum >= 0.8)[0][0] + 1

    return n_components_80


def load_signal(
    spks: np.ndarray,
    session: Cached2pSession,
    rewarded: bool | None,
) -> np.ndarray:

    signal = []
    for trial in session.trials:
        if not trial_is_imaged(trial):
            continue
        if rewarded is not None and trial.texture_rewarded != rewarded:
            continue

        data = activity_trial_position(
            trial=trial,
            flu=spks,
            wheel_circumference=get_wheel_circumference_from_rig("2P"),
            bin_size=grosmark_config.bin_size,
            start=grosmark_config.start,
            max_position=grosmark_config.end,
            verbose=False,
            do_shuffle=False,
            threshold_speed=False,
        )
        signal.append(data)
    return np.array(signal)


def batch_process() -> None:

    for mouse_name, dates in tqdm(SESSIONS_KEEP.items()):
        for stage, date in dates.items():
            if date is None:
                continue
            if (
                SERVER_PATH
                / "viral_caches"
                / "response_profiles"
                / f"{mouse_name}_{date}_rewarded_None.npy"
            ).exists():
                continue

            with open(CACHE_PATH / f"{mouse_name}_{date}.json", "r") as f:
                session = Cached2pSession.model_validate_json(f.read())
            spks_path = (
                TIFF_UMBRELLA / session.date / session.mouse_name / "suite2p" / "plane0"
            )

            assert (
                spks_path / "full_grosmark_oasis_preprocessed.npy"
            ).exists(), f"File {spks_path / 'full_grosmark_oasis_preprocessed.npy'} does not exist"
            spks = np.load(spks_path / "oasis_spikes.npy")
            is_cell = np.load(spks_path / "iscell.npy")[:, 0].astype(bool)
            spks = spks[is_cell, :]

            for rewarded in [True, False, None]:
                save_path = Path(
                    SERVER_PATH
                    / "viral_caches"
                    / "response_profiles"
                    / f"{mouse_name}_{date}_rewarded_{rewarded}.npy",
                )
                if save_path.exists():
                    continue

                signal = load_signal(session=session, spks=spks, rewarded=rewarded)
                np.save(save_path, signal)


def plot_variance_explained() -> None:

    results = {"genotype": [], "stage": [], "variance_explained": []}

    for mouse_name, dates in SESSIONS_KEEP.items():
        if get_genotype(mouse_name) not in ["WT", "NLGF"]:
            continue
        genotype = get_genotype(mouse_name)
        for stage, date in dates.items():
            if date is None:
                continue

            res = get_variance_explained(mouse_name, date, rewarded=False)
            results["genotype"].append(genotype)
            results["stage"].append(
                "Trained" if stage in ["learning", "learned"] else "Baseline"
            )
            results["variance_explained"].append(res)

    df = pd.DataFrame(results)

    sns.boxplot(
        data=df,
        x="stage",
        y="variance_explained",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        showfliers=False,
    )

    sns.stripplot(
        data=df,
        x="stage",
        y="variance_explained",
        hue="genotype",
        hue_order=["WT", "NLGF"],
        dodge=True,
        linewidth=1,
        edgecolor="black",
    )
    ttest_ind(
        df[(df["genotype"] == "WT") & (df["stage"] == "Trained")]["variance_explained"],
        df[(df["genotype"] == "NLGF") & (df["stage"] == "Trained")][
            "variance_explained"
        ],
    )
    1 / 0


if __name__ == "__main__":
    plot_variance_explained()
