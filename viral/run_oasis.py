import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import sys
from pathlib import Path
from typing import Tuple

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

import pywt
from tqdm import tqdm
import concurrent.futures

from viral.constants import CACHE_PATH, TIFF_UMBRELLA


from viral.models import Cached2pSession, WheelFreeze
from viral.utils import remove_consecutive_ones


from oasis.functions import (
    deconvolve,
)

from scipy.stats import median_abs_deviation
from scipy.ndimage import percentile_filter

""" 
This now does every step in the Calcium activity detection section in Grosmark.
Though i've found the deconvolution to be better without the wavelet denoising.
"""


def modwt_denoise(
    signal: np.ndarray, wavelet: str = "sym4", level: int = 5
) -> np.ndarray:
    """
    Perform MODWT-based wavelet denoising similar to MATLAB's wden with 'modwtsqtwolog'.

    Parameters:
        signal (array): Input noisy signal.
        wavelet (str): Type of wavelet (default: 'sym4').
        level (int): Decomposition level (default: 5).

    Returns:
        array: Denoised signal.

    N.B Have checked with real data that this matches matlab and it does.

    This is not currently used as I've found the deconvolution to be better without it.
    """
    # Perform MODWT (Maximal Overlap Discrete Wavelet Transform)
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode="periodization")

    # Estimate noise standard deviation using Median Absolute Deviation (MAD)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Robust estimate of noise

    # Compute the universal threshold (SqTwolog: sqrt(2 * log(N)) * sigma)
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))

    # Apply soft thresholding to detail coefficients (exclude approximation coefficients)
    coeffs_thresh = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]
    ]

    # Reconstruct the signal using inverse MODWT
    denoised_signal = pywt.waverec(coeffs_thresh, wavelet, mode="periodization")

    return denoised_signal[: len(signal)]  # Ensure same length as input


def subtract_neuropil(f_raw: np.ndarray, f_neu: np.ndarray) -> np.ndarray:
    return f_raw - f_neu * 0.7


def compute_dff_percentile_filter(
    f: np.ndarray, percentile: float, window_size_seconds: int
) -> Tuple[np.ndarray, np.ndarray]:
    window_size = int(window_size_seconds * 30)
    baseline = percentile_filter(f, percentile, size=window_size)
    # return (f - baseline) / baseline, baseline
    return f - baseline, baseline


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """For plotting only"""
    return np.convolve(arr, np.ones(window), "same") / window


def process_cell(
    cell: np.ndarray,
    wheel_freeze: WheelFreeze,
    plot: bool = False,
    figure_path: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.array([])
    baselined = np.array([])
    baseline = np.array([])
    spikes = np.array([])
    denoised = np.array([])

    for chunk_name, chunk in zip(
        ["pre", "online", "post"],
        [
            cell[
                wheel_freeze.pre_training_start_frame : wheel_freeze.pre_training_end_frame
            ],
            cell[
                wheel_freeze.pre_training_end_frame : wheel_freeze.post_training_start_frame
            ],
            cell[
                wheel_freeze.post_training_start_frame : wheel_freeze.post_training_end_frame
            ],
        ],
    ):
        raw = np.append(raw, chunk)
        chunk_baselined, chunk_baseline = compute_dff_percentile_filter(
            chunk, percentile=5, window_size_seconds=90
        )
        # Grosmark does this but I don't think it changes anything
        chunk_baselined = chunk_baselined - np.median(chunk_baselined)

        # Grosmark does this, but I've found it makes the spike inference worse
        # wavelet_denoised = wavelet_denoised - np.median(wavelet_denoised)

        chunk_denoised, chunk_spikes, b, g, lam = deconvolve(
            chunk_baselined,
            penalty=1,
            b_nonneg=False,
        )

        # Normalize spike estimates by MAD of the residual (abs(T - Test))
        residual = np.abs(chunk_baselined - (chunk_denoised + b))
        mad_residual = median_abs_deviation(residual)

        chunk_spikes_norm = chunk_spikes / mad_residual

        threshold = 1.5 if chunk_name == "online" else 1.25

        chunk_spikes_norm[chunk_spikes_norm < threshold] = 0
        chunk_spikes_norm[chunk_spikes_norm >= threshold] = 1

        spikes = np.append(spikes, chunk_spikes_norm)
        baselined = np.append(baselined, chunk_baselined)
        baseline = np.append(baseline, chunk_baseline)
        denoised = np.append(denoised, chunk_denoised + b)

    assert (
        baselined.shape
        == spikes.shape
        == baseline.shape
        == denoised.shape
        == cell.shape
    )
    if plot:
        fig = plot_result(raw, baselined, baseline, denoised, spikes)
        plt.title(
            f"baseline {round(b, 2)} Firing rate {round(np.sum(spikes) / (len(spikes) / 30), 2)}, mad residual {round(mad_residual, 2)}"
        )
        assert figure_path is not None, "Need to provide a figure path if plot = True"
        with open(
            figure_path,
            "wb",
        ) as f:
            pickle.dump(fig, f)

    return spikes, denoised


def correct_f(f: np.ndarray, s2p_path: Path) -> np.ndarray:

    if "JB036" in str(s2p_path) and "2025-07-05" in str(s2p_path):
        # Two small grabs with the PMT off
        # Hack: replace them with a duplicate of a neighbouring chunk
        bad_frames = (27000, 27060)
        n_bad_frames = bad_frames[1] - bad_frames[0]
        f[:, bad_frames[0] : bad_frames[1]] = f[
            :, bad_frames[0] - n_bad_frames : bad_frames[0]
        ]

        bad_frames_end = (137940, 138020)
        n_bad_frames_end = bad_frames_end[1] - bad_frames_end[0]
        f[:, bad_frames_end[0] : bad_frames_end[1]] = f[
            :, bad_frames_end[1] : bad_frames_end[1] + n_bad_frames_end
        ]

    return f


def _process_cell_no_plot_with_index(
    args: tuple[int, np.ndarray, WheelFreeze],
) -> tuple[int, np.ndarray, np.ndarray]:
    """Driver for parallel processing, ensure cells are returned in the correct order"""
    idx, cell, wheel_freeze = args
    from pathlib import Path  # Needed for process_cell signature in subprocesses
    import numpy as np

    spikes, denoised = process_cell(cell, wheel_freeze, plot=False, figure_path=None)
    return idx, spikes, denoised


def preprocess_and_run(
    s2p_path: Path,
    wheel_freeze: WheelFreeze,
    plot: bool = False,
    parallel: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Set parallel to True to run across all cores available on your system (no plotting)"""

    f_raw = np.load(s2p_path / "F.npy")
    f_neu = np.load(s2p_path / "Fneu.npy")
    f = subtract_neuropil(f_raw, f_neu)
    f = correct_f(f, s2p_path)

    t1 = time.time()

    if parallel:
        # Parallel processing, ignore plotting
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    _process_cell_no_plot_with_index,
                    ((idx, cell, wheel_freeze) for idx, cell in enumerate(f)),
                )
            )
        # Sort results by index to preserve order
        results.sort(key=lambda x: x[0])
        all_spikes = np.stack([r[1] for r in results])
        all_denoised = np.stack([r[2] for r in results])
        assert all_spikes.shape == all_denoised.shape == f.shape
        print(f"Time taken: {time.time() - t1} seconds")
        return all_spikes, all_denoised

    all_spikes = []
    all_denoised = []
    for idx, cell in tqdm(enumerate(f)):
        mouse, date = get_mouse_and_date_from_path(s2p_path)
        (
            spikes,
            denoised,
        ) = process_cell(
            cell,
            wheel_freeze,
            plot=plot if idx < 10 else False,
            figure_path=HERE.parent
            / "data"
            / "oasis_examples"
            / f"{mouse}_{date}_{idx}.pkl",
        )

        all_spikes.append(spikes)
        all_denoised.append(denoised)

        if idx == 10 and plot:
            plt.show()
    all_spikes = np.array(all_spikes)
    all_denoised = np.array(all_denoised)

    assert all_spikes.shape == all_denoised.shape == f.shape
    print(f"Time taken: {time.time() - t1} seconds")

    return all_spikes, all_denoised


def get_mouse_and_date_from_path(s2p_path: Path) -> Tuple[str, str]:
    return s2p_path.parts[-3], s2p_path.parts[-4]


def plot_from_cache(s2p_path: Path) -> None:

    mouse, date = get_mouse_and_date_from_path(s2p_path)
    files = (HERE.parent / "data" / "oasis_examples").glob("*.pkl")
    files = sorted(list(files))

    for idx, file in enumerate(files):
        if mouse in str(file) and date in str(file):
            with open(file, "rb") as f:
                fig = pickle.load(f)
                plt.figure(fig.number)

    plt.show()


def plot_result(
    raw: np.ndarray,
    cell_baselined: np.ndarray,
    baseline: np.ndarray,
    oasis_denoised: np.ndarray,
    spikes: np.ndarray,
) -> Figure:

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()
    (p1,) = ax.plot(
        spikes,
        color="red",
        label=f"spikes",
        alpha=1,
    )
    # ax.hlines(1.25, color="red", linestyle="--", xmin=0, xmax=27000)
    # ax.hlines(1.5, color="red", linestyle="--", xmin=27000, xmax=len(cell) - 27000)
    # ax.hlines(1.25, color="red", linestyle="--", xmin=len(cell) - 27000, xmax=len(cell))

    (p2,) = ax2.plot(cell_baselined, color="blue", label="baselined")
    (p5,) = ax2.plot(oasis_denoised, color="green", label="denoised")
    # (p6,) = ax.plot(wavelet_denoised, color="orange", label="wavelet denoised")
    (p3,) = ax2.plot(raw, color="black", label="raw", alpha=0.01)
    (p4,) = ax2.plot(baseline, color="pink", label="baseline", alpha=0.01)
    lines = [p1, p2, p3, p4, p5]
    labels = [line.get_label() for line in lines]
    ax.set_ylabel("spikes")
    ax2.set_ylabel("flu")
    ax.legend(lines, labels, loc="upper right")
    return fig


def main(
    s2p_path: Path, wheel_freeze: WheelFreeze, parallel: bool = True, plot: bool = False
) -> None:

    all_spikes, all_denoised = preprocess_and_run(
        s2p_path,
        wheel_freeze=wheel_freeze,
        plot=plot,
        parallel=parallel,
    )

    all_spikes = remove_consecutive_ones(all_spikes)

    np.save(s2p_path / "oasis_spikes.npy", all_spikes)
    np.save(s2p_path / "oasis_denoised.npy", all_denoised)


if __name__ == "__main__":
    mouse_name = "JB036"
    date = "2025-07-05"
    cache_path = CACHE_PATH / f"{mouse_name}_{date}.json"
    s2p_path = TIFF_UMBRELLA / date / mouse_name / "suite2p" / "plane0"
    cached_session = Cached2pSession.model_validate_json(cache_path.read_text())
    assert cached_session.wheel_freeze is not None
    main(s2p_path, cached_session.wheel_freeze, parallel=True, plot=False)
