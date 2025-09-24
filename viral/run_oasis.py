import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import sys
from pathlib import Path
from typing import Tuple

import pywt

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


def grosmark_preprocess(
    s2p_path: Path, plot: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    f = get_f(s2p_path)

    all_spikes = []
    all_spikes_norm = []
    all_denoised = []
    for idx, cell in enumerate(f):
        wavelet_denoised = modwt_denoise(cell, wavelet="sym4", level=5)
        wavelet_denoised = wavelet_denoised - np.median(wavelet_denoised)
        denoised, spikes, b, g, lam = deconvolve(
            wavelet_denoised, penalty=1, b_nonneg=False
        )
        if plot and idx < 30:
            _, ax1 = plt.subplots()
            baseobj = BaselineRemoval(cell)
            cell_baselined = baseobj.ZhangFit()

            residual = np.sum(np.abs(cell - cell_baselined))

            print(f"residual is {residual}")

            # ax1.plot(cell, color="pink")
            ax1.plot(cell_baselined, color="blue")
            ax1.plot(spikes, color="black")
            # ax2 = ax1.twinx()
            # ax2.plot(wavelet_denoised, color="pink")

        """The deconvolution noise was taken as the m.a.d. of the residual of the observed trace T and the reconstructed trace Test.
        Spike estimates, Csp, were normalized by the deconvolution noise [...]"""
        spikes_norm = spikes / median_abs_deviation(residual)
        all_spikes.append(spikes)
        all_spikes_norm.append(spikes_norm)
        all_denoised.append(oasis_denoised)

    if plot:
        plt.show()

    return np.array(all_spikes), np.array(all_denoised)


def main(mouse: str, date: str, grosmark: bool = False) -> None:
    print(f"Running OASIS deconvolution on session data: {mouse} - {date}")

    plot = False

    s2p_path = TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0"

    if grosmark:
        print("Using grosmark preprocessing")
        spikes, denoised = grosmark_preprocess(s2p_path, plot)

    else:
        dff = compute_dff(get_f(s2p_path))

        all_denoised = list()
        all_spikes = list()

        for idx, cell in enumerate(dff):
            baseobj = BaselineRemoval(cell)
            cell_baselined = baseobj.ZhangFit()
            denoised, spikes, b, g, lam = deconvolve(
                cell_baselined, penalty=1, b_nonneg=False
            )

            if plot and idx < 30:

                _, ax1 = plt.subplots(figsize=(20, 10))

                ax1.plot(cell_baselined, color="pink", alpha=0.5)
                ax1.plot(cell, "--", color="blue")
                ax2 = ax1.twinx()
                ax2.plot(spikes, color="black", alpha=0.7)
                # ax2.plot(wavelet_denoised, color="pink")

            all_denoised.append(denoised)
            all_spikes.append(spikes)

        denoised = np.array(all_denoised)
        spikes = np.array(all_spikes)

    np.save(s2p_path / "oasis_spikes.npy", spikes)
    np.save(s2p_path / "oasis_denoised.npy", denoised)
    print("Saved oasis spikes and denoised data")
    if plot:
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
    main(mouse="JB031", date="2025-03-07", grosmark=False)
    # main(mouse="JB027", date="2025-02-26", grosmark=True)
