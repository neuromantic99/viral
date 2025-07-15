import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Tuple

import pywt
from tqdm import tqdm

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from oasis.functions import (
    deconvolve,
)

from oasis.plotting import simpleaxis
from oasis.oasis_methods import oasisAR1, oasisAR2
from BaselineRemoval import BaselineRemoval

from scipy.stats import median_abs_deviation

from scipy.ndimage import percentile_filter


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
    cell: np.ndarray, plot: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.array([])
    baselined = np.array([])
    baseline = np.array([])
    spikes = np.array([])
    denoised = np.array([])

    for chunk_name, chunk in zip(
        ["pre", "online", "post"], [cell[:27000], cell[27000:-27000], cell[-27000:]]
    ):

        # This is necessary as sometimes the baseline is very close to zero, thus inflating
        chunk = chunk - np.min(chunk)
        raw = np.append(raw, chunk)
        chunk_baselined, chunk_baseline = compute_dff_percentile_filter(
            chunk, percentile=10, window_size_seconds=90
        )
        # Grosmark does this, but I've found it makes the spike inference worse
        # wavelet_denoised = modwt_denoise(cell_baselined, wavelet="sym4", level=3)
        # wavelet_denoised = wavelet_denoised - np.median(wavelet_denoised)
        # chunk_baselined = chunk_baselined - np.median(chunk_baselined)

        # Lol
        chunk_baselined = chunk_baselined - np.min(chunk_baselined)

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
        plot_result(raw, baselined, baseline, denoised, spikes)

    return spikes, denoised, baselined, baseline


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


def preprocess_and_run(
    s2p_path: Path, plot: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    # Delete me
    cell = np.load("cell.npy")
    process_cell(cell, plot=True)

    f_raw = np.load(s2p_path / "F.npy")
    f_neu = np.load(s2p_path / "Fneu.npy")
    f = subtract_neuropil(f_raw, f_neu)
    f = correct_f(f, s2p_path)

    np.save("cell.npy", f[0, :])

    all_spikes = []
    all_denoised = []

    for cell in tqdm(f):
        spikes, denoised, baselined, baseline = process_cell(cell, plot=plot)
        all_spikes.append(spikes)
        all_denoised.append(denoised)

    all_spikes = np.array(all_spikes)
    all_denoised = np.array(all_denoised)
    assert all_spikes.shape == all_denoised.shape == f.shape

    return all_spikes, all_denoised


def plot_result(
    cell: np.ndarray,
    cell_baselined: np.ndarray,
    baseline: np.ndarray,
    oasis_denoised: np.ndarray,
    spikes: np.ndarray,
) -> None:

    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()
    (p1,) = ax.plot(
        spikes,
        color="red",
        label=f"spikes",
        alpha=0.5,
    )
    # ax.hlines(1.25, color="red", linestyle="--", xmin=0, xmax=27000)
    # ax.hlines(1.5, color="red", linestyle="--", xmin=27000, xmax=len(cell) - 27000)
    # ax.hlines(1.25, color="red", linestyle="--", xmin=len(cell) - 27000, xmax=len(cell))

    (p2,) = ax2.plot(moving_average(cell_baselined, 5), color="blue", label="baselined")
    (p5,) = ax2.plot(oasis_denoised, color="green", label="denoised")
    # (p6,) = ax.plot(wavelet_denoised, color="orange", label="wavelet denoised")
    (p3,) = ax2.plot(cell, color="black", label="raw", alpha=0.01)
    (p4,) = ax2.plot(baseline, color="pink", label="baseline", alpha=0.01)
    lines = [p1, p2, p3, p4, p5]
    labels = [line.get_label() for line in lines]
    ax.set_ylabel("spikes")
    ax2.set_ylabel("flu")
    ax.legend(lines, labels, loc="upper right")
    plt.show()


def main() -> None:

    s2p_path = Path("/Volumes/hard_drive/VR-2p/2025-07-05/JB036/suite2p/plane0")
    all_spikes, all_denoised = preprocess_and_run(s2p_path, plot=True)

    np.save(s2p_path / "oasis_spikes.npy", all_spikes)
    np.save(s2p_path / "oasis_denoised.npy", all_denoised)


if __name__ == "__main__":
    main()
