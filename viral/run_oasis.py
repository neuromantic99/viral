import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Tuple

import pywt

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))
sys.path.append(str(HERE.parent / "OASIS"))

from OASIS.oasis.functions import deconvolve
from OASIS.oasis.plotting import simpleaxis

# from OASIS.oasis.oasis_methods import oasisAR1, oasisAR2

from BaselineRemoval import BaselineRemoval

from viral.constants import TIFF_UMBRELLA


def modwt_denoise(
    signal: np.ndarray, wavelet: str = "sym4", level: int = 3
) -> np.ndarray:
    """
    Perform MODWT-based wavelet denoising similar to MATLAB's wden with 'modwtsqtwolog'.

    Parameters:
        signal (array): Input noisy signal.
        wavelet (str): Type of wavelet (default: 'sym4').
        level (int): Decomposition level (default: 5).

    Returns:
        array: Denoised signal.
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


def compute_dff(f: np.ndarray) -> np.ndarray:
    flu_mean = np.expand_dims(np.mean(f, 1), 1)
    return (f - flu_mean) / flu_mean


def subtract_neuropil(f_raw: np.ndarray, f_neu: np.ndarray) -> np.ndarray:
    return f_raw - f_neu * 0.7


def grosmark_preprocess(
    s2p_path: Path, plot: bool = False
) -> Tuple[np.ndarray, np.ndarray]:

    # Probably remove the is cell

    iscell = np.load(s2p_path / "iscell.npy")[:, 0].astype(bool)
    f_raw = np.load(s2p_path / "F.npy")[iscell, :]
    f_neu = np.load(s2p_path / "Fneu.npy")[iscell, :]
    f = subtract_neuropil(f_raw, f_neu)

    print("Loaded fluorescence data and substracted neuropil")

    all_spikes = []
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
            ax2 = ax1.twinx()
            # ax2.plot(wavelet_denoised, color="pink")

            # ax2.plot(spikes, color="pink")

        all_spikes.append(denoised)
        all_denoised.append(spikes)

    if plot:
        plt.show()

    return np.array(all_spikes), np.array(all_denoised)


def get_dff(s2p_path: Path) -> np.ndarray:
    f_raw = np.load(s2p_path / "F.npy")
    f_neu = np.load(s2p_path / "Fneu.npy")
    dff = compute_dff(subtract_neuropil(f_raw, f_neu))
    return dff


def main(mouse: str, date: str, grosmark: bool = False) -> None:
    print(f"Running OASIS deconvolution on session data: {mouse} - {date}")

    plot = False

    s2p_path = TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0"

    if grosmark:
        print("Using grosmark preprocessing")
        spikes, denoised = grosmark_preprocess(s2p_path, plot)

    else:
        dff = get_dff(s2p_path)

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
    plt.show()


if __name__ == "__main__":
    # main(mouse="JB027", date="2025-02-26", grosmark=False)
    main(mouse="JB027", date="2025-02-26", grosmark=True)
