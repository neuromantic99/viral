import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Tuple

import pywt
from tqdm import tqdm
from scipy.stats import median_abs_deviation

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))
sys.path.append(str(HERE.parent / "OASIS"))

from OASIS.oasis.functions import deconvolve

from BaselineRemoval import BaselineRemoval

from viral.constants import TIFF_UMBRELLA
from viral.imaging_utils import subtract_neuropil, compute_dff


def get_f(s2p_path: Path) -> np.ndarray:
    """
    Returns fluorescence with neuropil signal subtracted.
    """
    # not using iscell as it will be used in later analysis steps
    f_raw = np.load(s2p_path / "F.npy")
    f_neu = np.load(s2p_path / "Fneu.npy")
    print("Loaded fluorescence data and subtracted neuropil")
    return subtract_neuropil(f_raw, f_neu)


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


def grosmark_preprocess(
    s2p_path: Path, plot: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """To reduce calcium white 'scatter' noise and to improve signal detection, a wavelet-based denoising algorithm (Matlab 2019a, wden function,
    using the parameters modwtsqtwolog, s, mln, 5 and sym4) was applied to the calcium activity traces resulting in the denoised activity trace vector Tdn.
    This algorithm was adopted as, when compared to other algorithms such as Savitsky-Golay filtering,
    it resulted in robust white noise reduction while minimally altering the underlying waveforms of observable calcium transient events.
    Subsequently, the per-trace median was subtracted from each of the smoothed traces and they were deconvolved using a
    first-order autoregressive model as implemented in the OASIS software package.
    This algorithm outputs the deconvolved (spike estimate) of the trace Csp, and a denoised trace reconstruction Test based on the re-convolution of the spike estimates.
    The deconvolution noise was taken as the m.a.d. of the residual of the observed trace T and the reconstructed trace Test.
    Spike estimates, Csp, were normalized by the deconvolution noise [...]"""
    f = get_f(s2p_path)

    all_spikes = []
    all_spikes_norm = []
    all_denoised = []

    for idx, cell in enumerate(tqdm(f)):
        wavelet_denoised = modwt_denoise(cell, wavelet="sym4", level=5)
        wavelet_denoised = wavelet_denoised - np.median(wavelet_denoised)
        oasis_denoised, spikes, b, g, lam = deconvolve(
            wavelet_denoised, penalty=1, b_nonneg=False
        )

        residual = np.abs(cell - oasis_denoised)

        # if plot and idx < 30:
        #     baseobj = BaselineRemoval(cell)
        #     cell_baselined = baseobj.ZhangFit()
        #     print(f"residual is {residual}")

        #     _, ax1 = plt.subplots()
        #     # ax1.plot(cell, color="pink")
        #     ax1.plot(cell_baselined, color="blue")
        #     ax1.plot(spikes, color="black")
        #     # ax2 = ax1.twinx()
        #     # ax2.plot(wavelet_denoised, color="pink")

        """The deconvolution noise was taken as the m.a.d. of the residual of the observed trace T and the reconstructed trace Test.
        Spike estimates, Csp, were normalized by the deconvolution noise [...]"""
        spikes_norm = spikes / median_abs_deviation(residual)
        all_spikes.append(spikes)
        all_spikes_norm.append(spikes_norm)
        all_denoised.append(oasis_denoised)

    if plot:
        plt.show()

    return np.array(all_spikes), np.array(all_spikes_norm), np.array(all_denoised)


def main(mouse: str, date: str, grosmark: bool = False) -> None:
    print(f"Running OASIS deconvolution on session data: {mouse} - {date}")

    plot = False

    s2p_path = TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0"

    if grosmark:
        print("Using grosmark preprocessing")
        spikes, spikes_norm, denoised = grosmark_preprocess(s2p_path, plot)

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
    if grosmark:
        np.save(s2p_path / "oasis_spikes_norm.npy", spikes_norm)
    print("Saved oasis spikes and denoised data")
    if plot:
        plt.show()


if __name__ == "__main__":
    main(mouse="JB031", date="2025-03-25", grosmark=True)
    # main(mouse="JB027", date="2025-02-26", grosmark=True)
