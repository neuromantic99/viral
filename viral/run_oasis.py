import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import sys
from pathlib import Path
from typing import Tuple

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent / "OASIS"))
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
    wheel_freeze: WheelFreeze | None,
    plot: bool = False,
    figure_path: Path | None = None,
) -> Tuple[np.ndarray, np.ndarray, list[dict]]:
    """Deconvolve one cell, chunk by chunk.

    Also returns one diagnostics dict per chunk. The point of these is that every
    correction here is applied WITHIN a chunk - the percentile-filter baseline, and
    the normalisation of spike estimates by that chunk's own mad_residual. That fixes
    scale and slow drift, but it cannot fix signal-to-noise: if transients shrink
    relative to shot noise later in the session, a fixed 1.25 m.a.d. threshold admits
    a different mixture of real and noise-driven events, the binary Ssp becomes a
    noisier measurement of the same activity, and every pairwise correlation is
    attenuated by a common multiplicative factor. That is indistinguishable from a
    global reactivation difference in the pre/post comparison, so it needs measuring.

    The per-chunk normalisation also partly pins the event rate: the threshold is
    1.25 x that chunk's own noise, so the fraction of frames crossing it is somewhat
    fixed by construction, and a genuine global firing change between pre and post is
    partly normalised away before reactivation is ever computed.

    Returns spikes (binary, one per supra-threshold frame) AND amplitudes (the same
    frames, carrying the normalised spike estimate rather than a 1). Binarising makes
    the measure a count of EVENTS; a burst of ten spikes and a single spike both become
    one. Amplitude tracks spike count instead. Report both: agreement means the result
    is robust to burst structure, disagreement localises the difference to it.

    Note that amplitudes are NOT passed through remove_consecutive_ones, so their
    support differs from the saved oasis_spikes.npy. Sum amplitudes over an epoch to get
    total activity; do not compare the two arrays frame by frame.
    """
    raw = np.array([])
    baselined = np.array([])
    baseline = np.array([])
    spikes = np.array([])
    amplitudes = np.array([])
    denoised = np.array([])
    diagnostics: list[dict] = []

    chunk_names = ["pre", "online", "post"] if wheel_freeze is not None else ["online"]
    chunks = (
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
        ]
        if wheel_freeze is not None
        else [cell]
    )

    for chunk_name, chunk in zip(chunk_names, chunks, strict=True):
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

        # Stats on the NORMALISED spike estimate before it is binarised. How far the
        # real events sit above the threshold is the signal-to-noise measure that the
        # per-chunk mad normalisation cannot restore.
        supra = chunk_spikes_norm[chunk_spikes_norm >= threshold]
        diagnostics.append(
            {
                "chunk": chunk_name,
                "n_frames": int(chunk.size),
                "threshold": float(threshold),
                # noise scale: the attenuation driver
                "mad_residual": float(mad_residual),
                # raw brightness, for a direct look at bleaching
                "baseline_median": float(np.median(chunk_baseline)),
                "baselined_mad": float(median_abs_deviation(chunk_baselined)),
                # signal-to-noise: how far above the threshold events actually sit
                "spike_norm_p99": float(np.percentile(chunk_spikes_norm, 99)),
                "spike_norm_p999": float(np.percentile(chunk_spikes_norm, 99.9)),
                "supra_median": float(np.median(supra)) if supra.size else np.nan,
                "supra_max": float(np.max(supra)) if supra.size else np.nan,
                # detection rate, before remove_consecutive_ones sparsification
                "n_supra": int(supra.size),
                "event_rate_hz": float(supra.size / (chunk.size / 30)),
                # Amplitude tracks spike count where the binary count tracks event
                # count. On synthetic data holding the spike count fixed at 900 while
                # varying burst size from 1 to 10, the binarised count swung 6-fold
                # (1135 -> 188) while summed amplitude held within 15%. If the groups
                # differ in burst structure the binary count can invert the answer.
                "summed_amplitude": float(supra.sum()),
                # summed amplitude per event: a proxy for spikes per burst
                "amplitude_per_event": float(supra.mean()) if supra.size else np.nan,
            }
        )

        # Threshold but do NOT binarise, so the amplitude survives
        chunk_amplitudes = np.where(
            chunk_spikes_norm >= threshold, chunk_spikes_norm, 0
        )
        amplitudes = np.append(amplitudes, chunk_amplitudes)

        chunk_spikes_norm[chunk_spikes_norm < threshold] = 0
        chunk_spikes_norm[chunk_spikes_norm >= threshold] = 1

        spikes = np.append(spikes, chunk_spikes_norm)
        baselined = np.append(baselined, chunk_baselined)
        baseline = np.append(baseline, chunk_baseline)
        denoised = np.append(denoised, chunk_denoised + b)

    assert (
        baselined.shape
        == spikes.shape
        == amplitudes.shape
        == baseline.shape
        == denoised.shape
        == cell.shape
    ), "Shape of deconvolved does not match shape of original cell"

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

    return spikes, amplitudes, denoised, diagnostics


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
    args: tuple[int, np.ndarray, WheelFreeze | None],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Driver for parallel processing, ensure cells are returned in the correct order"""
    idx, cell, wheel_freeze = args
    from pathlib import Path  # Needed for process_cell signature in subprocesses
    import numpy as np

    spikes, amplitudes, denoised, diagnostics = process_cell(
        cell, wheel_freeze, plot=False, figure_path=None
    )
    return idx, spikes, amplitudes, denoised, diagnostics


def preprocess_and_run(
    s2p_path: Path,
    wheel_freeze: WheelFreeze | None,
    plot: bool = False,
    parallel: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Set parallel to True to run across all cores available on your system (no plotting)

    Returns spikes, amplitudes, denoised, and a tidy per-cell per-chunk diagnostics
    frame. See process_cell for why amplitudes are worth keeping.
    """

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
        all_amplitudes = np.stack([r[2] for r in results])
        all_denoised = np.stack([r[3] for r in results])
        diagnostics = pd.DataFrame(
            [dict(cell=r[0], **d) for r in results for d in r[4]]
        )
        assert all_spikes.shape == all_amplitudes.shape == all_denoised.shape == f.shape
        print(f"Time taken: {time.time() - t1} seconds")
        return all_spikes, all_amplitudes, all_denoised, diagnostics

    all_spikes = []
    all_amplitudes = []
    all_denoised = []
    all_diagnostics = []
    for idx, cell in tqdm(enumerate(f)):
        mouse, date = get_mouse_and_date_from_path(s2p_path)
        (
            spikes,
            amplitudes,
            denoised,
            diagnostics,
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
        all_amplitudes.append(amplitudes)
        all_denoised.append(denoised)
        all_diagnostics.extend(dict(cell=idx, **d) for d in diagnostics)

        if idx == 10 and plot:
            plt.show()
    all_spikes = np.array(all_spikes)
    all_amplitudes = np.array(all_amplitudes)
    all_denoised = np.array(all_denoised)

    assert all_spikes.shape == all_amplitudes.shape == all_denoised.shape == f.shape
    print(f"Time taken: {time.time() - t1} seconds")

    return all_spikes, all_amplitudes, all_denoised, pd.DataFrame(all_diagnostics)


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
    s2p_path: Path,
    wheel_freeze: WheelFreeze | None,
    parallel: bool = True,
    plot: bool = False,
) -> None:

    all_spikes, all_amplitudes, all_denoised, diagnostics = preprocess_and_run(
        s2p_path,
        wheel_freeze=wheel_freeze,
        plot=plot,
        parallel=parallel,
    )

    all_spikes = remove_consecutive_ones(all_spikes)

    np.save(s2p_path / "oasis_spikes.npy", all_spikes)
    # Thresholded but neither binarised nor sparsified, so summing over an epoch gives
    # total activity rather than an event count. See process_cell.
    np.save(s2p_path / "oasis_amplitudes.npy", all_amplitudes)
    np.save(s2p_path / "oasis_denoised.npy", all_denoised)
    diagnostics.to_csv(s2p_path / "oasis_diagnostics.csv", index=False)
    report_chunk_diagnostics(diagnostics, s2p_path)
    np.save(s2p_path / "full_grosmark_oasis_preprocessed.npy", np.array([True]))


def report_chunk_diagnostics(diagnostics: pd.DataFrame, s2p_path: Path) -> None:
    """Print the pre versus post comparison that decides whether the two freeze epochs
    are measured equally well.

    What to look for. If post has a larger mad_residual, or a lower baseline_median,
    or lower supra_median / spike_norm_p99, then the post epoch is a noisier
    measurement of the same activity. Pairwise correlations are then attenuated there
    by a common multiplicative factor, which looks exactly like a uniform pre/post
    difference in the offline correlation-versus-peak-distance curve, with no change
    in its shape.

    The rough attenuation this predicts is printed as a ratio. Compare it against the
    observed pre/post ratio of the correlation curves: if they match, the difference
    is measurement quality, not reactivation.
    """
    if diagnostics.empty or "pre" not in set(diagnostics["chunk"]):
        print("No pre/post chunks in this session, skipping diagnostics report")
        return

    columns = [
        "mad_residual",
        "baseline_median",
        "baselined_mad",
        "spike_norm_p99",
        "supra_median",
        "event_rate_hz",
        # If these two disagree the difference is in burst structure: the same number
        # of events carrying more spikes each. The binary event rate is blind to that.
        "summed_amplitude",
        "amplitude_per_event",
    ]

    print(f"\nOASIS per-chunk diagnostics: {s2p_path}")
    print(f"  {diagnostics['cell'].nunique()} cells\n")
    header = f"{'metric':>17}" + "".join(f"{c:>12}" for c in ["pre", "online", "post"])
    print(header + f"{'post/pre':>11}")
    for column in columns:
        medians = diagnostics.groupby("chunk")[column].median()
        row = f"{column:>17}"
        for chunk in ("pre", "online", "post"):
            row += f"{medians.get(chunk, float('nan')):>12.4g}"
        ratio = medians.get("post", np.nan) / medians.get("pre", np.nan)
        print(row + f"{ratio:>11.3f}")

    # Paired within cell, which is the comparison that matters
    wide = diagnostics.pivot(index="cell", columns="chunk", values="mad_residual")
    if {"pre", "post"}.issubset(wide.columns):
        worse = float((wide["post"] > wide["pre"]).mean())
        print(
            f"\n  cells with a noisier post epoch: {worse:.1%} "
            f"(50% = no systematic difference)"
        )
        # Correlation attenuation scales with the reliability of each signal, so a
        # first-order guess at the correlation ratio is the inverse noise ratio.
        print(
            f"  predicted attenuation of post correlations vs pre: "
            f"~{float((wide['pre'] / wide['post']).median()):.3f}x"
        )
        print(
            "  -> compare with the observed pre/post ratio of the offline "
            "correlation curves\n"
        )


if __name__ == "__main__":
    cache_files = list(CACHE_PATH.glob("*.json"))
    for cache_file in cache_files:
        print("Processing", cache_file)
        file_parts = cache_file.stem.split("_")
        date = file_parts[1]
        mouse_name = file_parts[0]
        s2p_path = TIFF_UMBRELLA / date / mouse_name / "suite2p" / "plane0"
        cached_session = Cached2pSession.model_validate_json(cache_file.read_text())

        if (s2p_path / "full_grosmark_oasis_preprocessed.npy").exists():
            print(f"Already processed {mouse_name} {date}, skipping")
            continue
        try:
            main(s2p_path, cached_session.wheel_freeze, parallel=False, plot=False)
        except Exception as e:
            print(f"Error processing {mouse_name} {date}: {e}")
            continue
