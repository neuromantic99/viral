r"""
Bayesian reconstruction of virtual position from place-cell activity.
Same-day, online (running) epochs only. 200-cm linear track, 100 x 2-cm bins,
position resets to 0 at the end of each lap.

Model
-----
    Pr(pos | spikes) \propto \left( \prod_{i=1}^{n} f_i(pos)^{sp_i} \right)
                             \exp\!\left( -\tau \sum_{i=1}^{n} f_i(pos) \right)

    Pr(pos | spikes) \leftarrow \frac{Pr(pos|spikes)}{\sum_{j=1}^{P_n} Pr(pos_j|spikes)}

evaluated in the log domain as

    \log Pr(pos|spikes) = \sum_i sp_i \log f_i(pos) - \tau \sum_i f_i(pos) + C

Parameters
----------
imaging rate        60 Hz
tau                 20 frames (~333 ms)
track               200 cm, 100 x 2-cm bins, LINEAR (0 and 200 are distinct)
smoothing           7.5-cm Gaussian on the firing-rate-by-position vector
                    (spatial, applied after occupancy normalisation)
template            PCs only
running epochs      velocity smoothed with a 0.5-s Gaussian, > 5 cm/s,
                    sustained for >= 3 consecutive seconds
cross-validation    fivefold, split by lap number

Data format
-----------
X / spks : (n_cells, n_frames)  sparsified binary spike estimates, Ssp
y        : (n_frames,)          position in cm, 0-200, resets each lap
lap_id   : (n_frames,)          lap number per frame
"""

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

FRAME_RATE: float = 30  # Hz
FRAMES_PER_BIN: int = 10  # ~333 ms
TAU: float = FRAMES_PER_BIN / FRAME_RATE
BIN_SIZE_CM: float = 2.0
TRACK_LENGTH_CM: float = 200.0
N_POS_BINS: int = int(TRACK_LENGTH_CM / BIN_SIZE_CM)  # 100
SMOOTH_CM: float = 7.5  # Gaussian kernel
SIGMA_BINS: float = SMOOTH_CM / BIN_SIZE_CM  # 3.75 bins


VEL_SIGMA_S: float = 0.5  # velocity smoothing kernel
# I have reduced this slightly compared to the original (5 cm/s + 3s) because our mice often stop very briefly and when smoothing this ruins it
VEL_THRESHOLD: float = 3.0  # cm/s, online running
VEL_MIN_DURATION_S: float = 2.0  # sustained for at least this long


# ------------------------------------------------------------- binning ------
def digitize_position(
    y: np.ndarray,
    bin_size: float = BIN_SIZE_CM,
    track_length: float = TRACK_LENGTH_CM,
) -> np.ndarray:
    """Transforms raw position into 2-cm bin index. So e.g. a pos of 0-1.999 cm -> bin 0, 2-3.999 cm -> bin 1, ..., 198-199.999 cm -> bin 99."""
    n_pos_bins = int(track_length / bin_size)
    return np.clip((np.asarray(y) / bin_size).astype(int), 0, n_pos_bins - 1)


def bin_time(
    X: np.ndarray,
    y_bins: np.ndarray,
    lap_id: np.ndarray,
    frames_per_bin: int = FRAMES_PER_BIN,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sum spikes into tau-long bins *within each lap*, so no bin straddles the
    position reset. Leftover frames at the end of a lap are dropped.

    Returns counts (n_time_bins, n_cells), true_pos (n_time_bins,) taken at the
    centre frame, and bin_lap (n_time_bins,).
    """
    X, y_bins, lap_id = np.asarray(X), np.asarray(y_bins), np.asarray(lap_id)
    if mask is not None:
        X, y_bins, lap_id = X[:, mask], y_bins[mask], lap_id[mask]

    counts: list[np.ndarray] = []
    true_pos: list[np.ndarray] = []
    bin_lap: list[np.ndarray] = []
    for lap in np.unique(lap_id):
        idx = np.flatnonzero(lap_id == lap)
        n = (len(idx) // frames_per_bin) * frames_per_bin
        if n == 0:
            continue
        chunks = idx[:n].reshape(-1, frames_per_bin)
        counts.append(X[:, chunks].sum(axis=2).T)
        true_pos.append(y_bins[chunks[:, frames_per_bin // 2]])
        bin_lap.append(np.full(chunks.shape[0], lap))
    return np.concatenate(counts), np.concatenate(true_pos), np.concatenate(bin_lap)


# --------------------------------------------------------- running mask -----
def velocity(
    y: np.ndarray,
    lap_id: np.ndarray,
    frame_rate: float = FRAME_RATE,
    sigma_s: float = VEL_SIGMA_S,
) -> np.ndarray:
    """
    Velocity in cm/s, smoothed with a Gaussian kernel. Computed within each lap
    so the position reset is never read as movement.
    """
    y = np.asarray(y, dtype=float)
    lap_id = np.asarray(lap_id)
    v = np.zeros_like(y)
    for lap in np.unique(lap_id):
        i = np.flatnonzero(lap_id == lap)
        if i.size > 1:
            v[i] = gaussian_filter1d(
                np.gradient(y[i]) * frame_rate, sigma_s * frame_rate, mode="nearest"
            )
    return v


def _sustained(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Keep only runs of True lasting at least `min_frames`."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    out = np.zeros_like(mask)
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s >= min_frames:
            out[s:e] = True
    return out


def running_mask(
    y: np.ndarray,
    lap_id: np.ndarray,
    frame_rate: float = FRAME_RATE,
    threshold: float = VEL_THRESHOLD,
    min_duration_s: float = VEL_MIN_DURATION_S,
    sigma_s: float = VEL_SIGMA_S,
) -> np.ndarray:
    """
    Online running epochs: smoothed velocity (0.5s window) > 5 cm/s for >= 3 consecutive s.
    The duration criterion is applied across the whole session, so a bout that
    continues through a lap reset is not artificially split.
    """
    v = velocity(y, lap_id, frame_rate, sigma_s)
    mask = _sustained(v > threshold, int(round(min_duration_s * frame_rate)))
    # x = np.arange(len(mask)) / frame_rate
    # plt.axhline(y=threshold, color="k", linestyle="--", label="running threshold")
    # plt.plot(x, v, label="velocity (cm/s)")
    # plt.plot(x, mask * 10, label="running mask (5 cm/s for 3 s)")
    # plt.plot(x, y, label="position (cm)")
    # plt.legend()

    return mask


# ------------------------------------------------------------ template ------
def rate_map(
    X: np.ndarray,
    y_bins: np.ndarray,
    n_pos_bins: int = N_POS_BINS,
    frame_rate: float = FRAME_RATE,
    sigma_bins: float = SIGMA_BINS,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    f_i(pos) -> (n_cells, n_pos_bins).

    The within-spatial-bin firing rate is computed first (spikes / occupancy),
    then that rate-by-position vector is smoothed with the 7.5-cm Gaussian.

    `sigma_bins` is SPATIAL - it runs along the position axis, in 2-cm bins.
    Edges use `nearest`, since the track is linear.
    """
    spikes = np.zeros((X.shape[0], n_pos_bins))
    np.add.at(spikes.T, y_bins, X.T)  # spikes per bin
    occupancy = (
        np.bincount(y_bins, minlength=n_pos_bins) / frame_rate
    )  # seconds per bin

    rate = np.divide(spikes, occupancy, out=np.zeros_like(spikes), where=occupancy > 0)
    return gaussian_filter1d(rate, sigma_bins, axis=1, mode="nearest") + eps


# ------------------------------------------------------------- decoder ------
def decode(
    counts: np.ndarray,
    template: np.ndarray,
    tau: float = TAU,
) -> np.ndarray:
    """counts (n_time, n_cells), template (n_cells, n_pos) -> posterior (n_time, n_pos)."""
    ll = counts @ np.log(template) - tau * template.sum(axis=0)
    return np.exp(ll - logsumexp(ll, axis=1, keepdims=True))


def map_estimate(posterior: np.ndarray) -> np.ndarray:
    """Position bin with maximal posterior probability."""
    return np.argmax(posterior, axis=1)


# -------------------------------------------------------------- errors ------
def position_error(
    decoded: np.ndarray,
    true: np.ndarray,
    bin_size: float = BIN_SIZE_CM,
) -> np.ndarray:
    """Absolute reconstruction error in cm."""
    return np.abs(np.asarray(decoded) - np.asarray(true)) * bin_size


def distance_to_reward(
    decoded: np.ndarray,
    reward_bin: int,
    bin_size: float = BIN_SIZE_CM,
) -> np.ndarray:
    """Signed reconstructed distance to the reward, in cm (negative = before it)."""
    return (np.asarray(decoded) - reward_bin) * bin_size


# ------------------------------------------------- same-day 5-fold by lap ----
def cross_validate_same_day(
    X: np.ndarray,
    y_bins: np.ndarray,
    lap_id: np.ndarray,
    n_pos_bins: int = N_POS_BINS,
    tau: float = TAU,
    frames_per_bin: int = FRAMES_PER_BIN,
    frame_rate: float = FRAME_RATE,
    n_folds: int = 5,
    sigma_bins: float = SIGMA_BINS,
    mask: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Fivefold cross-validation applied by lap number. The template is rebuilt from
    the training laps' frames inside every fold, so no test lap leaks into it.
    """
    counts, true_pos, bin_lap = bin_time(X, y_bins, lap_id, frames_per_bin, mask)
    X_f, y_f, lap_f = (
        (X, y_bins, lap_id)
        if mask is None
        else (X[:, mask], y_bins[mask], lap_id[mask])
    )

    laps = np.unique(lap_id)
    posteriors = np.empty((len(counts), n_pos_bins))
    for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(laps):
        train = np.isin(lap_f, laps[tr])
        test = np.isin(bin_lap, laps[te])
        template = rate_map(
            X_f[:, train], y_f[train], n_pos_bins, frame_rate, sigma_bins
        )
        posteriors[test] = decode(counts[test], template, tau)

    decoded = map_estimate(posteriors)
    errors = position_error(decoded, true_pos)

    confusion = np.zeros((n_pos_bins, n_pos_bins))
    np.add.at(confusion, (true_pos, decoded), 1)
    confusion /= np.maximum(confusion.sum(axis=1, keepdims=True), 1)

    return dict(
        posteriors=posteriors,
        decoded=decoded,
        true_pos=true_pos,
        bin_lap=bin_lap,
        errors=errors,
        median_error=np.median(errors),
        mean_error=errors.mean(),
        confusion=confusion,
    )


# ----------------------------------------------------------------- main -----
def decode_main(
    all_frame_positions: np.ndarray,
    all_positions: np.ndarray,
    lap_id: np.ndarray,
    spks: np.ndarray,
    pc_mask: np.ndarray | None = None,
    frame_rate: float = FRAME_RATE,
    restrict_to_running: bool = True,
    n_folds: int = 5,
    sigma_bins: float = SIGMA_BINS,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    End-to-end same-day decoding from the raw behaviour vectors.

    Parameters
    ----------
    all_frame_positions : (n_samples,) frame index of each position sample
    all_positions       : (n_samples,) position in cm, 0-200, resets each lap
    lap_id              : (n_samples,) lap number of each position sample
    spks                : (n_cells, n_frames) sparsified binary spike estimates, Ssp
    pc_mask             : (n_cells,) bool. The template is built from PCs only,
                          so this should be supplied; all cells used if None.
    restrict_to_running : apply the >5 cm/s, >=3 s online running criterion.

    Returns the cross_validate_same_day dict plus alignment diagnostics.
    """
    frames = np.asarray(all_frame_positions).astype(int)
    pos = np.asarray(all_positions, dtype=float)
    laps = np.asarray(lap_id)
    spks = np.asarray(spks)

    X = spks[:, frames]
    if pc_mask is not None:
        X = X[np.asarray(pc_mask, dtype=bool)]
    y_bins = digitize_position(pos)

    mask = running_mask(pos, laps, frame_rate) if restrict_to_running else None

    res = cross_validate_same_day(
        X,
        y_bins,
        laps,
        n_pos_bins=N_POS_BINS,
        frame_rate=frame_rate,
        n_folds=n_folds,
        sigma_bins=sigma_bins,
        mask=mask,
        seed=seed,
    )

    # a tau-bin assumes 20 *consecutive* frames; censored frames break that
    gap_frac = float(np.mean(np.diff(frames) != 1)) if frames.size > 1 else 0.0
    res.update(
        n_cells_used=X.shape[0],
        n_frames_used=frames.size,
        n_laps=int(np.unique(laps).size),
        frame_gap_fraction=gap_frac,
        frames_used=frames,
        running=mask,
    )

    if verbose:
        if pc_mask is None:
            print(
                "NOTE: no pc_mask given, template built from all cells "
                "(the methods use PCs only)"
            )
        print(
            f"{res['n_cells_used']} cells, {res['n_laps']} laps, "
            f"{frames.size} frames -> {len(res['decoded'])} time bins ({TAU * 1000:.0f} ms)"
        )
        if mask is not None:
            print(
                f"running epochs (>{VEL_THRESHOLD} cm/s for >={VEL_MIN_DURATION_S}s) "
                f"kept {mask.mean():.1%} of frames"
            )
        if gap_frac > 0.01:
            print(
                f"WARNING: {gap_frac:.1%} of consecutive samples are not adjacent "
                f"frames (censored motion frames?); those tau bins span >333 ms"
            )
        print(
            f"median error {res['median_error']:.1f} cm | "
            f"mean {res['mean_error']:.1f} cm | chance ~{TRACK_LENGTH_CM / 3:.0f} cm"
        )
    return res
