"""Chang et al. (2020) analyses, adapted to the wheel-freeze sessions.

Chang H, Esteves IM, Neumann AR, Sun J, Mohajerani MH, McNaughton BL.
"Coordinated activities of retrosplenial ensembles during resting-state encode
spatial landmarks." Phil Trans R Soc B 375:20190228.

Their preparation is close to ours: Thy1-GCaMP6s, dysgranular RSC layers II-III at
100-200 um, two-photon at 19 Hz, a 150 cm FAMILIAR linear treadmill carrying four
tactile landmarks (one at reward), and sessions run as REST1 (10 min) -> RUN ->
REST2 (10 min) with the belt clamped during rest. That is the wheel-freeze design on a
familiar track, in the same region and layer.

The reason to implement it separately from ensemble_reactivation is that they measure
something we have not. Our pre/post contrast is reactivation STRENGTH, and it came out
flat. Chang et al. never report a strength difference - their pre/post claim is about
CONTENT: which positions the resting synchronous events decode to, and whether those
positions sit near landmarks. On a familiar track the ensembles exist before the run, so
a null on strength is the expected result rather than a contradiction.

Two further design differences worth keeping:

  * Ensembles are clustered independently WITHIN each rest epoch, not carried over from
    the running data. That sidesteps the circularity that dogs the run-template
    approach, where cells correlated during running are correlated offline for
    anatomical and neuropil reasons and no weight shuffle can tell that from
    reinstatement. Here ensemble membership comes from rest and spatial tuning from run,
    so the link between them is the thing being tested.

  * Explained variance (Kudrimoti et al. 1999) asks whether REST2's correlation
    structure resembles RUN more than REST1's does, partialling REST1 out. It is a
    different question from mean reactivation strength and is cheap to compute.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import squareform
from scipy.stats import linregress, zscore

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))

from viral.bayesian_decoder import decode, map_estimate, rate_map
from viral.constants import CACHE_PATH, LOCAL_DFF_PATH, TIFF_UMBRELLA, grosmark_config
from viral.ensemble_reactivation import freeze_immobility_masks
from viral.grosmark_analysis import get_place_cells
from viral.imaging_utils import (
    get_online_position_and_frames,
    load_imaging_data,
    trial_is_imaged,
)
from viral.models import Cached2pSession
from viral.utils import (
    degrees_to_cm,
    get_genotype,
    get_wheel_circumference_from_rig,
    upper_triangle_no_diagonal,
)


@dataclass
class ChangConfig:
    """Parameters as published, converted to our 30 Hz where they were given in seconds.

    Chang et al. imaged at 19 Hz; every parameter here is specified in time or in
    correlation units, so the only conversion needed is the frame rate.
    """

    fs: int = 30
    # "smoothed using a sigma = 1 s Gaussian kernel to reduce temporal jitter"
    rate_sigma_seconds: float = 1.0
    # "any continuous segment of time in which the MUA exceeded 3 s.d. above the mean"
    sce_threshold_sd: float = 3.0
    # "SCEs that occurred less than 250 ms apart were identified as part of the same SCE"
    sce_merge_seconds: float = 0.25
    # "d < 0.75, which corresponds to an average correlation coefficient of r > 0.25"
    cluster_distance: float = 0.75
    # "clusters containing fewer than 10 members were rejected"
    min_ensemble_size: int = 10
    # decoding
    # Decoding is restricted to the range place cells were DEFINED over. get_place_cells
    # uses grosmark_config, so a cell enters pcs_mask by having a field between 20 and
    # 160 cm; outside that the template is unselected residual activity, and at the
    # corridor start it is also divided by a near-zero occupancy because the running
    # criterion strips the acceleration phase. Decoding 0-180 put 42% of bins below
    # 20 cm and pushed the median landmark distance above chance.
    position_start_cm: float = grosmark_config.start
    position_end_cm: float = grosmark_config.end
    position_bin_cm: float = 2.0
    decode_frames_per_bin: int = 2
    # Online decoding is conventionally done in ~333 ms bins (Grosmark uses 20 frames);
    # scoring the held-out accuracy check in the 2-frame offline bins measures the
    # decoder at 67 ms resolution, which is much harder and understates it.
    decode_frames_per_bin_online: int = 20
    # Grosmark: "only bins with non-zero firing rates were used for offline Bayesian
    # decoding", and a PSE required "at least 5 distinct PCs each fired at least one
    # estimated spike". Without this the posterior for a near-empty bin is driven by
    # -tau * sum(f), which peaks wherever the population fires least - the track ends.
    # Measured: an empty bin decodes to 0 cm, one active cell to 166 cm.
    min_active_cells: int = 5
    # Position bins the animal barely visits get a firing rate divided by a near-zero
    # occupancy, which inflates the template there and pulls the decoder in. It bites at
    # the corridor start, because the running criterion (>5 cm/s sustained for 3 s)
    # removes the beginning of every trial while the animal accelerates. Measured
    # without this: 42% of decoded bins landed below 20 cm, pushing the median distance
    # to a landmark ABOVE the uniform-chance value of 15 cm.
    min_occupancy_seconds: float = 1.0
    # An absolute floor is far too weak on its own: summed across ~40 trials, even a
    # bin the animal barely enters accumulates more than a second. The binding
    # criterion is relative - a bin must hold at least this fraction of the MEDIAN bin's
    # occupancy for its spikes/occupancy rate to be worth trusting.
    min_occupancy_fraction: float = 0.25
    # Chang et al. correlate continuous deconvolved rates. Our spks are binarised and
    # then thinned by remove_consecutive_ones, which discards amplitude - and amplitude
    # matters: binarising swings activity measures sixfold with burst structure alone.
    # Setting this makes get_df_summary pass oasis_denoised as the activity array for
    # the correlation and SCE measures. Place cells and the decoding template still come
    # from spks. It lives here rather than as a bare argument so it lands in the cache
    # key and cannot be confused with results computed the other way.
    use_denoised: bool = False
    # Same values as PlaceCellResults.LANDMARK_LOCATIONS, held here rather than
    # imported: learning_stages.py does not reliably import (its __main__ block has no
    # body), and a three-element constant is not worth that coupling.
    landmarks_cm: Tuple[float, ...] = (45.0, 90.0, 135.0)

    @property
    def rate_sigma_frames(self) -> float:
        return self.rate_sigma_seconds * self.fs

    @property
    def sce_merge_frames(self) -> int:
        return int(round(self.sce_merge_seconds * self.fs))

    @property
    def n_position_bins(self) -> int:
        return int(
            (self.position_end_cm - self.position_start_cm) / self.position_bin_cm
        )

    @property
    def bin_centres_cm(self) -> np.ndarray:
        return (
            np.arange(self.n_position_bins) * self.position_bin_cm
            + self.position_start_cm
        )


# Bump when the analysis changes in a way ChangConfig does not capture
ANALYSIS_VERSION = "8"


def config_key(config: ChangConfig) -> str:
    """Short deterministic tag for a ChangConfig, for use in cache filenames.

    Every cache in this project so far has been keyed on mouse and date alone, so a
    changed parameter leaves stale results behind a filename that still looks valid.
    That has cost a re-run more than once; this makes the parameters part of the key.

    ANALYSIS_VERSION covers changes the config cannot see - a new output column, or a
    fix to how the template is built. Bump it whenever the numbers or the columns
    change for reasons other than a config value.
    """
    return hashlib.md5(f"{ANALYSIS_VERSION}{asdict(config)!r}".encode()).hexdigest()[:8]


def smoothed_rates(spks: np.ndarray, config: ChangConfig) -> np.ndarray:
    """Deconvolved activity smoothed with the 1 s Gaussian, then z-scored per cell.

    Smoothing happens on the CONTINUOUS recording before any epoch is selected. The
    rest epochs are contiguous blocks so this matters less than it does for the ITI
    masks, but selecting first and smoothing after would still blur the boundary
    between the end of one epoch and the start of the next.

    Silent cells z-score to NaN and are zeroed.
    """
    smoothed = gaussian_filter1d(spks, sigma=config.rate_sigma_frames, axis=1)
    return np.nan_to_num(zscore(smoothed, axis=1))


def detect_sce(rates: np.ndarray, config: ChangConfig) -> List[Tuple[int, int]]:
    """Synchronous co-activation events, as (start, end) frame pairs.

    "Any continuous segment of time in which the MUA exceeded 3 s.d. above the mean
    were classified as SCEs", with events less than 250 ms apart merged into one.

    MUA is the mean across cells of the z-scored smoothed rates, then z-scored again,
    so the threshold is in units of the population signal's own variability. Note this
    makes the SCE rate partly self-normalising within an epoch: an epoch with globally
    more synchrony raises its own threshold. That is the paper's definition, and it is
    also why their pre/post claim is about decoded content rather than SCE rate.

    `rates` must already be restricted to the epoch being scored, since the mean and
    s.d. are taken over it.
    """
    if rates.shape[1] == 0:
        return []

    mua = np.nan_to_num(zscore(rates.mean(axis=0)))
    above = mua > config.sce_threshold_sd
    if not above.any():
        return []

    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(np.int8), 0]))
    events = list(zip(edges[::2], edges[1::2]))

    merged = [events[0]]
    for start, end in events[1:]:
        if start - merged[-1][1] < config.sce_merge_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def cluster_ensembles(rates: np.ndarray, config: ChangConfig) -> List[np.ndarray]:
    """Ensembles as groups of co-active cells, clustered within a single epoch.

    "Correlation coefficients r were converted to a distance metric d = 1 - r",
    agglomerative clustering with the "unweighted average distance linkage criterion",
    cut at "d < 0.75, which corresponds to an average correlation coefficient of
    r > 0.25 within a cluster", and "clusters containing fewer than 10 members were
    rejected".

    Clustering each epoch independently is the point of the design: ensemble membership
    comes from rest, spatial tuning comes from run, and the relationship between them is
    what the analysis tests. Carrying a run template into rest instead makes the two
    sides share a source of correlation.

    Returns one array of cell indices per surviving ensemble.
    """
    if rates.shape[1] < 2:
        return []

    correlation = np.nan_to_num(np.corrcoef(rates))
    distance = 1.0 - correlation
    np.fill_diagonal(distance, 0.0)
    # squareform needs exact symmetry; corrcoef can be off by floating point
    distance = np.clip((distance + distance.T) / 2, 0, None)

    links = linkage(squareform(distance, checks=False), method="average")
    labels = fcluster(links, t=config.cluster_distance, criterion="distance")

    ensembles = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    return [e for e in ensembles if e.size >= config.min_ensemble_size]


def explained_variance(
    run: np.ndarray, rest_pre: np.ndarray, rest_post: np.ndarray
) -> Tuple[float, float]:
    """Explained variance and its reverse control (Kudrimoti et al. 1999).

    EV is the squared partial correlation between the RUN and REST2 pairwise-correlation
    structures, controlling for REST1:

        EV = [ (r_run,post - r_run,pre * r_pre,post)
               / sqrt((1 - r_run,pre^2)(1 - r_pre,post^2)) ]^2

    REV swaps REST1 and REST2, so it asks how much of the RUN structure was already
    present beforehand. EV > REV is the reactivation claim; EV == REV means the
    structure was there all along, which is what static coupling predicts and what a
    familiar track would produce.

    Note the formula as printed in the paper omits the square root in the denominator;
    this is the standard partial-correlation form.

    Each input is (n_cells, n_frames) for one epoch.
    """
    vectors = {}
    for name, activity in (("run", run), ("pre", rest_pre), ("post", rest_post)):
        vectors[name] = upper_triangle_no_diagonal(np.nan_to_num(np.corrcoef(activity)))

    def r(a: str, b: str) -> float:
        return float(np.corrcoef(vectors[a], vectors[b])[0, 1])

    def partial(target: str, control: str) -> float:
        numerator = r("run", target) - r("run", control) * r(control, target)
        denominator = np.sqrt(
            (1 - r("run", control) ** 2) * (1 - r(control, target) ** 2)
        )
        return 0.0 if denominator == 0 else (numerator / denominator) ** 2

    return partial("post", "pre"), partial("pre", "post")


def running_frames(session: Cached2pSession, config: ChangConfig) -> np.ndarray:
    """Every imaged running frame in the session.

    Distinct from the frames run_rate_map returns, which are odd trials only so the
    decoding template stays independent of the even-trial accuracy check. Explained
    variance has no such requirement and wants all the data - computing it on half the
    running frames just makes the RUN correlation matrix noisier, which pushes both EV
    and REV toward zero.
    """
    frames: List[int] = []
    wheel_circumference = get_wheel_circumference_from_rig("2P")

    for trial in session.trials:
        if not trial_is_imaged(trial):
            continue
        position, frame_position = get_online_position_and_frames(
            trial=trial,
            wheel_circumference=wheel_circumference,
            threshold_speed=True,
        )
        if len(position) != len(frame_position) or position.size == 0:
            continue
        keep = (position >= config.position_start_cm) & (
            position < config.position_end_cm
        )
        frames.extend(frame_position[keep].astype(int))

    return np.unique(np.array(frames, dtype=int))


def running_frames_and_positions(
    session: Cached2pSession, config: ChangConfig, parity: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Running frames and their positions, for trials of the given parity.

    parity=1 gives odd trials, which build the decoding template; parity=0 gives even
    trials, held out so decoding_error is independent of it. Restricted to the range
    place cells were defined over, since a template outside it is unselected activity.
    """
    frames: List[int] = []
    positions: List[float] = []
    wheel_circumference = get_wheel_circumference_from_rig("2P")

    for index, trial in enumerate(session.trials):
        if not trial_is_imaged(trial) or index % 2 != parity:
            continue
        position, frame_position = get_online_position_and_frames(
            trial=trial,
            wheel_circumference=wheel_circumference,
            threshold_speed=True,
        )
        if len(position) != len(frame_position) or position.size == 0:
            continue
        keep = (position >= config.position_start_cm) & (
            position < config.position_end_cm
        )
        frames.extend(frame_position[keep].astype(int))
        positions.extend(position[keep])

    return np.array(frames, dtype=int), np.array(positions, dtype=float)


def decoding_error(
    session: Cached2pSession,
    spks: np.ndarray,
    template: np.ndarray,
    valid_bins: np.ndarray,
    config: ChangConfig,
) -> float:
    """Median absolute decoding error on held-out EVEN trials, in cm.

    The check the odd/even split exists for, and the one that says whether any offline
    decoding result means anything. Chang et al. report a "mean decoding error during
    running of 11.8 cm (+/- 0.3 s.e.m.)" - if ours is near that the template works and a
    null offline result is real; if it is closer to chance the offline decoding is
    uninterpretable and the landmark measure says nothing either way.

    Chance for reference is the mean absolute difference between two uniform draws over
    the decoded range, which is (end - start) / 3 = 46.7 cm over 20-160.
    """
    frames, positions = running_frames_and_positions(session, config, parity=0)
    if frames.size < config.decode_frames_per_bin_online:
        return float("nan")

    per_bin = config.decode_frames_per_bin_online
    n_bins = frames.size // per_bin
    usable = n_bins * per_bin
    counts = (
        spks[:, frames[:usable]].reshape(spks.shape[0], n_bins, per_bin).sum(axis=2).T
    )
    true_position = positions[:usable].reshape(n_bins, per_bin).mean(axis=1)

    active = (counts > 0).sum(axis=1)
    keep = active >= config.min_active_cells
    if not keep.any():
        return float("nan")

    posterior = decode(counts[keep], template, tau=per_bin / config.fs)
    posterior = np.where(valid_bins[np.newaxis, :], posterior, -np.inf)
    decoded = (
        map_estimate(posterior) * config.position_bin_cm + config.position_start_cm
    )
    return float(np.median(np.abs(decoded - true_position[keep])))


def run_rate_map(
    session: Cached2pSession, spks: np.ndarray, config: ChangConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place-field template from the running epoch, plus the frames it was built from.

    "f_i(x) is the mean deconvolved fluorescence of neuron i as a function of position
    derived from training data". Built from odd trials only so the template and any
    decoding accuracy check are independent, mirroring their even/odd split.

    Restricted to RUNNING frames, via the same criterion get_place_cells uses. Without
    that the template is built partly from the ITI, where the animal sits at the end of
    the corridor - which loads occupancy into one position bin and puts immobile
    activity into what is supposed to be a place field.
    """
    frames_array, positions = running_frames_and_positions(session, config, parity=1)

    if frames_array.size == 0:
        return (
            np.zeros((spks.shape[0], config.n_position_bins)),
            np.array([], dtype=int),
            np.zeros(config.n_position_bins, dtype=bool),
        )

    bins = np.clip(
        ((positions - config.position_start_cm) / config.position_bin_cm).astype(int),
        0,
        config.n_position_bins - 1,
    )
    template = rate_map(
        spks[:, frames_array],
        bins,
        n_pos_bins=config.n_position_bins,
        frame_rate=config.fs,
    )
    occupancy = np.bincount(bins, minlength=config.n_position_bins) / config.fs
    valid = (occupancy >= config.min_occupancy_seconds) & (
        occupancy >= config.min_occupancy_fraction * np.median(occupancy[occupancy > 0])
    )
    return template, frames_array, valid


def decode_sce_positions(
    spks: np.ndarray,
    sce_events: List[Tuple[int, int]],
    epoch_frames: np.ndarray,
    template: np.ndarray,
    valid_bins: np.ndarray,
    config: ChangConfig,
) -> np.ndarray:
    """Decoded position (cm) for each time bin inside each SCE.

    Bayesian reconstruction with the running template, in two-frame bins as Grosmark
    uses offline. sce_events index into epoch_frames, not into the session.

    Bins with fewer than config.min_active_cells contributing are dropped. This is not
    optional tidying: with few or no spikes the log posterior reduces to -tau * sum(f),
    whose maximum sits wherever the population fires least, so sparse bins pile up at
    the track ends. Left in, they push the median distance-to-landmark ABOVE the
    uniform-chance value of 15 cm on a 180 cm track with landmarks at 45/90/135, which
    is how the bug announced itself.
    """
    decoded: List[float] = []
    tau = config.decode_frames_per_bin / config.fs

    for start, end in sce_events:
        frames = epoch_frames[start:end]
        if frames.size < config.decode_frames_per_bin:
            continue
        n_bins = frames.size // config.decode_frames_per_bin
        counts = (
            spks[:, frames[: n_bins * config.decode_frames_per_bin]]
            .reshape(spks.shape[0], n_bins, config.decode_frames_per_bin)
            .sum(axis=2)
            .T
        )
        active = (counts > 0).sum(axis=1)
        counts = counts[active >= config.min_active_cells]
        if counts.shape[0] == 0:
            continue

        posterior = decode(counts, template, tau=tau)
        # Only positions the animal occupied enough to estimate a rate for
        posterior = np.where(valid_bins[np.newaxis, :], posterior, -np.inf)
        decoded.extend(
            map_estimate(posterior) * config.position_bin_cm + config.position_start_cm
        )

    return np.array(decoded)


def landmark_regression(
    decoded: np.ndarray, config: ChangConfig
) -> Tuple[float, float]:
    """Chang et al.'s landmark measure: a per-spatial-bin regression, not a per-event one.

    "The fraction of frames that significantly decoded position (p < 0.05) is compared
    with the distance from the closest landmark", fitted with a linear regression.

    Taking the median distance from each decoded event to its nearest landmark - which
    is what this module did first - cannot work at our decoding accuracy. With a 27.8 cm
    error and landmarks 45 cm apart, events decoding PERFECTLY to landmarks measure at
    14.5 cm, above the uniform-chance value of 11.7, so the measure cannot separate
    perfect coding from none. Even at Chang's 11.8 cm error the gap is 10.9 versus 11.7.
    A spatial profile is far more robust, because decoder noise blurs the profile rather
    than saturating a summary statistic.

    Returns (slope, p). A NEGATIVE slope is the claim: more decoding closer to landmarks.
    """
    if decoded.size < 10:
        return float("nan"), float("nan")

    edges = np.arange(
        config.position_start_cm,
        config.position_end_cm + config.position_bin_cm,
        config.position_bin_cm,
    )
    counts, _ = np.histogram(decoded, bins=edges)
    fraction = counts / counts.sum()
    distance = distance_to_nearest_landmark(
        config.bin_centres_cm, list(config.landmarks_cm)
    )

    result = linregress(distance, fraction)
    return float(result.slope), float(result.pvalue)


def distance_to_nearest_landmark(
    positions: np.ndarray, landmarks: List[float]
) -> np.ndarray:
    """Distance in cm from each decoded position to the closest landmark."""
    if positions.size == 0:
        return positions
    return np.min(
        np.abs(positions[:, np.newaxis] - np.array(landmarks)[np.newaxis, :]), axis=1
    )


def chang_session(
    session: Cached2pSession,
    spks: np.ndarray,
    config: ChangConfig = ChangConfig(),
    use_place_cell_cache: bool = True,
    activity: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run the Chang et al. analyses on one wheel-freeze session.

    Returns one row per epoch (pre, post) carrying the SCE count, the number of
    ensembles clustered within that epoch, the median distance from decoded SCE
    positions to the nearest landmark, and the session's EV and REV.

    The landmark measure is the paper's central claim: their REST2 decoded positions sat
    closer to landmarks than REST1's. A negative here is meaningful, since this is the
    one published result from a design that matches ours.

    activity, if given, replaces spks for the correlation and SCE measures only; place
    cells and the decoding template still come from spks. Use it to run the analysis on
    continuous deconvolved traces rather than binarised spikes.
    """
    masks = freeze_immobility_masks(session, n_frames=spks.shape[1])
    if not masks:
        print(f"Skipping {session.mouse_name} {session.date}: no wheel freeze")
        return pd.DataFrame()

    pcs_mask, _, _ = get_place_cells(
        session=session,
        spks=spks,
        rewarded=None,
        config=grosmark_config,
        plot=False,
        use_cache=use_place_cell_cache,
    )
    place_cells = spks[pcs_mask, :]
    # Chang et al. correlate continuous deconvolved rates; our spks are binarised and
    # then sparsified by remove_consecutive_ones, which discards amplitude. Pass
    # activity=denoised (the third return of load_imaging_data) to use the continuous
    # trace for the correlation structure while place cells still come from spks.
    rates = smoothed_rates(
        place_cells if activity is None else activity[pcs_mask, :], config
    )

    template, run_frames, valid_bins = run_rate_map(session, place_cells, config)
    if run_frames.size == 0:
        print(f"  {session.mouse_name} {session.date}: no running frames, skipping")
        return pd.DataFrame()

    # All running frames, not the odd-trial subset the template is built from
    ev_run_frames = running_frames(session, config)
    error = decoding_error(session, place_cells, template, valid_bins, config)

    ev, rev = explained_variance(
        run=rates[:, ev_run_frames],
        rest_pre=rates[:, masks["pre"]],
        rest_post=rates[:, masks["post"]],
    )

    records = []
    for epoch, mask in masks.items():
        epoch_frames = np.flatnonzero(mask)
        epoch_rates = rates[:, epoch_frames]

        sce_events = detect_sce(epoch_rates, config)

        if sce_events:
            ensembles = cluster_ensembles(epoch_rates, config)
            decoded = decode_sce_positions(
                place_cells, sce_events, epoch_frames, template, valid_bins, config
            )
            distances = distance_to_nearest_landmark(decoded, list(config.landmarks_cm))
            landmark_slope, landmark_slope_p = landmark_regression(decoded, config)
        else:
            ensembles = []
            decoded = np.array([], dtype=float)
            distances = np.array([], dtype=float)
            landmark_slope, landmark_slope_p = float("nan"), float("nan")

        records.append(
            {
                "mouse": session.mouse_name,
                "date": session.date,
                "session_type": session.session_type,
                "genotype": get_genotype(session.mouse_name),
                "epoch": epoch,
                "epoch_seconds": epoch_frames.size / config.fs,
                "n_place_cells": int(np.sum(pcs_mask)),
                "n_sce": len(sce_events),
                "sce_rate_hz": len(sce_events) / (epoch_frames.size / config.fs),
                "n_ensembles": len(ensembles),
                "mean_ensemble_size": (
                    float(np.mean([e.size for e in ensembles])) if ensembles else np.nan
                ),
                "n_decoded_bins": decoded.size,
                "median_landmark_distance": (
                    float(np.median(distances)) if distances.size else np.nan
                ),
                # Chang's actual measure - negative slope means more decoding near
                # landmarks. Robust to decoder blur in a way the median distance is not.
                "landmark_slope": landmark_slope,
                "landmark_slope_p": landmark_slope_p,
                # Where the decoded positions actually sit. A median landmark distance
                # above the uniform-chance value means they are concentrated in the
                # outer track, which no real signal should do - these say whether that
                # is happening and where.
                "median_decoded_position": (
                    float(np.median(decoded)) if decoded.size else np.nan
                ),
                "frac_decoded_below_start": (
                    float(np.mean(decoded < grosmark_config.start))
                    if decoded.size
                    else np.nan
                ),
                "frac_decoded_above_end": (
                    float(np.mean(decoded > grosmark_config.end))
                    if decoded.size
                    else np.nan
                ),
                "decoded_p10": (
                    float(np.percentile(decoded, 10)) if decoded.size else np.nan
                ),
                "decoded_p90": (
                    float(np.percentile(decoded, 90)) if decoded.size else np.nan
                ),
                # Occupancy-normalised rates blow up where the animal spends little
                # time, and those bins then attract the decoder
                "template_mass_min": float(template.sum(axis=0).min()),
                "template_mass_max": float(template.sum(axis=0).max()),
                "template_argmax_cm": float(
                    np.argmax(template.sum(axis=0)) * config.position_bin_cm
                ),
                "n_valid_bins": int(valid_bins.sum()),
                "decoding_error_cm": error,
                "valid_min_cm": (
                    float(config.bin_centres_cm[valid_bins].min())
                    if valid_bins.any()
                    else np.nan
                ),
                "valid_max_cm": (
                    float(config.bin_centres_cm[valid_bins].max())
                    if valid_bins.any()
                    else np.nan
                ),
                # EV and REV are session-level, repeated on both rows
                "explained_variance": ev,
                "reverse_explained_variance": rev,
            }
        )

    return pd.DataFrame(records)


def _chance_landmark_distance(config: ChangConfig = ChangConfig()) -> float:
    """Median distance to the nearest landmark if decoded positions were uniform.

    The reference the landmark measure has to beat. On a 180 cm track with landmarks at
    45/90/135 it is 15 cm.
    """
    positions = np.linspace(config.position_start_cm, config.position_end_cm, 200000)
    return float(
        np.median(distance_to_nearest_landmark(positions, list(config.landmarks_cm)))
    )


def report_chang(df: pd.DataFrame) -> pd.DataFrame:
    """Print the pre/post comparisons and return the per-session wide frame."""
    if df.empty:
        print("No sessions")
        return df

    n_sessions = df.groupby(["mouse", "date"]).ngroups
    print(f"Chang analyses: {n_sessions} sessions, {df.mouse.nunique()} mice\n")

    chance = _chance_landmark_distance()
    if not np.isnan(chance):
        print(
            f"  uniform-decoding chance for median_landmark_distance: {chance:.1f} cm\n"
            f"  (above chance means decoded positions AVOID landmarks, which for a real\n"
            f"   signal should not happen - check n_decoded_bins and min_active_cells)\n"
        )

    for value in (
        "sce_rate_hz",
        "n_sce",
        "n_decoded_bins",
        "n_valid_bins",
        "decoding_error_cm",
        "valid_min_cm",
        "valid_max_cm",
        "median_decoded_position",
        "frac_decoded_below_start",
        "frac_decoded_above_end",
        "decoded_p10",
        "decoded_p90",
        "n_ensembles",
        "mean_ensemble_size",
        "median_landmark_distance",
        "landmark_slope",
        "epoch_seconds",
    ):
        wide = df.pivot_table(index=["mouse", "date"], columns="epoch", values=value)
        if not {"pre", "post"}.issubset(wide.columns):
            continue
        delta = (wide["post"] - wide["pre"]).dropna()
        print(
            f"  {value:>26}  pre {wide['pre'].median():>8.3f}  "
            f"post {wide['post'].median():>8.3f}  "
            f"post-pre {delta.median():>+8.3f}  "
            f"{int((delta > 0).sum())}/{len(delta)} sessions up"
        )

    session_level = df.groupby(["mouse", "date"])[
        ["explained_variance", "reverse_explained_variance"]
    ].first()
    print(
        f"\n  {'explained variance':>26}  EV {session_level.explained_variance.median():.4f}"
        f"   REV {session_level.reverse_explained_variance.median():.4f}"
        f"   EV>REV in "
        f"{int((session_level.explained_variance > session_level.reverse_explained_variance).sum())}"
        f"/{len(session_level)} sessions"
    )
    print(
        "\n  EV > REV is the reactivation claim. EV == REV means the RUN correlation\n"
        "  structure was already present before the run, which is what a familiar\n"
        "  track and static coupling both predict."
    )
    return session_level


def load_spks(
    mouse: str, date: str, need_denoised: bool = False
) -> Tuple[np.ndarray, np.ndarray | None]:
    """Deconvolved spikes for a session, iscell-filtered whichever source they came from.

    Returns (spks, denoised). denoised is None unless need_denoised, since the local
    copy under LOCAL_DFF_PATH holds only spikes and asking for the continuous trace
    means going to the server.

    The local copy may hold every suite2p ROI or only the curated ones, while
    load_imaging_data always filters. Getting that wrong is silent rather than an error:
    get_place_cells would run on a different cell set and pcs_mask would index the wrong
    rows, so the count is checked against iscell explicitly.
    """
    local = LOCAL_DFF_PATH / f"{mouse}_{date}_spks.npy"
    if not local.exists():
        _, spks, denoised = load_imaging_data(mouse, date)
        np.save(local, spks)
        np.save(LOCAL_DFF_PATH / f"{mouse}_{date}_denoised.npy", denoised)
        return spks, (denoised if need_denoised else None)

    spks = np.load(local)
    iscell = np.load(
        TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0" / "iscell.npy"
    )[:, 0].astype(bool)

    assert spks.shape[0] == int(iscell.sum()), (
        f"{mouse} {date}: local spks has {spks.shape[0]} rows, which matches neither "
        f"the {iscell.size} suite2p ROIs nor the {int(iscell.sum())} curated cells"
    )

    denoised_local = LOCAL_DFF_PATH / f"{mouse}_{date}_denoised.npy"
    if need_denoised and not denoised_local.exists():
        _, _, denoised = load_imaging_data(mouse, date)
        np.save(denoised_local, denoised)
        return spks, denoised
    elif need_denoised:
        denoised = np.load(denoised_local)
        assert denoised.shape[0] == int(iscell.sum()), (
            f"{mouse} {date}: local denoised has {denoised.shape[0]} rows, which "
            f"matches neither the {iscell.size} suite2p ROIs nor the "
            f"{int(iscell.sum())} curated cells"
        )
        return spks, denoised
    return spks, None


def get_df_summary(
    genotype: str,
    config: ChangConfig = ChangConfig(),
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Run chang_session across every cached session of one genotype.

    Per-session results are cached as CSVs keyed by the ChangConfig, so changing a
    threshold or the landmark positions invalidates them rather than silently reusing
    values computed under different parameters.

    Set config.use_denoised to run the correlation and SCE measures on the continuous
    deconvolved trace instead of binarised spikes, which is what Chang et al. did. That
    forces loading from the server, since the local cache holds only spikes.
    """
    cache_files = list(CACHE_PATH.glob("*.json"))

    chang_cache_path = CACHE_PATH.parent / "ensemble_caches" / "chang"
    assert chang_cache_path.exists(), f"{chang_cache_path} does not exist"

    all_df = []

    for cache_file in cache_files:
        mouse, date = cache_file.stem.split("_")[:2]

        if get_genotype(mouse) != genotype:
            continue

        session_cache = chang_cache_path / f"{mouse}_{date}_{config_key(config)}.csv"
        if session_cache.exists():
            print(f"Loading cached {mouse} {date}")
            all_df.append(pd.read_csv(session_cache, index_col=False))
            continue

        session = Cached2pSession.model_validate_json(cache_file.read_text())

        if not session.session_type.startswith(
            "reversal learning"
        ) and not session.session_type.startswith("learning"):
            continue

        if session.wheel_freeze is None:
            print(f"Skipping {mouse} {date}: no wheel freeze")
            continue

        print(f"Processing {mouse} {date}")

        try:
            spks, denoised = load_spks(mouse, date, need_denoised=config.use_denoised)
        except Exception as e:
            print(f"Error loading {mouse} {date}: {e}")
            continue

        if spks.shape[0] < 5:
            print("No cells :'( probably wrong pmt")
            continue

        try:
            df = chang_session(session, spks, config=config, activity=denoised)
        except Exception as e:
            print(f"Error processing {mouse} {date}: {e}")
            if raise_on_error:
                raise
            continue

        if df.empty:
            continue

        print(f"Saving cached {mouse} {date}")
        df.to_csv(session_cache, index=False)
        all_df.append(df)

    df_summary = pd.concat(all_df, ignore_index=True)
    suffix = "_denoised" if config.use_denoised else ""
    df_summary.to_csv(f"df_summary_chang_{genotype}{suffix}.csv", index=False)
    return df_summary


def load_cached(genotype: str, config: ChangConfig) -> pd.DataFrame:
    """Load all the per-session CSVs for the current ChangConfig."""
    chang_cache_path = CACHE_PATH.parent / "ensemble_caches" / "chang"
    assert chang_cache_path.exists(), f"{chang_cache_path} does not exist"

    all_df = []
    for session_cache in chang_cache_path.glob(f"*_{config_key(config)}.csv"):
        print(session_cache.name)
        all_df.append(pd.read_csv(session_cache, index_col=False))

    df_summary = pd.concat(all_df, ignore_index=True)
    suffix = "_denoised" if config.use_denoised else ""
    df_summary.to_csv(f"df_summary_chang_{genotype}{suffix}.csv", index=False)
    return pd.concat(all_df, ignore_index=True)


def temp_spks_copier() -> None:
    for cache_file in CACHE_PATH.glob("*.json"):
        mouse, date = cache_file.stem.split("_")[:2]
        # if not get_genotype(mouse).startswith("WT") and not get_genotype(
        #     mouse
        # ).startswith("NLGF"):
        if not get_genotype(mouse) == "Oligo-BACE1-KO":
            continue
            # print(f"Skipping {mouse} {date}: not WT or NLGF")
            # continue
        local = LOCAL_DFF_PATH / f"{mouse}_{date}_spks.npy"
        if not local.exists():
            print(f"Copying {mouse} {date}")
            try:
                _, spks, denoised = load_imaging_data(mouse, date)
                np.save(local, spks)
                np.save(LOCAL_DFF_PATH / f"{mouse}_{date}_denoised.npy", denoised)
            except Exception as e:
                print(f"Error copying {mouse} {date}: {e}")

        else:
            print(f"{mouse} {date} already copied")


def temp_stat_ops_copier() -> None:
    for cache_file in CACHE_PATH.glob("*.json"):
        mouse, date = cache_file.stem.split("_")[:2]
        # if not get_genotype(mouse).startswith("WT") and not get_genotype(
        #     mouse
        # ).startswith("NLGF"):
        #     print(f"Skipping {mouse} {date}: not WT or NLGF")
        #     continue
        if not get_genotype(mouse) == "Oligo-BACE1-KO":
            continue

        local = LOCAL_DFF_PATH / f"{mouse}_{date}_stat.npy"
        if not local.exists():
            print(f"Copying {mouse} {date}")
            try:
                stat = np.load(
                    TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0" / "stat.npy",
                    allow_pickle=True,
                )
                ops = np.load(
                    TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0" / "ops.npy",
                    allow_pickle=True,
                )
                iscell = np.load(
                    TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0" / "iscell.npy",
                    allow_pickle=True,
                )
                np.save(local, stat)
                np.save(LOCAL_DFF_PATH / f"{mouse}_{date}_ops.npy", ops)
                np.save(LOCAL_DFF_PATH / f"{mouse}_{date}_iscell.npy", iscell)
            except Exception as e:
                print(f"Error copying {mouse} {date}: {e}")

        else:
            print(f"{mouse} {date} already copied")


if __name__ == "__main__":
    temp_spks_copier()
    temp_stat_ops_copier()
    # spks_df = get_df_summary("WT")
    # # denoised_df = get_df_summary("WT", config=ChangConfig(use_denoised=True))

    # # df = get_df_summary("WT")
    # df = load_cached("WT", config=ChangConfig(use_denoised=True))

    # # df = pd.read_csv(f"df_summary_chang_WT.csv")
    # report_chang(df)
