import itertools
import math
from pathlib import Path
import sys
import warnings
from matplotlib import pyplot as plt
from scipy.stats import median_abs_deviation, zscore, pearsonr, ttest_ind
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm

# Allow you to run the file directly, remove if exporting as a proper module
HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


from viral.constants import (
    CACHE_PATH,
    HERE,
    SERVER_PATH,
    TIFF_UMBRELLA,
    grosmark_config,
)
from viral.imaging_utils import (
    get_ITI_matrix,
    load_imaging_data,
    trial_is_imaged,
    activity_trial_position,
    split_fluoresence_online_freeze,
)
from viral.models import Cached2pSession, GrosmarkConfig, WheelFreeze

from viral.utils import (
    compute_linear_slope,
    cross_correlation_pandas,
    degrees_to_cm,
    find_n_consecutive_trues_center,
    get_movement_bool,
    get_wheel_circumference_from_rig,
    has_n_consecutive_trues,
    interpolate_nans_vector,
    remove_consecutive_ones,
    remove_diagonal,
    session_is_unsupervised,
    shaded_line_plot,
    shuffle_rows,
    sort_matrix_peak,
    get_speed_positions,
)


def grosmark_place_field(
    session: Cached2pSession,
    spks_raw: np.ndarray,
    rewarded: bool | None,
    config: GrosmarkConfig,
    plot: bool = True,
    cache_file_additional_info: str | None = None,
    use_cache: bool = True,
) -> None:
    """
    Grosmark et al. place field analysis.
    1. get place cell mask
    2. get peak indices and peak positions
    3. do pair-wise correlations
    """
    if session.wheel_freeze is None:
        spks = spks_raw
    else:
        offline_spks_pre, online_spks, offline_spks_post = (
            split_fluoresence_online_freeze(
                flu=spks_raw, wheel_freeze=session.wheel_freeze
            )
        )

        spks = np.hstack([offline_spks_pre, online_spks, offline_spks_post])
        assert spks_raw.shape == spks.shape

    pcs, smoothed_matrix, _ = get_place_cells(
        session=session,
        spks=spks,
        rewarded=rewarded,
        config=config,
        plot=plot,
        cache_file_additional_info=cache_file_additional_info,
        use_cache=use_cache,
    )

    spks = spks[pcs, :]
    smoothed_matrix = smoothed_matrix[pcs, :]

    peak_indices = np.argmax(smoothed_matrix, axis=1)
    peak_position_cm = peak_indices * config.bin_size + config.start
    sorted_order = np.argsort(peak_indices)
    peak_position_cm = peak_position_cm[sorted_order]
    smoothed_matrix = smoothed_matrix[sorted_order, :]
    spks = spks[sorted_order, :]

    offline_spks_pre = offline_spks_pre[pcs, :]
    offline_spks_post = offline_spks_post[pcs, :]
    offline_spks_pre = offline_spks_pre[sorted_order, :]
    offline_spks_post = offline_spks_post[sorted_order, :]

    if plot:
        plot_circular_distance_matrix(smoothed_matrix)

    offline_correlations(
        offline_spks_pre=offline_spks_pre,
        offline_spks_post=offline_spks_post,
        peak_position_cm=peak_position_cm,
        wheel_freeze=session.wheel_freeze,
    )


def get_place_cells(
    session: Cached2pSession,
    spks: np.ndarray,
    config: GrosmarkConfig,
    rewarded: bool | None,
    use_cache: bool = True,
    bin_occupancy_divide: bool = False,
    plot: bool = True,
    cache_file_additional_info: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From Grosmark et al.:
    The position of the animal during online running epochs on the 2-m-long run belts was binned into 100,
    2-cm spatial bins. For each cell, the as within spatial-bin firing rate was calculated across all bins
    based on its sparsified spike estimate vector, Ssp. This firing rate by position vector was subsequently
    smoothed with a 7.5-cm Gaussian kernel leading to the smoothed firing rate by position vector.
    In addition, for each cell, 2,000 shuffled smoothed firing rate by position vectors were computed for each
    cell following the per-lap randomized circular permutation of estimated activity vector, Ssp. For each cell,
    putative PFs were defined as those in which the observed smoothed firing rate by position vectors exceeded the
    99th percentile of their shuffled smoothed firing rate by position vectors as assessed on a per spatial-bin basis for
    at least five consecutive spatial bins.

    As an additional control, only those putative PFs in which the cell had a greater within-PF than outside-of-PF firing rates
    in at least 3 or 15% of laps (whichever was greater for each session) were considered bona fide PFs and kept for further analysis.

    Returns:
    - place_cell_mask: boolean mask of shape (n_cells,) where True indicates a place cell
    - smoothed_matrix: smoothed firing rate by position matrix of shape (n_cells, n_bins)
    """

    n_cells_total = spks.shape[0]

    sigma_cm = 7.5  # Desired smoothing in cm
    sigma_bins = sigma_cm / config.bin_size  # Convert to bin units

    n_shuffles = 2000
    if n_shuffles < 2000:
        warnings.warn(
            "n_shuffles is less than 2000. This may not be enough to get a good estimate of the place cell distribution."
        )

    all_trials = np.array(
        [
            activity_trial_position(
                trial=trial,
                flu=spks,
                wheel_circumference=get_wheel_circumference_from_rig("2P"),
                bin_size=config.bin_size,
                start=config.start,
                max_position=config.end,
                verbose=False,
                do_shuffle=False,
                threshold_speed=False if bin_occupancy_divide else True,
                bin_occupancy_divide=bin_occupancy_divide,
            )
            for trial in session.trials
            if trial_is_imaged(trial)
            and (rewarded is None or trial.texture_rewarded == rewarded)
        ]
    )

    smoothed_matrix = gaussian_filter1d(
        np.nanmean(all_trials, 0), sigma=sigma_bins, axis=1
    )

    if not cache_file_additional_info:
        get_cache_path = lambda variable_name: (
            SERVER_PATH
            / "viral_caches"
            / "place_cells"
            / variable_name
            / f"{session.mouse_name}_{session.date}_rewarded_{rewarded}_{config}_{variable_name}_BOD_{bin_occupancy_divide}.npy"
        )
    else:
        # e.g. train-test split
        get_cache_path = lambda variable_name: (
            SERVER_PATH
            / "viral_caches"
            / "place_cells"
            / variable_name
            / f"{session.mouse_name}_{session.date}_rewarded_{rewarded}_{config}_{cache_file_additional_info}_{variable_name}_BOD_{bin_occupancy_divide}.npy"
        )

    if use_cache and get_cache_path("place_threshold").exists():
        print("Found cached place threshold")
        place_threshold = np.load(get_cache_path("place_threshold"))
    else:
        print("No cached place threshold, calculating")
        # Create array of shape (n_shuffles, n_cells, n_bins)
        # where each (n_cells x bins) matrix is trial averaged but shuffled on a per-trial basis (as in Grosmark)
        # You can then apply percentiles along the first dimension to find "real" place cells
        shuffled_matrices = np.empty(
            (n_shuffles, n_cells_total, smoothed_matrix.shape[1])
        )
        for shuffle_idx in tqdm(
            range(n_shuffles), desc="Calculating shuffled place fields"
        ):
            shuffle_result = np.nanmean(
                np.array(
                    [
                        activity_trial_position(
                            trial=trial,
                            flu=spks,
                            wheel_circumference=get_wheel_circumference_from_rig("2P"),
                            bin_size=config.bin_size,
                            start=config.start,
                            max_position=config.end,
                            verbose=False,
                            do_shuffle=True,
                            threshold_speed=False if bin_occupancy_divide else True,
                            bin_occupancy_divide=bin_occupancy_divide,
                        )
                        for trial in session.trials
                        if trial_is_imaged(trial)
                        and (rewarded is None or trial.texture_rewarded == rewarded)
                    ]
                ),
                0,
            )
            smoothed_shuffle = gaussian_filter1d(
                shuffle_result, sigma=sigma_bins, axis=1
            )
            shuffled_matrices[shuffle_idx, :, :] = smoothed_shuffle

        place_threshold = np.nanpercentile(shuffled_matrices, 99, axis=0)
        np.save(get_cache_path("place_threshold"), place_threshold)

    # 5 if the bin size matches grosmark, otherwise adjust
    n_consecutive_trues = int((2 / config.bin_size) * 5)

    # Got a load of logic downstream that only works with odd numbers, as even numbers
    # don't have a center. Probably fine to do this but not ideal
    if n_consecutive_trues % 2 == 0:
        n_consecutive_trues += 1

    pcs = has_n_consecutive_trues(
        smoothed_matrix > place_threshold, n_consecutive_trues
    )

    print(f"percent place cells before extra check {np.sum(pcs) / n_cells_total}")

    pcs_additional = filter_additional_check(
        all_trials=all_trials[:, pcs, :],
        place_threshold=place_threshold[pcs, :],
        smoothed_matrix=smoothed_matrix[pcs, :],
        n_consecutive_trues=n_consecutive_trues,
    )

    # Cells that pass both the original and additional checks
    pcs_combined = pcs.copy()
    pcs_combined[pcs] = pcs_additional

    print(
        f"percent place cells after extra check {np.sum(pcs_combined) / n_cells_total}"
    )
    if plot:
        plot_place_cell_heatmap(
            smoothed_matrix=smoothed_matrix[pcs_combined, :],
            config=config,
        )
        plt.savefig(
            SERVER_PATH
            / "viral_plots"
            / "place_cells"
            / f"{session.mouse_name}_{session.date}_rewarded_{rewarded}.png"
        )

    np.save(get_cache_path("smoothed_matrix"), smoothed_matrix)
    np.save(get_cache_path("pcs_combined"), pcs_combined)
    return pcs_combined, smoothed_matrix, place_threshold


def plot_speed(
    session: Cached2pSession, rewarded: bool | None, config: GrosmarkConfig
) -> None:
    speeds = [
        np.array(
            [
                speed.speed
                for speed in get_speed_positions(
                    degrees_to_cm(
                        np.array(trial.rotary_encoder_position),
                        get_wheel_circumference_from_rig("2P"),
                    ),
                    config.start,
                    config.end,
                    config.bin_size,
                    sampling_rate=30,
                )
            ]
        )
        for trial in session.trials
        if trial_is_imaged(trial)
        and (rewarded is None or trial.texture_rewarded == rewarded)
    ]

    plt.figure()
    shaded_line_plot(
        np.array(speeds),
        x_axis=np.arange(config.start, config.end, config.bin_size),
        color="red",
        label="speed",
    )
    plt.xlabel("Position (cm)")
    plt.ylabel("Speed (cm/s)")


def offline_correlations(
    offline_spks_pre: np.ndarray,
    offline_spks_post: np.ndarray,
    peak_position_cm: np.ndarray,
    wheel_freeze: WheelFreeze,
) -> None:
    """Correlates offline activity with running sequences. There used to be a lot of alternative definitions of offline activity
    that can be found in the commit history (e.g. d2e7852f54282a52722767e52cca1ab71e56851b) if you need them

    From Grosmark:
    For pair-wise reactivation analysis, the run PF peak distance between pairs of PCs was compared to their offline firing-rate
    Pearsons correlation coefficients in either the pre or post epochs. For calculating offline firing rate correlations,
    the sparsified binary spike estimate vectors Ssp were restricted to periods of immobility during either the pre or post epoch
    and convolved with a 150-ms Gaussian kernel.

    """
    movement_pre, movement_post = get_movement_bool(wheel_freeze=wheel_freeze)

    offline_spks_pre = offline_spks_pre[:, ~movement_pre]
    offline_spks_post = offline_spks_post[:, ~movement_post]

    pre_corrs_real = get_offline_correlation_matrix(
        offline=offline_spks_pre, wheel_freeze=True, do_shuffle=False, plot=True
    )
    pre_corrs_shuffled = get_offline_correlation_matrix(
        offline=offline_spks_pre, wheel_freeze=True, do_shuffle=True, plot=False
    )
    post_corrs_real = get_offline_correlation_matrix(
        offline=offline_spks_post, wheel_freeze=True, do_shuffle=False, plot=True
    )
    post_corrs_shuffled = get_offline_correlation_matrix(
        offline=offline_spks_post, wheel_freeze=True, do_shuffle=True, plot=False
    )
    plt.figure()
    plt.xlabel("Distance between peaks")
    plt.ylabel("Average pearson correlation")
    r_pre, p_pre = correlations_vs_peak_distance(
        pre_corrs_real,
        peak_position_cm=peak_position_cm,
        colour="blue",
        label="pre-epoch",
        plot=True,
    )
    r_post, p_post = correlations_vs_peak_distance(
        post_corrs_real,
        peak_position_cm=peak_position_cm,
        colour="red",
        label="post-epoch",
        plot=True,
    )
    plt.legend()
    plt.title(
        f"pre: r={r_pre:.2f}, p={p_pre:.2f}\npost: r={r_post:.2f}, p={p_post:.2f}"
    )
    # plt.savefig("plots/correlations_peak_distance.png", dpi=300)


def get_offline_correlation_matrix(
    offline: np.ndarray,
    wheel_freeze: bool,
    do_shuffle: bool = False,
    plot: bool = True,
) -> np.ndarray:
    """Reproducing Grosmark et al. figure 4. c/d."""
    if not wheel_freeze:
        n_trials = offline.shape[0]
        all_corrs = []
        for trial in range(n_trials):
            trial_matrix = offline[trial, :, :]
            if do_shuffle:
                trial_matrix = shuffle_rows(trial_matrix)
            # 150-ms kernel convolution
            ITI_trial = gaussian_filter1d(trial_matrix, sigma=4.5, axis=1)
            all_corrs.append(cross_correlation_pandas(ITI_trial.T))
    else:
        # 150-ms kernel convolution
        if do_shuffle:
            offline = shuffle_rows(offline)
        offline = gaussian_filter1d(offline, sigma=4.5, axis=1)
        all_corrs = [cross_correlation_pandas(offline.T)]

    # Silent cells, correctly are correlated to NaN, so we take the mean across trials and ignore NaNs
    corrs = np.nanmean(np.array(all_corrs), 0)

    if plot:
        plt.figure()
        plt.title("shuffled" if do_shuffle else "real")
        if do_shuffle:
            np.random.shuffle(corrs)
        plt.imshow(
            gaussian_filter1d(remove_diagonal(corrs), sigma=2.5),
            vmin=0,
            vmax=0.2,
            cmap="bwr",
        )
    return corrs


def correlations_vs_peak_distance(
    corrs: np.ndarray,
    peak_position_cm: np.ndarray,
    colour: str | None = None,
    label: str | None = None,
    plot: bool = False,
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """Figure 4. e/f in Grosmark. Computes the pairwise offline correlations between neurons as a function of the
        distance between their place field peaks.
    Args:
        corrs: the Pearson correlation matrix between neurons during offline periods of shape (n_cells, n_cells)
        peak_position_cm: the position of the peak firing rate of each neuron in cm
        colour: colour for the plot
        label: label for the plot
        plot: whether to plot
    """

    n_cells = corrs.shape[0]
    peak_distances = []
    cell_corrs = []

    for i, j in itertools.combinations(range(n_cells), r=2):
        assert i != j
        cell1_peak = peak_position_cm[i]
        cell2_peak = peak_position_cm[j]
        peak_distances.append(abs(cell1_peak - cell2_peak))
        cell_corrs.append(corrs[i, j])

    peak_distances = np.array(peak_distances)
    cell_corrs = np.array(cell_corrs)

    x = []
    y = []
    bin_starts = np.arange(0, 100, 5)

    bin_width = bin_starts[1] - bin_starts[0]
    for bin_start in bin_starts:
        in_bin = np.logical_and(
            peak_distances >= bin_start, peak_distances < bin_start + bin_width
        )
        x.append(bin_start)
        y.append(np.nanmean(cell_corrs[in_bin]))

    if plot:
        plt.figure()
        plt.plot(np.array(x), y, color=colour, label=label)
        plt.legend()

    # Need to put this back if grosmarking
    # r, p = pearsonr(x, y)
    # return r, p
    x = np.array(x)
    y = np.array(y)

    y = interpolate_nans_vector(y)

    m = compute_linear_slope((x / 60), y / y[0])

    return m, (np.array(x), np.array(y))


def plot_circular_distance_matrix(smoothed_matrix: np.ndarray) -> None:

    plt.figure()
    plt.imshow(circular_distance_matrix(smoothed_matrix), cmap="RdYlBu")
    plt.colorbar()
    plt.ylabel("Cell number")
    plt.xlabel("Cell number")


def plot_place_cell_heatmap(
    smoothed_matrix: np.ndarray,
    config: GrosmarkConfig,
) -> None:
    plt.figure()
    plt.imshow(
        zscore(sort_matrix_peak(smoothed_matrix), axis=1, nan_policy="omit"),
        aspect="auto",
        cmap="bwr",
        vmin=-1,
        vmax=2,
    )

    plt.xlabel("Corridor position (cm)")
    plt.ylabel("Cell number")

    plt.xticks(
        np.linspace(0, smoothed_matrix.shape[1], 5),
        [str(x) for x in np.linspace(config.start, config.end, 5).astype(int)],
    )

    plt.colorbar()
    plt.tight_layout()


def filter_additional_check(
    all_trials: np.ndarray,
    place_threshold: np.ndarray,
    smoothed_matrix: np.ndarray,
    n_consecutive_trues: int,
) -> np.ndarray:
    """Runs the following check from the Grosmark paper:
    As an additional control, only those putative PFs in which the cell had a greater within-PF than outside-of-PF firing rates
    in at least 3 or 15% of laps (whichever was greater for each session) were considered bona fide PFs and kept for further analysis.
    """

    centers = find_n_consecutive_trues_center(
        smoothed_matrix > place_threshold, n_consecutive_trues
    )

    n_trials, n_cells, n_bins = all_trials.shape

    valid_pcs = np.array([False] * n_cells)
    for cell in range(n_cells):
        center = centers[cell]
        assert center + math.ceil(n_consecutive_trues / 2) <= n_bins
        assert center - math.floor(n_consecutive_trues / 2) >= 0

        cell_place_field = np.array([False] * n_bins)
        cell_place_field[
            center
            - math.floor(n_consecutive_trues) : center
            + math.ceil(n_consecutive_trues)
        ] = True
        cell_out_of_place_field = np.logical_not(cell_place_field)

        cell_place_activity = all_trials[:, cell, cell_place_field]
        cell_not_place_activity = all_trials[:, cell, cell_out_of_place_field]
        count = 0
        for trial in range(n_trials):
            if np.nanmean(cell_place_activity[trial, :]) > np.nanmean(
                cell_not_place_activity[trial, :]
            ):
                count += 1

        if count / n_trials > 0.15:
            valid_pcs[cell] = True

    return np.array(valid_pcs)


def circular_distance_matrix(activity_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the pairwise circular distance matrix of place field (PF) peaks.

    Parameters:
    - activity_matrix: np.array of shape (n_cells, n_positions)
      Each row corresponds to the neural activity of a single cell across positions.

    Returns:
    - circular_dist_matrix: np.array of shape (n_cells, n_cells)
      The pairwise circular peak distance matrix.

    I haven't unit tested this but the results look good.
    """

    n_positions = activity_matrix.shape[1]  # Number of spatial bins

    # Step 1: Find peak firing positions for each cell
    peak_positions = np.argmax(activity_matrix, axis=1)

    # Step 2: Compute pairwise circular distances
    pairwise_distances = cdist(
        peak_positions[:, None], peak_positions[:, None], metric="cityblock"
    )

    # Step 3: Apply circular distance correction
    circular_dist_matrix = np.minimum(
        pairwise_distances, n_positions - pairwise_distances
    )

    return circular_dist_matrix


def batch_runner() -> None:

    cache_files = list(CACHE_PATH.glob("*.json"))
    data_type = "denoised"
    use_cache = False

    for cache_file in cache_files:
        print("Processing", cache_file)
        file_parts = cache_file.stem.split("_")
        date = file_parts[1]
        mouse = file_parts[0]
        if mouse not in ["J034", "J035", "J037", "J038"]:
            continue
        s2p_path = TIFF_UMBRELLA / date / mouse / "suite2p" / "plane0"
        cached_session = Cached2pSession.model_validate_json(cache_file.read_text())

        with open(
            SERVER_PATH / "viral_caches" / "cached_2p" / f"{mouse}_{date}.json", "r"
        ) as f:
            session = Cached2pSession.model_validate_json(f.read())

        print(f"Total number of trials: {len(session.trials)}")
        print(
            f"number of trials imaged {len([trial for trial in session.trials if trial_is_imaged(trial)])}"
        )

    if (HERE / f"{mouse}_{date}_dff.npy").exists():
        dff = np.load(HERE / f"{mouse}_{date}_dff.npy")
        spks = np.load(HERE / f"{mouse}_{date}_spks.npy")
        denoised = np.load(HERE / f"{mouse}_{date}_denoised.npy")
    else:
        dff, spks, denoised = load_imaging_data(mouse, date)
        np.save(HERE / f"{mouse}_{date}_dff.npy", dff)
        np.save(HERE / f"{mouse}_{date}_spks.npy", spks)
        np.save(HERE / f"{mouse}_{date}_denoised.npy", denoised)

        assert (
            max(
                trial.states_info[-1].closest_frame_start
                for trial in session.trials
                if trial.states_info[-1].closest_frame_start is not None
            )
            < dff.shape[1]
        ), "Tiff is too short"

        is_unsupervised = session_is_unsupervised(session)

        for rewarded in [None, True, False]:
            try:
                grosmark_place_field(
                    session,
                    spks if data_type == "spks" else denoised,
                    rewarded=None if is_unsupervised else rewarded,
                    config=grosmark_config,
                    cache_file_additional_info=data_type,
                    use_cache=use_cache,
                )
            except Exception as e:
                print(f"Error processing {mouse} {date} rewarded={rewarded}: {e}")

            if is_unsupervised:
                break
