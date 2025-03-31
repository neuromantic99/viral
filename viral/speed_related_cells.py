import numpy as np
import sys
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))


def shuffle_neuron_spikes(spks: np.ndarray) -> np.ndarray:
    """Shuffle spikes within each neuron independently."""
    spks_shuffled = spks.copy()
    for i in range(spks.shape[0]):  # Loop over neurons
        np.random.shuffle(spks_shuffled[i, :])  # Shuffle each neuron's spikes
    return spks_shuffled


def bin_activity(
    spks: np.ndarray, matrix: np.ndarray, start: int, max: int, bin_size: float
) -> np.ndarray:
    """Bin activity of neurons into bins determined by matrix (e.g. speed, position)"""
    spks_matrix = list()
    # Including upper bound
    bins = np.arange(start, max + bin_size, bin_size)
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        print(bin_start)
        if i == len(bins) - 2:
            # Include upper bound in last bin
            frame_idx_bin = np.where((matrix >= bin_start) & (matrix <= bin_end))[0]
        else:
            frame_idx_bin = np.where((matrix >= bin_start) & (matrix < bin_end))[0]
        if frame_idx_bin.size == 0:
            spks_matrix.append(np.full(spks.shape[0], np.nan))
        else:
            spks_bin = spks[:, frame_idx_bin]
            spks_matrix.append(np.nanmean(spks_bin, axis=1))
    return np.stack(spks_matrix, axis=1)


def get_mean_value_per_bin(start: int, max: int, bin_size: float) -> np.ndarray:
    bin_edges = np.arange(start, max, bin_size)
    return np.array([(start + start + bin_size) / 2 for start in bin_edges])


def get_speed_pearsonr(
    activity_speed: np.ndarray, mean_speed_per_bin: np.ndarray
) -> np.ndarray:
    assert (
        activity_speed.shape[1] == mean_speed_per_bin.shape[0]
    ), f"There is {activity_speed.shape[1]} speed bins, but mean speeds for {mean_speed_per_bin.shape} speed bins"
    if np.isnan(activity_speed).any():
        print("Warning! NaN in activity_speed matrix")
    return np.array(
        [
            pearsonr(activity_speed[i, :], mean_speed_per_bin)[0]
            for i in range(activity_speed.shape[0])
        ]
    )


def get_speed_spearmanr(
    activity_speed: np.ndarray, mean_speed_per_bin: np.ndarray
) -> np.ndarray:
    assert (
        activity_speed.shape[1] == mean_speed_per_bin.shape[0]
    ), f"There is {activity_speed.shape[1]} speed bins, but mean speeds for {mean_speed_per_bin.shape} speed bins"
    if np.isnan(activity_speed).any():
        print("Warning! NaN in activity_speed matrix")
    return np.array(
        [
            spearmanr(activity_speed[i, :], mean_speed_per_bin)[0]
            for i in range(activity_speed.shape[0])
        ]
    )


def speed_related_cells(spks: np.ndarray, speed: np.ndarray, bin_size: int = 2) -> None:
    """Plot speed related cells (correlation histograms)"""
<<<<<<< HEAD
    spks_shuffled = shuffle_neuron_spikes(spks)
=======
>>>>>>> main
    max_speed = np.max(speed)
    start = 0
    end = np.ceil(max_speed / bin_size) * bin_size
    mean_speed_per_bin = get_mean_value_per_bin(start, end, bin_size)
    spks_binned = bin_activity(spks, speed, start, end, bin_size)
<<<<<<< HEAD
    spks_shuffled_binned = bin_activity(spks_shuffled, speed, start, end, bin_size)
    correlation_distribution = get_speed_spearmanr(spks_binned, mean_speed_per_bin)
    correlation_distribution_shuffled = get_speed_spearmanr(
        spks_shuffled_binned, mean_speed_per_bin
=======
    spks_binned_shuffled = shuffle_neuron_spikes(spks_binned)
    correlation_distribution = get_speed_spearmanr(spks_binned, mean_speed_per_bin)
    correlation_distribution_shuffled = get_speed_spearmanr(
        spks_binned_shuffled, mean_speed_per_bin
>>>>>>> main
    )
    # neuron_sort = np.argsort(pearson_distribution, axis=0)
    percentile_99 = np.percentile(correlation_distribution_shuffled, 99)
    percentile_1 = np.percentile(correlation_distribution_shuffled, 1)
    bins = 20
    print(f"99th percentile of shuffled correlations: {percentile_99:.4f}")
    plt.figure(figsize=(10, 6))
    plt.hist(correlation_distribution, color="blue", bins=bins, label="Real")
    plt.hist(
        correlation_distribution_shuffled,
        color="red",
        bins=bins,
        alpha=0.6,
        label="Shuffled",
    )
    plt.axvline(0, color="gray", linestyle="--")
    plt.axvline(
        percentile_99, color="red", linestyle="--", label="99th percentile of shuffled"
    )
    plt.axvline(
        percentile_1, color="red", linestyle="--", label="99th percentile of shuffled"
    )
    plt.title("Neuron-Speed Correlation")
    plt.ylabel("Neurons")
    plt.xlabel("Correlation Coefficient")
    plt.legend()
    plt.show()
