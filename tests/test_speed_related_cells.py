import numpy as np
from viral.speed_related_cells import bin_activity, get_mean_value_per_bin


def test_bin_activity() -> None:
    spks = np.array([[0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0]])
    position = np.array([0, 2, 2, 3, 4, 4, 4, 4])
    bin_start = 0
    bin_end = 4
    bin_size = 2
    expected = np.array([[0, 4 / 7], [1, 3 / 7]])
    result = bin_activity(spks, position, bin_start, bin_end, bin_size)
    print(expected.shape)
    print(result.shape)
    print(result.dtype)
    print(expected.dtype)
    assert np.all(np.isclose(result, expected))


def test_get_mean_value_per_bin() -> None:
    start = 0
    max = 5
    bin_size = 1
    expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    result = get_mean_value_per_bin(start, max, bin_size)
    assert np.array_equal(expected, result)
