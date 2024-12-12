import numpy as np

from viral.two_photon import sort_matrix_peak


def test_sort_matrix_peak_no_change() -> None:
    input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(3, 3)
    result = sort_matrix_peak(input)
    assert np.array_equal(input, result)


def test_sort_matrix_peak_order_change() -> None:
    input = np.array([1, 2, 3, 10, 5, 6, 7, 8, 9]).reshape(3, 3)
    result = sort_matrix_peak(input)
    expected = np.array([10, 5, 6, 1, 2, 3, 7, 8, 9]).reshape(3, 3)
    assert np.array_equal(expected, result)


def test_sort_matrix_peak_order_change_not_square() -> None:
    input = np.array([2, 3, 10, 1, 100, 7, 8, 9]).reshape(2, 4)
    result = sort_matrix_peak(input)
    expected = np.array([100, 7, 8, 9, 2, 3, 10, 1]).reshape(2, 4)
    assert result.shape == input.shape
    assert np.array_equal(expected, result)
