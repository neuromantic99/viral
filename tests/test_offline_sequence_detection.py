import pytest
import numpy as np
from unittest.mock import Mock

from viral.offline_sequence_detection import array_bin_mean


def test_bin_offline_activity_bin_size_1() -> None:
    input = np.array([[0, 0, 0, 1], [1, 0, 0, 0]])
    bayesian_config = Mock()
    bayesian_config.configure_mock(bin_size_time_offline=1)  # frame
    result = array_bin_mean(input, bayesian_config.bin_size_time_offline)
    assert np.array_equal(input, result)


def test_bin_offline_activity_bin_size_2() -> None:
    input = np.array([[0, 0, 0, 1], [1, 0, 0, 0]])
    bayesian_config = Mock()
    bayesian_config.configure_mock(bin_size_time_offline=2)  # frame
    output = array_bin_mean(input, bayesian_config.bin_size_time_offline)
    expected = np.array([[0, 0.5], [0.5, 0]])
    assert np.array_equal(expected, output)
