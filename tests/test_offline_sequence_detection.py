import pytest
import numpy as np
from unittest.mock import Mock

from viral.offline_sequence_detection import array_bin_mean, find_pse_events


# TODO: remove? currently not used in offline_sequence_detection.py
# def test_bin_offline_activity_bin_size_1() -> None:
#     input = np.array([[0, 0, 0, 1], [1, 0, 0, 0]])
#     bayesian_config = Mock()
#     bayesian_config.configure_mock(bin_size_time_offline=1)  # frame
#     result = array_bin_mean(input, bayesian_config.bin_size_time_offline)
#     assert np.array_equal(input, result)


# def test_bin_offline_activity_bin_size_2() -> None:
#     input = np.array([[0, 0, 0, 1], [1, 0, 0, 0]])
#     bayesian_config = Mock()
#     bayesian_config.configure_mock(bin_size_time_offline=2)  # frame
#     output = array_bin_mean(input, bayesian_config.bin_size_time_offline)
#     expected = np.array([[0, 0.5], [0.5, 0]])
#     assert np.array_equal(expected, output)


def test_find_pse_events_easy() -> None:
    population_vector = np.array([-1, 0, 3.5, 0, -1])
    reactivation = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
        ]
    )
    bayesian_config = Mock()
    bayesian_config.configure_mock(
        peak_threshold=2, edge_threshold=1, event_duration=(3, 3)
    )
    events = find_pse_events(population_vector, reactivation, bayesian_config)
    expected_events = [(1, 3)]
    assert events == expected_events


def test_find_pse_events_no_events() -> None:
    population_vector = np.array([0, 0, 0, 0, 0])
    reactivation = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
        ]
    )
    bayesian_config = Mock()
    bayesian_config.configure_mock(
        peak_threshold=2, edge_threshold=1, event_duration=(3, 3)
    )
    events = find_pse_events(population_vector, reactivation, bayesian_config)
    expected_events = []
    assert events == expected_events


def test_find_pse_events_event_too_short() -> None:
    population_vector = np.array([-1, 0, 3.5, 0, -1])
    reactivation = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
        ]
    )
    bayesian_config = Mock()
    bayesian_config.configure_mock(
        peak_threshold=2, edge_threshold=1, event_duration=(4, 5)
    )
    events = find_pse_events(population_vector, reactivation, bayesian_config)
    expected_events = []
    assert events == expected_events


def test_find_pse_events_event_pcs_no_pass() -> None:
    """PSE event duration is ok, but not enough distinct PCs fired."""
    population_vector = np.array([-1, 0, 3.5, 0, -1])
    reactivation = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 0, 1, 0],
        ]
    )
    bayesian_config = Mock()
    bayesian_config.configure_mock(
        peak_threshold=2, edge_threshold=1, event_duration=(3, 3)
    )
    events = find_pse_events(population_vector, reactivation, bayesian_config)
    expected_events = []
    assert events == expected_events
