import pytest
import numpy as np
from unittest.mock import Mock, patch

from viral.sequence_utils import (
    detect_candidate_events,
    merge_close_events,
    filter_candidate_events_by_duration,
    filter_candidate_events_by_inter_event_time,
    additional_pc_check,
    bin_for_classification,
)


# TODO: naive question, but what should happen if only ever above peak threshold? filtered out by duration only?
def test_detect_candidate_events() -> None:
    # the second-to-last candidate does not exceed the peak threshold, so should not appear
    # the last candidate exceeds the peak threshold but doesn't have a second edge threshold crossing, so should not appear
    population_vector = np.array(
        [
            0,
            0,
            0,
            1.5,
            2,
            0.5,
            0,
            0,
            1.2,
            1.5,
            2.5,
            0.7,
            0,
            0,
            1.5,
            1.8,
            1.5,
            0,
            0,
            1.8,
            2,
            0.7,
            0,
            0,
            0.9,
            2.5,
            2.1,
        ]
    )
    expected_events = [(3, 5), (8, 11), (19, 21)]

    peak_threshold = 2.0
    edge_threshold = 1.0
    mean = 0
    std = 1
    config = Mock()
    config.configure_mock(
        peak_threshold=peak_threshold,
        edge_threshold=edge_threshold,
        event_duration=(2, 5),
    )

    with patch("numpy.std", return_value=std):
        with patch("numpy.mean", return_value=mean):
            candidate_events = detect_candidate_events(population_vector, config)

    assert candidate_events == expected_events


# TODO: think about whether these candidate events should be considered or not!!!
def test_detect_candidate_events_no_edge_before_peak() -> None:
    population_vector = np.array(
        [
            2,
            0.5,
            0,
            0,
        ]
    )
    expected_events = []

    peak_threshold = 2.0
    edge_threshold = 1.0
    mean = 0
    std = 1
    config = Mock()
    config.configure_mock(
        peak_threshold=peak_threshold,
        edge_threshold=edge_threshold,
        event_duration=(2, 5),
    )

    with patch("numpy.std", return_value=std):
        with patch("numpy.mean", return_value=mean):
            candidate_events = detect_candidate_events(population_vector, config)

    assert candidate_events == expected_events


# TODO: what about something like this?
# should there be a rule that it can't jump over the edge threshold and peak threshold with the same frame value?
# population_vector = np.array(
#     [
#         0,
#         0,
#         0,
#         0,
#         2,
#         0.5,
#         0,
#         0,
#     ]
# )


def test_merge_close_events() -> None:
    candidate_events = [
        (0, 3),
        (5, 8),
        (15, 17),
        (18, 22),
        (30, 35),
        (50, 55),
        (57, 60),
        (65, 70),
        (80, 85),
        (91, 95),
    ]
    # merge less than 6 frames apart
    expected_merged_events = [(0, 8), (15, 22), (30, 35), (50, 70), (80, 85), (91, 95)]

    merged_events = merge_close_events(candidate_events)
    assert merged_events == expected_merged_events


def test_filter_candidate_events_by_inter_event_time() -> None:
    candidate_events = [
        (0, 3),
        (5, 8),
        (13, 17),
        (18, 22),
        (30, 35),
        (50, 55),
        (57, 60),
        (65, 70),
        (80, 85),
        (91, 95),
    ]
    # reject the following event if it is more than 6 frames apart
    min_inter_event_time_frames = 6
    # expected = [(0, 3), (30, 35), (50, 55), (80, 85), (91, 95)]
    # see TODO comment in the actual function
    expected = [(0, 3), (13, 17), (30, 35), (50, 55), (65, 70), (80, 85), (91, 95)]

    result = filter_candidate_events_by_inter_event_time(
        candidate_events, min_inter_event_time_frames
    )
    assert result == expected


def test_filter_candidate_events_by_duration() -> None:
    candidate_events = [
        (0, 2),
        (5, 8),
        (15, 17),
        (18, 22),
        (30, 35),
        (50, 55),
        (57, 60),
        (65, 70),
        (80, 87),
        (91, 95),
    ]
    event_duration_thresholds = (4, 5)
    expected = [
        (5, 8),
        (18, 22),
        (57, 60),
        (91, 95),
    ]
    result = filter_candidate_events_by_duration(
        candidate_events=candidate_events,
        event_duration_thresholds=event_duration_thresholds,
    )
    assert result == expected


def test_additional_pc_check() -> None:
    filtered_events = [
        (0, 5),
        (10, 15),
        (20, 25),
    ]
    ssp = np.zeros((10, 30))
    ssp[2:7, 2:5] = 1
    ssp[0:5, 10:12] = 1
    ssp[4:9, 20:23] = 1
    expected = [
        (0, 5),
        (10, 15),
        (20, 25),
    ]

    additionally_checked = additional_pc_check(filtered_events, ssp)
    assert additionally_checked == expected


def test_additional_pc_check_one_non_pass() -> None:
    filtered_events = [
        (0, 5),
        (10, 15),
        (20, 25),
    ]
    # one doesn't have any pc activity
    ssp = np.zeros((10, 30))
    ssp[2:7, 2:5] = 1
    ssp[4:9, 20:23] = 1
    expected = [
        (0, 5),
        (20, 25),
    ]

    additionally_checked = additional_pc_check(filtered_events, ssp)
    assert additionally_checked == expected

    filtered_events = [
        (0, 5),
        (10, 15),
        (20, 25),
    ]
    # one doesn't have enough place cells
    ssp = np.zeros((10, 30))
    ssp[3:6, 2:5] = 1
    ssp[0:5, 10:12] = 1
    ssp[4:9, 20:23] = 1
    expected = [
        (10, 15),
        (20, 25),
    ]

    additionally_checked = additional_pc_check(filtered_events, ssp)
    assert additionally_checked == expected


# TODO: what do we actually expect?
# TODO: do we want the upper edge to be included?
def test_bin_for_classification() -> None:
    position_array = np.array([0, 5, 10, 15, 20, 25, 30])
    bayesian_config = Mock()
    bayesian_config.configure_mock(
        start_spatial=0,
        end_spatial=30,
        bin_size_spatial=10,
    )
    binned_positions = bin_for_classification(position_array, bayesian_config)
    expected_binned_positions = np.array([0, 0, 1, 1, 2, 2, 2])
    assert np.array_equal(binned_positions, expected_binned_positions)


def test_bin_for_classification_code_rabbit() -> None:
    # Issue raised by CodeRabbit:
    # Example: max position ≈ 179 cm, bin_size=5 → max // bin_size = 35.
    # Valid bin indices are 0..35, but you pass n_bins=35, so clipping is to 0..34.
    # -> perhaps Code Rabbit commented on an old commit?
    position_array = np.array([0, 10, 20, 55, 78, 100, 120, 179])
    bayesian_config = Mock()
    bayesian_config.configure_mock(
        start_spatial=0,
        end_spatial=180,
        bin_size_spatial=5,
    )
    binned_positions = bin_for_classification(position_array, bayesian_config)
    expected_binned_positions = np.array([0, 2, 4, 11, 15, 20, 24, 35])
    assert np.array_equal(binned_positions, expected_binned_positions)
