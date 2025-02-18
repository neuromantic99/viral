from typing import List
import pytest
import numpy as np
from unittest.mock import Mock, patch

from viral.speed_cells import activity_trial_speed, get_speed_activity


def test_activity_trial_speed() -> None:
    trial = Mock()
    dff = np.array([[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1]])  # 2 cells with signal
    trial_frames = np.array([0, 1, 2, 3, 4, 5])
    wheel_circumference = 10  # Doesn't matter anyway as it would have been used in the patched function get_frame_position
    bin_size = 5
    mocked_frame_positions = np.array([[0, 0], [1, 1], [2, 2], [3, 4], [4, 6], [5, 7]])
    mocked_frame_speeds = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 0]])

    # expected_speeds = np.array([30, 30, 30, 30, 30, 0])
    # converting from cm/frame to cm/second (@30fps)
    # expecting 6 bins, 4 of which will be NaN

    expected = np.array(
        [
            [1, np.nan, np.nan, np.nan, np.nan, 1],
            [1, np.nan, np.nan, np.nan, np.nan, 0],
        ]
    )

    with patch(
        "viral.speed_cells.get_frame_position",
        side_effect=mocked_frame_positions,
    ):
        with patch(
            "viral.speed_cells.get_speed_frame", return_value=mocked_frame_speeds
        ):
            result = activity_trial_speed(
                trial, dff, trial_frames, wheel_circumference, bin_size
            )
    assert np.array_equal(result, expected, equal_nan=True)


# TODO: more testing needed, as in tests for two_photon.py?

# TODO: Test the padding functions?


def test_get_speed_activity():

    with patch("viral.speed_cells.random.sample", return_value = [0,1])
    @patch("your_module.random.sample", return_value=[0, 1])  # Control train/test split
    @patch("your_module.get_signal_for_trials", return_value=np.array([[0.1, 0.2], [0.3, 0.4]]))  # Mock extracted signals
    @patch("your_module.pad_to_max_length_bins", side_effect=lambda x: x)  # Identity function
    @patch("your_module.normalize", side_effect=lambda x, axis: x)  # Identity function
    @patch("your_module.activity_trial_speed", return_value=np.array([[0.1, 0.3], [0.2, 0.4]]))  # Controlled output
    trials = [Mock(), Mock()]  # Mock trial objects
    aligned_trial_frames = np.array([[0, 10], [5, 15]])
    dff = np.random.rand(2, 10)
    wheel_circumference = 10.0
    bin_size = 5

    result = get_speed_activity(trials, aligned_trial_frames, dff, wheel_circumference, bin_size)
    
    assert isinstance(result, np.ndarray)  # Ensure it returns an array
    assert result.shape[1] == 2  # Assuming two bins from mocks
    assert np.all(result >= 0)  # Activity values should be non-negative

    # old approahc
    trial = Mock()
    dff = np.array([[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1]])  # 2 cells with signal
    trial_frames = np.array([0, 1, 2, 3, 4, 5])
    wheel_circumference = 10  # Doesn't matter anyway as it would have been used in the patched function get_frame_position
    bin_size = 5
    mocked_frame_positions = np.array([[0, 0], [1, 1], [2, 2], [3, 4], [4, 6], [5, 7]])
    mocked_frame_speeds = np.array([[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 0]])

    expected_speed_dff = np.array(
        [
            [1, np.nan, np.nan, np.nan, np.nan, 1],
            [1, np.nan, np.nan, np.nan, np.nan, 0],
        ]
    )
    
    with patch("viral.rastermap_utils.get_signal_for_trials", return_value=dff):
        # return the input array, as we have only one trial and hence there is no need for padding
        with patch("viral.rastermap_utils.pad_to_max_length_bins", side_effect=lambda x: x, axis=x):

            with patch("viral.rastermap_utils.normalize", side_effect=lambda x: x, axis=x):

    assert np.array_equal(result, expected)

    # look at jimmy's test
    # look at how it is dealing with NaN values
    # look at if the speed bins will be correct after padding
