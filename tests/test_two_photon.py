from typing import List
import numpy as np
from unittest.mock import patch


from viral.imaging_utils import activity_trial_position
from viral.models import StateInfo, TrialInfo
from viral.two_photon import (
    get_position_activity,
)
from viral.utils import sort_matrix_peak

WHEEL_CIRCUMFERENCE = 10


def get_state_info(closest_frame: int) -> StateInfo:
    """Only cares about the closest frame"""
    return StateInfo(
        name="trigger_panda",
        start_time=0,
        end_time=0,
        start_time_daq=0,
        end_time_daq=0,
        closest_frame_start=closest_frame,
        closest_frame_end=closest_frame,
    )


def create_test_trial(
    rotary_encoder_position: List, states_info: List[StateInfo]
) -> TrialInfo:
    return TrialInfo(
        trial_start_time=0,
        trial_end_time=1,
        pc_timestamp="timestamp",
        states_info=states_info,
        events_info=[],
        rotary_encoder_position=rotary_encoder_position,
        texture="texture",
        texture_rewarded=True,
    )


def test_activity_trial_position_simple() -> None:

    # Evenly spaced full corridor
    rotary_encoder_position = list(range(200 * 360 // WHEEL_CIRCUMFERENCE))

    trial = create_test_trial(
        rotary_encoder_position=rotary_encoder_position,
        states_info=[get_state_info(i) for i in range(len(rotary_encoder_position))],
    )

    # cells x frames
    dff = np.ones((10, len(trial.rotary_encoder_position)))
    result = activity_trial_position(
        trial, dff, WHEEL_CIRCUMFERENCE, smoothing_sigma=None
    )
    assert np.array_equal(np.ones((dff.shape[0], 160)), result)


def test_activity_trial_position_multiple_frames_per_bin() -> None:

    # First three frames are in the first 10 cm, last is between 10 and 20
    rotary_encoder_position = [0, 180, 359, 700]
    states_info = [get_state_info(i) for i in [0, 1, 2, 3]]

    # cells x time
    dff = np.array([[5, 10, 15, 20], [1, 2, 3, 4]])

    trial = create_test_trial(
        rotary_encoder_position=rotary_encoder_position, states_info=states_info
    )

    # cells x frames
    result = activity_trial_position(
        trial,
        dff,
        WHEEL_CIRCUMFERENCE,
        bin_size=10,
        start=0,
        max_position=20,
        smoothing_sigma=None,
    )
    expected = np.array([[10, 20], [2, 4]])
    assert np.array_equal(result, expected)


def test_activity_trial_position_uneven_frame_spacing() -> None:

    # First three frames are in the first 10 cm, last is between 10 and 20
    rotary_encoder_position = [0, 180, 359, 700]
    states_info = [get_state_info(i) for i in [0, 0, 1, 3]]

    # cells x time
    dff = np.array([[5, 10, 15, 20], [1, 2, 3, 4]])

    trial = create_test_trial(
        rotary_encoder_position=rotary_encoder_position, states_info=states_info
    )

    # cells x frames
    result = activity_trial_position(
        trial,
        dff,
        WHEEL_CIRCUMFERENCE,
        bin_size=10,
        start=0,
        max_position=20,
        smoothing_sigma=None,
    )
    # Expect the first bin to be the mean of the first two frames
    # Expect the second bin to be only the final frame
    expected = np.array([[7.5, 20], [1.5, 4]])
    assert np.array_equal(result, expected)


def test_get_position_activity_all_trials_same_dont_reorder() -> None:

    # Two frames in the first bin, two in the second
    rotary_encoder_position = [0, 180, 361, 700]
    states_info = [get_state_info(i) for i in [0, 1, 2, 3]]

    trials = [
        create_test_trial(
            rotary_encoder_position=rotary_encoder_position, states_info=states_info
        )
        for _ in range(100)
    ]

    # cells x time
    dff = np.array([[5, 10, 15, 20], [1, 2, 3, 4]])

    # All the trials are the same so this should be the same result as activity_trial_position
    expected = np.array([[7.5, 17.5], [1.5, 3.5]])

    # Remove the normalize step to make the test easier to interpret
    with patch("viral.two_photon.normalize", side_effect=lambda x, axis: x):

        result = get_position_activity(
            trials=trials,
            dff=dff,
            wheel_circumference=WHEEL_CIRCUMFERENCE,
            bin_size=10,
            start=0,
            max_position=20,
            remove_landmarks=False,
            ITI_bin_size=None,
        )

    assert np.array_equal(result, expected)


def test_get_position_activity_all_trials_same_reorder() -> None:

    # Two frames in the first bin, two in the second
    rotary_encoder_position = [0, 180, 361, 700]
    states_info = [get_state_info(i) for i in [0, 1, 2, 3]]

    trials = [
        create_test_trial(
            rotary_encoder_position=rotary_encoder_position, states_info=states_info
        )
        for _ in range(100)
    ]

    # cells x time
    dff = np.array([[1, 2, 3, 4], [20, 15, 10, 5]])

    # Order of the cells should be flipped
    expected = np.array([[17.5, 7.5], [1.5, 3.5]])

    # removes the normalize step to make it clearer
    with patch("viral.two_photon.normalize", side_effect=lambda x, axis: x):

        result = get_position_activity(
            trials=trials,
            dff=dff,
            wheel_circumference=WHEEL_CIRCUMFERENCE,
            bin_size=10,
            start=0,
            max_position=20,
            remove_landmarks=False,
            ITI_bin_size=None,
        )

    assert np.array_equal(result, expected)


def test_get_position_activity_trials_different() -> None:
    rotary_encoder_position = [0, 180, 361, 700]

    train_trials = [
        create_test_trial(
            rotary_encoder_position=rotary_encoder_position,
            states_info=[get_state_info(i) for i in [0, 1, 2, 3]],
        )
        for _ in range(2)
    ]

    test_trials = [
        create_test_trial(
            rotary_encoder_position=rotary_encoder_position,
            states_info=[get_state_info(i) for i in [4, 5, 6, 7]],
        )
        for _ in range(2)
    ]

    # cells x time
    dff = np.array([[1, 2, 3, 4, 8, 7, 6, 5], [20, 15, 10, 5, 5, 10, 15, 20]])

    # Should give the sort order
    # expected_train = np.array([[17.5, 7.5], [1.5, 3.5]])

    # Expect the cell with the largest activity (that's second in the dff matrix) to be first,
    # as it peaks first in the train matrix, although it peaks second in the test matrix
    expected_test = np.array([[7.5, 17.5], [7.5, 5.5]])

    with patch("viral.two_photon.normalize", side_effect=lambda x, axis: x):
        with patch(
            "random.sample", side_effect=lambda x, y: [0, 1]
        ):  # Always take the first two trials in random.sample
            result = get_position_activity(
                trials=train_trials + test_trials,
                dff=dff,
                wheel_circumference=WHEEL_CIRCUMFERENCE,
                bin_size=10,
                start=0,
                max_position=20,
                remove_landmarks=False,
                ITI_bin_size=None,
            )
    assert np.array_equal(result, expected_test)


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
