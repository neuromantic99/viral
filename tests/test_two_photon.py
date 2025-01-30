from typing import List
import numpy as np

from viral.models import StateInfo, TrialInfo
from viral.two_photon import activity_trial_position, sort_matrix_peak
import matplotlib.pyplot as plt

WHEEL_CIRCUMPHERENCE = 10


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
    rotary_encoder_position = list(range(200 * 360 // WHEEL_CIRCUMPHERENCE))

    trial = create_test_trial(
        rotary_encoder_position=rotary_encoder_position,
        states_info=[get_state_info(i) for i in range(len(rotary_encoder_position))],
    )

    # cells x frames
    dff = np.ones((10, len(trial.rotary_encoder_position)))
    result = activity_trial_position(trial, dff, WHEEL_CIRCUMPHERENCE)
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
        trial, dff, WHEEL_CIRCUMPHERENCE, bin_size=10, start=0, max_position=20
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
        WHEEL_CIRCUMPHERENCE,
        bin_size=10,
        start=0,
        max_position=20,
    )
    # Expect the first bin to be the mean of the first two frames
    # Expect the second bin to be only the final frame
    expected = np.array([[7.5, 20], [1.5, 4]])
    assert np.array_equal(result, expected)


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
