import numpy as np

from viral.models import StateInfo, TrialInfo
from viral.two_photon import activity_trial_position, sort_matrix_peak


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


def create_test_trial() -> TrialInfo:
    return TrialInfo(
        trial_start_time=0,
        trial_end_time=1,
        pc_timestamp="timestamp",
        states_info=
        events_info=[]
        rotary_encoder_position=List[float],
        texture=str,
        texture_rewarded=bool,
    )


def test_activity_trial_position_SOMETHING() -> None:

    result = activity_trial_position()
    pass


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
