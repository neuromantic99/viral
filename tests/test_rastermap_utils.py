from typing import List
import pytest
import numpy as np
from unittest.mock import Mock, patch

from viral.rastermap_utils import (
    align_trial_frames,
    get_frame_position,
    get_speed_frame,
    get_lick_index,
    get_reward_index,
    create_frame_mapping,
    remap_to_continuous_indices,
)


@pytest.fixture
def mock_trials() -> list[Mock]:
    """Creates a list of mocked TrialInfo objects for testing."""
    trial1 = Mock()
    trial1.trial_start_closest_frame = 10
    trial1.trial_end_closest_frame = 50
    trial1.texture_rewarded = True
    state1 = Mock()
    state1.configure_mock(name="ITI", closest_frame_start=40)
    trial1.states_info = [state1]

    trial2 = Mock()
    trial2.trial_start_closest_frame = 60
    trial2.trial_end_closest_frame = 100
    trial2.texture_rewarded = False

    state2 = Mock()
    state2.configure_mock(name="ITI", closest_frame_start=90)
    trial2.states_info = [state2]

    return [trial1, trial2]


def test_align_trial_frames_no_ITI(mock_trials: List[Mock]) -> None:
    """Test align_trial_frames function without ITI."""

    result = align_trial_frames(mock_trials, ITI=False)
    expected = np.array([[10, 40, True], [60, 90, False]])

    assert np.array_equal(
        result, expected
    ), "align_trial_frames output does not match expected values without ITI"


def test_align_trial_frames_with_ITI(mock_trials: Mock) -> None:
    """Test align_trial_frames function with ITI included."""
    result = align_trial_frames(mock_trials, ITI=True)
    expected = np.array([[10, 50, True], [60, 100, False]])

    assert np.array_equal(
        result, expected
    ), "align_trial_frames output does not match expected values with ITI"


def test_align_trial_frames_overlapping() -> None:
    """Test that align_trial_frames raises an assertion error for overlapping trials."""
    trial1 = Mock()
    trial1.trial_start_closest_frame = 10
    trial1.trial_end_closest_frame = 50
    trial1.texture_rewarded = True
    state1 = Mock()
    state1.configure_mock(name="ITI", closest_frame_start=45)
    trial1.states_info = [state1]

    trial2 = Mock()
    trial2.trial_start_closest_frame = 40  # Overlapping frame
    trial2.trial_end_closest_frame = 100
    trial2.texture_rewarded = False

    state2 = Mock()
    state2.configure_mock(name="ITI", closest_frame_start=90)
    trial2.states_info = [state2]

    with pytest.raises(AssertionError, match="Overlapping frames for trials"):
        align_trial_frames([trial1, trial2], ITI=False)


def test_get_frame_position() -> None:
    """Test that the position for each frame is given correctly."""
    trial = Mock()
    trial.rotary_encoder_position = [0, 90, 180]
    state1 = Mock()
    state1.name = "trigger_panda"
    state1.closest_frame_start = 0
    state1.closest_frame_end = 0
    state2 = Mock()
    state2.name = "trigger_panda"
    state2.closest_frame_start = 1
    state2.closest_frame_end = 2
    state3 = Mock()
    state3.name = "trigger_panda"
    state3.closest_frame_start = 4
    state3.closest_frame_end = 4
    trial.states_info = [state1, state2, state3]

    trial_frames = np.array([0, 1, 2, 3, 4])

    wheel_circumference = 100
    manual_position = np.array([0, 25, 50])  # 0, 0; 90, 25; 180, 50

    expected_positions = np.array([0, 25, 25, 37.5, 50])
    expected = np.column_stack((trial_frames, expected_positions))

    with patch("viral.utils.degrees_to_cm", return_value=manual_position):
        result = get_frame_position(trial, trial_frames, wheel_circumference)

    assert np.array_equal(result, expected)

    # TODO: another test with ITI?


def test_get_speed_frame() -> None:
    """Test that the speed for a given array of frame indices and positions will return the right speeds."""
    frame_position = np.array([[0, 10], [1, 20], [2, 30]])
    expected = np.array([[0, 10], [1, 10], [2, 10]])
    result = get_speed_frame(frame_position)
    assert np.array_equal(result, expected)

    # Test with no change of position in between two frames
    frame_position = np.array([[0, 0], [1, 10], [2, 10]])
    expected = np.array([[0, 10], [1, 0], [2, 0]])
    result = get_speed_frame(frame_position)
    assert np.array_equal(result, expected)


def test_get_lick_index() -> None:
    """Test that frame indices of frames with licks will be returned."""
    trial = Mock()
    event1 = Mock()
    event1.name = "Port1In"
    event1.closest_frame = 0
    event2 = Mock()
    event2.name = "Port1Out"
    event2.closest_frame = 1
    event3 = Mock()
    event3.name = "Port1In"
    event3.closest_frame = 4
    event4 = Mock()
    event4.name = "Port1Out"
    event4.closest_frame = 5
    # Creating a lick which spans several frames
    event5 = Mock()
    event5.name = "Port1In"
    event5.closest_frame = 7
    event6 = Mock()
    event6.name = "Port1Out"
    event6.closest_frame = 10
    trial.events_info = [event1, event2, event3, event4, event5, event6]

    expected = np.array([0, 1, 4, 5, 7, 8, 9, 10])
    result = get_lick_index(trial)
    assert np.array_equal(result, expected)


def test_get_lick_index_no_licks() -> None:
    """Test that no frame indices will be returned, if there were not any licks."""
    trial = Mock()
    trial.events_info = []

    expected = None
    result = get_lick_index(trial)
    assert np.array_equal(result, expected)


def test_get_lick_index_missing_end_frame() -> None:
    """Test that frame indices of frames with licks will be returned, but just if both start and end are in the events info."""
    trial = Mock()
    event1 = Mock()
    event1.name = "Port1In"
    event1.closest_frame = 0
    event2 = Mock()
    event2.name = "Port1Out"
    event2.closest_frame = 1
    event3 = Mock()
    event3.name = "Port1In"
    event3.closest_frame = 4
    event4 = Mock()
    event4.name = "Port1Out"
    event4.closest_frame = 5
    # Creating a lick which misses the end frame (i.e. missing 'Port1Out' event)
    event5 = Mock()
    event5.name = "Port1In"
    event5.closest_frame = 7
    trial.events_info = [event1, event2, event3, event4, event5]

    expected = np.array([0, 1, 4, 5])
    result = get_lick_index(trial)
    assert np.array_equal(result, expected)


def test_get_reward_index() -> None:
    """Test that frame indices for frames with rewards will be returned."""
    trial = Mock()
    state1 = Mock()
    state1.name = "reward_on1"
    state1.closest_frame_start = 0
    state2 = Mock()
    state2.name = "reward_off3"
    state2.closest_frame_start = 1
    state3 = Mock()
    state3.name = "reward_on1"
    state3.closest_frame_start = 3
    state4 = Mock()
    state4.name = "reward_off3"
    state4.closest_frame_start = 4
    state5 = Mock()
    state5.name = "reward_on1"
    state5.closest_frame_start = 7
    state6 = Mock()
    state6.name = "reward_off3"
    state6.closest_frame_start = 9
    trial.states_info = [state1, state2, state3, state4, state5, state6]

    expected = np.array([0, 1, 3, 4, 7, 8, 9])
    result = get_reward_index(trial)
    assert np.array_equal(result, expected)


def test_get_reward_index_no_rewards() -> None:
    """Test no frame indices will be returned if there were not any rewards."""
    trial = Mock()
    trial.states_info = []

    expected = None
    result = get_reward_index(trial)
    assert np.array_equal(result, expected)


def test_get_reward_index_missing_end_frame() -> None:
    """Test that frame indices for frames with rewards will be returned, but just if both start and end are in the states info."""
    trial = Mock()
    state1 = Mock()
    state1.name = "reward_on1"
    state1.closest_frame_start = 0
    state2 = Mock()
    state2.name = "reward_off3"
    state2.closest_frame_start = 1
    state3 = Mock()
    state3.name = "reward_on1"
    state3.closest_frame_start = 3
    state4 = Mock()
    state4.name = "reward_off3"
    state4.closest_frame_start = 4
    # Creating one reward without an end (i.e. missing the 'reward_off3' state)
    state5 = Mock()
    state5.name = "reward_on1"
    state5.closest_frame_start = 7
    trial.states_info = [state1, state2, state3, state4, state5]

    expected = np.array([0, 1, 3, 4])
    result = get_reward_index(trial)
    assert np.array_equal(result, expected)


def test_create_frame_mapping() -> None:
    """Test that the frame mapping dictionary is created correctly."""
    positions_combined = np.array(
        [
            [10],
            [11],
            [12],
            [13],
        ]
    )
    expected = {10: 0, 11: 1, 12: 2, 13: 3}
    assert create_frame_mapping(positions_combined) == expected

    # Gaps in the original frames do not have to be tested
    # as the get_speed_frame function interpolates and makes sure that there are no gaps


def test_remap_to_continuous_indices() -> None:
    positions_combined = np.arange(20, 70).reshape(-1, 1)
    original_indices = np.array([24, 30, 60])
    frame_mapping = create_frame_mapping(positions_combined)
    continuous_indices = remap_to_continuous_indices(original_indices, frame_mapping)
    expected = np.array([4, 10, 40])
    assert np.array_equal(continuous_indices, expected)
