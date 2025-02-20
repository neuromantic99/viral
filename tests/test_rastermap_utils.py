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


def test_get_frame_position_no_start_no_end() -> None:
    """Test that the position for each frame is given correctly even though there are no positions for the beginning and start frames."""
    trial = Mock()
    trial.rotary_encoder_position = [0, 90, 180]
    state1 = Mock()
    state1.name = "trigger_panda"
    state1.closest_frame_start = 1
    state1.closest_frame_end = 1
    state2 = Mock()
    state2.name = "trigger_panda"
    state2.closest_frame_start = 2
    state2.closest_frame_end = 3
    state3 = Mock()
    state3.name = "trigger_panda"
    state3.closest_frame_start = 5
    state3.closest_frame_end = 5
    trial.states_info = [state1, state2, state3]

    trial_frames = np.array([0, 1, 2, 3, 4, 5, 6])

    wheel_circumference = 100
    manual_position = np.array([0, 25, 50])  # 0, 0; 90, 25; 180, 50

    expected_positions = np.array([0, 0, 25, 25, 37.5, 50, 50])
    expected = np.column_stack((trial_frames, expected_positions))

    with patch("viral.utils.degrees_to_cm", return_value=manual_position):
        result = get_frame_position(trial, trial_frames, wheel_circumference)

    assert np.array_equal(result, expected)


def test_get_speed_frame() -> None:
    """Test that the speed for a given array of frame indices and positions will return the right speeds."""
    frame_position = np.array(
        [[0, 10], [1, 20], [2, 30], [3, 40], [4, 50], [5, 80], [6, 100]]
    )
    expected = np.array([[0, 10], [1, 10], [2, 10], [3, 10], [4, 10], [5, 20], [6, 20]])
    result = get_speed_frame(frame_position)
    assert np.array_equal(result, expected)

    # Test with no change of position in between two frames
    frame_position = np.array(
        [
            [0, 10],
            [1, 20],
            [2, 30],
            [3, 40],
            [4, 50],
            [5, 60],
            [6, 60],
        ]
    )
    expected = np.array(
        [
            [0, 10],
            [1, 10],
            [2, 10],
            [3, 10],
            [4, 10],
            [5, 0],
            [6, 0],
        ]
    )
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
    assert result is expected is None


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
    state1.closest_frame_start = 1
    state2 = Mock()
    state2.name = "reward_on2"
    state2.closest_frame_start = 2
    state3 = Mock()
    state3.name = "reward_on3"
    state3.closest_frame_start = 3
    state4 = Mock()
    state4.name = "reward_off3"
    state4.closest_frame_start = 4
    state5 = Mock()
    state5.name = "reward_on1"
    state5.closest_frame_start = 6
    state6 = Mock()
    state6.name = "reward_on2"
    state6.closest_frame_start = 7
    state7 = Mock()
    state7.name = "reward_on3"
    state7.closest_frame_start = 8
    state8 = Mock()
    state8.name = "reward_off3"
    state8.closest_frame_start = 9
    state9 = Mock()
    state9.name = "reward_on1"
    state9.closest_frame_start = 11
    state10 = Mock()
    state10.name = "reward_on2"
    state10.closest_frame_start = 12
    state11 = Mock()
    state11.name = "reward_on3"
    state11.closest_frame_start = 13
    state12 = Mock()
    state12.name = "reward_off3"
    state12.closest_frame_start = 14
    trial.states_info = [
        state1,
        state2,
        state3,
        state4,
        state5,
        state6,
        state7,
        state8,
        state9,
        state10,
        state11,
        state12,
    ]

    expected = np.array([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
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
    state1.closest_frame_start = 1
    state2 = Mock()
    state2.name = "reward_on2"
    state2.closest_frame_start = 2
    state3 = Mock()
    state3.name = "reward_on3"
    state3.closest_frame_start = 3
    state4 = Mock()
    state4.name = "reward_off3"
    state4.closest_frame_start = 4
    state5 = Mock()
    state5.name = "reward_on1"
    state5.closest_frame_start = 6
    state6 = Mock()
    state6.name = "reward_on2"
    state6.closest_frame_start = 7
    state7 = Mock()
    state7.name = "reward_on3"
    state7.closest_frame_start = 8
    # Creating one reward without an end (i.e. missing the 'reward_off3' state)
    trial.states_info = [state1, state2, state3, state4, state5, state6, state7]

    expected = np.array([1, 2, 3, 4])
    result = get_reward_index(trial)
    assert np.array_equal(result, expected)


def test_missing_reward_state() -> None:
    """Test that an AssertionError is raised if one of the valid reward_on states is not present."""
    trial = Mock()
    state1 = Mock()
    state1.name = "reward_on1"
    state1.closest_frame_start = 1
    state2 = Mock()
    state2.name = "reward_on2"
    state2.closest_frame_start = 2
    state3 = Mock()
    state3.name = "reward_off3"
    state3.closest_frame_start = 4
    trial.states_info = [state1, state2, state3]

    try:
        allowed_rewards = {"reward_on1", "reward_on2", "reward_on3"}
        for reward in allowed_rewards:
            assert any(
                state.name == reward for state in trial.states_info
            ), f"'{reward}' is not in states_info"
    except AssertionError as e:
        print(f"test_missing_reward_state passed. Caught exception: {e}")
        return

    raise AssertionError(f"test_missing_reward_state failed. Exception not caught.")


def test_unexpected_reward_state() -> None:
    """Test that an AssertionError is raised if one of the valid reward_on states is not present."""
    trial = Mock()
    state1 = Mock()
    state1.name = "reward_on1"
    state1.closest_frame_start = 1
    state2 = Mock()
    state2.name = "reward_on2"
    state2.closest_frame_start = 2
    state3 = Mock()
    state3.name = "reward_on3"
    state3.closest_frame_start = 3
    state4 = Mock()
    state4.name = "reward_on4"
    state4.closest_frame_start = 3
    state5 = Mock()
    state5.name = "reward_off3"
    state5.closest_frame_start = 4
    trial.states_info = [state1, state2, state3, state4, state5]

    try:
        allowed_rewards = {"reward_on1", "reward_on2", "reward_on3"}
        for state in trial.states_info:
            if state.name.startswith("reward_on") and state.name not in allowed_rewards:
                raise AssertionError(f"Unexpected reward state found: {state.name}")
    except AssertionError as e:
        print(f"test_unexpected_reward_state passed. Caught exception: {e}")
        return

    raise AssertionError(f"test_unexpected_reward_state failed. Exception not caught.")


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
