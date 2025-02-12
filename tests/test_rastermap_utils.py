from typing import List
import pytest
import numpy as np
from unittest.mock import Mock

from viral.rastermap_utils import align_trial_frames, get_ITI_start_frame


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
