from typing import List
from viral.multiple_sessions import create_metric_dict
from viral.models import TrialSummary
from unittest.mock import Mock


def create_mock_mouse() -> Mock:
    """Creates a list of mocked MouseSummary objects for testing."""
    trial1 = Mock()
    trial1.rewarded = True
    trial2 = Mock()
    trial2.rewarded = True
    trial3 = Mock()
    trial3.rewarded = False
    trial4 = Mock()
    trial4.rewarded = False
    trial5 = Mock()
    trial5.rewarded = False

    mock_trials = [trial1, trial2, trial3, trial4, trial5]
    mock_session_names = ["learning", "reversal", "recall", "recall_reversal"]
    mock_sessions = list()
    for name in mock_session_names:
        session = Mock()
        session.name = name
        session.trials = mock_trials
        mock_sessions.append(session)

    mock_mouse = Mock()
    mock_mouse.name = "mock_mouse"
    mock_mouse.sessions = mock_sessions
    return mock_mouse


def mock_metric_fcn_trials(
    trials: List[TrialSummary], rewarded: bool | None = None
) -> int:
    """Count trials"""
    if rewarded is not None:
        return len([trial for trial in trials if trial.rewarded == rewarded])
    else:
        return len(trials)


def mock_metric_fcn_sessions(sessions):
    """Count trials/sessions"""
    return len(sessions)


def test_create_metric_dict_include_reward_status():
    """Test that the trials get sorted correctly by session type and reward condition"""
    mock_mouse = create_mock_mouse()
    metric_dict = create_metric_dict(
        mice=[mock_mouse],
        metric_fn=mock_metric_fcn_trials,
        flat_sessions=True,
        include_reward_status=True,
    )
    assert "mock_mouse" in metric_dict
    mouse_metric = metric_dict["mock_mouse"]
    expected_keys = {
        "learning_rewarded",
        "learning_unrewarded",
        "reversal_rewarded",
        "reversal_unrewarded",
        "recall_rewarded",
        "recall_unrewarded",
        "recall_reversal_rewarded",
        "recall_reversal_unrewarded",
    }
    # Test that the expected keys are in the dictionary
    assert set(mouse_metric.keys()) == expected_keys
    # Test that the number of trials for each session_type and condition in the sorted data is still the same as in the raw data
    assert mouse_metric["learning_rewarded"] == 2
    assert mouse_metric["learning_unrewarded"] == 3
    assert mouse_metric["reversal_rewarded"] == 2
    assert mouse_metric["reversal_unrewarded"] == 3
    assert mouse_metric["recall_rewarded"] == 2
    assert mouse_metric["recall_unrewarded"] == 3
    assert mouse_metric["recall_reversal_rewarded"] == 2
    assert mouse_metric["recall_reversal_unrewarded"] == 3


def test_create_metric_dict_no_reward_status():
    """Test that the trials get sorted correctly by session type."""
    mock_mouse = create_mock_mouse()
    metric_dict = create_metric_dict(
        mice=[mock_mouse],
        metric_fn=mock_metric_fcn_trials,
        flat_sessions=True,
        include_reward_status=False,
    )
    assert "mock_mouse" in metric_dict
    mouse_metric = metric_dict["mock_mouse"]
    expected_keys = {"learning", "reversal", "recall", "recall_reversal"}
    # Test that the expected keys are in the dicitonary
    assert set(mouse_metric.keys()) == expected_keys
    # Test that the number of trials for each session_type in the sorted data is still the same as in the raw data
    assert mouse_metric["learning"] == 5
    assert mouse_metric["reversal"] == 5
    assert mouse_metric["recall"] == 5
    assert mouse_metric["recall_reversal"] == 5


def test_create_metric_dict_sessions():
    """Test that the sessions get sorted correctly by session type."""
    mock_mouse = create_mock_mouse()
    metric_dict = create_metric_dict(
        mice=[mock_mouse],
        metric_fn=mock_metric_fcn_sessions,
        flat_sessions=False,
        include_reward_status=False,
    )
    # Realised that flat_sessions and include_reward_status have to be set to False so that the session dictionary works
    assert "mock_mouse" in metric_dict
    mouse_metric = metric_dict["mock_mouse"]
    expected_keys = {"learning", "reversal", "recall", "recall_reversal"}
    # Test that the expected keys are in the dictionary
    assert set(mouse_metric.keys()) == expected_keys
    # Test that the number of sessions for each session_type in the sorted data is still the same as in the raw data
    assert mouse_metric["learning"] == 1
    assert mouse_metric["reversal"] == 1
    assert mouse_metric["recall"] == 1
    assert mouse_metric["recall_reversal"] == 1
