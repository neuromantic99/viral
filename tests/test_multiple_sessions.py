import numpy as np
import pytest

from typing import List

from viral.utils import d_prime
from viral.multiple_sessions import (
    speed_difference,
    licking_difference,
    rolling_performance,
    get_chance_level,
    filter_sessions_by_session_type,
    create_metric_dict,
    prepare_plot_data,
    get_num_to_x,
)
from viral.models import TrialSummary, TrialInfo, StateInfo, SessionSummary


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


# def rolling_performance(trials: List[TrialSummary], window: int) -> List[float]:
#     return [
#         learning_metric(trials[idx - window : idx])
#         for idx in range(window, len(trials))
#     ]

# def test_rolling_performance() -> None:
#     trials = [create_test_trial()]
#     result =
#     expected =
#     assert
