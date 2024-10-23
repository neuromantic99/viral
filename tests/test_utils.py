from typing import List
import numpy as np
from pydantic import BaseModel

from viral.models import SpeedPosition
from viral.utils import get_speed_positions, threshold_detect_edges


def compare_pydantic_models(
    a: BaseModel | List[BaseModel], b: BaseModel | List[BaseModel]
) -> bool:

    if isinstance(a, list):
        assert isinstance(b, list)
        return all(compare_pydantic_models(x, y) for x, y in zip(a, b))

    assert not isinstance(b, list)
    return a.model_dump() == b.model_dump()


def test_get_speed_positions_basic() -> None:

    position = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    result = get_speed_positions(
        position=position,
        first_position=0,
        last_position=10,
        step_size=5,
        sampling_rate=10,
    )

    assert compare_pydantic_models(
        result,
        [
            SpeedPosition(position_start=0, position_stop=5, speed=10.0),
            SpeedPosition(position_start=5, position_stop=10, speed=10.0),
        ],
    )


def test_get_speed_positions_speed_change() -> None:

    position = np.array([1, 1, 1, 1, 1, 5, 6])

    result = get_speed_positions(
        position=position,
        first_position=0,
        last_position=10,
        step_size=5,
        sampling_rate=10,
    )

    assert compare_pydantic_models(
        result,
        [
            SpeedPosition(position_start=0, position_stop=5, speed=10),
            SpeedPosition(position_start=5, position_stop=10, speed=25),
        ],
    )


def test_get_speed_positions_one_bin() -> None:

    position = np.array([1, 1, 1, 1, 1, 5, 6])

    result = get_speed_positions(
        position=position,
        first_position=0,
        last_position=3,
        step_size=3,
        sampling_rate=10,
    )

    assert compare_pydantic_models(
        result,
        [
            SpeedPosition(position_start=0, position_stop=3, speed=6),
        ],
    )


def test_no_crossings() -> None:
    # Signal never crosses the threshold
    signal = np.array([0, 0.5, 0.7, 0.8])
    threshold = 1.0
    rising, falling = threshold_detect_edges(signal, threshold)
    assert len(rising) == 0, "There should be no rising edges"
    assert len(falling) == 0, "There should be no falling edges"


def test_single_rising_and_falling() -> None:
    # Single crossing above and then below the threshold
    signal = np.array([0, 1, 2, 1, 0])
    threshold = 1.5
    rising, falling = threshold_detect_edges(signal, threshold)
    assert np.array_equal(rising, [2]), "Expected a single rising edge at index 2"
    assert np.array_equal(falling, [3]), "Expected a single falling edge at index 3"


def test_multiple_crossings() -> None:
    # Multiple crossings
    signal = np.array([0, 2, 0, 3, 0, 4, 0])
    threshold = 1.5
    rising, falling = threshold_detect_edges(signal, threshold)
    assert np.array_equal(
        rising, [1, 3, 5]
    ), "Expected rising edges at indices 1, 3, and 5"
    assert np.array_equal(
        falling, [2, 4, 6]
    ), "Expected falling edges at indices 2, 4, and 6"


def test_threshold_edge_case() -> None:
    # Signal values exactly at the threshold
    signal = np.array([1.5, 1.5, 2.0, 1.5, 0])
    threshold = 1.5
    rising, falling = threshold_detect_edges(signal, threshold)
    assert np.array_equal(rising, [2]), "Expected a rising edge at index 2"
    assert np.array_equal(falling, [3]), "Expected a falling edge at index 3"


def test_signal_just_meets_threshold() -> None:
    # Signal that exactly touches the threshold but doesn't cross it
    signal = np.array([1.5, 1.5, 1.5, 1.5])
    threshold = 1.5
    rising, falling = threshold_detect_edges(signal, threshold)
    assert (
        len(rising) == 0
    ), "There should be no rising edges as the signal is at the threshold"
    assert (
        len(falling) == 0
    ), "There should be no falling edges as the signal is at the threshold"
