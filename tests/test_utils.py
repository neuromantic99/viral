from typing import List
import numpy as np
from pydantic import BaseModel

from viral.models import SpeedPosition
from viral.utils import get_speed_positions


def compare_pydantic_models(
    a: BaseModel | List[BaseModel], b: BaseModel | List[BaseModel]
) -> bool:

    if isinstance(a, list):
        assert isinstance(b, list)
        return all(compare_pydantic_models(x, y) for x, y in zip(a, b))

    assert not isinstance(b, list)
    return a.model_dump() == b.model_dump()


def test_get_speed_positions_basic():

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


def test_get_speed_positions_speed_change():

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


def test_get_speed_positions_one_bin():

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
