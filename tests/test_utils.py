from pathlib import Path
from typing import List
import numpy as np
from pydantic import BaseModel

from viral.imaging_utils import extract_TTL_chunks
from viral.models import SpeedPosition
from viral.utils import (
    array_bin_mean,
    get_speed_positions,
    remove_consecutive_ones,
    threshold_detect_edges,
    get_session_type,
)


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


def test_regular_frame_clock_intervals() -> None:
    sampling_rate = 1000
    frame_clock = np.zeros(1000)
    frame_clock[::30] = 5

    frame_times, chunk_lens = extract_TTL_chunks(frame_clock, sampling_rate)

    # Check that the length of the detected frames is as expected
    assert len(frame_times) == len(frame_clock[::30])
    assert np.array_equal(np.array([34]), chunk_lens)


def test_frame_clock_with_single_gap() -> None:
    # Frame clock with a single larger gap, should split into two chunks
    sampling_rate = 1000
    frame_clock1 = np.zeros(2000)
    frame_clock1[::30] = 5

    frame_clock2 = np.zeros(5000)

    frame_clock3 = np.zeros(3000)
    frame_clock3[::30] = 5

    frame_clock = np.hstack((frame_clock1, frame_clock2, frame_clock3))

    frame_times, chunk_lens = extract_TTL_chunks(frame_clock, sampling_rate)

    # Check that chunk length splits into two parts as expected
    assert len(chunk_lens) == 2
    # Ensure chunks are roughly half each of 33 expected intervals
    assert len(frame_times) == len(frame_clock1[::30]) + len(frame_clock3[::30])


def test_multiple_gaps_in_frame_clock() -> None:
    # Frame clock with multiple large gaps, expecting multiple chunks
    sampling_rate = 1000
    frame_clock = np.zeros(30000)
    frame_clock[::30] = 5
    frame_clock[500:1600] = 0  # First gap
    frame_clock[12000:14000] = 0  # Second gap
    frame_clock[25000:28000] = 0  # Third gap

    frame_times, chunk_lens = extract_TTL_chunks(frame_clock, sampling_rate)

    # Expect 4 chunks due to 3 gaps introduced
    assert len(chunk_lens) == 4
    # Sum of chunk lengths should equal total frames detected
    assert sum(chunk_lens) == len(frame_times)
    assert len(frame_times) == np.sum(frame_clock == 5)


def test_chunk_len_correct() -> None:
    sampling_rate = 10
    frame_clock = np.zeros(100)

    frame_clock[1] = 5
    frame_clock[2] = 0
    frame_clock[3] = 5
    frame_clock[4] = 0
    frame_clock[5] = 5

    frame_clock[50] = 5
    frame_clock[51] = 0
    frame_clock[52] = 5
    frame_clock[53] = 0
    frame_clock[54] = 5
    frame_clock[55] = 0
    frame_clock[56] = 5

    frame_clock[70] = 5
    frame_clock[71] = 0
    frame_clock[72] = 5
    frame_clock[73] = 0
    frame_clock[74] = 5

    frame_times, chunk_lens = extract_TTL_chunks(frame_clock, sampling_rate)

    assert len(chunk_lens) == 3
    assert chunk_lens[0] == 3 and chunk_lens[1] == 4 and chunk_lens[2] == 3
    assert len(frame_times) == np.sum(frame_clock == 5)


def test_get_session_type() -> None:
    session_name = "Learning day 1"
    expected = "learning"
    result = get_session_type(session_name)
    assert result == expected

    session_name = "Reversal learning day 2"
    expected = "reversal"
    result = get_session_type(session_name)
    assert result == expected

    session_name = "Recall learning day 3"
    expected = "recall"
    result = get_session_type(session_name)
    assert result == expected

    session_name = "Recall reversal learning day 4"
    expected = "recall_reversal"
    result = get_session_type(session_name)
    assert result == expected


def test_array_bin_mean() -> None:
    input = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
    result = array_bin_mean(input, bin_size=2)
    expected = np.array([[1.5, 3.5], [4.5, 6.5]])
    np.testing.assert_array_equal(result, expected)


def test_array_bin_mean_bin_size_array_not_divisible_along_bin_size() -> None:
    input = np.array([[1, 2, 3, 4, 5], [4, 5, 6, 7, 8]])
    result = array_bin_mean(input, bin_size=3)
    expected = np.array([[2, 4.5], [5, 7.5]])
    np.testing.assert_array_equal(result, expected)


def test_array_bin_mean_other_axis() -> None:
    input = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])
    result = array_bin_mean(input, bin_size=2, axis=0)
    expected = np.array([[2.5, 3.5, 4.5, 5.5]])
    np.testing.assert_array_equal(result, expected)


def test_remove_consecutive_ones() -> None:
    matrix = np.array([[0, 1, 1, 1, 0], [1, 1, 0, 1, 1]])
    result = remove_consecutive_ones(matrix)
    assert np.array_equal(result, np.array([[0, 1, 0, 0, 0], [1, 0, 0, 1, 0]]))
