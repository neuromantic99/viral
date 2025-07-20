from pathlib import Path
from typing import List
import numpy as np
from pydantic import BaseModel
import matplotlib.pyplot as plt

from viral.constants import BEHAVIOUR_DATA_PATH, CACHE_PATH
from viral.imaging_utils import (
    compute_speed_grosmark,
    compute_windowed_speed_1d,
    extract_TTL_chunks,
    get_online_position_and_frames,
)
from viral.models import Cached2pSession, SpeedPosition, TrialInfo
from viral.utils import (
    above_threshold_for_n_consecutive_samples,
    array_bin_mean,
    degrees_to_cm,
    get_speed_positions,
    get_wheel_circumference_from_rig,
    has_n_consecutive_trues,
    remove_consecutive_ones,
    shuffle_rows,
    threshold_detect_edges,
    get_session_type,
    trial_is_imaged,
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


def test_has_n_consecutive_trues_basic() -> None:
    matrix = np.array(
        [[True, True, True, True, True, False], [True, True, True, True, False, False]]
    )
    result = has_n_consecutive_trues(matrix)
    assert np.array_equal(result, np.array([True, False]))


def test_has_n_consecutive_trues_none_have() -> None:
    matrix = np.array(
        [[False, True, True, True, True, False], [True, True, True, True, False, False]]
    )
    result = has_n_consecutive_trues(matrix)
    assert np.array_equal(result, np.array([False, False]))


def test_has_n_consecutive_trues_loads_have() -> None:
    matrix = np.array([[True, True, True, True, True, False]] * 100)
    result = has_n_consecutive_trues(matrix)
    assert np.array_equal(result, np.array([True] * 100))


def test_has_n_consecutive_trues_another_random_one() -> None:
    matrix = np.array(
        [
            [True, True, False, True, True, False, False, False, True, True],
            [False, False, False, True, True, True, True, True, False, True],
            [True, True, False, True, True, False, False, False, False, False],
            [False, False, False, True, True, True, True, True, False, True],
        ]
    )
    result = has_n_consecutive_trues(matrix)
    assert np.array_equal(result, np.array([False, True, False, True]))


def test_shuffle_rows() -> None:
    matrix = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    result = shuffle_rows(matrix)
    # This actually could be true with low probabilty
    assert not np.array_equal(result, matrix)

    assert np.array_equal(np.sort(result[0, :]), np.array([1, 2, 3, 4, 5, 6]))
    assert np.array_equal(np.sort(result[1, :]), np.array([7, 8, 9, 10, 11, 12]))


def test_shuffle_rows_make_sure_not_seeded() -> None:
    matrix = np.array([np.arange(100), np.arange(100)])

    result1 = shuffle_rows(matrix)
    result2 = shuffle_rows(matrix)

    assert not np.array_equal(result1, result2)


def test_above_threshold_for_n_consecutive_samples_basic_zeros() -> None:
    arr = np.zeros(100)

    for n_samples in range(1, 100):
        result = above_threshold_for_n_consecutive_samples(
            arr, threshold=1, n_samples=n_samples
        )
        assert np.array_equal(result, np.zeros(100))


def test_above_threshold_for_n_consecutive_samples_all_above() -> None:
    arr = np.ones(100)

    for n_samples in range(1, 100):
        result = above_threshold_for_n_consecutive_samples(
            arr, threshold=0.5, n_samples=n_samples
        )
        assert np.array_equal(result, np.ones(100))


def test_above_threshold_for_n_consecutive_samples_chunk_in_the_middle() -> None:
    arr = np.zeros(100)
    arr[10:20] = 1
    arr[30:33] = 1

    result = above_threshold_for_n_consecutive_samples(arr, threshold=0.2, n_samples=5)
    expected = np.zeros(100)
    expected[10:20] = 1
    assert np.array_equal(result, expected)

    result = above_threshold_for_n_consecutive_samples(arr, threshold=0.2, n_samples=2)
    expected = np.zeros(100)
    expected[10:20] = 1
    expected[30:33] = 1
    assert np.array_equal(result, expected)


def test_above_threshold_for_n_consecutive_samples_edge_cases() -> None:
    arr = np.zeros(100)
    arr[98:99] = 1
    result = above_threshold_for_n_consecutive_samples(arr, threshold=0.2, n_samples=5)
    expected = np.zeros(100)
    assert np.array_equal(result, expected)

    arr[95:100] = 1
    result = above_threshold_for_n_consecutive_samples(arr, threshold=0.2, n_samples=5)
    expected = np.zeros(100)
    expected[95:100] = 1
    assert np.array_equal(result, expected)


def test_compute_windowed_speed_1d() -> None:
    # Travels 90 cm in 3 seconds
    position = np.arange(90)
    expected = np.repeat(30, 90)
    for window_duration in [0.1, 0.5, 1, 2, 3]:
        result = compute_windowed_speed_1d(position, window_duration=window_duration)
        # Can get a floating point error here
        np.testing.assert_allclose(result, expected, atol=1e-10)


def test_compute_windowed_speed_no_movement() -> None:
    position = np.repeat(0, 100)
    expected = np.repeat(0, 100)
    for window_duration in [0.1, 0.5, 1, 2, 3]:
        result = compute_windowed_speed_1d(position, window_duration=window_duration)
        # Can get a floating point error here
        np.testing.assert_allclose(result, expected, atol=1e-10)


def test_compute_windowed_speed_movement_in_the_middle() -> None:
    position = np.repeat(0, 20)

    position[10:] = 1
    window_duration = 0.2  # 6 samples

    expected = np.repeat(0, 20)

    # travelled 1 cm in 0.2 seconds
    # between 10 - (window / 2) and 10 + (window / 2)
    expected[7:13] = 5

    result = compute_windowed_speed_1d(position, window_duration=window_duration)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_compute_windowed_speed_movement_and_at_the_edges() -> None:
    position = np.repeat(0, 40)

    position[5:20] = 1

    position[20:] = 2

    position[39:] = 3

    window_duration = 0.2  # 6 samples

    expected = np.repeat(0, 40)
    # window is full not halved at the edges, so full window is used
    expected[:6] = 5
    # Then we get to the half window, which also looks backwards
    expected[6:8] = 5

    # movement in the middle with the half window
    expected[17:23] = 5

    # Full window at the end
    expected[36:] = 5

    result = compute_windowed_speed_1d(position, window_duration=window_duration)
    np.testing.assert_allclose(result, expected, atol=1e-10)


# def test_compare_speed_functions() -> None:
#     """Dont have this as part of the actual test suite as it requires server access"""
#     # Get a random real trial
#     trial_file = BEHAVIOUR_DATA_PATH / "JB035" / "2025-07-16" / "002" / "trial0.json"
#     with open(trial_file) as f:
#         trial = TrialInfo.model_validate_json(f.read())

#     position = degrees_to_cm(
#         np.array(trial.rotary_encoder_position), get_wheel_circumference_from_rig("2P")
#     )
#     our_speed = compute_windowed_speed_1d(position, window_duration=0.5)
#     grosmark_speed = compute_speed_grosmark(position)
#     _, ax = plt.subplots(figsize=(20, 10))

#     ax2 = ax.twinx()
#     ax.plot(position, color="blue")
#     ax.set_ylabel("Position (cm)")
#     ax2.plot(our_speed, label="Our speed", color="red")
#     ax2.plot(grosmark_speed, label="Grosmark speed", color="green")
#     ax2.set_ylabel("Speed (cm/s)")
#     ax2.legend()
#     plt.axhline(y=5, color="black", linestyle="--")
#     plt.show()


# def load_test_trial() -> TrialInfo:
#     path = CACHE_PATH / "JB036_2025-07-05.json"
#     with open(path) as f:
#         session = Cached2pSession.model_validate_json(f.read())

#     trial = session.trials[19]

#     assert trial_is_imaged(trial)

#     return trial


# def test_compare_speed_functions() -> None:
#     """Dont have this as part of the actual test suite as it requires server access"""
#     trial = load_test_trial()
#     position = degrees_to_cm(
#         np.array(trial.rotary_encoder_position), get_wheel_circumference_from_rig("2P")
#     )
#     our_speed = compute_windowed_speed_1d(position, window_duration=0.5)
#     grosmark_speed = compute_speed_grosmark(position)
#     _, ax = plt.subplots(figsize=(20, 10))
#     ax2 = ax.twinx()
#     ax.plot(position, color="blue")
#     ax.set_ylabel("Position (cm)")
#     ax2.plot(our_speed, label="Our speed", color="red")
#     ax2.plot(grosmark_speed, label="Grosmark speed", color="green")
#     ax2.set_ylabel("Speed (cm/s)")
#     ax2.legend()
#     plt.axhline(y=5, color="black", linestyle="--")
#     plt.show()


# def test_get_online_position_and_frames() -> None:
#     """Also best done with a real trial"""

#     trial = load_test_trial()

#     original_position = degrees_to_cm(
#         np.array(trial.rotary_encoder_position), get_wheel_circumference_from_rig("2P")
#     )
#     original_frames = np.array(
#         [
#             state.closest_frame_start
#             for state in trial.states_info
#             if state.name
#             in ["trigger_panda", "trigger_panda_post_reward", "trigger_panda_ITI"]
#         ]
#     )

#     position, frames = get_online_position_and_frames(
#         trial, get_wheel_circumference_from_rig("2P")
#     )
#     speed = compute_speed_grosmark(original_position)

#     _, ax = plt.subplots(figsize=(20, 10))
#     ax2 = ax.twinx()

#     ax.plot(original_frames, original_position, ".", color="red", label="Original")
#     ax.plot(frames, position, ".", color="blue", label="Online filtered")

#     ax.legend()
#     ax.set_ylabel("Position (cm)")
#     ax2.plot(original_frames, speed, color="green", label="Speed")
#     ax2.set_ylabel("Speed (cm/s)")
#     ax2.axhline(y=5, color="black", linestyle="--")

#     time = ((original_frames - original_frames[0]) / 30).astype(int)

#     plt.xticks(original_frames[::40], time[::40])
#     ax.set_xlabel("Time (s)")
#     ax2.set_xlabel("Time (s)")

#     plt.show()
