import pytest
import numpy as np

from viral.cache_2p_sessions import extract_frozen_wheel_chunks


def test_extract_frozen_wheel_chunks_standard_case() -> None:
    """Test with grab stopped automatically and everything in order"""
    stack_lengths_tiff = np.array([27000, 10000, 27000])
    chunk_lengths_daq = np.array([27000, 10000, 27000])
    frame_times_daq = np.arange(sum(chunk_lengths_daq))
    behaviour_times = np.arange(27100, 36900)

    expected = (0, 26999), (36999, 63999)
    result = extract_frozen_wheel_chunks(
        stack_lengths_tiffs=stack_lengths_tiff,
        chunk_lengths_daq=chunk_lengths_daq,
        frame_times_daq=frame_times_daq,
        behaviour_times=behaviour_times,
    )
    assert expected == tuple(tuple(int(x) for x in t) for t in result)


def test_extract_frozen_wheel_chunks_manual_stop() -> None:
    """Test with grab stopped manually and everything in order"""
    stack_lengths_tiff = np.array([27054, 10000, 28098])
    chunk_lengths_daq = np.array([27056, 10000, 29000])
    frame_times_daq = np.arange(sum(chunk_lengths_daq))
    behaviour_times = np.arange(27100, 36900)

    expected = (0, 27053), (37053, 65151)
    result = extract_frozen_wheel_chunks(
        stack_lengths_tiffs=stack_lengths_tiff,
        chunk_lengths_daq=chunk_lengths_daq,
        frame_times_daq=frame_times_daq,
        behaviour_times=behaviour_times,
    )
    assert expected == tuple(tuple(int(x) for x in t) for t in result)


def test_extract_frozen_wheel_chunks_behaviour_pulses_pre() -> None:
    """Test with behaviour pulses in the pre-training neural only chunk."""
    stack_lengths_tiff = np.array([27000, 10000, 27000])
    chunk_lengths_daq = np.array([27000, 10000, 27000])
    frame_times_daq = np.arange(sum(chunk_lengths_daq))
    behaviour_times = np.arange(26000, 36900)
    try:
        result = extract_frozen_wheel_chunks(
            stack_lengths_tiffs=stack_lengths_tiff,
            chunk_lengths_daq=chunk_lengths_daq,
            frame_times_daq=frame_times_daq,
            behaviour_times=behaviour_times,
        )
    except AssertionError as e:
        print(
            f"test_extract_frozen_wheel_chunks_behaviour_pulses_pre passed. Caught expection: {e}"
        )
        return

    raise AssertionError(
        "test_extract_frozen_wheel_chunks_behaviour_pulses_pre failed. Expection not caught."
    )


# Should be the exact same logic, as the pre-training neural only chunk, but still testing
def test_extract_frozen_wheel_chunks_behaviour_pulses_post() -> None:
    """Test with behaviour pulses in the post-training neural only chunk."""
    stack_lengths_tiff = np.array([27000, 10000, 27000])
    chunk_lengths_daq = np.array([27000, 10000, 27000])
    frame_times_daq = np.arange(sum(chunk_lengths_daq))
    behaviour_times = np.arange(37000, 40000)
    try:
        result = extract_frozen_wheel_chunks(
            stack_lengths_tiffs=stack_lengths_tiff,
            chunk_lengths_daq=chunk_lengths_daq,
            frame_times_daq=frame_times_daq,
            behaviour_times=behaviour_times,
        )
    except AssertionError as e:
        print(
            f"test_extract_frozen_wheel_chunks_behaviour_pulses_post passed. Caught expection: {e}"
        )
        return

    raise AssertionError(
        "test_extract_frozen_wheel_chunks_behaviour_pulses_post failed. Expection not caught."
    )


def test_extract_frozen_wheel_chunks_unexpected_chunk_len() -> None:
    """Test with chunk lengths that don't match with the expected lengths."""
    # Just testing for the first, is the same logic for pre- and post-training
    stack_lengths_tiff = np.array([26000, 10000, 27000])
    chunk_lengths_daq = np.array([26000, 10000, 27000])
    frame_times_daq = np.arange(sum(chunk_lengths_daq))
    behaviour_times = np.arange(27100, 36900)

    try:
        result = extract_frozen_wheel_chunks(
            stack_lengths_tiffs=stack_lengths_tiff,
            chunk_lengths_daq=chunk_lengths_daq,
            frame_times_daq=frame_times_daq,
            behaviour_times=behaviour_times,
        )
    except AssertionError as e:
        print(
            f"test_extract_frozen_wheel_chunks_unexpected_chunk_len passed. Caught expection: {e}"
        )
        return

    raise AssertionError(
        "test_extract_frozen_wheel_chunks_unexpected_chunk_len failed. Expection not caught."
    )
