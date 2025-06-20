import pytest
import numpy as np

from viral.cache_2p_sessions import extract_frozen_wheel_chunks


def test_extract_frozen_wheel_chunks_standard_case() -> None:
    """Test with grab stopped automatically and everything in order"""
    stack_lengths_tiff = np.array([27000, 10000, 27000])
    sampling_rate = 10000
    valid_frame_times = np.arange(sum(stack_lengths_tiff)) * (sampling_rate / 30)
    behaviour_times = np.arange(27100, 36900) * (sampling_rate / 30)

    expected = (0, 27000), (37000, 64000)
    result = extract_frozen_wheel_chunks(
        stack_lengths_tiffs=stack_lengths_tiff,
        valid_frame_times=valid_frame_times,
        behaviour_times=behaviour_times,
        sampling_rate=sampling_rate,
    )
    assert expected == result


def test_extract_frozen_wheel_chunks_manual_stop() -> None:
    """Test with grab stopped manually and everything in order"""
    stack_lengths_tiff = np.array([27054, 10000, 28098])
    sampling_rate = 10000
    valid_frame_times = np.arange(sum(stack_lengths_tiff)) * (sampling_rate / 30)
    behaviour_times = np.arange(27100, 36900) * (sampling_rate / 30)

    expected = (0, 27054), (37054, 65152)
    result = extract_frozen_wheel_chunks(
        stack_lengths_tiffs=stack_lengths_tiff,
        valid_frame_times=valid_frame_times,
        behaviour_times=behaviour_times,
        sampling_rate=sampling_rate,
    )
    assert expected == result


def test_extract_frozen_wheel_chunks_behaviour_pulses_pre() -> None:
    """Test with behaviour pulses in the pre-training neural only chunk."""
    stack_lengths_tiff = np.array([27000, 10000, 27000])
    sampling_rate = 10000
    valid_frame_times = np.arange(sum(stack_lengths_tiff)) * (sampling_rate / 30)
    behaviour_times = np.arange(26000, 36900) * (sampling_rate / 30)

    with pytest.raises(
        AssertionError, match="Behavioral pulses detected in pre-training period!"
    ):
        extract_frozen_wheel_chunks(
            stack_lengths_tiffs=stack_lengths_tiff,
            valid_frame_times=valid_frame_times,
            behaviour_times=behaviour_times,
            sampling_rate=sampling_rate,
        )


# Should be the exact same logic, as the pre-training neural only chunk, but still testing
def test_extract_frozen_wheel_chunks_behaviour_pulses_post() -> None:
    """Test with behaviour pulses in the post-training neural only chunk."""
    stack_lengths_tiff = np.array([27000, 10000, 27000])
    sampling_rate = 10000
    valid_frame_times = np.arange(sum(stack_lengths_tiff)) * (sampling_rate / 30)
    behaviour_times = np.arange(37000, 40000) * (sampling_rate / 30)
    with pytest.raises(
        AssertionError, match="Behavioral pulses detected in post-training period!"
    ):
        extract_frozen_wheel_chunks(
            stack_lengths_tiffs=stack_lengths_tiff,
            valid_frame_times=valid_frame_times,
            behaviour_times=behaviour_times,
            sampling_rate=sampling_rate,
        )


def test_extract_frozen_wheel_chunks_unexpected_chunk_len() -> None:
    """Test with chunk lengths that don't match with the expected lengths."""
    # Just testing for the first, is the same logic for pre- and post-training
    stack_lengths_tiff = np.array([26000, 10000, 27000])
    sampling_rate = 10000
    valid_frame_times = np.arange(sum(stack_lengths_tiff)) * (sampling_rate / 30)
    behaviour_times = np.arange(27100, 36900) * (sampling_rate / 30)

    with pytest.raises(
        AssertionError, match="First chunk length does not match expected length"
    ):
        extract_frozen_wheel_chunks(
            stack_lengths_tiffs=stack_lengths_tiff,
            valid_frame_times=valid_frame_times,
            behaviour_times=behaviour_times,
            sampling_rate=sampling_rate,
        )


def test_extract_frozen_wheel_chunks_first_chunk() -> None:
    """Check the logic for ignoring the first chunk if the DAQ has not been started."""
    stack_lengths_tiff = np.array([27000, 10000, 27000])
    sampling_rate = 10000
    behaviour_times = np.arange(10, 2367) * (sampling_rate / 30)

    # mimick manual correction
    valid_frame_times = np.arange(0, sum([10000, 27000])) * (sampling_rate / 30)
    stack_lengths_tiff = np.delete(stack_lengths_tiff, 0)

    expected = (None, (10000, 37000))
    result = extract_frozen_wheel_chunks(
        stack_lengths_tiffs=stack_lengths_tiff,
        valid_frame_times=valid_frame_times,
        behaviour_times=behaviour_times,
        sampling_rate=sampling_rate,
        check_first_chunk=False,
    )

    assert expected == result
