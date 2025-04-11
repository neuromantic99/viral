from typing import Tuple
import numpy as np
from viral.models import TrialInfo
from viral.utils import threshold_detect


def get_ITI_start_frame(trial: TrialInfo) -> int:
    for state in trial.states_info:
        if state.name in {"ITI", "trigger_ITI"}:
            assert (
                state.closest_frame_start is not None
            ), "Imaging data not added to trial"
            return state.closest_frame_start
    raise ValueError("ITI state not found")


def get_sampling_rate(frame_clock: np.ndarray) -> int:
    """Bit of a hack as the sampling rate is not stored in the tdms file I think. I've used
    two different sampling rates: 1,000 and 10,000. The sessions should be between 30 and 100 minutes.
    """
    if 30 < len(frame_clock) / 1000 / 60 < 120:
        return 1000
    elif 30 < len(frame_clock) / 10000 / 60 < 120:
        return 10000
    raise ValueError("Could not determine sampling rate")


def trial_is_imaged(trial: TrialInfo) -> bool:
    trigger_panda_states = [
        state
        for state in trial.states_info
        if state.name in {"trigger_panda", "trigger_panda_post_reward"}
    ]
    start_times_bpod = [state.start_time for state in trigger_panda_states]
    length_trial_bpod = start_times_bpod[-1] - start_times_bpod[0]

    start_time_frames = [state.closest_frame_start for state in trigger_panda_states]

    assert start_time_frames[-1] is not None
    assert start_time_frames[0] is not None

    length_trial_frames = (start_time_frames[-1] - start_time_frames[0]) / 30

    # A little but of a error in this calculation is allowed here as the frame rate is not exactly 30.
    # Possibly you will get a false positive if the trial is stopped exactly at the end but unlikely
    return length_trial_bpod - 0.5 <= length_trial_frames <= length_trial_bpod + 0.5


def extract_TTL_chunks(
    frame_clock: np.ndarray, sampling_rate: int
) -> Tuple[np.ndarray, np.ndarray]:
    frame_times = threshold_detect(frame_clock, 4)
    diffed = np.diff(frame_times)
    chunk_starts = np.where(diffed > sampling_rate)[0] + 1
    # The first chunk starts on the first frame detected
    chunk_starts = np.insert(chunk_starts, 0, 0)
    # Add the final frame to allow the diff to work on the last chunk
    chunk_starts = np.append(chunk_starts, len(frame_times))
    return frame_times, np.diff(chunk_starts)
