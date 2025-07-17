"""cache_2p_sessions.py is responsible for adding frame stamps and time stamps to behavioural event data.
It processes imaging files (.tiff), behavioural events (trial.json), and the corresponding synchronisation file (DAQami) as inputs,
performing a series of checks before appending this information.
However, a range of user-dependent or experimental circumstances can occasionally cause the caching process to fail.
In these cases, manual correction is required."""

"""cache_2p_sessions.py is responsible for adding frame stamps and time stamps to behavioural event data.
It processes imaging files (.tiff), behavioural events (trial.json), and the corresponding synchronisation file (DAQami) as inputs,
performing a series of checks before appending this information.
However, a range of user-dependent or experimental circumstances can occasionally cause the caching process to fail.
In these cases, manual correction is required."""

import sys
import numpy as np
from pathlib import Path
from typing import Callable

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent))
sys.path.append(str(HERE.parent.parent))

from viral.models import SessionCorrection

session_corrections = dict()


def register_correction(mouse_name: str, date: str) -> Callable:
    """
    Decorator to register manual session corrections.
    """

    def decorator(func: Callable) -> Callable:
        session_corrections[(mouse_name, date)] = func
        return func

    return decorator


def apply_session_correction(
    mouse_name: str,
    date: str,
    epochs: np.ndarray,
    all_tiff_timestamps: np.ndarray,
    stack_lengths_tiffs: np.ndarray,
    chunk_lengths_daq: np.ndarray,
    frame_times_daq: np.ndarray,
) -> SessionCorrection:
    if (mouse_name, date) in session_corrections:
        print("Applying manual session corrections")
        correction_func = session_corrections[(mouse_name, date)]
        c = correction_func(
            SessionCorrection(
                epochs=epochs,
                all_tiff_timestamps=all_tiff_timestamps,
                stack_lengths_tiffs=stack_lengths_tiffs,
                chunk_lengths_daq=chunk_lengths_daq,
                frame_times_daq=frame_times_daq,
                offset_after_pre_epoch=0,
            )
        )
        assert sum(c.chunk_lengths_daq) == len(c.frame_times_daq)
        # I found a bug where if you deleted a column of the epochs array in a session correction,
        # all the assertions would pass and save the session cache regardless.
        # It is fixed now, but these assertions are here to ensure it does not happen again.
        assert c.epochs.shape[0] == len(
            c.stack_lengths_tiffs
        ), "There should be one epoch per tiff stack"
        assert (
            c.epochs.shape[1] == 6
        ), "Each epoch should have 6 values (year, month, ...)"
        return c

        assert sum(c.chunk_lengths_daq) == len(c.frame_times_daq)
        # I found a bug where if you deleted a column of the epochs array in a session correction,
        # all the assertions would pass and save the session cache regardless.
        # It is fixed now, but these assertions are here to ensure it does not happen again.
        assert c.epochs.shape[0] == len(
            c.stack_lengths_tiffs
        ), "There should be one epoch per tiff stack"
        assert (
            c.epochs.shape[1] == 6
        ), "Each epoch should have 6 values (year, month, ...)"
        return c

    return SessionCorrection(
        epochs=epochs,
        all_tiff_timestamps=all_tiff_timestamps,
        stack_lengths_tiffs=stack_lengths_tiffs,
        chunk_lengths_daq=chunk_lengths_daq,
        frame_times_daq=frame_times_daq,
        offset_after_pre_epoch=0,
    )


# epochs:               (n_tiffs, 6);   start time of each tiff file (they are in a matlab format, hence each epoch has 6 values)
# all_tiff_timestamps:  (n_frames);     timestamps of each frame in all tiff files (in seconds since start of the DAQ)
# stack_lengths_tiffs:  (n_tiffs,);     length of each tiff stack (in frames)
# chunk_lengths_daq:    (n_chunks,);    length of each daq chunk (in frames), i.e. all frame pulses emitted by the frame clock and recorded by the DAQ
# frame_times_daq:      (n_frames,);    timestamps of each frame in the DAQ (in time units of the DAQ, usually 10000 Hz, but depending on the DAQ sampling rate)

# In principle, the following rules apply to all sessions and are either checked by assertions directly or will cause other assertions to fail:
# 1. sum(chunk_lengths_daq) == len(frame_times_daq), i.e. total number of frames in the DAQ must match total number of frame timestamps
# 2. epochs.shape[0] == len(stack_lengths_tiffs), i.e. there must be one epoch per tiff stack
# 3. for each chunk of imaging, the chunk_length_daq can be 0-3 frames short of stack_length_tiff (see get_valid_frame_times())

# So, if applying certain corrections, it must be ensured that the above rules are still satisfied.
# See explanations in the individual corrections below.


# Session corrections ordered by mouse name and date.


# epochs:               (n_tiffs, 6);   start time of each tiff file (they are in a matlab format, hence each epoch has 6 values)
# all_tiff_timestamps:  (n_frames);     timestamps of each frame in all tiff files (in seconds since start of the DAQ)
# stack_lengths_tiffs:  (n_tiffs,);     length of each tiff stack (in frames)
# chunk_lengths_daq:    (n_chunks,);    length of each daq chunk (in frames), i.e. all frame pulses emitted by the frame clock and recorded by the DAQ
# frame_times_daq:      (n_frames,);    timestamps of each frame in the DAQ (in time units of the DAQ, usually 10000 Hz, but depending on the DAQ sampling rate)

# In principle, the following rules apply to all sessions and are either checked by assertions directly or will cause other assertions to fail:
# 1. sum(chunk_lengths_daq) == len(frame_times_daq), i.e. total number of frames in the DAQ must match total number of frame timestamps
# 2. epochs.shape[0] == len(stack_lengths_tiffs), i.e. there must be one epoch per tiff stack
# 3. for each chunk of imaging, the chunk_length_daq can be 0-3 frames short of stack_length_tiff (see get_valid_frame_times())

# So, if applying certain corrections, it must be ensured that the above rules are still satisfied.
# See explanations in the individual corrections below.


# Session corrections ordered by mouse name and date.


@register_correction("JB031", "2025-03-31")
def jb031_2025_03_31(c: SessionCorrection) -> SessionCorrection:
    # stack_lengths_tiffs
    # array([   161,  27000,    294, 113116,    165,  23307])
    # chunk_lengths_daq
    # array([  1325,   3224,    147,    296, 113118,    167,  23309,    657])
    # extremely troubled recording
    # I. daq was started after the pre-session epoch, with probably one accidental grab (very first tiff)
    # II. probably three "focus" without grabbing before the session
    # III. recording of the post-session epoch crashed
    # IV. probably one more "focus" without grabbing, which did not result in another grab
    c.stack_lengths_tiffs = np.array([294, 113116, 165, 23307])
    c.chunk_lengths_daq = np.array([296, 113118, 167, 23309])
    bad_tiff_len = sum([161, 27000])
    c.all_tiff_timestamps = c.all_tiff_timestamps[bad_tiff_len:]
    # epochs refer to the start time of each tiff file!!! they are in a matlab format, that's why each epoch has 6 values
    c.epochs = np.delete(c.epochs, [0, 1], axis=0)  # remove rows 1 and 2
    bad_daq_len_pre = sum([1325, 3224, 147])
    bad_daq_len_post = 657
    c.frame_times_daq = c.frame_times_daq[bad_daq_len_pre:-bad_daq_len_post]
    # TODO: the last chunk is shorter than 15 mins / 27,000 frames, do we need to remove the session?
    return SessionCorrection(
        epochs=c.epochs,
        all_tiff_timestamps=c.all_tiff_timestamps,
        stack_lengths_tiffs=c.stack_lengths_tiffs,
        chunk_lengths_daq=c.chunk_lengths_daq,
        frame_times_daq=c.frame_times_daq,
        offset_after_pre_epoch=27000,
    )


# Ex.: classic case of manual 'focus' without grabbing, resulting in a tiff stack with no associated DAQ chunk.
# The signals in the DAQ files have to be deleted, i.e. in chunk_lengths_daq and frame_times_daq.
# Ex.: classic case of manual 'focus' without grabbing, resulting in a tiff stack with no associated DAQ chunk.
# The signals in the DAQ files have to be deleted, i.e. in chunk_lengths_daq and frame_times_daq.
@register_correction("JB031", "2025-04-01")
def jb031_2025_04_01(c: SessionCorrection) -> SessionCorrection:
    # stack_lengths_tiffs
    # array([27000, 94469, 16183, 27000])
    # chunk_lengths_daq
    # array([27000,   246,   423, 94471, 16185, 27000])
    c.chunk_lengths_daq = np.delete(c.chunk_lengths_daq, [1, 2])
    c.frame_times_daq = np.concatenate(
        [c.frame_times_daq[:27000], c.frame_times_daq[sum([27000, 246, 423]) :]]
    )
    return SessionCorrection(
        epochs=c.epochs,
        all_tiff_timestamps=c.all_tiff_timestamps,
        stack_lengths_tiffs=c.stack_lengths_tiffs,
        chunk_lengths_daq=c.chunk_lengths_daq,
        frame_times_daq=c.frame_times_daq,
        offset_after_pre_epoch=0,
    )


# Ex.: Classic case of starting the DAQ after the pre-session epoch, resulting in a tiff stack with no associated DAQ chunk.
# The first tiff has to be removed from the syncing, i.e. in stack_lengths_tiffs, all_tiff_timestamps and epochs.
# The 'offset_after_pre_epoch' is set to the length of the first tiff stack,
# so that the DAQ signals keep on being aligned while the pre_session_epoch will be skipped.
# Ex.: Classic case of starting the DAQ after the pre-session epoch, resulting in a tiff stack with no associated DAQ chunk.
# The first tiff has to be removed from the syncing, i.e. in stack_lengths_tiffs, all_tiff_timestamps and epochs.
# The 'offset_after_pre_epoch' is set to the length of the first tiff stack,
# so that the DAQ signals keep on being aligned while the pre_session_epoch will be skipped.
@register_correction("JB031", "2025-04-02")
def jb031_2025_04_02(c: SessionCorrection) -> SessionCorrection:
    # stack_lengths_tiffs
    # array([ 27000, 147798,  27000])
    # chunk_lengths_daq
    # array([147800,  27000])
    c.stack_lengths_tiffs = np.delete(c.stack_lengths_tiffs, 0)
    c.all_tiff_timestamps = c.all_tiff_timestamps[27000:]
    c.epochs = np.delete(c.epochs, 0, axis=0)
    return SessionCorrection(
        epochs=c.epochs,
        all_tiff_timestamps=c.all_tiff_timestamps,
        stack_lengths_tiffs=c.stack_lengths_tiffs,
        chunk_lengths_daq=c.chunk_lengths_daq,
        frame_times_daq=c.frame_times_daq,
        offset_after_pre_epoch=27000,
    )


@register_correction("JB031", "2025-04-03")
def jb031_2025_04_03(c: SessionCorrection) -> SessionCorrection:
    # stack_lengths_tiffs
    # array([ 27000, 116771,  27000])
    # chunk_lengths_daq
    # array([ 27000, 116773,     46,  27000,   3028,   2634])
    c.chunk_lengths_daq = np.delete(c.chunk_lengths_daq, [2, 4, 5])
    c.frame_times_daq = np.concatenate(
        [
            c.frame_times_daq[0:27000],
            c.frame_times_daq[27000 : sum([27000, 116773])],
            c.frame_times_daq[
                sum([27000, 116773, 46]) : sum([27000, 116773, 46, 27000])
            ],
        ]
    )
    return SessionCorrection(
        epochs=c.epochs,
        all_tiff_timestamps=c.all_tiff_timestamps,
        stack_lengths_tiffs=c.stack_lengths_tiffs,
        chunk_lengths_daq=c.chunk_lengths_daq,
        frame_times_daq=c.frame_times_daq,
        offset_after_pre_epoch=0,
    )


# Ex.: Classic case of hitting 'focus' without grabbing before the session.
# Again, the signals in the DAQ files have to be deleted, i.e. in chunk_lengths_daq and frame_times_daq.
# Ex.: Classic case of hitting 'focus' without grabbing before the session.
# Again, the signals in the DAQ files have to be deleted, i.e. in chunk_lengths_daq and frame_times_daq.
@register_correction("JB031", "2025-04-04")
def jb031_2025_04_04(c: SessionCorrection) -> SessionCorrection:
    # stack_lengths_tiffs
    # array([ 27000, 108943,  27000])
    # chunk_lengths_daq
    # array([   104,  27000, 108945,  27000])
    c.chunk_lengths_daq = np.delete(c.chunk_lengths_daq, 0)
    c.frame_times_daq = c.frame_times_daq[104:]
    return SessionCorrection(
        epochs=c.epochs,
        all_tiff_timestamps=c.all_tiff_timestamps,
        stack_lengths_tiffs=c.stack_lengths_tiffs,
        chunk_lengths_daq=c.chunk_lengths_daq,
        frame_times_daq=c.frame_times_daq,
        offset_after_pre_epoch=0,
    )


@register_correction("JB032", "2025-03-27")
def jb032_2025_03_27(c: SessionCorrection) -> SessionCorrection:
    # stack_lengths_tiffs
    # array([ 27000,   6319, 104990,  12172,  27000,    113])
    # chunk_lengths_daq
    # array([ 27000,   6321, 104992,  12174,    868,  27000,    115])
    # TODO: check if all of this makes sense??
    # Deleting last tiff after post epoch, getting rid of focus w/o grabbing before post epoch
    c.stack_lengths_tiffs = np.delete(c.stack_lengths_tiffs, [5])
    c.epochs = np.delete(c.epochs, [5], axis=0)
    c.all_tiff_timestamps = c.all_tiff_timestamps[
        : sum([27000, 6319, 104990, 12172, 27000])
    ]
    c.chunk_lengths_daq = np.delete(c.chunk_lengths_daq, [4, 6])
    c.frame_times_daq = np.concatenate(
        [
            c.frame_times_daq[0 : sum([27000, 6321, 104992, 12174])],
            c.frame_times_daq[
                sum([27000, 6321, 104992, 12174, 868]) : sum(
                    [27000, 6321, 104992, 12174, 868, 27000]
                )
            ],
        ]
    )
    return SessionCorrection(
        epochs=c.epochs,
        all_tiff_timestamps=c.all_tiff_timestamps,
        stack_lengths_tiffs=c.stack_lengths_tiffs,
        chunk_lengths_daq=c.chunk_lengths_daq,
        frame_times_daq=c.frame_times_daq,
        offset_after_pre_epoch=0,
    )


@register_correction("JB032", "2025-04-01")
def jb032_2025_04_01(c: SessionCorrection) -> SessionCorrection:
    # stack_lengths_tiffs
    # array([27000,  3500,   132,  3914, 44414, 61165, 27000])
    # chunk_lengths_daq
    # array([27000,  3502,   134,  3916,  2657, 44416, 61167,    11, 27000])
    # the daq signals without any associated tiffs are due to focussing without grabbing
    c.chunk_lengths_daq = np.delete(c.chunk_lengths_daq, [4, 7])
    c.frame_times_daq = np.concatenate(
        [
            c.frame_times_daq[: sum([27000, 3502, 134, 3916])],
            c.frame_times_daq[
                sum([27000, 3502, 134, 3916, 2657]) : sum(
                    [27000, 3502, 134, 3916, 2657, 44416, 61167]
                )
            ],
            c.frame_times_daq[
                sum(
                    [
                        27000,
                        3502,
                        134,
                        3916,
                        2657,
                        44416,
                        61167,
                        11,
                    ]
                ) :
            ],
        ]
    )
    return SessionCorrection(
        epochs=c.epochs,
        all_tiff_timestamps=c.all_tiff_timestamps,
        stack_lengths_tiffs=c.stack_lengths_tiffs,
        chunk_lengths_daq=c.chunk_lengths_daq,
        frame_times_daq=c.frame_times_daq,
        offset_after_pre_epoch=0,
    )


@register_correction("JB033", "2025-03-20")
def jb033_2025_03_20(c: SessionCorrection) -> SessionCorrection:
    # stack_lengths_tiffs
    # array([  3215,    464,  28014, 122997,    105,  29148])
    # chunk_lengths_daq
    # array([   466,  28016, 122999,    107,  29151])
    # Deleting the first two grabs (3215; 464, 466 respectively) before the pre epoch
    c.stack_lengths_tiffs = np.delete(c.stack_lengths_tiffs, [0, 1])
    c.epochs = np.delete(c.epochs, [0, 1], axis=0)
    c.chunk_lengths_daq = np.delete(c.chunk_lengths_daq, [0])
    c.frame_times_daq = c.frame_times_daq[466:]
    c.all_tiff_timestamps = c.all_tiff_timestamps[3215 + 464 :]
    return SessionCorrection(
        epochs=c.epochs,
        all_tiff_timestamps=c.all_tiff_timestamps,
        stack_lengths_tiffs=c.stack_lengths_tiffs,
        chunk_lengths_daq=c.chunk_lengths_daq,
        frame_times_daq=c.frame_times_daq,
        offset_after_pre_epoch=0,
    )


@register_correction("JB033", "2025-06-17")
def jb033_2025_06_17(c: SessionCorrection) -> SessionCorrection:
    # stack_lengths_tiffs
    # array([ 1165, 27000, 54293, 17594, 49112, 27000])
    # chunk_lengths_daq
    # array([27000, 54295, 17596, 49115, 27000])
    c.stack_lengths_tiffs = np.delete(c.stack_lengths_tiffs, [0])
    c.epochs = np.delete(c.epochs, [0], axis=0)
    c.all_tiff_timestamps = c.all_tiff_timestamps[1165:]
    return SessionCorrection(
        epochs=c.epochs,
        all_tiff_timestamps=c.all_tiff_timestamps,
        stack_lengths_tiffs=c.stack_lengths_tiffs,
        chunk_lengths_daq=c.chunk_lengths_daq,
        frame_times_daq=c.frame_times_daq,
        offset_after_pre_epoch=0,
    )
