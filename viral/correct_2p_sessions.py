"""For some sessions, focussing without grabbing or late start of the DAQ occured, requiring manual fixing."""

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
        return correction_func(
            SessionCorrection(
                epochs=epochs,
                all_tiff_timestamps=all_tiff_timestamps,
                stack_lengths_tiffs=stack_lengths_tiffs,
                chunk_lengths_daq=chunk_lengths_daq,
                frame_times_daq=frame_times_daq,
                offset_after_pre_epoch=0,
            )
        )
    return SessionCorrection(
        epochs=epochs,
        all_tiff_timestamps=all_tiff_timestamps,
        stack_lengths_tiffs=stack_lengths_tiffs,
        chunk_lengths_daq=chunk_lengths_daq,
        frame_times_daq=frame_times_daq,
        offset_after_pre_epoch=0,
    )


@register_correction("JB031", "2025-03-31")
def jb031_2025_03_31(c: SessionCorrection) -> SessionCorrection:
    # stack_lengths_tiffs
    # array([   161,  27000,    294, 113116,    165,  23307])
    # chunk_lengths_daq
    # array([  1325,   3224,    147,    296, 113118,    167,  23309,    657])
    c.stack_lengths_tiffs = np.array([294, 113116, 165, 23307])
    c.chunk_lengths_daq = np.array([296, 113118, 167, 23309])
    bad_tiff_len = sum([161, 27000])
    c.all_tiff_timestamps = c.all_tiff_timestamps[bad_tiff_len:]
    # epochs refer to the start time of each tiff file!!! they are in a matlab format, that's why each epoch has 6 values
    c.epochs = np.delete(c.epochs, [0, 1], axis=0)  # remove rows 1 and 2
    bad_daq_len_pre = sum([1325, 3224, 147])
    bad_daq_len_post = 657
    c.frame_times_daq = c.frame_times_daq[bad_daq_len_pre:-bad_daq_len_post]
    assert sum(c.chunk_lengths_daq) == len(c.frame_times_daq)
    # TODO: the last chunk is shorter than 15 mins / 27,000 frames, do we need to remove the session?
    return SessionCorrection(
        epochs=c.epochs,
        all_tiff_timestamps=c.all_tiff_timestamps,
        stack_lengths_tiffs=c.stack_lengths_tiffs,
        chunk_lengths_daq=c.chunk_lengths_daq,
        frame_times_daq=c.frame_times_daq,
        offset_after_pre_epoch=27000,
    )


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
            c.frame_times_daq[27000:143773],
            c.frame_times_daq[143819:170819],
        ]
    )
    assert sum(c.chunk_lengths_daq) == len(c.frame_times_daq)
    return SessionCorrection(
        epochs=c.epochs,
        all_tiff_timestamps=c.all_tiff_timestamps,
        stack_lengths_tiffs=c.stack_lengths_tiffs,
        chunk_lengths_daq=c.chunk_lengths_daq,
        frame_times_daq=c.frame_times_daq,
        offset_after_pre_epoch=0,
    )


@register_correction("JB031", "2025-04-04")
def jb031_2025_04_04(c: SessionCorrection) -> SessionCorrection:
    # stack_lengths_tiffs
    # array([ 27000, 108943,  27000])
    # chunk_lengths_daq
    # array([   104,  27000, 108945,  27000])
    c.chunk_lengths_daq = np.delete(c.chunk_lengths_daq, 0)
    c.frame_times_daq = c.frame_times_daq[104:]
    assert sum(c.chunk_lengths_daq) == len(c.frame_times_daq)
    return SessionCorrection(
        epochs=c.epochs,
        all_tiff_timestamps=c.all_tiff_timestamps,
        stack_lengths_tiffs=c.stack_lengths_tiffs,
        chunk_lengths_daq=c.chunk_lengths_daq,
        frame_times_daq=c.frame_times_daq,
        offset_after_pre_epoch=0,
    )
