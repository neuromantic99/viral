"""Freeze-epoch movement vectors, with the malformed cases handled explicitly.

viral.utils.get_movement_bool accepts a movement vector whose length is the epoch
length or up to 2 short (movements come from diffs, sometimes of diffs) and pads the
remainder. Nine cached sessions fall outside that, in four distinct ways, and the
existing assert cannot tell them apart - two of them surface as an IndexError from
np.array(None).shape[0] rather than as the assert's message.

Three of the four are recoverable and are recovered here. This is deliberately a
separate implementation rather than an edit to get_movement_bool: the recovery makes
assumptions (below) that should not silently apply to the rest of the codebase, and
the WHOLE_SESSION case in particular is a bug at the caching stage that is better
fixed there than papered over everywhere.

  EXACT          len in {L, L-1, L-2}   pad to L, repeating the last value (as before)
  OFF_BY_ONE     len == L + 1           truncate. A diff that kept one sample too many.
  WHOLE_SESSION  len == epoch_end       slice [start:end]. Verified on all five
                                        affected sessions: len(movement_post) equals
                                        post_training_end_frame AND equals the frame
                                        count of the session's spks, so the vector is
                                        the whole recording and absolute frame indices
                                        address it correctly.
  MISSING        movement is None       unrecoverable; freeze_movement_type is also
                                        None on these, so movement was never extracted.
  SHORT          anything else          unrecoverable. J030_2026-05-11 is the only
                                        case: 20955 and 20556 samples for two 27000
                                        frame epochs, i.e. ~77% of each, with a
                                        different ratio per epoch. Not a slice or a
                                        rate mismatch, so it is not guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from viral.models import WheelFreeze

Repair = Literal["exact", "off_by_one", "whole_session", "missing", "short"]


@dataclass
class MovementResult:
    movement_pre: Optional[np.ndarray]
    movement_post: Optional[np.ndarray]
    repair_pre: Repair
    repair_post: Repair

    @property
    def usable(self) -> bool:
        return self.movement_pre is not None and self.movement_post is not None


def _one_epoch(
    movement: Optional[list], start: int, end: int
) -> Tuple[Optional[np.ndarray], Repair]:
    if movement is None:
        return None, "missing"

    m = np.asarray(movement)
    length = end - start

    if m.shape[0] in (length, length - 1, length - 2):
        return (
            np.pad(m, (0, length - m.shape[0]), mode="edge").astype(bool),
            "exact",
        )
    if m.shape[0] == length + 1:
        return m[:length].astype(bool), "off_by_one"
    if m.shape[0] == end:
        return m[start:end].astype(bool), "whole_session"
    return None, "short"


def movement_bools(wheel_freeze: WheelFreeze) -> MovementResult:
    pre, r_pre = _one_epoch(
        wheel_freeze.movement_pre_freeze,
        wheel_freeze.pre_training_start_frame,
        wheel_freeze.pre_training_end_frame,
    )
    post, r_post = _one_epoch(
        wheel_freeze.movement_post_freeze,
        wheel_freeze.post_training_start_frame,
        wheel_freeze.post_training_end_frame,
    )
    return MovementResult(pre, post, r_pre, r_post)
