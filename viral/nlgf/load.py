"""Reading digests back, and the session ledger.

One row per session, with what is available for it. Everything downstream selects
from this rather than globbing the drive, so the inclusion criteria for an analysis
are visible in one place.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.append(str(HERE.parent.parent))

from viral.nlgf.paths import DIGESTS, digest_path, spks_path
from viral.utils import get_session_type

TRIAL_COLUMNS = [
    "idx", "start_frame", "end_frame", "texture_rewarded", "imaged", "licks_AZ",
    "reward_drunk", "speed_AZ", "speed_nonAZ", "trial_speed", "trial_time",
    "n_licks", "reward_on",
]


@dataclass
class Digest:
    meta: Dict
    trials: pd.DataFrame
    run_position_cm: np.ndarray
    run_frame: np.ndarray
    run_lap: np.ndarray
    # Unthresholded samples: every encoder sample of every imaged trial, including the
    # slow ones. The thresholded arrays above drop frames below 5 cm/s, so anything
    # asking about SLOWING has to use these.
    run_position_cm_all: np.ndarray
    run_frame_all: np.ndarray
    lick_position_cm: np.ndarray
    lick_trial: np.ndarray
    freeze_bounds: Optional[np.ndarray]
    movement_pre: Optional[np.ndarray]
    movement_post: Optional[np.ndarray]

    @property
    def mouse(self) -> str:
        return self.meta["mouse"]

    @property
    def date(self) -> str:
        return self.meta["date"]

    @property
    def genotype(self) -> str:
        return self.meta["genotype"]

    def immobility_masks(self, n_frames: int) -> Dict[str, np.ndarray]:
        """Full-session immobility masks for the pre and post freeze epochs.

        Same construction as ensemble_reactivation.freeze_immobility_masks, but taking
        the movement vectors from the digest so the repairs in viral.nlgf.freeze apply.
        """
        if self.freeze_bounds is None or self.movement_pre is None:
            return {}
        pre_start, pre_end, post_start, post_end = self.freeze_bounds
        masks = {}
        for name, start, end, movement in (
            ("pre", pre_start, pre_end, self.movement_pre),
            ("post", post_start, post_end, self.movement_post),
        ):
            if end > n_frames:
                raise ValueError(
                    f"{self.mouse} {self.date}: {name} epoch ends at {end} but the "
                    f"imaging has {n_frames} frames"
                )
            mask = np.zeros(n_frames, dtype=bool)
            mask[start:end] = ~movement
            masks[name] = mask
        return masks


def load_digest(mouse: str, date: str) -> Digest:
    z = np.load(digest_path(mouse, date), allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    trials = pd.DataFrame(z["trials"], columns=list(z["trial_columns"]))
    return Digest(
        meta=meta,
        trials=trials,
        run_position_cm=z["run_position_cm"],
        run_frame=z["run_frame"],
        run_lap=z["run_lap"],
        run_position_cm_all=z["run_position_cm_all"],
        run_frame_all=z["run_frame_all"],
        lick_position_cm=z["lick_position_cm"],
        lick_trial=z["lick_trial"],
        freeze_bounds=z["freeze_bounds"] if "freeze_bounds" in z else None,
        movement_pre=z["movement_pre"] if "movement_pre" in z else None,
        movement_post=z["movement_post"] if "movement_post" in z else None,
    )


def ledger() -> pd.DataFrame:
    rows = []
    for p in sorted(DIGESTS.glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        try:
            stage = get_session_type(m["session_type"])
        except ValueError:
            stage = "unknown"
        rows.append(
            dict(
                mouse=m["mouse"],
                date=m["date"],
                genotype=m["genotype"],
                session_type=m["session_type"],
                stage=stage,
                n_trials=m["n_trials"],
                n_trials_imaged=m["n_trials_imaged"],
                has_freeze=m["has_freeze"],
                freeze_type=m.get("freeze_type"),
                freeze_usable=m.get("freeze_usable", False),
                repair_pre=m.get("repair_pre"),
                repair_post=m.get("repair_post"),
                has_spks=spks_path(m["mouse"], m["date"]).exists(),
            )
        )
    df = pd.DataFrame(rows)
    df["day"] = df.session_type.str.extract(r"day\s*(\d+)", expand=False).astype(float)
    return df.sort_values(["genotype", "mouse", "date"]).reset_index(drop=True)
