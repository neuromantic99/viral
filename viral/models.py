from dataclasses import dataclass
from typing import List, Optional, Dict
from pydantic import BaseModel, computed_field
from datetime import datetime
import numpy as np


class StateInfo(BaseModel):
    name: str
    start_time: float
    end_time: float
    start_time_daq: float | None = None  # Possibly redundant
    end_time_daq: float | None = None
    closest_frame_start: float | int | None = None
    closest_frame_end: float | int | None = None


class EventInfo(BaseModel):
    name: str
    start_time: float
    start_time_daq: float | None = None
    closest_frame: float | int | None = None


class SpeedPosition(BaseModel):
    position_start: int
    position_stop: int
    speed: float


class TrialSummary(BaseModel):
    speed_AZ: float
    speed_nonAZ: float
    trial_speed: float
    licks_AZ: int
    rewarded: bool
    reward_drunk: bool
    trial_time_overall: float


class SessionSummary(BaseModel):
    name: str
    trials: List[TrialSummary]
    rewarded_licks: List[int]
    unrewarded_licks: List[int]

    @computed_field
    @property
    def num_trials(self) -> int:
        return len(self.trials)

    @computed_field
    @property
    def num_rewarded_trials(self) -> int:
        return sum(trial.rewarded for trial in self.trials)

    @computed_field
    @property
    def num_unrewarded_trials(self) -> int:
        return self.num_trials - self.num_rewarded_trials


class MouseSummary(BaseModel):
    name: str
    genotype: str
    sex: str
    setup: Dict[str, str]
    rewarded_texture: Dict[str, str]
    sessions: List[SessionSummary]


class TrialInfo(BaseModel):
    trial_start_time: float
    trial_end_time: float
    trial_start_closest_frame: float | int | None = None
    trial_end_closest_frame: float | int | None = None
    pc_timestamp: str
    states_info: List[StateInfo]
    events_info: List[EventInfo]
    rotary_encoder_position: List[float]
    texture: str
    texture_rewarded: bool

    # Type is ignored due to  an open issue with pydantic-mypy interfacing
    # github.com/python/mypy/issues/14461
    @computed_field  # type: ignore
    @property
    def lick_start(self) -> List[float]:
        return [
            event.start_time for event in self.events_info if event.name == "Port1In"
        ]

    @computed_field  # type: ignore
    @property
    def lick_end(self) -> List[float]:
        return [
            event.start_time for event in self.events_info if event.name == "Port1Out"
        ]

    @computed_field  # type: ignore
    @property
    def reward_on(self) -> List[float]:
        return [
            state.start_time for state in self.states_info if state.name == "reward_on"
        ]


class WheelFreeze(BaseModel):
    pre_training_start_frame: int
    pre_training_end_frame: int
    post_training_start_frame: int
    post_training_end_frame: int


class Cached2pSession(BaseModel):
    trials: List[TrialInfo]
    mouse_name: str
    date: str
    session_type: str
    wheel_freeze: WheelFreeze | None = None


class ImagedTrialInfo(BaseModel):
    trial_start_frame: int
    trial_end_frame: int
    rewarded: int
    trial_frames: np.ndarray
    iti_start_frame: int
    iti_end_frame: int
    frames_positions: np.ndarray
    frames_speed: np.ndarray
    corridor_width: int
    lick_idx: Optional[np.ndarray] = None
    reward_idx: Optional[np.ndarray] = None
    signal: np.ndarray

    class Config:
        arbitrary_types_allowed = True


@dataclass
class GrosmarkConfig:
    bin_size: int
    start: int
    end: int

    def __repr__(self) -> str:
        return f"GrosmarkConfig(bin_size={self.bin_size}, start={self.start}, end={self.end})"


@dataclass
class SortedPlaceCells:
    sorted_indices: np.ndarray
    n_ensemble_a: int
    n_ensemble_b: int


@dataclass
class SessionImagingInfo:
    # 2p / ScanImage info
    stack_lengths_tiffs: np.ndarray
    epochs: np.ndarray
    all_tiff_timestamps: np.ndarray
    # DAQ info
    chunk_lengths_daq: np.ndarray
    daq_start_time: datetime
    # "results"
    valid_frame_times: np.ndarray
    behaviour_chunk_lens: np.ndarray
    behaviour_times: np.ndarray
    sampling_rate: int
    offset_after_pre_epoch: int


@dataclass
class SessionCorrection:
    epochs: np.ndarray
    all_tiff_timestamps: np.ndarray
    stack_lengths_tiffs: np.ndarray
    chunk_lengths_daq: np.ndarray
    frame_times_daq: np.ndarray
    offset_after_pre_epoch: int
