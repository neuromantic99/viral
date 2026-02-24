from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Literal
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


class Mouse2pSessions(BaseModel):
    mouse_name: str
    unsupervised: Cached2pSession
    learning: Cached2pSession
    learned: Cached2pSession


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


@dataclass
class EnsembleSessionResult:
    reactivation_triggered_response: Tuple[np.ndarray, np.ndarray]
    number_of_events: Tuple[np.ndarray, np.ndarray]
    sum_values_over_threshold: Tuple[np.ndarray, np.ndarray]


@dataclass
class MultipleSessionsConfig:
    window: int
    speed: float
    licking: float

    def __post_init__(self):
        if sum([self.speed, self.licking]) != 1:
            raise ValueError(
                f"Invalid weights for speed and licking in learning metric! Sum has to equal to 1, instead it is {sum([self.speed, self.licking])}"
            )


@dataclass
class SSPConfig:
    mode: Literal["below", "above"]
    speed_threshold: float
    n_consecutive_samples: int


@dataclass
class SSPVectorData:
    ssp_vectors: np.ndarray
    position_vectors: np.ndarray
    trial_start_indices: np.ndarray
    chunk_start_indices: List[
        List[int]
    ]  # list of chunk start indices with each trial being an element in the outer list


@dataclass
class BayesianDecodingConfig:
    peak_threshold: float  # SDs above mean
    edge_threshold: float  # SDs above mean
    event_duration: Tuple[float, float]  # min, max (frames)
    bin_size_time_offline: int  # frames
    bin_size_time_online: int  # frames
    start_spatial: int  # cms
    end_spatial: int  # cms
    bin_size_spatial: int  # cms
    en_bloc: bool  # whether to decode en bloc or pse event by pse event
    online: (
        bool  # True for the online epoch, False for the offline post wheel freeze epoch
    )
    sigma_offline: int  # frames (for the convolving with Gaussian kernel bit)
    sigma_online: int  # frames (for the convolving with Gaussian kernel bit)

    @computed_field
    @property
    def total_length(self) -> float:
        # TODO: Think about this! Grosmark used metres instead of centimetres
        return (self.end_spatial - self.start_spatial) / 100


# TODO: rethink this, currently unused
# @dataclass
# class ReplayEvent:
#     start_frame: int
#     end_frame: int
#     posterior_probability_matrix: np.ndarray
#     pr_max: np.ndarray
#     p_value: float
#     weighted_r: float
#     rz_score: float


@dataclass
class BayesianDecodingResult:
    posterior_probability_matrices: List[np.ndarray]
    pr_max_matrices: List[np.ndarray]
    linear_weighted_r: List[float]
    circular_weighted_r: List[float]
    actual_positions: List[np.ndarray] | None


@dataclass
class RadonLUT:
    path_length: np.ndarray
    xp: np.ndarray
    theta: np.ndarray
    n_radon_points: int
    point1x: np.ndarray
    point1y: np.ndarray
    point2x: np.ndarray
    point2y: np.ndarray
    slope: np.ndarray
    path_length_from_points: np.ndarray
    space_offset: np.ndarray
    temp_offset: np.ndarray
    space_offset_round: np.ndarray
    temp_offset_round: np.ndarray
    temp_offset_round_perc: np.ndarray


@dataclass
class RadonReplayResult:
    pos_mean: float  # mean posterior probability of best line
    # max_id: int  # linear index of best line
    path_length: float
    point1x: float
    point1y: float
    point2x: float
    point2y: float
    slope: float
    slope_metres_per_sec: float
    replay_type: Literal["forward", "reverse"]
