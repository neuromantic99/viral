from typing import List
from pydantic import BaseModel, computed_field


class StateInfo(BaseModel):
    name: str
    start_time: float
    end_time: float
    start_time_daq: float | None = None  # Possibly redundant
    end_time_daq: float | None = None
    closest_frame_start: int | None = None
    closest_frame_end: int | None = None


class EventInfo(BaseModel):
    name: str
    start_time: float
    start_time_daq: float | None = None
    closest_frame: int | None = None


class SpeedPosition(BaseModel):
    position_start: int
    position_stop: int
    speed: float


class TrialSummary(BaseModel):
    speed_trial: List[SpeedPosition]
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
    sessions: List[SessionSummary]


class TrialInfo(BaseModel):
    trial_start_time: float
    trial_end_time: float
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


class Cached2pSession(BaseModel):
    trials: List[TrialInfo]
    mouse_name: str
    date: str
    session_type: str
