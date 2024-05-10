from typing import List
from pydantic import BaseModel, computed_field


class StateInfo(BaseModel):
    name: str
    start_time: float
    end_time: float


class EventInfo(BaseModel):
    name: str
    start_time: float


class SpeedPosition(BaseModel):
    position_start: int
    position_stop: int
    speed: float


class TrialSummary(BaseModel):
    speed_AZ: float
    licks_AZ: int
    rewarded: bool


class SessionSummary(BaseModel):
    name: str
    trials: List[TrialSummary]


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
