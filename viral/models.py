from typing import List
from pydantic import BaseModel, computed_field


class StateInfo(BaseModel):
    name: str
    start_time: float
    end_time: float


class EventInfo(BaseModel):
    name: str
    start_time: float


class TrialInfo(BaseModel):
    trial_start_time: float
    trial_end_time: float
    pc_timestamp: str
    states_info: List[StateInfo]
    events_info: List[EventInfo]
    rotary_encoder_position: List[float]
    texture: str
    texture_rewarded: bool

    @computed_field
    def lick_start(self) -> List[float]:
        return [
            event.start_time for event in self.events_info if event.name == "Port1In"
        ]

    @computed_field
    def lick_end(self) -> List[float]:
        return [
            event.start_time for event in self.events_info if event.name == "Port1Out"
        ]
