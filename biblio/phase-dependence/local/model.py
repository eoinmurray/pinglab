
from pydantic import BaseModel
from pinglab.types import ExperimentConfig, LinspaceConfig


class PulseConfig(BaseModel):
    linspace: LinspaceConfig
    pre_window_ms: float
    post_window_ms: float
    width_ms: float
    amp: float
    repeats: int


class LocalConfig(ExperimentConfig):
    pulse: PulseConfig
