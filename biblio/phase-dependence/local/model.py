
from pinglab.types import ExperimentConfig, LinspaceConfig
from pydantic import BaseModel

class PulseConfig(BaseModel):
    linspace: LinspaceConfig
    pre_window_ms: float
    post_window_ms: float
    width_ms: float
    amp: float

class LocalConfig(ExperimentConfig):
    pulse: PulseConfig