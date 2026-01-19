from pydantic import BaseModel

from pinglab.types import ExperimentConfig, LinspaceConfig


class SweepConfig(BaseModel):
    I_E: LinspaceConfig


class PlvConfig(BaseModel):
    bin_ms: float
    burn_in_ms: float
    fmin: float
    fmax: float


class LocalConfig(ExperimentConfig):
    sweep: SweepConfig
    plv: PlvConfig
