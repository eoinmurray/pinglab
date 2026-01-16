from pydantic import BaseModel

from pinglab.types import ExperimentConfig, LinspaceConfig


class WeightSweepConfig(BaseModel):
    weight_sigma: LinspaceConfig


class WeightConfig(BaseModel):
    dist: str = "normal"
    sample_size: int = 10000
    clamp_min: float | None = 0.0
    sigma_relative: bool = True
    g_ee: float
    g_ei: float
    g_ie: float
    g_ii: float


class LocalConfig(ExperimentConfig):
    sweep: WeightSweepConfig
    weights: WeightConfig
