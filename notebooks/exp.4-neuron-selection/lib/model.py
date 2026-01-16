from pydantic import BaseModel

from pinglab.types import ExperimentConfig, LinspaceConfig


class SweepConfig(BaseModel):
    I_E: LinspaceConfig
    gif_levels: list[float]
    raster_levels: list[float]
    burn_in_ms: float
    bin_ms: float
    gif_frame_duration_s: float = 0.7


class ModelConfig(BaseModel):
    name: str
    neuron_model: str
    overrides: dict[str, float | int | list[float]] = {}
    input_scale_E: float = 1.0
    input_scale_I: float = 1.0


class LocalConfig(ExperimentConfig):
    sweep: SweepConfig
    models: list[ModelConfig]
