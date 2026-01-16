from pydantic import BaseModel

from pinglab.types import ExperimentConfig, LinspaceConfig


class SweepConfig(BaseModel):
    I_E: LinspaceConfig
    bin_ms: float
    burn_in_ms: float
    tau_min_ms: float
    tau_max_ms: float


class SweepOverrides(BaseModel):
    I_E: LinspaceConfig | None = None
    bin_ms: float | None = None
    burn_in_ms: float | None = None
    tau_min_ms: float | None = None
    tau_max_ms: float | None = None


class ModelConfig(BaseModel):
    name: str
    neuron_model: str
    overrides: dict[str, float | int | list[float]] = {}
    input_scale_E: float = 1.0
    input_scale_I: float = 1.0
    sweep: SweepOverrides | None = None


class LocalConfig(ExperimentConfig):
    sweep: SweepConfig
    models: list[ModelConfig]
