
from pinglab.types import ExperimentConfig, LinspaceConfig
from pydantic import BaseModel

class LocalExperimentConfig(BaseModel):
    I_E: float | None = None
    g_ei: float | None = None
    linspace: LinspaceConfig | None = None

class LocalConfig(ExperimentConfig):
    experiment_1: LocalExperimentConfig
