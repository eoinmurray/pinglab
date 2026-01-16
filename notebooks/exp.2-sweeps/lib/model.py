
from pinglab.types import ExperimentConfig, LinspaceConfig
from pydantic import BaseModel

class LocalExperimentConfig(BaseModel):
    I_E: float | None = None
    g_ei: float | None = None
    linspace: LinspaceConfig | None = None

class LocalExperiment3Config(BaseModel):
    g_ei: float
    linspace_large: LinspaceConfig
    linspace_small: LinspaceConfig

class LocalConfig(ExperimentConfig):
    experiment_1: LocalExperimentConfig
    experiment_2: LocalExperimentConfig
    experiment_3: LocalExperiment3Config
    experiment_4: LocalExperimentConfig
    experiment_5: LocalExperimentConfig
    experiment_6: LocalExperimentConfig