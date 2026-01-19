from pydantic import BaseModel

from pinglab.types import ExperimentConfig, LinspaceConfig


class InputScanConfig(BaseModel):
    I_I: float
    linspace: LinspaceConfig


class LocalConfig(ExperimentConfig):
    input_scan: InputScanConfig
