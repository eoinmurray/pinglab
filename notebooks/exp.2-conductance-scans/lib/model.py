from pydantic import BaseModel

from pinglab.types import ExperimentConfig, LinspaceConfig


class GEIScanConfig(BaseModel):
    I_E: float
    linspace: LinspaceConfig


class LocalConfig(ExperimentConfig):
    gei_scan: GEIScanConfig
    gie_scan: GEIScanConfig
    gee_scan: GEIScanConfig
    gii_scan: GEIScanConfig
