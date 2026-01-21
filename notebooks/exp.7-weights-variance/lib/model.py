from pydantic import BaseModel

from pinglab.types import ExperimentConfig, LinspaceConfig


class VarianceScanConfig(BaseModel):
    mean_ei: LinspaceConfig
    std_ei: LinspaceConfig
    mean_ie: LinspaceConfig
    std_ie: LinspaceConfig
    mean_ee: LinspaceConfig
    std_ee: LinspaceConfig
    mean_ii: LinspaceConfig
    std_ii: LinspaceConfig


class PlottingRasterConfig(BaseModel):
    start_time: float
    stop_time: float


class PlottingConfig(BaseModel):
    raster: PlottingRasterConfig


class LocalConfig(ExperimentConfig):
    plotting: PlottingConfig
    scan: VarianceScanConfig
