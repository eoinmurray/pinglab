from pydantic import BaseModel

from pinglab.types import ExperimentConfig, LinspaceConfig


class VarianceScanConfig(BaseModel):
    std_ei: LinspaceConfig


class PlottingRasterConfig(BaseModel):
    start_time: float
    stop_time: float


class PlottingConfig(BaseModel):
    raster: PlottingRasterConfig


class LocalConfig(ExperimentConfig):
    plotting: PlottingConfig
    scan: VarianceScanConfig
