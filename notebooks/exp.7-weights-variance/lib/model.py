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


class WindowConfig(BaseModel):
    start: float
    stop: float


class RateMatchConfig(BaseModel):
    I_E: LinspaceConfig
    window_ms: WindowConfig
    sim_T: float
    target_rate_e: float


class PlottingRasterConfig(BaseModel):
    start_time: float
    stop_time: float


class PlottingConfig(BaseModel):
    raster: PlottingRasterConfig


class ParallelConfig(BaseModel):
    n_jobs: int = 1


class LocalConfig(ExperimentConfig):
    plotting: PlottingConfig
    scan: VarianceScanConfig
    rate_match: RateMatchConfig
    parallel: ParallelConfig = ParallelConfig()
    ri_burn_in_ms: float = 200.0
