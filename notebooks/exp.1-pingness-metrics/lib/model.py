from pydantic import BaseModel

from pinglab.types import ExperimentConfig, LinspaceConfig


class ScanConfig(BaseModel):
    mean_ei: LinspaceConfig
    std_ei: float
    bin_ms: float
    burn_in_ms: float
    tau_min_ms: float
    tau_max_ms: float


class AnalysisConfig(BaseModel):
    autocorr_max_lag_ms: float = 200.0
    xcorr_max_lag_ms: float = 200.0
    xcorr_bin_ms: float = 5.0
    corr_min_lag_ms: float = 10.0
    corr_max_lag_ms: float = 150.0
    corr_peak_min: float = 0.1
    corr_peak_prominence: float = 0.02


class LocalConfig(ExperimentConfig):
    scan: ScanConfig
    analysis: AnalysisConfig = AnalysisConfig()
