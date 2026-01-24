from pydantic import BaseModel

from pinglab.types import ExperimentConfig, LinspaceConfig


class ScanConfig(BaseModel):
    mean_ei: LinspaceConfig
    std_ei: float
    bin_ms: float
    burn_in_ms: float
    tau_min_ms: float
    tau_max_ms: float


class LocalConfig(ExperimentConfig):
    scan: ScanConfig
    seed_scan: list[int] = []
    seed_scan_mean_ei: list[float] = [0.01]
    seed_scan_std_ei: float = 0.005
