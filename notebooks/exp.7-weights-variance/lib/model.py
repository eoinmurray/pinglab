from pydantic import BaseModel

from pinglab.types import ExperimentConfig, LinspaceConfig, WeightSpec


class WeightSweepConfig(BaseModel):
    weight_sigma: LinspaceConfig


class HeatmapConfig(BaseModel):
    mean_scale: LinspaceConfig
    std_scale: LinspaceConfig
    bin_ms: float
    burn_in_ms: float
    tau_min_ms: float
    tau_max_ms: float


class GEISweepConfig(BaseModel):
    g_ei: LinspaceConfig


class GIESweepConfig(BaseModel):
    g_ie: LinspaceConfig


class GEESweepConfig(BaseModel):
    g_ee: LinspaceConfig


class GIISweepConfig(BaseModel):
    g_ii: LinspaceConfig


class LocalConfig(ExperimentConfig):
    sweep: WeightSweepConfig
    gei_sweep: GEISweepConfig
    gie_sweep: GIESweepConfig
    gee_sweep: GEESweepConfig
    gii_sweep: GIISweepConfig
    heatmap: HeatmapConfig
    weights: WeightSpec
