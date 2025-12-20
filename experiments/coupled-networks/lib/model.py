from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

from pinglab.types import ExperimentConfig

ImageType = Literal["blobs", "bars", "checker"]
ReconMethod = Literal["linear_rescale"]


class HomeostasisConfig(BaseModel):
    """Configuration for the homeostatic rate controller."""

    target_rate_E: float = 20.0
    """Target mean E firing rate (Hz)."""

    burnin_ms: float = 300.0
    """Duration of each burn-in simulation for rate measurement (ms)."""

    max_iters: int = 10
    """Maximum tuning iterations."""

    eta: float = 0.05
    """Learning rate for I_E updates (nA / Hz)."""

    tol: float = 1.0
    """Convergence tolerance (Hz). Stop if |error| < tol."""


class SingleImageExperimentConfig(BaseModel):
    image_h: int = 16
    image_w: int = 16
    image_type: ImageType = "blobs"
    image_seed: int = 0
    image_contrast: float = 1.0

    group_size: int = 5
    mapping_seed: int = 0
    pixel_value_range: tuple[float, float] = (0.0, 1.0)

    warmup_ms: float = 400.0
    stim_ms: float = 600.0
    image_current_scale: float = 0.8

    readout_bin_ms: float = 600.0
    recon_method: ReconMethod = "linear_rescale"

class LocalConfig(ExperimentConfig):
    homeostasis: HomeostasisConfig = Field(default_factory=HomeostasisConfig)
    experiment_1: SingleImageExperimentConfig = Field(default_factory=SingleImageExperimentConfig)
