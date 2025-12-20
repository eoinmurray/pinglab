from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

from pinglab.types import ExperimentConfig, LinspaceConfig

PhaseSignal = Literal["g_i_mean_E", "rate_E", "rate_I"]


class PhaseConfig(BaseModel):
    signal: PhaseSignal = "g_i_mean_E"
    rate_bin_ms: float = 2.0
    smoothing_ms: float = 5.0


class InputProjectionConfig(BaseModel):
    num_fibers: int = 200
    targets_per_fiber: int = 80
    weight: float = 0.6
    seed: int = 1


class PacketConfig(BaseModel):
    times: LinspaceConfig
    delays_ms: LinspaceConfig
    width_ms: float = 3.0
    mean_spikes_per_fiber: float = 1.0
    jitter_ms: list[float] = Field(default_factory=lambda: [0.0, 1.0, 2.0, 5.0])
    trials_per_condition: int = 10
    seed: int = 2


class ReadoutConfig(BaseModel):
    window_ms: float = 20.0
    baseline_ms: float = 20.0


class LocalConfig(ExperimentConfig):
    phase: PhaseConfig = Field(default_factory=PhaseConfig)
    projection: InputProjectionConfig = Field(default_factory=InputProjectionConfig)
    packet: PacketConfig
    readout: ReadoutConfig = Field(default_factory=ReadoutConfig)
