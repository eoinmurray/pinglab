from typing import Literal
from pydantic import BaseModel
from pinglab.types import ExperimentConfig, LinspaceConfig


class SweepConfig(BaseModel):
    param: str
    linspace: LinspaceConfig
    outputs: list[Literal["raster", "psd", "metrics"]]


class LocalConfig(ExperimentConfig):
    sweep: SweepConfig
