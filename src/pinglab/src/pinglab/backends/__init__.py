"""Backend namespace with PyTorch as the default simulation backend."""

from .pytorch import lif_step, simulate_network, SimulationResult, SimulationState
from .types import *  # noqa: F403

__all__ = [
    "lif_step",
    "simulate_network",
    "SimulationResult",
    "SimulationState",
]
