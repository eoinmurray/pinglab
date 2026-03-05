"""PyTorch backend scaffold."""

from .simulate_network import (
    SimulationResult,
    SimulationState,
    lif_step,
    prepare_runtime_tensors,
    reset_simulation_state,
    simulate_network,
)
from .surrogate import SpikeFunction, surrogate_lif_step
from .training import get_device, run_batch
from .e_prop import run_batch_eprop, compute_eprop_gradients, train_epoch_eprop

__all__ = [
    "lif_step",
    "SimulationState",
    "SimulationResult",
    "simulate_network",
    "prepare_runtime_tensors",
    "reset_simulation_state",
    "SpikeFunction",
    "surrogate_lif_step",
    "get_device",
    "run_batch",
    "run_batch_eprop",
    "compute_eprop_gradients",
    "train_epoch_eprop",
]
