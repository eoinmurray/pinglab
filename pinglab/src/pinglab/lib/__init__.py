"""
Core computational components for neural simulation.
"""
from .lif import lif_step
from .synapse import decay_exponential

__all__ = ["lif_step", "decay_exponential"]
