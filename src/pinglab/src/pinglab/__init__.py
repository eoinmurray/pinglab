"""
pinglab: Conductance-based E/I neural network simulation library.

A framework for simulating leaky integrate-and-fire (LIF) networks with
excitatory and inhibitory populations, synaptic delays, and heterogeneous
neural parameters.
"""

from .backends import simulate_network
from . import backends
from . import io
from . import analysis
from . import plots
from . import service

__version__ = "0.1.0"

__all__ = [
    "simulate_network",
    "backends",
    "io",
    "analysis",
    "plots",
    "service",
]
