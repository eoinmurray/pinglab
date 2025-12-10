"""
pinglab: Conductance-based E/I neural network simulation library.

A framework for simulating leaky integrate-and-fire (LIF) networks with
excitatory and inhibitory populations, synaptic delays, and heterogeneous
neural parameters.
"""

from .run.run_network import run_network
from . import types
from . import inputs
from . import utils
from . import plots
from . import analysis
from . import multiprocessing

__version__ = "0.1.0"

__all__ = [
    "run_network",
    "types",
    "inputs",
    "utils",
    "plots",
    "analysis",
    "multiprocessing",
]
