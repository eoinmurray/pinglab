"""
pinglab: Conductance-based E/I neural network simulation library.

A framework for simulating leaky integrate-and-fire (LIF) networks with
excitatory and inhibitory populations, synaptic delays, and heterogeneous
neural parameters.
"""

from .run import run_network, build_model_from_config
from . import types
from . import inputs
from . import utils
from . import plots
from . import analysis
from . import multiprocessing

__version__ = "0.1.0"

__all__ = [
    "run_network",
    "build_model_from_config",
    "types",
    "inputs",
    "utils",
    "plots",
    "analysis",
    "multiprocessing",
]
