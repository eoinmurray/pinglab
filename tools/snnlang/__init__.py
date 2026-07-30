"""snnlang: deterministic authoring and inspection of spiking network graphs."""

from . import components, ops, readouts, training
from .compiler import Bundle, Diagnostic, compile, load_bundle, validate_graph
from .core import (
    AMPA,
    COBA_LIF,
    GABA,
    LIF,
    Constant,
    Hz,
    LeakyIntegrator,
    Modulatory,
    Network,
    NonNegative,
    Normal,
    ParameterRef,
    Population,
    Projection,
    Quantity,
    Signal,
    ms,
    mV,
    nS,
)
from .training import TrainSpec

__all__ = [
    "AMPA",
    "GABA",
    "COBA_LIF",
    "LIF",
    "Constant",
    "LeakyIntegrator",
    "Modulatory",
    "Network",
    "NonNegative",
    "Normal",
    "ParameterRef",
    "Population",
    "Projection",
    "Quantity",
    "Signal",
    "TrainSpec",
    "Bundle",
    "Diagnostic",
    "compile",
    "load_bundle",
    "validate_graph",
    "components",
    "ops",
    "readouts",
    "training",
    "Hz",
    "mV",
    "ms",
    "nS",
]

__version__ = "0.1.0"
