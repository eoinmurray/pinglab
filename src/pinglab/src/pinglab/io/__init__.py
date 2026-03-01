"""I/O and spec/graph tooling: inputs, graph compilation, scans, and utilities."""

from .compiler import RuntimeBundle, compile_graph, compile_graph_to_runtime
from .graph_utils import layer_bounds_from_spec, set_edges_enabled
from .external_spike_train import external_spike_train
from .oscillating import oscillating
from .pulse import add_pulse_to_input, add_pulse_train_to_input, compute_spike_delta
from .ramp import ramp
from .overrides import overwrite_spec_value, overwrite_spec_value_inplace, spec_with_overwrite
from .scans import collect_scans, linspace_from_scan, scan_variant
from .renderer import save_graph_diagram
from .slice_spikes import slice_spikes

__all__ = [
    "compile_graph",
    "compile_graph_to_runtime",
    "RuntimeBundle",
    "layer_bounds_from_spec",
    "set_edges_enabled",
    "external_spike_train",
    "oscillating",
    "add_pulse_to_input",
    "add_pulse_train_to_input",
    "compute_spike_delta",
    "ramp",
    "overwrite_spec_value",
    "overwrite_spec_value_inplace",
    "spec_with_overwrite",
    "collect_scans",
    "linspace_from_scan",
    "scan_variant",
    "save_graph_diagram",
    "slice_spikes",
]
