"""External input generation: tonic, oscillating, and pulse stimuli."""

from .tonic import tonic
from .oscillating import oscillating
from .pulse import add_pulse_to_input, compute_spike_delta

__all__ = [
    "tonic",
    "oscillating",
    "add_pulse_to_input",
    "compute_spike_delta",
]