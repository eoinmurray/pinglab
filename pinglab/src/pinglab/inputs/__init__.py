"""External input generation: ramp, oscillating, and pulse stimuli."""

from .oscillating import oscillating
from .pulse import add_pulse_to_input, add_pulse_train_to_input, compute_spike_delta
from .ramp import ramp

__all__ = [
    "oscillating",
    "add_pulse_to_input",
    "add_pulse_train_to_input",
    "compute_spike_delta",
    "ramp",
]
