from __future__ import annotations

import numpy as np


def add_fiber_spikes_to_input(
    *,
    baseline_input: np.ndarray,
    fiber_targets: np.ndarray,
    spikes_per_fiber: list[np.ndarray],
    weight: float,
    dt: float,
    pulse_width_ms: float,
) -> np.ndarray:
    """
    Add fiber spike events to an external input array.

    Each fiber spike adds `weight` to all target E neurons for a short pulse.
    """
    if baseline_input.ndim != 2:
        raise ValueError("baseline_input must be 2D (num_steps, N)")

    num_steps = baseline_input.shape[0]
    pulse_steps = max(1, int(round(pulse_width_ms / dt)))
    out = baseline_input.copy()

    for fiber_id, spike_times in enumerate(spikes_per_fiber):
        if spike_times.size == 0:
            continue
        targets = fiber_targets[fiber_id]
        for t_ms in spike_times:
            step0 = int(t_ms / dt)
            if step0 < 0 or step0 >= num_steps:
                continue
            step1 = min(num_steps, step0 + pulse_steps)
            out[step0:step1, targets] += weight
    return out
