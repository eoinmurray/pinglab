"""Pulse stimulus generation and spike response analysis."""

import numpy as np

from pinglab.backends.types import Spikes


def add_pulse_to_input(
    baseline_input: np.ndarray,
    target_neurons: np.ndarray,
    pulse_t: float,
    pulse_width_ms: float,
    pulse_amp: float,
    dt: float,
    num_steps: int | None = None,
) -> np.ndarray:
    """
    Add a pulse stimulus to target neurons at a specified time.

    Parameters:
        baseline_input: Base input array of shape (num_steps, N)
        target_neurons: Array of neuron IDs to stimulate
        pulse_t: Pulse onset time in milliseconds
        pulse_width_ms: Pulse duration in milliseconds
        pulse_amp: Pulse amplitude (added to baseline)
        dt: Time step in milliseconds
        num_steps: Total simulation steps (inferred from baseline_input if None)

    Returns:
        Modified input array with pulse added (copy of baseline_input)
    """
    pulse_input = baseline_input.copy()
    if num_steps is None:
        num_steps = baseline_input.shape[0]

    p0 = int(pulse_t / dt)
    p1 = int((pulse_t + pulse_width_ms) / dt)
    p0 = max(0, min(p0, num_steps - 1))
    p1 = max(p0 + 1, min(p1, num_steps))
    pulse_input[p0:p1, target_neurons] += pulse_amp

    return pulse_input


def add_pulse_train_to_input(
    baseline_input: np.ndarray,
    target_neurons: np.ndarray,
    pulse_t: float,
    pulse_width_ms: float,
    pulse_amp: float,
    pulse_interval_ms: float,
    dt: float,
    num_steps: int | None = None,
) -> np.ndarray:
    """
    Add a pulse train stimulus to target neurons.

    Parameters:
        baseline_input: Base input array of shape (num_steps, N)
        target_neurons: Array of neuron IDs to stimulate
        pulse_t: First pulse onset time in milliseconds
        pulse_width_ms: Pulse duration in milliseconds
        pulse_amp: Pulse amplitude (added to baseline)
        pulse_interval_ms: Interval between pulse onsets in milliseconds
        dt: Time step in milliseconds
        num_steps: Total simulation steps (inferred from baseline_input if None)

    Returns:
        Modified input array with pulse train added (copy of baseline_input)
    """
    if pulse_interval_ms <= 0:
        raise ValueError("pulse_interval_ms must be positive")

    pulse_input = baseline_input.copy()
    if num_steps is None:
        num_steps = baseline_input.shape[0]

    t = pulse_t
    while t < num_steps * dt:
        p0 = int(t / dt)
        p1 = int((t + pulse_width_ms) / dt)
        p0 = max(0, min(p0, num_steps - 1))
        p1 = max(p0 + 1, min(p1, num_steps))
        pulse_input[p0:p1, target_neurons] += pulse_amp
        t += pulse_interval_ms

    return pulse_input


def compute_spike_delta(
    spikes: Spikes,
    target_neurons: np.ndarray,
    pulse_t: float,
    pre_window_ms: float,
    post_window_ms: float,
) -> int:
    """
    Compute change in spike count before and after a pulse.

    Parameters:
        spikes: Spike data object with .times and .ids attributes
        target_neurons: Array of neuron IDs to analyze
        pulse_t: Time of pulse onset in milliseconds
        pre_window_ms: Duration of pre-pulse window in milliseconds
        post_window_ms: Duration of post-pulse window in milliseconds

    Returns:
        Difference in spike count (post - pre)
    """
    spike_t = spikes.times
    spike_ids = spikes.ids
    mask_target = np.isin(spike_ids, target_neurons)
    t_target = spike_t[mask_target]

    pre_mask = (t_target >= pulse_t - pre_window_ms) & (t_target < pulse_t)
    post_mask = (t_target >= pulse_t) & (t_target < pulse_t + post_window_ms)

    return int(np.sum(post_mask) - np.sum(pre_mask))
