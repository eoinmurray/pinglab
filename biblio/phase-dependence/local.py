
import numpy as np


def add_pulse_to_input(
    baseline_input: np.ndarray,
    target_E: np.ndarray,
    pulse_t: float,
    pulse_width_ms: float,
    pulse_amp: float,
    dt: float,
    num_steps: int,
) -> np.ndarray:
    """Add a pulse stimulus to target neurons at specified time."""
    pulse_input = baseline_input.copy()
    p0 = int(pulse_t / dt)
    p1 = int((pulse_t + pulse_width_ms) / dt)
    p0 = max(0, min(p0, num_steps - 1))
    p1 = max(p0 + 1, min(p1, num_steps))
    pulse_input[p0:p1, target_E] += pulse_amp
    return pulse_input


def compute_spike_delta(
    spikes,
    target_E: np.ndarray,
    pulse_t: float,
    pre_window_ms: float,
    post_window_ms: float,
) -> int:
    """Compute change in spike count before and after pulse."""
    spike_t = spikes.times
    spike_ids = spikes.ids
    mask_target = np.isin(spike_ids, target_E)
    t_target = spike_t[mask_target]

    pre_mask = (t_target >= pulse_t - pre_window_ms) & (t_target < pulse_t)
    post_mask = (t_target >= pulse_t) & (t_target < pulse_t + post_window_ms)

    return int(np.sum(post_mask) - np.sum(pre_mask))
