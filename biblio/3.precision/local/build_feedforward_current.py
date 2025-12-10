
import numpy as np
from pinglab.types import Spikes


def build_feedforward_current(
    spikes: Spikes,
    N_E: int,
    N_I: int,
    T: float,
    dt: float,
    delay_ms: float = 5.0,
    pulse_ms: float = 3.0,
    w_ff: float = 0.3,
    phase_ms: float = 0.0,
) -> np.ndarray:
    times = spikes.times
    ids = spikes.ids

    num_steps = int(T / dt)
    N = N_E + N_I
    ff = np.zeros((num_steps, N), dtype=np.float32)

    mask_E = ids < N_E
    pre_times = times[mask_E]
    pre_ids = ids[mask_E]

    for t, i in zip(pre_times, pre_ids):
        t_start = int((t + delay_ms + phase_ms) / dt)
        t_end = int((t + delay_ms + phase_ms + pulse_ms) / dt)
        if t_start >= num_steps:
            continue
        t_end = min(t_end, num_steps)
        ff[t_start:t_end, i] += w_ff

    return ff
