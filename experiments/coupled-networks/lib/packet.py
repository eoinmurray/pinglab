from __future__ import annotations

import numpy as np


def generate_packet_spike_times(
    *,
    num_fibers: int,
    t0_ms: float,
    width_ms: float,
    mean_spikes_per_fiber: float,
    jitter_ms: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Generate spike times per fiber for a single packet.

    Each fiber gets a Poisson number of spikes with mean mean_spikes_per_fiber,
    placed uniformly within [t0_ms, t0_ms + width_ms], then jittered.
    """
    spikes_per_fiber: list[np.ndarray] = []
    for _ in range(num_fibers):
        k = int(rng.poisson(mean_spikes_per_fiber))
        if k <= 0:
            spikes_per_fiber.append(np.array([], dtype=np.float32))
            continue
        times = rng.uniform(t0_ms, t0_ms + width_ms, size=k)
        if jitter_ms > 0.0:
            times = times + rng.normal(0.0, jitter_ms, size=k)
        spikes_per_fiber.append(times.astype(np.float32))
    return spikes_per_fiber


def stack_spike_times(
    spikes_per_fiber: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten per-fiber spike times into (times, fiber_ids).
    """
    times = []
    fiber_ids = []
    for fid, spikes in enumerate(spikes_per_fiber):
        if spikes.size == 0:
            continue
        times.append(spikes)
        fiber_ids.append(np.full(spikes.shape[0], fid, dtype=np.int32))
    if not times:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)
    return np.concatenate(times), np.concatenate(fiber_ids)
