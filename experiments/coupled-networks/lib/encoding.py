from __future__ import annotations
import numpy as np

def build_pixel_groups(
    N_E: int,
    h: int,
    w: int,
    group_size: int,
    seed: int,
) -> list[np.ndarray]:
    """
    Returns list of length P=h*w.
    Each entry is an array of E-neuron indices assigned to that pixel.
    Pixels are mapped to disjoint groups (no overlap). If you run out of neurons,
    it raises.
    """
    P = h * w
    total_needed = P * group_size
    if total_needed > N_E:
        raise ValueError(
            f"Not enough E neurons for pixel groups: need {total_needed}, have {N_E}. "
            f"Reduce image size or group_size."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_E)
    groups = []
    cursor = 0
    for _ in range(P):
        groups.append(np.sort(perm[cursor: cursor + group_size]))
        cursor += group_size
    return groups

def image_to_group_currents(
    img: np.ndarray,
    groups: list[np.ndarray],
    scale: float,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> dict[int, float]:
    """
    Produces per-neuron added current values for E neurons.
    Returns dict {neuron_id: added_current}.
    """
    lo, hi = value_range
    x = img.astype(np.float32).reshape(-1)
    x = np.clip(x, lo, hi)
    if hi > lo:
        x = (x - lo) / (hi - lo)

    neuron_add = {}
    for p, ids in enumerate(groups):
        amp = float(scale * x[p])
        for nid in ids:
            neuron_add[int(nid)] = amp
    return neuron_add
