import numpy as np

def reconstruct_image_from_spikes(
    spikes,
    N_E: int,
    T_window: tuple[float, float],
) -> np.ndarray:
    """
    Turn spikes from the E-pop into a rate image.

    T_window in ms, e.g. (500.0, 1000.0)
    """
    times = spikes.times
    ids = spikes.ids

    t_start, t_end = T_window
    mask_time = (times >= t_start) & (times < t_end)

    # Only E neurons: assume IDs 0..N_E-1 are excitatory
    mask_E = ids < N_E

    mask = mask_time & mask_E
    ids_E = ids[mask].astype(int)

    counts = np.bincount(ids_E, minlength=N_E)
    duration_sec = (t_end - t_start) / 1000.0
    rates = counts / duration_sec  # Hz, shape (N_E,)

    side = int(np.sqrt(N_E))
    assert side * side == N_E, "Assuming square image / E-pop"

    return rates.reshape(side, side)
