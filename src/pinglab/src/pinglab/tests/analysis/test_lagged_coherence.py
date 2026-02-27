import numpy as np

from pinglab.analysis import lagged_coherence
from pinglab.backends.types import Spikes


def _make_periodic_spikes(T_ms: float, N_E: int, freq_hz: float) -> Spikes:
    period_ms = 1000.0 / freq_hz
    times = np.arange(0.0, T_ms, period_ms, dtype=float)
    all_times = np.repeat(times, N_E)
    all_ids = np.tile(np.arange(N_E, dtype=int), times.size)
    order = np.argsort(all_times)
    return Spikes(times=all_times[order], ids=all_ids[order])


def _make_noisy_spikes(
    T_ms: float, N_E: int, rate_hz: float, rng: np.random.Generator
) -> Spikes:
    times_list = []
    ids_list = []
    for nid in range(N_E):
        t = 0.0
        while t < T_ms:
            isi = rng.exponential(1000.0 / rate_hz)
            t += isi
            if t < T_ms:
                times_list.append(t)
                ids_list.append(nid)
    if not times_list:
        return Spikes(times=np.array([]), ids=np.array([]))
    order = np.argsort(times_list)
    return Spikes(
        times=np.array(times_list, dtype=float)[order],
        ids=np.array(ids_list, dtype=int)[order],
    )


def test_lagged_coherence_bounds_and_rhythmicity():
    T_ms = 5000.0
    dt_ms = 5.0
    freq_hz = 5.0
    N_E = 20

    periodic = _make_periodic_spikes(T_ms, N_E, freq_hz)
    lam_periodic, t_ms, rate_hz, windows, coeffs, phase_mag = lagged_coherence(
        periodic,
        T_ms=T_ms,
        dt_ms=dt_ms,
        freq_hz=freq_hz,
        window_cycles=3.0,
        lag_cycles=3.0,
        pop="E",
        N_E=N_E,
        remove_mean=True,
        taper="hann",
    )

    rng = np.random.default_rng(123)
    noisy = _make_noisy_spikes(T_ms, N_E, rate_hz=5.0, rng=rng)
    lam_noisy, _, _, _, _, _ = lagged_coherence(
        noisy,
        T_ms=T_ms,
        dt_ms=dt_ms,
        freq_hz=freq_hz,
        window_cycles=3.0,
        lag_cycles=3.0,
        pop="E",
        N_E=N_E,
        remove_mean=True,
        taper="hann",
    )

    assert t_ms.size == rate_hz.size
    assert windows.ndim == 2 and windows.shape[1] == 2
    assert coeffs.size == phase_mag.shape[0]
    assert phase_mag.shape[1] == 3
    assert 0.0 <= lam_periodic <= 1.0
    assert 0.0 <= lam_noisy <= 1.0
    assert lam_periodic > 0.7
    assert lam_periodic > lam_noisy + 0.2
