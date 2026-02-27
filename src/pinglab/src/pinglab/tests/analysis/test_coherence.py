import numpy as np

from pinglab.analysis import coherence, coherence_peak
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


def test_coherence_curve_shapes_and_contrast():
    T_ms = 2000.0
    dt_ms = 5.0
    freq_hz = 10.0
    N_E = 20

    periodic = _make_periodic_spikes(T_ms, N_E, freq_hz)
    lags_p, corr_p = coherence(
        periodic,
        T_ms=T_ms,
        dt_ms=dt_ms,
        sigma_ms=10.0,
        max_lag_ms=200.0,
        N_E=N_E,
    )

    rng = np.random.default_rng(123)
    noisy = _make_noisy_spikes(T_ms, N_E, rate_hz=10.0, rng=rng)
    lags_n, corr_n = coherence(
        noisy,
        T_ms=T_ms,
        dt_ms=dt_ms,
        sigma_ms=10.0,
        max_lag_ms=200.0,
        N_E=N_E,
    )

    assert lags_p.shape == corr_p.shape
    assert lags_n.shape == corr_n.shape
    assert corr_p.size > 0
    assert corr_n.size > 0

    peak_p = float(np.max(corr_p))
    peak_n = float(np.max(corr_n))
    assert 0.0 <= peak_p <= 1.0
    assert 0.0 <= peak_n <= 1.0
    assert peak_p > peak_n


def test_coherence_peak_matches_curve_max():
    T_ms = 2000.0
    dt_ms = 5.0
    freq_hz = 10.0
    N_E = 20

    periodic = _make_periodic_spikes(T_ms, N_E, freq_hz)
    peak, lags, corr = coherence_peak(
        periodic,
        T_ms=T_ms,
        dt_ms=dt_ms,
        sigma_ms=10.0,
        max_lag_ms=200.0,
        N_E=N_E,
    )
    assert lags.shape == corr.shape
    assert np.isclose(peak, np.max(corr))
