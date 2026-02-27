import numpy as np

from pinglab.analysis import mean_pairwise_xcorr_peak
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


def test_mean_pairwise_xcorr_peak_periodic_vs_noise():
    T_ms = 2000.0
    dt_ms = 5.0
    freq_hz = 10.0
    N_E = 20

    periodic = _make_periodic_spikes(T_ms, N_E, freq_hz)
    peak_periodic, lags_ms, corr = mean_pairwise_xcorr_peak(
        periodic,
        T_ms=T_ms,
        dt_ms=dt_ms,
        N_E=N_E,
    )

    rng = np.random.default_rng(321)
    noisy = _make_noisy_spikes(T_ms, N_E, rate_hz=freq_hz, rng=rng)
    peak_noisy, _, _ = mean_pairwise_xcorr_peak(
        noisy,
        T_ms=T_ms,
        dt_ms=dt_ms,
        N_E=N_E,
    )

    assert lags_ms.size == corr.size
    assert peak_periodic > 0.1
    assert peak_periodic > peak_noisy
