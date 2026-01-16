import numpy as np

from pinglab.types import Spikes
from pinglab.analysis import rate_psd, gamma_metrics, calculate_regime_label


def make_spikes(times, ids):
    return Spikes(
        times=np.array(times, dtype=float),
        ids=np.array(ids, dtype=int),
    )


def test_rate_psd_peak_frequency():
    dt_ms = 1.0
    t = np.arange(0.0, 1000.0, dt_ms) / 1000.0
    rate = np.sin(2 * np.pi * 10.0 * t)
    freqs, psd = rate_psd(rate, dt_ms=dt_ms)
    peak_freq = freqs[int(np.argmax(psd))]
    assert abs(peak_freq - 10.0) < 1.0


def test_gamma_metrics_detects_peak():
    times = np.arange(0.0, 1000.0, 25.0)
    spikes = make_spikes(times, np.zeros_like(times))
    peak_freq, peak_power, gamma_q = gamma_metrics(spikes, T=1000.0, fs=1000.0, fmin=30.0, fmax=90.0)
    assert peak_freq is not None
    assert 35.0 <= peak_freq <= 45.0
    assert peak_power is not None
    assert peak_power > 0.0
    assert gamma_q is None or gamma_q > 0.0


def test_calculate_regime_label_silent():
    regime, reason = calculate_regime_label(
        mean_rate_E=0.0,
        mean_rate_I=0.0,
        cv_population=0.0,
        cv_per_neuron=np.array([]),
        corr_EE_mean=0.0,
        corr_II_mean=0.0,
        corr_EI_mean=0.0,
        lag_mean_ms=0.0,
        lag_std_ms=0.0,
        gamma_peak_freq=None,
        gamma_peak_power=None,
        gamma_Q=None,
        fano_E=0.0,
        fano_I=0.0,
        synchrony=0.0,
        g_e_mean=1.0,
        g_i_mean=1.0,
        g_ei_ratio=1.0,
        g_e_cv=0.0,
        g_i_cv=0.0,
    )
    assert regime == "silent"
    assert "Low firing" in reason


def test_calculate_regime_label_ping():
    regime, reason = calculate_regime_label(
        mean_rate_E=5.0,
        mean_rate_I=5.0,
        cv_population=1.0,
        cv_per_neuron=np.array([1.0, 1.1]),
        corr_EE_mean=0.0,
        corr_II_mean=0.0,
        corr_EI_mean=0.0,
        lag_mean_ms=2.0,
        lag_std_ms=0.5,
        gamma_peak_freq=40.0,
        gamma_peak_power=1e9,
        gamma_Q=3.0,
        fano_E=1.0,
        fano_I=1.0,
        synchrony=1.0,
        g_e_mean=1.0,
        g_i_mean=1.0,
        g_ei_ratio=1.0,
        g_e_cv=0.1,
        g_i_cv=0.1,
    )
    assert regime == "PING"
    assert "Gamma" in reason


def test_calculate_regime_label_ai():
    regime, _ = calculate_regime_label(
        mean_rate_E=5.0,
        mean_rate_I=5.0,
        cv_population=1.0,
        cv_per_neuron=np.array([1.0, 1.1]),
        corr_EE_mean=0.0,
        corr_II_mean=0.0,
        corr_EI_mean=0.0,
        lag_mean_ms=2.0,
        lag_std_ms=0.5,
        gamma_peak_freq=None,
        gamma_peak_power=None,
        gamma_Q=None,
        fano_E=1.0,
        fano_I=1.0,
        synchrony=1.0,
        g_e_mean=1.0,
        g_i_mean=1.0,
        g_ei_ratio=1.0,
        g_e_cv=0.1,
        g_i_cv=0.1,
    )
    assert regime == "AI"


def test_calculate_regime_label_burst():
    regime, _ = calculate_regime_label(
        mean_rate_E=5.0,
        mean_rate_I=5.0,
        cv_population=2.0,
        cv_per_neuron=np.array([2.0, 2.1]),
        corr_EE_mean=0.3,
        corr_II_mean=0.3,
        corr_EI_mean=0.3,
        lag_mean_ms=2.0,
        lag_std_ms=0.5,
        gamma_peak_freq=None,
        gamma_peak_power=None,
        gamma_Q=None,
        fano_E=6.0,
        fano_I=6.0,
        synchrony=6.0,
        g_e_mean=1.0,
        g_i_mean=1.0,
        g_ei_ratio=1.0,
        g_e_cv=0.1,
        g_i_cv=0.1,
    )
    assert regime == "burst"


def test_calculate_regime_label_unstable_ratio():
    regime, reason = calculate_regime_label(
        mean_rate_E=5.0,
        mean_rate_I=5.0,
        cv_population=1.0,
        cv_per_neuron=np.array([1.0]),
        corr_EE_mean=0.0,
        corr_II_mean=0.0,
        corr_EI_mean=0.0,
        lag_mean_ms=0.0,
        lag_std_ms=0.0,
        gamma_peak_freq=None,
        gamma_peak_power=None,
        gamma_Q=None,
        fano_E=1.0,
        fano_I=1.0,
        synchrony=1.0,
        g_e_mean=1.0,
        g_i_mean=1.0,
        g_ei_ratio=30.0,
        g_e_cv=0.1,
        g_i_cv=0.1,
    )
    assert regime == "unstable"
    assert "Extreme" in reason
