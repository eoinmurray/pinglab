import numpy as np

from pinglab.analysis import autocorr_rhythmicity


def test_autocorr_rhythmicity_zero_signal() -> None:
    rate_hz = np.zeros(200)
    rho = autocorr_rhythmicity(rate_hz, dt_ms=5.0, tau_min_ms=5.0, tau_max_ms=200.0)
    assert rho == 0.0


def test_autocorr_rhythmicity_sine_wave() -> None:
    dt_ms = 5.0
    t_ms = np.arange(0.0, 2000.0, dt_ms)
    rate_hz = 5.0 + np.sin(2.0 * np.pi * t_ms / 50.0)
    rho = autocorr_rhythmicity(rate_hz, dt_ms=dt_ms, tau_min_ms=10.0, tau_max_ms=200.0)
    assert 0.3 < rho < 1.01
