import numpy as np

from pinglab.analysis import decode_fit_metrics, envelope_rate_hz, minmax_normalize


def test_envelope_rate_hz_clamps_nonnegative() -> None:
    t_ms = np.arange(0.0, 100.0, 1.0)
    env = envelope_rate_hz(
        t_ms,
        lambda0_hz=10.0,
        mod_depth=2.0,
        envelope_freq_hz=5.0,
        phase_rad=0.0,
    )
    assert env.shape == t_ms.shape
    assert float(np.min(env)) >= 0.0


def test_decode_fit_metrics_perfect_match() -> None:
    t_ms = np.arange(0.0, 200.0, 5.0)
    env = envelope_rate_hz(
        t_ms,
        lambda0_hz=30.0,
        mod_depth=0.5,
        envelope_freq_hz=4.0,
        phase_rad=0.3,
    )
    corr, err = decode_fit_metrics(env, env, normalize=True)
    assert corr > 0.999
    assert err < 1e-9


def test_minmax_normalize_constant_series_is_zero() -> None:
    x = np.ones(8, dtype=float) * 3.5
    y = minmax_normalize(x)
    assert np.allclose(y, 0.0)
