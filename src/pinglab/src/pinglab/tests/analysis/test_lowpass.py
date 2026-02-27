import numpy as np
import pytest

from pinglab.analysis import lowpass_first_order


def test_lowpass_returns_empty_for_empty_input() -> None:
    out = lowpass_first_order(np.array([], dtype=float), dt_ms=5.0, cutoff_hz=10.0)
    assert out.size == 0


def test_lowpass_step_response_is_monotonic_and_bounded() -> None:
    x = np.concatenate([np.zeros(20, dtype=float), np.ones(100, dtype=float)])
    y = lowpass_first_order(x, dt_ms=5.0, cutoff_hz=10.0)
    # Step response should rise smoothly without overshoot for first-order LPF.
    tail = y[20:]
    assert np.all(np.diff(tail) >= -1e-12)
    assert float(np.max(y)) <= 1.0 + 1e-9
    assert float(y[-1]) > 0.95


def test_lowpass_rejects_invalid_params() -> None:
    with pytest.raises(ValueError):
        lowpass_first_order(np.array([1.0, 2.0]), dt_ms=0.0, cutoff_hz=10.0)
    with pytest.raises(ValueError):
        lowpass_first_order(np.array([1.0, 2.0]), dt_ms=5.0, cutoff_hz=0.0)
    with pytest.raises(ValueError):
        lowpass_first_order(np.array([[1.0, 2.0]]), dt_ms=5.0, cutoff_hz=10.0)
