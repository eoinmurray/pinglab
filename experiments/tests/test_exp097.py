from __future__ import annotations

import numpy as np

from experiments import exp097


def test_inputs_match_registered_scout_scale() -> None:
    inputs = exp097.make_inputs()
    assert inputs.shape == (5000, 5, 128)
    assert inputs.dtype == np.uint8


def test_phase_series_marks_only_complete_cycles() -> None:
    phase, next_ms = exp097.phase_series(np.array([10, 20, 35]), 50)
    assert np.isnan(phase[:10]).all()
    assert np.isclose(phase[10], 0.0)
    assert np.isclose(next_ms[10], 1.0)
    assert np.isnan(phase[35:]).all()


def test_circular_error_wraps_at_cycle_boundary() -> None:
    error = exp097.circular_error(np.array([0.98, 0.2]), np.array([0.02, 0.3]))
    assert np.allclose(error, [0.04, 0.1])
