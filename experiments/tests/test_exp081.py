"""Focused unit tests for the standalone EXP081 analytical runner."""

from __future__ import annotations

import numpy as np
from experiments import exp081


def test_zero_drive_has_zero_operating_point_and_variance() -> None:
    _, voltage = exp081.linear_operating_point(0.0, 1.2)
    variance = exp081.predicted_variance(np.asarray([0.0]), np.asarray([1.2]))
    assert float(voltage) == exp081.PARAMETERS["E_L_mV"]
    assert variance[0] == 0.0


def test_finite_window_has_first_zero_at_five_hz() -> None:
    transfer = exp081.complete_transfer(np.asarray([5.0]), 3.0, 1.2)
    assert abs(transfer[0]) < 1e-10


def test_empirical_simulator_replays_and_zero_drive_rests() -> None:
    expected = np.asarray([0.0, 0.5])
    probes = np.asarray([1.2, 1.2])
    first = exp081.simulate_features(expected, probes, 8, 123)
    replay = exp081.simulate_features(expected, probes, 8, 123)
    assert np.array_equal(first, replay)
    assert np.all(first[0] == 0.0)
    assert np.all(first[1] >= 0.0)
