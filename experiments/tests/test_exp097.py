from __future__ import annotations

import numpy as np

from experiments import exp097


def test_inputs_match_registered_scout_scale() -> None:
    assert exp097.INPUT_RATE_HZ == 50.0
    assert (exp097.N_E, exp097.N_I) == (800, 200)
    inputs = exp097.make_inputs()
    assert inputs.shape == (5000, 5, 128)
    assert inputs.dtype == np.uint8


def test_linear_ramp_has_silent_bounds_and_frozen_peak() -> None:
    rate = exp097.input_rate_schedule("ramp")
    assert len(rate) == 12000
    assert np.all(rate[:2000] == 0.0)
    assert rate[6000] == 50.0
    assert np.all(rate[10000:] == 0.0)
    assert np.all(np.diff(rate[2000:6001]) >= 0.0)
    assert np.all(np.diff(rate[6000:10001]) <= 0.0)


def test_scheduled_inputs_are_deterministic_and_follow_schedule_shape() -> None:
    first, rates = exp097.make_scheduled_inputs("ramp")
    second, _ = exp097.make_scheduled_inputs("ramp")
    assert first.shape == (12000, 1, 128)
    assert np.array_equal(first, second)
    assert first[:2000].sum() == 0
    assert first[10000:].sum() == 0
    assert rates.max() == 50.0


def test_population_rate_is_normalized_by_cells_and_bin_duration() -> None:
    spikes = np.zeros((4, 1, 10), dtype=np.uint8)
    spikes[0:2, 0, :2] = 1
    rates = exp097.population_rate_hz(spikes, np.array([0, 2, 4]), 0)
    assert rates == [2000.0, 0.0]


def test_fixed_raster_samples_match_large_population_scale() -> None:
    assert len(exp097.E_RASTER_CELLS) == 40
    assert len(exp097.I_RASTER_CELLS) == 20
    assert exp097.E_RASTER_CELLS[-1] == 799
    assert exp097.I_RASTER_CELLS[-1] == 199


def test_native_input_raster_preserves_channel_and_timestamp() -> None:
    spikes = np.zeros((8, 1, 5), dtype=np.uint8)
    spikes[1, 0, 1] = 1
    spikes[6, 0, 3] = 1
    times, rows = exp097.native_raster_events(spikes, 0, 8, 0, (1, 3))
    assert times == [0.1, 0.6]
    assert rows == [0, 1]


def test_phase_series_marks_only_complete_cycles() -> None:
    phase, next_ms = exp097.phase_series(np.array([10, 20, 35]), 50)
    assert np.isnan(phase[:10]).all()
    assert np.isclose(phase[10], 0.0)
    assert np.isclose(next_ms[10], 1.0)
    assert np.isnan(phase[35:]).all()


def test_circular_error_wraps_at_cycle_boundary() -> None:
    error = exp097.circular_error(np.array([0.98, 0.2]), np.array([0.02, 0.3]))
    assert np.allclose(error, [0.04, 0.1])
