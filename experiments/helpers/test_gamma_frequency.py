from __future__ import annotations

from dataclasses import replace

import numpy as np
from experiments.helpers.gamma_frequency import (
    DEFAULT_PING_GAMMA,
    GammaFrequencyConfig,
    estimate_gamma_frequency,
    estimate_gamma_from_raster,
)


def _sinusoid(frequency_hz: float, *, seconds: float = 2.0, amplitude: float = 1.0):
    sample_hz = 1000.0
    time = np.arange(round(seconds * sample_hz)) / sample_hz
    return sample_hz, amplitude * np.sin(2.0 * np.pi * frequency_hz * time)


def test_stationary_off_bin_frequency_is_interpolated():
    sample_hz, trace = _sinusoid(43.7)
    result = estimate_gamma_frequency(
        trace,
        sample_hz=sample_hz,
        config=replace(DEFAULT_PING_GAMMA, min_events=0),
    )
    assert result.resolved
    assert result.frequency_hz is not None
    assert abs(result.frequency_hz - 43.7) < 0.15
    assert result.trials[0].discrete_frequency_hz == 43.5


def test_silent_raster_is_unresolved_without_fabricated_frequency():
    result = estimate_gamma_from_raster(
        np.zeros((2_000, 80)),
        dt_ms=0.1,
        config=DEFAULT_PING_GAMMA,
    )
    assert not result.resolved
    assert result.frequency_hz is None
    assert result.reason == "insufficient_activity"


def test_featureless_noise_can_be_rejected_by_declared_prominence():
    rng = np.random.default_rng(154)
    trace = rng.standard_normal(10_000)
    config = GammaFrequencyConfig(
        name="strict-noise-rejection",
        min_events=0,
        min_prominence_ratio=20.0,
    )
    result = estimate_gamma_frequency(trace, sample_hz=1000.0, config=config)
    assert not result.resolved
    assert result.reason == "no_resolved_trials"
    assert result.trials[0].reason == "inadequate_prominence"


def test_weak_homogeneous_poisson_raster_is_not_called_gamma():
    rng = np.random.default_rng(155)
    raster = rng.random((20_000, 40)) < 0.0005
    config = replace(
        DEFAULT_PING_GAMMA,
        name="weak-poisson-rejection",
        min_prominence_ratio=20.0,
    )
    result = estimate_gamma_from_raster(raster, dt_ms=0.1, config=config)
    assert not result.resolved
    assert result.frequency_hz is None


def test_subharmonic_policy_recovers_fundamental_from_dominant_harmonic():
    sample_hz, fundamental = _sinusoid(40.0, amplitude=1.0)
    _, harmonic = _sinusoid(80.0, amplitude=2.0)
    config = GammaFrequencyConfig(
        name="subharmonic-aware",
        band_hz=(30.0, 100.0),
        min_events=0,
        subharmonic_ratio=0.2,
    )
    result = estimate_gamma_frequency(
        fundamental + harmonic,
        sample_hz=sample_hz,
        config=config,
    )
    assert result.resolved
    assert result.frequency_hz is not None
    assert abs(result.frequency_hz - 40.0) < 0.1


def test_trial_aggregation_is_explicit_and_changes_the_estimand():
    sample_hz, low = _sinusoid(40.0)
    _, high = _sinusoid(70.0, amplitude=4.0)
    traces = np.stack([low, low, high])
    base = GammaFrequencyConfig(
        name="trial-median",
        band_hz=(30.0, 80.0),
        min_events=0,
        aggregation="median",
    )
    median = estimate_gamma_frequency(traces, sample_hz=sample_hz, config=base)
    mean_psd = estimate_gamma_frequency(
        traces,
        sample_hz=sample_hz,
        config=replace(base, name="mean-psd", aggregation="mean_psd_peak"),
    )
    assert median.frequency_hz is not None
    assert mean_psd.frequency_hz is not None
    assert abs(median.frequency_hz - 40.0) < 0.1
    assert abs(mean_psd.frequency_hz - 70.0) < 0.1


def test_chirp_is_reported_with_lower_prominence_than_stationary_rhythm():
    sample_hz = 1000.0
    time = np.arange(2_000) / sample_hz
    chirp = np.sin(2.0 * np.pi * (30.0 * time + 10.0 * time**2))
    _, stationary = _sinusoid(50.0)
    config = GammaFrequencyConfig(
        name="broad-rhythm",
        band_hz=(25.0, 80.0),
        min_events=0,
        min_prominence_ratio=None,
    )
    chirp_result = estimate_gamma_frequency(chirp, sample_hz=sample_hz, config=config)
    stationary_result = estimate_gamma_frequency(
        stationary,
        sample_hz=sample_hz,
        config=config,
    )
    assert chirp_result.resolved and stationary_result.resolved
    assert chirp_result.trials[0].prominence_ratio is not None
    assert stationary_result.trials[0].prominence_ratio is not None
    assert (
        chirp_result.trials[0].prominence_ratio
        < stationary_result.trials[0].prominence_ratio
    )


def test_configuration_and_result_are_provenance_ready():
    sample_hz, trace = _sinusoid(50.0)
    result = estimate_gamma_frequency(
        trace,
        sample_hz=sample_hz,
        config=replace(DEFAULT_PING_GAMMA, min_events=0),
    )
    payload = result.json(include_spectrum=True)
    assert payload["config"]["name"] == "default-ping-gamma-v1"
    assert payload["config"]["band_hz"] == [30.0, 80.0]
    assert payload["config"]["spectral_method"] == "welch"
    assert payload["config"]["segment_length"] == "trial"
    assert payload["resolved_trials"] == 1
    assert len(payload["frequencies_hz"]) == len(payload["mean_psd"])
