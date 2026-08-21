"""Configurable gamma peak-frequency analysis for experiment recordings.

The simulator and SNNLANG own recordings; experiment runners own the scientific
policy used to interpret them.  This module supplies the shared numerical
machinery while keeping every consequential choice serialisable and explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import gaussian_filter1d

Aggregation = Literal["median", "mean", "mean_psd_peak"]
PopulationReduction = Literal["mean", "sum"]
SpectralMethod = Literal["welch"]
SegmentLength = Literal["trial"]


@dataclass(frozen=True)
class GammaFrequencyConfig:
    """Complete, provenance-ready policy for one spectral estimate."""

    name: str
    source_population: str = "E"
    band_hz: tuple[float, float] = (30.0, 80.0)
    burn_ms: float = 0.0
    bin_ms: float | None = None
    smooth_sigma_ms: float | None = None
    population_reduction: PopulationReduction = "mean"
    mean_center: bool = True
    spectral_method: SpectralMethod = "welch"
    segment_length: SegmentLength = "trial"
    window: str = "hann"
    detrend: str | bool = False
    scaling: str = "density"
    interpolate: bool = True
    min_events: int = 1
    min_prominence_ratio: float | None = 3.0
    reject_band_edges: bool = True
    subharmonic_ratio: float | None = None
    aggregation: Aggregation = "median"

    def __post_init__(self) -> None:
        low, high = self.band_hz
        if not self.name:
            raise ValueError("gamma-frequency configuration requires a name")
        if not (0 < low < high):
            raise ValueError("gamma-frequency band must satisfy 0 < low < high")
        if self.burn_ms < 0:
            raise ValueError("gamma-frequency burn_ms must be non-negative")
        if self.bin_ms is not None and self.bin_ms <= 0:
            raise ValueError("gamma-frequency bin_ms must be positive")
        if self.smooth_sigma_ms is not None and self.smooth_sigma_ms <= 0:
            raise ValueError("gamma-frequency smooth_sigma_ms must be positive")
        if self.min_events < 0:
            raise ValueError("gamma-frequency min_events must be non-negative")
        if self.min_prominence_ratio is not None and self.min_prominence_ratio <= 0:
            raise ValueError("gamma-frequency prominence ratio must be positive")
        if self.subharmonic_ratio is not None and not 0 < self.subharmonic_ratio <= 1:
            raise ValueError("gamma-frequency subharmonic ratio must be in (0, 1]")
        if self.spectral_method != "welch":
            raise ValueError(f"unsupported spectral method {self.spectral_method}")
        if self.segment_length != "trial":
            raise ValueError(f"unsupported segment length {self.segment_length}")
        if self.aggregation not in {"median", "mean", "mean_psd_peak"}:
            raise ValueError(
                f"unsupported gamma-frequency aggregation {self.aggregation}"
            )

    def json(self) -> dict:
        row = asdict(self)
        row["band_hz"] = list(self.band_hz)
        return row


@dataclass(frozen=True)
class GammaPeak:
    resolved: bool
    frequency_hz: float | None
    reason: str | None
    discrete_frequency_hz: float | None
    peak_power: float | None
    band_median_power: float | None
    prominence_ratio: float | None
    frequencies_hz: np.ndarray = field(repr=False)
    psd: np.ndarray = field(repr=False)

    def json(self, *, include_spectrum: bool = False) -> dict:
        row = {
            "resolved": self.resolved,
            "frequency_hz": self.frequency_hz,
            "reason": self.reason,
            "discrete_frequency_hz": self.discrete_frequency_hz,
            "peak_power": self.peak_power,
            "band_median_power": self.band_median_power,
            "prominence_ratio": self.prominence_ratio,
        }
        if include_spectrum:
            row["frequencies_hz"] = self.frequencies_hz.tolist()
            row["psd"] = self.psd.tolist()
        return row


@dataclass(frozen=True)
class GammaFrequencyEstimate:
    config: GammaFrequencyConfig
    resolved: bool
    frequency_hz: float | None
    reason: str | None
    trials: tuple[GammaPeak, ...]
    frequencies_hz: np.ndarray = field(repr=False)
    mean_psd: np.ndarray = field(repr=False)

    @property
    def resolved_trials(self) -> int:
        return sum(trial.resolved for trial in self.trials)

    def json(self, *, include_spectrum: bool = False) -> dict:
        row = {
            "config": self.config.json(),
            "resolved": self.resolved,
            "frequency_hz": self.frequency_hz,
            "reason": self.reason,
            "resolved_trials": self.resolved_trials,
            "total_trials": len(self.trials),
            "trials": [trial.json() for trial in self.trials],
        }
        if include_spectrum:
            row["frequencies_hz"] = self.frequencies_hz.tolist()
            row["mean_psd"] = self.mean_psd.tolist()
        return row


DEFAULT_PING_GAMMA = GammaFrequencyConfig(name="default-ping-gamma-v1")


def _empty_peak(
    reason: str,
    frequencies: np.ndarray | None = None,
    psd: np.ndarray | None = None,
) -> GammaPeak:
    return GammaPeak(
        resolved=False,
        frequency_hz=None,
        reason=reason,
        discrete_frequency_hz=None,
        peak_power=None,
        band_median_power=None,
        prominence_ratio=None,
        frequencies_hz=np.array([], dtype=float)
        if frequencies is None
        else frequencies,
        psd=np.array([], dtype=float) if psd is None else psd,
    )


def _peak_from_spectrum(
    frequencies: np.ndarray,
    psd: np.ndarray,
    config: GammaFrequencyConfig,
) -> GammaPeak:
    finite = np.isfinite(frequencies) & np.isfinite(psd)
    if not np.all(finite):
        return _empty_peak("non_finite_spectrum", frequencies, psd)
    band_indices = np.flatnonzero(
        (frequencies >= config.band_hz[0]) & (frequencies <= config.band_hz[1])
    )
    if band_indices.size < 3:
        return _empty_peak("insufficient_band_bins", frequencies, psd)
    peak_index = int(band_indices[int(np.argmax(psd[band_indices]))])
    peak_power = float(psd[peak_index])
    median_power = float(np.median(psd[band_indices]))
    if peak_power <= 0:
        return _empty_peak("zero_band_power", frequencies, psd)
    prominence = float("inf") if median_power <= 0 else peak_power / median_power
    if config.reject_band_edges and peak_index in {band_indices[0], band_indices[-1]}:
        return GammaPeak(
            False,
            None,
            "band_edge_peak",
            float(frequencies[peak_index]),
            peak_power,
            median_power,
            prominence,
            frequencies,
            psd,
        )
    if (
        config.min_prominence_ratio is not None
        and prominence <= config.min_prominence_ratio
    ):
        return GammaPeak(
            False,
            None,
            "inadequate_prominence",
            float(frequencies[peak_index]),
            peak_power,
            median_power,
            prominence,
            frequencies,
            psd,
        )

    selected_index = peak_index
    if config.subharmonic_ratio is not None:
        half_frequency = frequencies[peak_index] / 2.0
        half_index = int(np.argmin(np.abs(frequencies - half_frequency)))
        if (
            frequencies[half_index] >= config.band_hz[0]
            and psd[half_index] >= config.subharmonic_ratio * peak_power
        ):
            selected_index = half_index

    frequency = float(frequencies[selected_index])
    if config.interpolate and 0 < selected_index < len(psd) - 1:
        y0, y1, y2 = (float(psd[selected_index + offset]) for offset in (-1, 0, 1))
        denominator = y0 - 2.0 * y1 + y2
        offset = 0.5 * (y0 - y2) / denominator if denominator != 0 else 0.0
        offset = float(np.clip(offset, -0.5, 0.5))
        frequency += offset * float(frequencies[1] - frequencies[0])
    return GammaPeak(
        True,
        frequency,
        None,
        float(frequencies[selected_index]),
        float(psd[selected_index]),
        median_power,
        prominence,
        frequencies,
        psd,
    )


def _spectrum(
    trace: np.ndarray,
    sample_hz: float,
    config: GammaFrequencyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(trace, dtype=np.float64)
    if config.mean_center:
        values = values - values.mean()
    frequencies, psd = sp_signal.welch(
        values,
        fs=sample_hz,
        window=config.window,
        nperseg=len(values),
        detrend=config.detrend,
        scaling=config.scaling,
    )
    return frequencies, psd


def estimate_gamma_frequency(
    traces: np.ndarray,
    *,
    sample_hz: float,
    config: GammaFrequencyConfig = DEFAULT_PING_GAMMA,
    event_counts: np.ndarray | None = None,
) -> GammaFrequencyEstimate:
    """Estimate gamma frequency from one trace or a `[trial, time]` matrix."""

    if not np.isfinite(sample_hz) or sample_hz <= 0:
        raise ValueError("gamma-frequency sample_hz must be positive and finite")
    values = np.asarray(traces, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError(
            "gamma-frequency traces must have shape [time] or [trial, time]"
        )
    counts = (
        np.count_nonzero(values, axis=1)
        if event_counts is None
        else np.asarray(event_counts)
    )
    if counts.shape != (values.shape[0],):
        raise ValueError(
            "gamma-frequency event_counts must contain one value per trial"
        )
    burn_samples = int(round(config.burn_ms * sample_hz / 1000.0))
    trial_results: list[GammaPeak] = []
    for trial, count in zip(values, counts):
        if not np.all(np.isfinite(trial)):
            trial_results.append(_empty_peak("non_finite_input"))
            continue
        if count < config.min_events:
            trial_results.append(_empty_peak("insufficient_activity"))
            continue
        trimmed = trial[burn_samples:]
        if trimmed.size < 4:
            trial_results.append(_empty_peak("insufficient_duration"))
            continue
        frequencies, psd = _spectrum(trimmed, sample_hz, config)
        trial_results.append(_peak_from_spectrum(frequencies, psd, config))

    spectra = [trial for trial in trial_results if trial.psd.size]
    if not spectra:
        reason = next(
            (trial.reason for trial in trial_results if trial.reason), "unresolved"
        )
        return GammaFrequencyEstimate(
            config,
            False,
            None,
            reason,
            tuple(trial_results),
            np.array([]),
            np.array([]),
        )
    frequencies = spectra[0].frequencies_hz
    if any(not np.array_equal(trial.frequencies_hz, frequencies) for trial in spectra):
        raise ValueError("gamma-frequency trials produced incompatible frequency axes")
    mean_psd = np.mean(np.stack([trial.psd for trial in spectra]), axis=0)
    resolved = [trial.frequency_hz for trial in trial_results if trial.resolved]
    if config.aggregation == "mean_psd_peak":
        aggregate_peak = _peak_from_spectrum(frequencies, mean_psd, config)
        frequency = aggregate_peak.frequency_hz
        aggregate_resolved = aggregate_peak.resolved
        reason = aggregate_peak.reason
    elif resolved:
        frequency = float(
            np.median(resolved) if config.aggregation == "median" else np.mean(resolved)
        )
        aggregate_resolved = True
        reason = None
    else:
        frequency = None
        aggregate_resolved = False
        reason = "no_resolved_trials"
    return GammaFrequencyEstimate(
        config,
        aggregate_resolved,
        frequency,
        reason,
        tuple(trial_results),
        frequencies,
        mean_psd,
    )


def estimate_gamma_from_raster(
    spikes: np.ndarray,
    *,
    dt_ms: float,
    config: GammaFrequencyConfig = DEFAULT_PING_GAMMA,
) -> GammaFrequencyEstimate:
    """Reduce `[time, cells]` or `[time, trial, cells]` spikes and estimate gamma."""

    if not np.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("gamma-frequency dt_ms must be positive and finite")
    values = np.asarray(spikes)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError(
            "gamma-frequency raster must have shape [time, cells] or [time, trial, cells]"
        )
    if not np.all(np.isfinite(values)):
        traces = np.full((values.shape[1], values.shape[0]), np.nan)
        return estimate_gamma_frequency(traces, sample_hz=1000.0 / dt_ms, config=config)
    event_counts = values.sum(axis=(0, 2))
    bin_steps = 1
    if config.bin_ms is not None:
        bin_steps = max(1, int(round(config.bin_ms / dt_ms)))
    bins = values.shape[0] // bin_steps
    if bins == 0:
        traces = np.empty((values.shape[1], 0))
    else:
        binned = values[: bins * bin_steps].reshape(
            bins, bin_steps, values.shape[1], values.shape[2]
        )
        binned = binned.sum(axis=1)
        traces = (
            binned.mean(axis=-1)
            if config.population_reduction == "mean"
            else binned.sum(axis=-1)
        ).T
    effective_bin_ms = dt_ms * bin_steps
    if config.smooth_sigma_ms is not None and traces.shape[1]:
        traces = gaussian_filter1d(
            traces,
            config.smooth_sigma_ms / effective_bin_ms,
            axis=1,
        )
    return estimate_gamma_frequency(
        traces,
        sample_hz=1000.0 / effective_bin_ms,
        config=config,
        event_counts=event_counts,
    )
