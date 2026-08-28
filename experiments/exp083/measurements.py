"""Original exp083 estimators and aggregation, unchanged by stage separation."""

import numpy as np
from experiments.helpers.gamma_frequency import GammaFrequencyEstimate
from experiments.helpers.rhythmicity import (
    iei_histogram,
    population_event_times,
    rhythmicity_scalars,
    spike_autocorrelogram,
)

from .recipe import BURN_MS, DT_MS, N_E, N_I, T_MS, TRIAL_SEEDS


def _phase_lag_ms(e_spikes: np.ndarray, i_spikes: np.ndarray) -> float | None:
    burn = round(BURN_MS / DT_MS)
    bin_steps = round(1.0 / DT_MS)
    e = e_spikes[burn:].mean(axis=1)
    i = i_spikes[burn:].mean(axis=1)
    usable = len(e) // bin_steps * bin_steps
    if usable == 0:
        return None
    e = e[:usable].reshape(-1, bin_steps).sum(axis=1)
    i = i[:usable].reshape(-1, bin_steps).sum(axis=1)
    if e.std() == 0 or i.std() == 0:
        return None
    max_lag = 20
    correlation = np.correlate(e - e.mean(), i - i.mean(), mode="full")
    lags = np.arange(-len(e) + 1, len(e))
    keep = np.abs(lags) <= max_lag
    return float(lags[keep][np.argmax(correlation[keep])])


def _rhythmicity_contrast(e_spikes: np.ndarray) -> float | None:
    """Canonical exp054 lobe-trough contrast on the post-burn E raster."""
    burn = round(BURN_MS / DT_MS)
    spikes = e_spikes[burn:]
    ac_lags, ac = spike_autocorrelogram(spikes, DT_MS, 100.0, 1.0)
    iei_lags, iei = iei_histogram(
        population_event_times(spikes, DT_MS),
        100.0,
        1.0,
    )
    scalars = rhythmicity_scalars(ac_lags, ac, iei_lags, iei, 1.0)
    contrast = scalars["contrast"]
    return None if contrast is None or not np.isfinite(contrast) else float(contrast)


def _trial_rows(
    rate_hz: float,
    e_spikes: np.ndarray,
    i_spikes: np.ndarray,
    gamma: GammaFrequencyEstimate,
) -> list[dict]:
    burn = round(BURN_MS / DT_MS)
    duration_s = (T_MS - BURN_MS) / 1_000.0
    rows = []
    for index, seed in enumerate(TRIAL_SEEDS):
        peak = gamma.trials[index]
        rows.append(
            {
                "trial": index,
                "seed": seed,
                "input_rate_hz": rate_hz,
                "e_rate_hz": float(e_spikes[burn:, index].sum() / N_E / duration_s),
                "i_rate_hz": float(i_spikes[burn:, index].sum() / N_I / duration_s),
                "frequency": peak.json(),
                "rhythmicity_contrast": _rhythmicity_contrast(e_spikes[:, index]),
                "e_i_peak_lag_ms": _phase_lag_ms(
                    e_spikes[:, index], i_spikes[:, index]
                ),
            }
        )
    return rows


def summarize_condition(rate_hz: float, rows: list[dict]) -> dict:
    resolved = [
        row["frequency"]["frequency_hz"] for row in rows if row["frequency"]["resolved"]
    ]
    lags = [
        row["e_i_peak_lag_ms"] for row in rows if row["e_i_peak_lag_ms"] is not None
    ]
    rhythmicity = [
        row["rhythmicity_contrast"]
        for row in rows
        if row["rhythmicity_contrast"] is not None
    ]
    return {
        "input_rate_hz": rate_hz,
        "e_rate_mean_hz": float(np.mean([row["e_rate_hz"] for row in rows])),
        "e_rate_std_hz": float(np.std([row["e_rate_hz"] for row in rows], ddof=1)),
        "i_rate_mean_hz": float(np.mean([row["i_rate_hz"] for row in rows])),
        "i_rate_std_hz": float(np.std([row["i_rate_hz"] for row in rows], ddof=1)),
        "frequency_resolved_fraction": len(resolved) / len(rows),
        "rhythmicity_score_median": 0.0
        if not rhythmicity
        else float(np.median(rhythmicity)),
        "rhythmicity_score_iqr": 0.0
        if not rhythmicity
        else float(np.percentile(rhythmicity, 75) - np.percentile(rhythmicity, 25)),
        "rhythm_frequency_median_hz": None
        if not resolved
        else float(np.median(resolved)),
        "e_i_peak_lag_median_ms": None if not lags else float(np.median(lags)),
        "trials": rows,
    }
