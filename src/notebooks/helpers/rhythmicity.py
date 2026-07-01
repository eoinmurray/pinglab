"""Pure-numpy rhythmicity analysis for notebook figures.

Population autocorrelogram, inter-event-interval histogram and the derived
rhythmicity scalars (central-lobe / trough contrast, etc.). These are analysis
functions — they take spike rasters and return numbers — so they belong with the
notebooks, not the CLI. The CLI keeps its own copy in cli.metrics for its metrics
reporting; this module lets notebooks compute rhythmicity from CLI-emitted rasters
without importing from src/cli.
"""

from __future__ import annotations

import numpy as np


def population_event_times(spikes, dt):
    """Pooled population spike-event times in ms from a [T_steps, N] raster."""
    spikes = np.asarray(spikes)
    step_idx = np.nonzero(spikes)[0]
    return np.sort(step_idx).astype(float) * dt


def iei_histogram(event_times_ms, max_lag_ms=100.0, bin_ms=1.0):
    """Inter-event-interval histogram of a pooled event train → (centers, counts)."""
    t = np.sort(np.asarray(event_times_ms, dtype=float))
    edges = np.arange(0.0, max_lag_ms + bin_ms, bin_ms)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if t.size < 2:
        return centers, np.zeros(centers.size)
    counts, _ = np.histogram(np.diff(t), bins=edges)
    return centers, counts.astype(float)


def spike_autocorrelogram(spikes, dt, max_lag_ms=100.0, bin_ms=1.0):
    """Normalised population spike-time autocorrelogram → (lags_ms, ac).

    Asymptotic floor is 1.0 (rate-matched independence); a rhythm shows a central
    lobe > 1, a trough at the half-period, and a secondary peak at the period.
    Zero-lag is returned NaN (self-pairs dominate).
    """
    spikes = np.asarray(spikes)
    bin_steps = max(1, int(round(bin_ms / dt)))
    n_bins = spikes.shape[0] // bin_steps
    max_lag_bins = max(1, int(round(max_lag_ms / bin_ms)))
    lags = np.arange(max_lag_bins + 1) * bin_ms
    if n_bins <= max_lag_bins + 1:
        return lags, np.full(max_lag_bins + 1, np.nan)
    r = (
        spikes[: n_bins * bin_steps]
        .reshape(n_bins, bin_steps, -1)
        .sum(axis=(1, 2))
        .astype(float)
    )
    nfft = 1 << int(np.ceil(np.log2(2 * n_bins)))
    f = np.fft.rfft(r, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[: max_lag_bins + 1].astype(float)
    overlap = n_bins - np.arange(max_lag_bins + 1)
    floor = r.mean() ** 2
    if floor <= 0:
        return lags, np.full(max_lag_bins + 1, np.nan)
    ac = ac / overlap / floor
    ac[0] = np.nan
    return lags, ac


def _smooth_ac(ac):
    """3-point-smoothed copy of the autocorrelogram (lag-0 NaN filled), or None."""
    a = np.asarray(ac, dtype=float).copy()
    if a.size > 1 and not np.isfinite(a[0]):
        a[0] = a[1]
    if not np.all(np.isfinite(a)):
        return None
    return np.convolve(a, [0.25, 0.5, 0.25], mode="same")


def rhythmicity_scalars(ac_lags, ac, iei_lags, iei_counts, bin_ms=1.0, bio_lag_ms=None):
    """Rhythmicity scalars from an autocorrelogram + IEI histogram.

    Reports the IEI-anchored central-lobe height, the lobe/trough ratio and the
    bounded Mexican-hat contrast (lobe−trough)/(lobe+trough) ∈ [0, 1), plus the
    lobe/trough lags and an optional biophysical-lag reading.
    """
    ac = np.asarray(ac, dtype=float)

    def ac_at(lag_ms):
        if lag_ms is None or not np.isfinite(lag_ms):
            return None
        i = int(np.clip(round(lag_ms / bin_ms), 1, ac.size - 1))
        return float(ac[i]) if np.isfinite(ac[i]) else None

    iei_mode_lag = (
        float(iei_lags[int(np.argmax(iei_counts))]) if np.sum(iei_counts) > 0 else None
    )

    sm = _smooth_ac(ac)
    lobe_to_trough = contrast = trough_lag = lobe_lag = None
    if sm is not None:
        trough_i = None
        for i in range(2, sm.size - 1):
            if sm[i] <= sm[i - 1] and sm[i] < sm[i + 1]:
                trough_i = i
                break
        if trough_i is not None and trough_i > 1:
            trough_lag = float(ac_lags[trough_i])
            lobe_i = 1 + int(np.argmax(sm[1:trough_i]))
            lobe_lag = float(ac_lags[lobe_i])
            lobe_v, trough_v = float(sm[lobe_i]), float(sm[trough_i])
            lobe_to_trough = lobe_v / max(trough_v, 0.05)
            denom = lobe_v + trough_v
            contrast = (lobe_v - trough_v) / denom if denom > 0 else None

    return {
        "iei_mode_lag": iei_mode_lag,
        "lobe_lag": lobe_lag,
        "trough_lag": trough_lag,
        "iei_anchored": ac_at(iei_mode_lag),
        "lobe_to_trough": lobe_to_trough,
        "contrast": contrast,
        "biophysical": ac_at(bio_lag_ms),
    }
