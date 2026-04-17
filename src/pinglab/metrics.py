"""Reusable analysis functions for SNN spike data.

Population rate, PSD, oscillatory power, peak frequency detection,
and reporting helpers.
"""
from __future__ import annotations

import numpy as np


def population_rate_nondiff(spikes, n_neurons, bin_ms=1.0, dt_ms=0.25):
    """Compute population firing rate from spike array [T_steps, N_neurons].
    Returns (t_bins_ms, rate_hz)."""
    bin_steps = max(1, int(bin_ms / dt_ms))
    T = len(spikes)
    n_bins = T // bin_steps
    counts = spikes[:n_bins * bin_steps].reshape(n_bins, bin_steps, -1).sum(axis=(1, 2))
    rate_hz = counts / (n_neurons * bin_ms / 1000.0)
    t_bins = np.arange(n_bins) * bin_ms
    return t_bins, rate_hz


def find_fundamental_nondiff(psd, freqs, f_lo=5.0, f_hi=80.0, snr_threshold=3.0):
    """Find fundamental frequency as the lowest prominent harmonic in the PSD.
    Checks if the highest peak has a sub-harmonic at f/2 -- if so, that's the
    true fundamental. Returns f0 in Hz, or 0.0 if no oscillation detected."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not mask.any():
        return 0.0
    psd_band = psd[mask]
    freqs_band = freqs[mask]
    idx = np.argmax(psd_band)
    peak_val = psd_band[idx]
    peak_freq = freqs_band[idx]
    median_val = np.median(psd_band)
    if median_val > 0 and peak_val / median_val < snr_threshold:
        return 0.0
    half_freq = peak_freq / 2
    if half_freq >= f_lo:
        half_mask = np.abs(freqs_band - half_freq) < (freqs_band[1] - freqs_band[0]) * 2
        if half_mask.any():
            sub_power = psd_band[half_mask].max()
            if sub_power > 0.3 * peak_val:
                return float(freqs_band[half_mask][np.argmax(psd_band[half_mask])])
    return float(peak_freq)


# =========================================================================
# Oscilloscope-level metric helpers (moved from oscilloscope.py)
# =========================================================================

def compute_pop_rate(spk, n_neurons, dt, bin_ms=2.0):
    """Compute population firing rate in Hz."""
    return population_rate_nondiff(spk, n_neurons, bin_ms=bin_ms, dt_ms=dt)


def compute_psd(spk_e, n_neurons, dt, bin_ms=2.0,
                step_on_ms=200.0, step_off_ms=300.0, burn_in_ms=100.0):
    """PSD of population rate during stimulus window (or full raster if no window)."""
    s0 = int((step_on_ms - burn_in_ms) / dt)
    s1 = int((step_off_ms - burn_in_ms) / dt)
    spk_stim = spk_e[max(0, s0):min(len(spk_e), s1)]
    if len(spk_stim) < 10:
        # No valid stimulus window -- use full raster
        spk_stim = spk_e
    _, rate_hz = compute_pop_rate(spk_stim, n_neurons, dt, bin_ms)
    rate_centered = rate_hz - rate_hz.mean()
    n = len(rate_centered)
    freqs = np.fft.rfftfreq(n, d=bin_ms / 1000.0)
    psd = np.abs(np.fft.rfft(rate_centered)) ** 2
    psd = psd / psd.max() if psd.max() > 0 else psd
    return freqs, psd


def compute_metrics(spk_e, spk_i, dt, model_name="ping",
                    n_e=1024, n_i=256,
                    step_on_ms=200.0, step_off_ms=300.0, burn_in_ms=100.0):
    """Compute population metrics from spike rasters. Returns a plain dict."""
    t_sec = len(spk_e) * dt / 1000.0
    rate_e = float(spk_e.sum() / (n_e * t_sec))
    rate_i = float(spk_i.sum() / (n_i * t_sec)) if spk_i is not None else 0.0

    freqs, psd = compute_psd(spk_e, n_e, dt,
                             step_on_ms=step_on_ms, step_off_ms=step_off_ms,
                             burn_in_ms=burn_in_ms)
    f0 = float(find_fundamental_nondiff(psd, freqs)) if model_name == "ping" else 0.0

    # Population spike count CV in 2ms bins
    bin_steps = max(1, int(2.0 / dt))
    n_bins = len(spk_e) // bin_steps
    if n_bins > 1:
        pop_counts = np.array([spk_e[i * bin_steps:(i + 1) * bin_steps].sum()
                               for i in range(n_bins)])
        pop_cv = float(pop_counts.std() / max(pop_counts.mean(), 1e-9))
    else:
        pop_cv = 0.0

    per_neuron_counts = spk_e.sum(axis=0)
    active_frac = float((per_neuron_counts > 0).sum()) / n_e

    return {
        "rate_e": rate_e,
        "rate_i": rate_i,
        "cv": pop_cv,
        "act": active_frac,
        "f0": f0,
    }


def format_metrics(m):
    """Render a metrics dict as the canonical fixed-width log line."""
    f0_str = f" f0={m['f0']:>3.0f}" if m["f0"] > 0 else ""
    return (f"E={m['rate_e']:>3.0f} I={m['rate_i']:>3.0f} "
            f"CV={m['cv']:>4.2f} act={m['act']:>4.0%}{f0_str}")


def report_metrics(spk_e, spk_i, dt, model_name="ping",
                   n_e=1024, n_i=256,
                   step_on_ms=200.0, step_off_ms=300.0, burn_in_ms=100.0,
                   quiet=False):
    """Compute simulation metrics, log to stdout (unless quiet), return string.

    Kept for backward compatibility. New code should call compute_metrics
    and format the result with format_metrics.
    """
    m = compute_metrics(spk_e, spk_i, dt, model_name=model_name,
                        n_e=n_e, n_i=n_i,
                        step_on_ms=step_on_ms, step_off_ms=step_off_ms,
                        burn_in_ms=burn_in_ms)
    s = format_metrics(m)
    if not quiet:
        import logging
        logging.getLogger("oscilloscope").info(f"  {s}")
    return s


def metrics_str(spk_e, spk_i, dt, model_name="ping", n_e=1024, n_i=256):
    """Compact one-line metrics string for scan frame logging."""
    t_sec = len(spk_e) * dt / 1000.0
    if t_sec == 0:
        return "no spikes"
    rate_e = spk_e.sum() / (n_e * t_sec)
    rate_i = spk_i.sum() / (n_i * t_sec) if spk_i is not None else 0
    bin_steps = max(1, int(2.0 / dt))
    n_bins = len(spk_e) // bin_steps
    if n_bins > 1:
        pop_counts = np.array([spk_e[i * bin_steps:(i + 1) * bin_steps].sum()
                               for i in range(n_bins)])
        pop_cv = float(pop_counts.std() / max(pop_counts.mean(), 1e-9))
    else:
        pop_cv = 0.0
    ie = rate_i / max(rate_e, 1e-9) if spk_i is not None else 0
    return f"E={rate_e:.0f} I={rate_i:.0f} CV={pop_cv:.2f} I/E={ie:.1f}"
