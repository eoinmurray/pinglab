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
    counts = (
        spikes[: n_bins * bin_steps].reshape(n_bins, bin_steps, -1).sum(axis=(1, 2))
    )
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
# CLI-level metric helpers (moved from cli.py)
# =========================================================================


def compute_pop_rate(spk, n_neurons, dt, bin_ms=2.0):
    """Compute population firing rate in Hz."""
    return population_rate_nondiff(spk, n_neurons, bin_ms=bin_ms, dt_ms=dt)


def compute_psd(
    spk_e,
    n_neurons,
    dt,
    bin_ms=2.0,
    step_on_ms=200.0,
    step_off_ms=300.0,
    burn_in_ms=100.0,
):
    """PSD of population rate during stimulus window (or full raster if no window)."""
    s0 = int((step_on_ms - burn_in_ms) / dt)
    s1 = int((step_off_ms - burn_in_ms) / dt)
    spk_stim = spk_e[max(0, s0) : min(len(spk_e), s1)]
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


def compute_metrics(
    spk_e,
    spk_i,
    dt,
    model_name="ping",
    n_e=1024,
    n_i=256,
    step_on_ms=200.0,
    step_off_ms=300.0,
    burn_in_ms=100.0,
):
    """Compute population metrics from spike rasters. Returns a plain dict."""
    t_sec = len(spk_e) * dt / 1000.0
    rate_e = float(spk_e.sum() / (n_e * t_sec))
    rate_i = float(spk_i.sum() / (n_i * t_sec)) if spk_i is not None else 0.0

    freqs, psd = compute_psd(
        spk_e,
        n_e,
        dt,
        step_on_ms=step_on_ms,
        step_off_ms=step_off_ms,
        burn_in_ms=burn_in_ms,
    )
    f0 = float(find_fundamental_nondiff(psd, freqs)) if model_name == "ping" else 0.0

    # Population spike count CV in 2ms bins
    bin_steps = max(1, int(2.0 / dt))
    n_bins = len(spk_e) // bin_steps
    if n_bins > 1:
        pop_counts = np.array(
            [spk_e[i * bin_steps : (i + 1) * bin_steps].sum() for i in range(n_bins)]
        )
        pop_cv = float(pop_counts.std() / max(pop_counts.mean(), 1e-9))
    else:
        pop_cv = 0.0

    per_neuron_counts = spk_e.sum(axis=0)
    active_frac = float((per_neuron_counts > 0).sum()) / n_e

    # nb054 lobe–trough contrast (pingness): rhythmicity of the E population,
    # 0 = flat/asynchronous, → 1 as sharp volleys separate against near-silence.
    try:
        contrast = rhythmicity_metrics(spk_e, dt).get("contrast")
    except Exception:
        contrast = None

    return {
        "rate_e": rate_e,
        "rate_i": rate_i,
        "cv": pop_cv,
        "act": active_frac,
        "f0": f0,
        "contrast": float(contrast) if contrast is not None else 0.0,
    }


def format_metrics(m):
    """Render a metrics dict as the canonical fixed-width log line."""
    f0_str = f" f0={m['f0']:>3.0f}" if m["f0"] > 0 else ""
    return (
        f"E={m['rate_e']:>3.0f} I={m['rate_i']:>3.0f} "
        f"CV={m['cv']:>4.2f} act={m['act']:>4.0%}{f0_str}"
    )


def report_metrics(
    spk_e,
    spk_i,
    dt,
    model_name="ping",
    n_e=1024,
    n_i=256,
    step_on_ms=200.0,
    step_off_ms=300.0,
    burn_in_ms=100.0,
    quiet=False,
):
    """Compute simulation metrics, log to stdout (unless quiet), return string.

    Kept for backward compatibility. New code should call compute_metrics
    and format the result with format_metrics.
    """
    m = compute_metrics(
        spk_e,
        spk_i,
        dt,
        model_name=model_name,
        n_e=n_e,
        n_i=n_i,
        step_on_ms=step_on_ms,
        step_off_ms=step_off_ms,
        burn_in_ms=burn_in_ms,
    )
    s = format_metrics(m)
    if not quiet:
        import logging

        logging.getLogger("cli").info(f"  {s}")
    return s


# =========================================================================
# Rhythmicity: spike-time autocorrelation + IEI histogram (nb054)
# =========================================================================


def population_event_times(spikes, dt):
    """Pooled population spike-event times in ms from a [T_steps, N] raster.

    Every (timestep, neuron) spike contributes one event at time step*dt;
    simultaneous spikes across neurons give repeated times. Returned sorted.
    """
    spikes = np.asarray(spikes)
    step_idx = np.nonzero(spikes)[0]
    return np.sort(step_idx).astype(float) * dt


def iei_histogram(event_times_ms, max_lag_ms=100.0, bin_ms=1.0):
    """Inter-event-interval histogram of a pooled event train.

    Intervals between consecutive (sorted) events. Returns (lag_centers_ms,
    counts). The first mode is the short-interval pile-up; a rhythm shows a
    second mode near the period.
    """
    t = np.sort(np.asarray(event_times_ms, dtype=float))
    edges = np.arange(0.0, max_lag_ms + bin_ms, bin_ms)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if t.size < 2:
        return centers, np.zeros(centers.size)
    counts, _ = np.histogram(np.diff(t), bins=edges)
    return centers, counts.astype(float)


def spike_autocorrelogram(spikes, dt, max_lag_ms=100.0, bin_ms=1.0):
    """Normalised population spike-time autocorrelogram.

    Bins the population count r(t) at bin_ms, forms the pair-count
    autocorrelation Σ_t r(t)·r(t+ℓ), divides by the per-lag overlap count and
    by mean(r)² so the asymptotic floor is 1.0 (rate-matched independence).
    The zero-lag bin is dominated by self-pairs and returned as NaN.

    Returns (lags_ms, ac). A rhythm shows ac > 1 in a central lobe near ℓ = 0,
    a trough (ac < 1) around the half-period, and a secondary peak at the
    period — the "Mexican hat". Flat (asynchronous) firing gives ac ≈ 1.
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
    # Linear (non-circular) autocorrelation Σ_t r(t)·r(t+ℓ) via zero-padded
    # FFT — O(n log n), so long single-train traces stay fast.
    nfft = 1 << int(np.ceil(np.log2(2 * n_bins)))
    f = np.fft.rfft(r, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[: max_lag_bins + 1].astype(float)
    overlap = n_bins - np.arange(max_lag_bins + 1)  # terms summed at each lag
    floor = r.mean() ** 2
    if floor <= 0:
        return lags, np.full(max_lag_bins + 1, np.nan)
    ac = ac / overlap / floor
    ac[0] = np.nan  # self-pairs dominate zero-lag
    return lags, ac


def _smooth_ac(ac):
    """3-point-smoothed copy of the autocorrelogram (lag-0 NaN filled), or None
    if it cannot be made finite. Smoothing controls the finite-count shot noise
    a single Poisson spike train carries, so lobe/trough extraction is stable.
    """
    a = np.asarray(ac, dtype=float).copy()
    if a.size > 1 and not np.isfinite(a[0]):
        a[0] = a[1]  # lag-0 is NaN by construction
    if not np.all(np.isfinite(a)):
        return None
    return np.convolve(a, [0.25, 0.5, 0.25], mode="same")


def rhythmicity_scalars(ac_lags, ac, iei_lags, iei_counts, bin_ms=1.0, bio_lag_ms=None):
    """Candidate rhythmicity scalars from an autocorrelogram + IEI histogram.

    Split out from rhythmicity_metrics so the same extraction can run on a
    single train's curves or on curves trial-averaged across seeds (the latter
    gives an unbiased lobe/trough on noisy single-train data). Reports:

      - iei_anchored: ac at the IEI primary-mode lag (central-lobe height).
      - lobe_to_trough: central-lobe peak / first-trough depth, both read from
        a lightly smoothed autocorrelogram (the trough is its first local
        minimum), with the trough floored at 0.05 of the unity baseline.
      - biophysical: ac at a supplied network lag bio_lag_ms, or None.
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
            # Unbounded ratio (kept for reference) and the bounded Mexican-hat
            # contrast (lobe−trough)/(lobe+trough) ∈ [0, 1): 0 = flat (lobe=trough),
            # → 1 as the trough goes silent. No trough floor needed.
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


def rhythmicity_metrics(spikes, dt, max_lag_ms=100.0, bin_ms=1.0, bio_lag_ms=None):
    """Spike-time-autocorrelation rhythmicity scalars for a [T, N] raster.

    Builds the IEI histogram and the normalised autocorrelogram, extracts the
    candidate scalars (see rhythmicity_scalars), and returns them together with
    the curves and located lags for plotting. All scalars are calibrated so
    flat/Poisson firing sits near the baseline value 1.0.
    """
    event_times = population_event_times(spikes, dt)
    iei_lags, iei_counts = iei_histogram(event_times, max_lag_ms, bin_ms)
    ac_lags, ac = spike_autocorrelogram(spikes, dt, max_lag_ms, bin_ms)
    out = rhythmicity_scalars(ac_lags, ac, iei_lags, iei_counts, bin_ms, bio_lag_ms)
    out.update(
        iei_lags=iei_lags,
        iei_counts=iei_counts,
        ac_lags=ac_lags,
        ac=ac,
        n_events=int(event_times.size),
    )
    return out


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
        pop_counts = np.array(
            [spk_e[i * bin_steps : (i + 1) * bin_steps].sum() for i in range(n_bins)]
        )
        pop_cv = float(pop_counts.std() / max(pop_counts.mean(), 1e-9))
    else:
        pop_cv = 0.0
    ie = rate_i / max(rate_e, 1e-9) if spk_i is not None else 0
    return f"E={rate_e:.0f} I={rate_i:.0f} CV={pop_cv:.2f} I/E={ie:.1f}"
