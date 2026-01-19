"""Phase-locking value (PLV) analysis for population spiking."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from pinglab.types import Spikes
from pinglab.analysis.population_rate import population_rate


def plv_from_phase_series(
    phase_t_ms: np.ndarray,
    phase: np.ndarray,
    spike_times_ms: np.ndarray,
) -> float:
    """
    Compute PLV from a phase time series and spike times.

    Parameters:
        phase_t_ms: Time axis (ms) for phase array.
        phase: Instantaneous phase in radians.
        spike_times_ms: Spike times in ms to sample phases at.

    Returns:
        PLV in [0, 1].
    """
    if spike_times_ms.size == 0:
        return 0.0
    if phase_t_ms.size == 0:
        return 0.0

    sample_phase = np.interp(spike_times_ms, phase_t_ms, phase)
    return float(np.abs(np.mean(np.exp(1j * sample_phase))))


def plv_phase_series(
    spikes: Spikes,
    T_ms: float,
    dt_ms: float,
    fmin: float,
    fmax: float,
    pop: str = "E",
    N_E: int | None = None,
    N_I: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Compute intermediate PLV series for plotting and debugging.

    Returns:
        (t_ms, rate_hz, filtered, phase, spike_times_ms) or None if invalid.
    """
    if T_ms <= 0:
        raise ValueError(f"T_ms must be positive, got {T_ms}")
    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be positive, got {dt_ms}")
    if fmin <= 0 or fmax <= 0 or fmax <= fmin:
        raise ValueError("fmin and fmax must be positive with fmax > fmin")

    t_ms, rate_hz = population_rate(
        spikes=spikes,
        T_ms=T_ms,
        dt_ms=dt_ms,
        pop=pop,
        N_E=N_E,
        N_I=N_I,
    )

    if rate_hz.size < 3:
        return None

    fs = 1000.0 / dt_ms
    nyq = 0.5 * fs
    if fmax >= nyq:
        raise ValueError(f"fmax must be < Nyquist ({nyq:.3f} Hz)")

    rate_centered = rate_hz - float(np.mean(rate_hz))
    if float(np.std(rate_centered)) == 0.0:
        return None

    sos = butter(4, [fmin / nyq, fmax / nyq], btype="bandpass", output="sos")
    padlen = 3 * (2 * sos.shape[0] + 1)
    if rate_centered.size <= padlen:
        return None

    filtered = sosfiltfilt(sos, rate_centered)
    phase = np.angle(hilbert(filtered))

    spike_times_ms = spikes.times
    if pop == "E" and N_E is not None:
        spike_times_ms = spike_times_ms[spikes.ids < N_E]
    elif pop == "I" and N_E is not None:
        spike_times_ms = spike_times_ms[spikes.ids >= N_E]

    return t_ms, rate_hz, filtered, phase, spike_times_ms


def population_plv(
    spikes: Spikes,
    T_ms: float,
    dt_ms: float,
    fmin: float,
    fmax: float,
    pop: str = "E",
    N_E: int | None = None,
    N_I: int | None = None,
) -> float:
    """
    Compute phase-locking value (PLV) of spikes to population rhythm.

    The population rate is bandpass-filtered and its instantaneous phase is
    computed via the Hilbert transform. PLV is the vector strength of spike
    phases sampled from that phase series.
    """
    series = plv_phase_series(
        spikes=spikes,
        T_ms=T_ms,
        dt_ms=dt_ms,
        fmin=fmin,
        fmax=fmax,
        pop=pop,
        N_E=N_E,
        N_I=N_I,
    )
    if series is None:
        return 0.0
    t_ms, _, _, phase, spike_times_ms = series
    return plv_from_phase_series(t_ms, phase, spike_times_ms)
