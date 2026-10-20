"""Measure the spectrum, participation and rhythm contrast used in the report.

Inputs are time-by-neuron E/I spike arrays and resolved analysis settings.
SNNLab supplies the spectrum, cycle counts and rhythm contrast; inhibitory
volleys define complete interior cycles for the participation measurement.

Return only trial acceptance, E participation, finite rhythm contrast and the
E spectrum needed to locate the condition's pooled spectral peak. Rejected
trials have undefined participation rather than zero participation.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from snnlab import analysis


def dynamics(e, i, cfg):
    dt = cfg["dt_ms"]
    spectrum = analysis.power_spectrum(
        e.sum(axis=1),
        dt,
        method="welch",
        center=True,
        nperseg=len(e),
        window="hann",
        scaling="density",
        detrend=False,
    )
    inhibitory = gaussian_filter1d(
        i.sum(axis=1).astype(float), sigma=cfg["volley_smoothing_ms"] / dt
    )
    volleys, _ = find_peaks(
        inhibitory,
        distance=round(cfg["volley_minimum_spacing_ms"] / dt),
        prominence=max(
            cfg["volley_minimum_prominence_spikes"],
            cfg["volley_relative_prominence"] * inhibitory.max(),
        ),
    )
    # Discard partial edge cycles; interior volleys are bracketed by midpoints.
    boundaries = analysis.cycle_boundaries(volleys, len(e), policy="midpoints")[
        "edges"
    ][1:-1]
    if len(boundaries) < 2:
        boundaries = np.empty(0, dtype=np.int64)
    inhibitory_cycles = analysis.cycle_spike_counts(i, boundaries)["counts"]
    valid = bool(
        len(inhibitory_cycles) >= cfg["minimum_cycles"]
        and (100 * (inhibitory_cycles > 0).mean(axis=1)).mean()
        >= cfg["minimum_mean_i_participation_pct"]
    )
    participation = None
    if valid:
        excitatory_cycles = analysis.cycle_spike_counts(e, boundaries)["counts"]
        participation = float((100 * (excitatory_cycles > 0).mean(axis=1)).mean())
    rhythm = analysis.rhythmicity_metrics(
        e,
        dt,
        max_lag_ms=cfg["rhythmicity_max_lag_ms"],
        bin_ms=cfg["population_bin_ms"],
    )
    contrast = analysis.rhythmicity_scalars(
        rhythm["ac_lags"],
        rhythm["ac"],
        rhythm["iei_lags"],
        rhythm["iei_counts"],
        bin_ms=cfg["population_bin_ms"],
    )["contrast"]
    return dict(
        valid_cycles=valid,
        participation_e_pct=participation,
        rhythmicity_contrast=float(contrast)
        if contrast is not None and np.isfinite(contrast)
        else None,
    ), dict(frequency_hz=spectrum["frequencies_hz"], psd=spectrum["power"])
