"""Original phase estimators and trajectory selection; no execution or plotting."""

import numpy as np
from experiments.exp085 import (
    detect_volleys,
    interpolated_phase,
    population_rate,
    rhythm_summary,
)
from scipy.ndimage import gaussian_filter1d

from .recipe import (
    ANALYSIS_START_MS,
    DT_MS,
    K_VALUES,
    N_E,
    N_I,
    PHASE_BINS,
    VELOCITY_SMOOTH_MS,
)


def instantaneous_frequency(peaks: np.ndarray, steps: int) -> np.ndarray:
    """Assign each inter-volley interval its instantaneous frequency."""
    frequency = np.full(steps, np.nan)
    for left, right in zip(peaks[:-1], peaks[1:], strict=True):
        if right > left:
            frequency[left:right] = 1_000.0 / ((right - left) * DT_MS)
    return frequency


def circular_distance(a: float, b: float) -> float:
    """Return the absolute shortest angular distance in radians."""
    return float(abs(np.angle(np.exp(1j * (a - b)))))


def analyse_trajectory(
    recordings: dict[str, np.ndarray],
    *,
    k: float,
) -> dict[str, object]:
    """Measure position, velocity, slips, and preferred phase for one K."""
    e_a = population_rate(recordings["population_0"], N_E)
    i_a = population_rate(recordings["population_1"], N_I)
    e_b = population_rate(recordings["population_2"], N_E)
    i_b = population_rate(recordings["population_3"], N_I)
    peaks_a = detect_volleys(e_a, burn_ms=0.0)
    peaks_b = detect_volleys(e_b, burn_ms=0.0)
    phase_a = interpolated_phase(peaks_a, len(e_a))
    phase_b = interpolated_phase(peaks_b, len(e_b))
    frequency_a = instantaneous_frequency(peaks_a, len(e_a))
    frequency_b = instantaneous_frequency(peaks_b, len(e_b))

    wrapped = np.angle(np.exp(1j * (phase_a - phase_b)))
    velocity = 2.0 * np.pi * (frequency_a - frequency_b)
    valid = np.isfinite(wrapped) & np.isfinite(velocity)
    valid &= np.arange(len(wrapped)) * DT_MS >= ANALYSIS_START_MS
    if valid.sum() < 100:
        raise RuntimeError(f"K={k:g} produced too little valid phase data")

    time_ms = np.arange(len(wrapped))[valid] * DT_MS
    wrapped_valid = wrapped[valid]
    velocity_valid = velocity[valid]
    unwrapped = np.unwrap(wrapped_valid)
    unwrapped -= unwrapped[0]
    net_cycles = float((unwrapped[-1] - unwrapped[0]) / (2.0 * np.pi))
    slips = int(np.floor(abs(net_cycles) + 1e-9))
    concentration = float(abs(np.mean(np.exp(1j * wrapped_valid))))

    edges = np.linspace(-np.pi, np.pi, PHASE_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(wrapped_valid, bins=edges)
    density = counts / counts.sum() / np.diff(edges)
    bin_index = np.clip(np.digitize(wrapped_valid, edges) - 1, 0, PHASE_BINS - 1)
    mean_velocity = np.full(PHASE_BINS, np.nan)
    for index in range(PHASE_BINS):
        values = velocity_valid[bin_index == index]
        if values.size:
            mean_velocity[index] = float(values.mean())

    preferred_index = int(np.argmax(density))
    preferred_phase = float(centres[preferred_index])
    populated = np.isfinite(mean_velocity)
    slow_index = int(
        np.flatnonzero(populated)[np.argmin(np.abs(mean_velocity[populated]))]
    )
    slow_phase = float(centres[slow_index])
    alignment = circular_distance(preferred_phase, slow_phase)
    density_ratio = float(density.max() / density.mean())
    mean_abs_velocity = float(np.mean(np.abs(velocity_valid)))
    slow_abs_velocity = float(abs(mean_velocity[slow_index]))
    slowing_fraction = (
        1.0 - slow_abs_velocity / mean_abs_velocity if mean_abs_velocity > 0 else 0.0
    )

    return {
        "k": float(k),
        "time_ms": time_ms,
        "rate_e_a": e_a,
        "rate_i_a": i_a,
        "rate_e_b": e_b,
        "rate_i_b": i_b,
        "peaks_a": peaks_a,
        "peaks_b": peaks_b,
        "wrapped_phase": wrapped_valid,
        "unwrapped_phase": unwrapped,
        "relative_velocity_rad_s": velocity_valid,
        "relative_velocity_smoothed_rad_s": gaussian_filter1d(
            velocity_valid,
            sigma=VELOCITY_SMOOTH_MS / DT_MS,
        ),
        "phase_bin_centres": centres,
        "phase_density": density,
        "mean_velocity_by_phase": mean_velocity,
        "preferred_phase_rad": preferred_phase,
        "slow_phase_rad": slow_phase,
        "phase_alignment_error_rad": alignment,
        "phase_concentration": concentration,
        "density_peak_to_mean": density_ratio,
        "slowing_fraction": slowing_fraction,
        "net_phase_change_cycles": net_cycles,
        "phase_slips": slips,
        "network_a": rhythm_summary(peaks_a),
        "network_b": rhythm_summary(peaks_b),
    }


def public_summary(trajectory: dict[str, object]) -> dict[str, object]:
    """Drop large arrays before JSON serialization."""
    array_keys = {
        "time_ms",
        "rate_e_a",
        "rate_i_a",
        "rate_e_b",
        "rate_i_b",
        "peaks_a",
        "peaks_b",
        "wrapped_phase",
        "unwrapped_phase",
        "relative_velocity_rad_s",
        "relative_velocity_smoothed_rad_s",
        "phase_bin_centres",
        "phase_density",
        "mean_velocity_by_phase",
    }
    return {key: value for key, value in trajectory.items() if key not in array_keys}


def choose_intermediate(trajectories: list[dict[str, object]]) -> dict[str, object]:
    """Choose the slipping nonzero-K trajectory with the clearest attraction."""
    candidates = [
        row
        for row in trajectories
        if 0.0 < float(row["k"]) < float(K_VALUES.max())
        and int(row["phase_slips"]) >= 2
    ]
    if not candidates:
        raise RuntimeError("the coupling sweep produced no intermediate slipping case")

    def score(row: dict[str, object]) -> float:
        alignment_score = np.exp(-float(row["phase_alignment_error_rad"]))
        return (
            float(row["phase_concentration"])
            * float(row["density_peak_to_mean"])
            * max(float(row["slowing_fraction"]), 0.0)
            * alignment_score
        )

    return max(candidates, key=score)
