from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
from pinglab.plots.styles import save_both


def estimate_trough_peak_and_period(
    lfp: np.ndarray,
    dt: float,
    baseline_ms: float = 500.0,
    min_cycle_ms: float = 10.0,
    max_cycle_ms: float = 40.0,
) -> tuple[float, float]:
    start_idx = int(baseline_ms / dt)
    t = np.arange(len(lfp)) * dt

    segment = lfp[start_idx:]
    t_seg = t[start_idx:]

    dist_samples = int(min_cycle_ms / dt)
    peaks_idx, _ = find_peaks(segment, distance=dist_samples)
    troughs_idx, _ = find_peaks(-segment, distance=dist_samples)

    if len(peaks_idx) == 0 or len(troughs_idx) == 0:
        raise RuntimeError("Could not find peaks/troughs in LFP")

    peak_t_ms = t_seg[peaks_idx[0]]

    if len(peaks_idx) >= 2:
        T_gamma_ms = float(t_seg[peaks_idx[1]] - t_seg[peaks_idx[0]])
    else:
        T_gamma_ms = max_cycle_ms

    return float(peak_t_ms), float(T_gamma_ms)


def add_pulse_to_input(
    baseline_input: np.ndarray,
    target_E: np.ndarray,
    pulse_t: float,
    pulse_width_ms: float,
    pulse_amp: float,
    dt: float,
    num_steps: int,
) -> np.ndarray:
    """Add a pulse stimulus to target neurons at specified time."""
    pulse_input = baseline_input.copy()
    p0 = int(pulse_t / dt)
    p1 = int((pulse_t + pulse_width_ms) / dt)
    p0 = max(0, min(p0, num_steps - 1))
    p1 = max(p0 + 1, min(p1, num_steps))
    pulse_input[p0:p1, target_E] += pulse_amp
    return pulse_input


def compute_spike_delta(
    spikes,
    target_E: np.ndarray,
    pulse_t: float,
    pre_window_ms: float,
    post_window_ms: float,
) -> int:
    """Compute change in spike count before and after pulse."""
    spike_t = spikes.times
    spike_ids = spikes.ids
    mask_target = np.isin(spike_ids, target_E)
    t_target = spike_t[mask_target]

    pre_mask = (t_target >= pulse_t - pre_window_ms) & (t_target < pulse_t)
    post_mask = (t_target >= pulse_t) & (t_target < pulse_t + post_window_ms)

    return int(np.sum(post_mask) - np.sum(pre_mask))


def plot_phase_gain_curve(
    phase_times: list[float],
    deltas: list[float],
    T_gamma_ms: float,
    lfp_proxy: np.ndarray,
    peak_t_ms: float,
    dt: float,
    save_path: Path,
) -> float:
    """Plot phase-dependent gain curve and compute modulation index."""
    phase_times_arr = np.array(phase_times)
    deltas_arr = np.array(deltas, dtype=float)

    phases_rad = (phase_times_arr / T_gamma_ms) * 2 * np.pi

    order = np.argsort(phases_rad)
    phases_sorted = phases_rad[order]
    deltas_sorted = deltas_arr[order]

    x_min = phases_sorted.min()
    x_max = phases_sorted.max()

    t_all = np.arange(len(lfp_proxy)) * dt
    phases_lfp = ((t_all - peak_t_ms) / T_gamma_ms) * 2 * np.pi

    mask_lfp = (phases_lfp >= x_min) & (phases_lfp <= x_max)
    phases_lfp_window = phases_lfp[mask_lfp]
    lfp_window = lfp_proxy[mask_lfp]

    def plot_fn():
        fig, ax1 = plt.subplots(figsize=(8, 8))

        # Get the color cycle colors
        prop_cycle = mpl.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        # First line uses first color (black/white)
        ax1.plot(phases_sorted, deltas_sorted, "o-", lw=2, label="Δ-spikes", color=colors[0])
        ax1.set_xlabel("Pulse phase (unwrapped radians)")
        ax1.set_ylabel("Δ-spikes (target E)")
        ax1.set_xlim(x_min, x_max)

        if phases_lfp_window.size > 0:
            order_lfp = np.argsort(phases_lfp_window)
            phases_lfp_sorted = phases_lfp_window[order_lfp]
            lfp_sorted = lfp_window[order_lfp]
            lfp_norm = (lfp_sorted - lfp_sorted.min()) / (
                lfp_sorted.max() - lfp_sorted.min() + 1e-9
            )

            ax2 = ax1.twinx()
            # Second line uses second color (red/light red)
            ax2.plot(phases_lfp_sorted, lfp_norm, "--", alpha=0.6, label="LFP proxy", color=colors[1])
            ax2.set_ylabel("Normalized inhibitory conductance (LFP proxy)")

        plt.title("Phase-dependent gain (Δ-spikes) in probed phase window")
        plt.tight_layout()

    save_both(save_path, plot_fn)

    modulation_index = (deltas_arr.max() - deltas_arr.min()) / (
        deltas_arr.max() + deltas_arr.min() + 1e-9
    )

    return modulation_index
