from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
from pinglab.plots.styles import save_both


def estimate_gamma_period(
    lfp: np.ndarray,
    dt: float,
    baseline_ms: float = 500.0,
    min_cycle_ms: float = 10.0,
) -> float:
    """
    Estimate the gamma oscillation period from LFP.

    Args:
        lfp: LFP proxy signal (typically inhibitory conductance)
        dt: Time step (ms)
        baseline_ms: Ignore initial transient (ms)
        min_cycle_ms: Minimum expected cycle duration (ms)

    Returns:
        Gamma period in milliseconds
    """
    start_idx = int(baseline_ms / dt)
    t = np.arange(len(lfp)) * dt

    segment = lfp[start_idx:]
    t_seg = t[start_idx:]

    dist_samples = int(min_cycle_ms / dt)
    peaks_idx, _ = find_peaks(segment, distance=dist_samples)

    if len(peaks_idx) < 2:
        return 25.0  # Default gamma period if can't detect

    # Average period across detected cycles
    periods = np.diff(t_seg[peaks_idx])
    return float(np.median(periods))


def extract_spike_volley(
    spikes,
    target_neurons: np.ndarray,
    pulse_t: float,
    window_ms: float = 50.0,
) -> tuple[np.ndarray, int]:
    """
    Extract spike times from target neurons after pulse.

    Args:
        spikes: Spike data object with .times and .ids
        target_neurons: Array of neuron IDs to extract from
        pulse_t: Time of pulse delivery (ms)
        window_ms: Window after pulse to collect spikes (ms)

    Returns:
        Array of spike times and count
    """
    spike_t = spikes.times
    spike_ids = spikes.ids

    # Filter for target neurons
    mask_target = np.isin(spike_ids, target_neurons)
    t_target = spike_t[mask_target]

    # Filter for post-pulse window
    mask_window = (t_target >= pulse_t) & (t_target < pulse_t + window_ms)
    volley_times = t_target[mask_window]

    return volley_times, len(volley_times)


def measure_transfer_gain(
    spikes,
    target_neurons: np.ndarray,
    volley_t: float,
    pre_window_ms: float,
    post_window_ms: float,
) -> tuple[int, int, float]:
    """
    Measure transfer gain: response of downstream neurons to upstream volley.

    Args:
        spikes: Spike data from downstream population B
        target_neurons: Downstream E neurons
        volley_t: Time when volley arrives (ms)
        pre_window_ms: Pre-volley baseline window (ms)
        post_window_ms: Post-volley response window (ms)

    Returns:
        (pre_count, post_count, gain)
    """
    spike_t = spikes.times
    spike_ids = spikes.ids

    # Filter for target neurons in B
    mask_target = np.isin(spike_ids, target_neurons)
    t_target = spike_t[mask_target]

    # Count spikes before and after volley
    pre_mask = (t_target >= volley_t - pre_window_ms) & (t_target < volley_t)
    post_mask = (t_target >= volley_t) & (t_target < volley_t + post_window_ms)

    pre_count = int(np.sum(pre_mask))
    post_count = int(np.sum(post_mask))
    gain = float(post_count - pre_count)

    return pre_count, post_count, gain


def add_pulse_to_input(
    baseline_input: np.ndarray,
    target_neurons: np.ndarray,
    pulse_t: float,
    pulse_width_ms: float,
    pulse_amp: float,
    dt: float,
    num_steps: int,
) -> np.ndarray:
    """
    Add a brief pulse to create spike volley.

    Args:
        baseline_input: Base input array (num_steps, N_E)
        target_neurons: Neuron IDs to stimulate
        pulse_t: Pulse onset time (ms)
        pulse_width_ms: Pulse duration (ms)
        pulse_amp: Pulse amplitude
        dt: Time step (ms)
        num_steps: Total simulation steps

    Returns:
        Modified input array
    """
    pulse_input = baseline_input.copy()
    p0 = int(pulse_t / dt)
    p1 = int((pulse_t + pulse_width_ms) / dt)
    p0 = max(0, min(p0, num_steps - 1))
    p1 = max(p0 + 1, min(p1, num_steps))
    pulse_input[p0:p1, target_neurons] += pulse_amp
    return pulse_input


def add_volley_as_input(
    baseline_input: np.ndarray,
    target_neurons: np.ndarray,
    volley_times: np.ndarray,
    phase_shift_ms: float,
    g_AB: float,
    tau_ampa: float,
    dt: float,
) -> np.ndarray:
    """
    Add upstream spike volley as synaptic input to downstream population.

    Args:
        baseline_input: Base input for population B (num_steps, N_E_B)
        target_neurons: Downstream neurons receiving input
        volley_times: Spike times from upstream volley (ms)
        phase_shift_ms: Time shift to apply to volley (for phase manipulation)
        g_AB: Coupling strength from A to B
        tau_ampa: AMPA time constant (ms)
        dt: Time step (ms)

    Returns:
        Modified input array with volley
    """
    num_steps = baseline_input.shape[0]
    volley_input = baseline_input.copy()

    # Shift volley times by phase offset
    shifted_times = volley_times + phase_shift_ms

    # For each spike in volley, add exponentially decaying conductance
    for spike_t in shifted_times:
        if spike_t < 0 or spike_t >= num_steps * dt:
            continue

        spike_idx = int(spike_t / dt)
        # Create exponential decay starting from spike time
        decay_steps = int(5 * tau_ampa / dt)  # 5 time constants
        decay_steps = min(decay_steps, num_steps - spike_idx)

        t_array = np.arange(decay_steps) * dt
        conductance = g_AB * np.exp(-t_array / tau_ampa)

        # Add to all target neurons
        volley_input[spike_idx:spike_idx + decay_steps, target_neurons] += conductance[:, np.newaxis]

    return volley_input


def plot_baseline_traces(
    t: np.ndarray,
    V_A: np.ndarray,
    g_i_A: np.ndarray,
    V_B: np.ndarray,
    g_i_B: np.ndarray,
    save_path: Path,
) -> None:
    """
    Plot combined baseline traces for both populations.

    Args:
        t: Time array (ms)
        V_A: Voltage trace for population A (mV)
        g_i_A: Inhibitory conductance for population A
        V_B: Voltage trace for population B (mV)
        g_i_B: Inhibitory conductance for population B
        save_path: Path to save figure
    """
    def plot_fn():
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Population A
        axes[0, 0].plot(t, V_A, lw=1.5)
        axes[0, 0].set_ylabel("V_A (mV)")
        axes[0, 0].set_title("Upstream Population A")
        axes[0, 0].set_xlim(t[0], t[-1])

        axes[1, 0].plot(t, g_i_A, lw=1.5)
        axes[1, 0].set_ylabel("g_i_A")
        axes[1, 0].set_xlabel("Time (ms)")
        axes[1, 0].set_xlim(t[0], t[-1])

        # Population B
        axes[0, 1].plot(t, V_B, lw=1.5)
        axes[0, 1].set_ylabel("V_B (mV)")
        axes[0, 1].set_title("Downstream Population B")
        axes[0, 1].set_xlim(t[0], t[-1])

        axes[1, 1].plot(t, g_i_B, lw=1.5)
        axes[1, 1].set_ylabel("g_i_B")
        axes[1, 1].set_xlabel("Time (ms)")
        axes[1, 1].set_xlim(t[0], t[-1])

        plt.tight_layout()

    save_both(save_path, plot_fn)


def plot_coupled_raster(
    spikes_A,
    spikes_B,
    N_E_A: int,
    N_E_B: int,
    pulse_t: float,
    save_path: Path,
    title: str = "Coupled Population Raster",
) -> None:
    """
    Plot raster showing both populations with pulse marker.

    Args:
        spikes_A: Spikes from population A
        spikes_B: Spikes from population B
        N_E_A: Number of E neurons in A
        N_E_B: Number of E neurons in B
        pulse_t: Time of pulse to A
        save_path: Path to save figure
        title: Plot title
    """
    def plot_fn():
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Get color cycle
        prop_cycle = mpl.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        # Population A raster
        mask_E_A = spikes_A.ids < N_E_A
        axes[0].scatter(spikes_A.times[mask_E_A], spikes_A.ids[mask_E_A],
                       s=1, c=colors[0], alpha=0.5)
        axes[0].axvline(pulse_t, color=colors[1], linestyle='--', alpha=0.7, label='Pulse')
        axes[0].set_ylabel("Neuron ID (A)")
        axes[0].set_title(f"{title} - Upstream A")
        axes[0].set_ylim(-10, N_E_A + 10)
        axes[0].legend()

        # Population B raster
        mask_E_B = spikes_B.ids < N_E_B
        axes[1].scatter(spikes_B.times[mask_E_B], spikes_B.ids[mask_E_B],
                       s=1, c=colors[0], alpha=0.5)
        axes[1].set_ylabel("Neuron ID (B)")
        axes[1].set_xlabel("Time (ms)")
        axes[1].set_title(f"{title} - Downstream B")
        axes[1].set_ylim(-5, N_E_B + 5)

        plt.tight_layout()

    save_both(save_path, plot_fn)


def plot_phase_transfer_curve(
    phase_lags_deg: np.ndarray,
    gains: np.ndarray,
    save_path: Path,
) -> None:
    """
    Plot transfer gain as function of phase lag.

    Args:
        phase_lags_deg: Phase lags in degrees (0°, 45°, 90°, 135°, 180°)
        gains: Transfer gain values
        save_path: Path to save figure
    """
    def plot_fn():
        fig, ax = plt.subplots(figsize=(10, 10))

        # Get color cycle
        prop_cycle = mpl.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        ax.plot(phase_lags_deg, gains, 'o-', lw=3, markersize=10, color=colors[0])
        ax.set_xlabel("Relative Phase Lag (degrees)")
        ax.set_ylabel("Transfer Gain (Δ-spikes in B)")
        ax.set_title("Phase-Dependent Transfer Gain (A → B)")
        ax.set_xticks(phase_lags_deg)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()

    save_both(save_path, plot_fn)

