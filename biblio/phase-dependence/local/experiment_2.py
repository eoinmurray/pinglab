
from __future__ import annotations

from matplotlib import pyplot as plt
import matplotlib as mpl
from pathlib import Path
import numpy as np

from pinglab.plots.styles import save_both
from pinglab.plots import save_raster, save_instrument_traces
from pinglab.inputs import tonic, add_pulse_to_input
from pinglab.utils import slice_spikes
from pinglab.multiprocessing import parallel
from pinglab import run_network

from local.model import LocalConfig


def compute_precision_metrics_for_trial(
    spikes,
    target_E: np.ndarray,
    pulse_t_ms: float,
    pre_window_ms: float,
    post_window_ms: float,
    dt_ms: float,
) -> tuple[float, int, np.ndarray]:
    """
    Compute Δ-spikes, spike count in post window, and first-spike latencies
    for a single trial.

    Works with Spikes objects (times, ids arrays).
    """
    times = spikes.times
    ids = spikes.ids

    # Mask for target E neurons
    target_mask = np.isin(ids, target_E)
    target_times = times[target_mask]
    target_ids = ids[target_mask]

    # Pre and post windows
    pre_start = pulse_t_ms - pre_window_ms
    pre_end = pulse_t_ms
    post_start = pulse_t_ms
    post_end = pulse_t_ms + post_window_ms

    # Count spikes in pre and post windows
    pre_mask = (target_times >= pre_start) & (target_times < pre_end)
    post_mask = (target_times >= post_start) & (target_times < post_end)

    pre_counts = int(pre_mask.sum())
    post_counts = int(post_mask.sum())

    rate_pre  = pre_counts  / pre_window_ms
    rate_post = post_counts / post_window_ms
    delta_spikes = rate_post - rate_pre
    spike_count_post = post_counts

    # First-spike latencies in post window, per neuron
    post_times = target_times[post_mask]
    post_ids = target_ids[post_mask]

    latencies: list[float] = []
    for neuron_id in target_E:
        neuron_mask = post_ids == neuron_id
        if neuron_mask.any():
            first_spike_time = post_times[neuron_mask].min()
            latency = first_spike_time - pulse_t_ms
            latencies.append(latency)

    return delta_spikes, spike_count_post, np.asarray(latencies, dtype=float)


def inner(cfg: dict):
    """
    Single trial:
    - build noisy tonic input with a trial-specific seed
    - add a pulse to target_E at pulse_t
    - run the network
    - return spikes + indexing metadata
    """
    config: LocalConfig = cfg["config"]
    pulse_t = cfg["pulse_t"]
    phase_idx = cfg["phase_idx"]
    trial_idx = cfg["trial_idx"]
    target_E = cfg["target_E"]
    base_seed = cfg["base_seed"]

    num_steps = int(np.ceil(config.base.T / config.base.dt))

    # Trial-specific seed so repeats see different noise
    # (but still deterministic given base_seed)
    seed = int(base_seed + phase_idx * config.pulse.repeats + trial_idx)

    baseline_input = tonic(
        N_E=config.base.N_E,
        N_I=config.base.N_I,
        I_E=config.default_inputs.I_E,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
        num_steps=num_steps,
        seed=seed,
    )

    pulse_input = add_pulse_to_input(
        baseline_input,
        target_E,
        pulse_t,
        config.pulse.width_ms,
        config.pulse.amp,
        config.base.dt,
        num_steps=num_steps,
    )

    result = run_network(config.base, pulse_input)

    return {
        "spikes": result.spikes,
        "phase_idx": phase_idx,
        "trial_idx": trial_idx,
        "pulse_t": pulse_t,
        # If you want per-trial rasters later, you can also return pulse_input here,
        # but we keep this minimal for now.
    }


def experiment_2(config: LocalConfig, data_path: Path) -> None:
    num_steps = int(np.ceil(config.base.T / config.base.dt))
    base_seed = config.base.seed or 0

    # -------------------------------------------------------------------------
    # 1. Run a single baseline network for LFP proxy and sanity rasters
    # -------------------------------------------------------------------------
    baseline_input = tonic(
        N_E=config.base.N_E,
        N_I=config.base.N_I,
        I_E=config.default_inputs.I_E,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
        num_steps=num_steps,
        seed=base_seed,
    )

    print("Run baseline")
    baseline_results = run_network(config.base, baseline_input)

    if not baseline_results.instruments:
        raise RuntimeError("No baseline_results.instruments recorded in baseline")

    save_instrument_traces(baseline_results.instruments, data_path)
    save_raster(
        baseline_results.spikes,
        path=data_path / "raster_baseline.png",
        label="baseline",
        dt=config.base.dt,
        external_input=baseline_input,
    )

    rng = np.random.default_rng(base_seed)
    target_E = rng.choice(
        config.base.N_E,
        size=max(1, int(0.1 * config.base.N_E)),
        replace=False,
    )

    # LFP proxy (normalized inhibitory conductance)
    lfp_proxy = np.asarray(baseline_results.instruments.g_i_mean_E)
    time = np.arange(len(lfp_proxy)) * config.base.dt

    # -------------------------------------------------------------------------
    # 2. Build pulse conditions: phases × repeats
    # -------------------------------------------------------------------------
    pulses = np.linspace(
        config.pulse.linspace.start,
        config.pulse.linspace.stop,
        config.pulse.linspace.num,
    )
    n_phases = len(pulses)
    n_repeats = config.pulse.repeats

    cfgs: list[dict] = []
    for phase_idx, pulse_t in enumerate(pulses):
        for trial_idx in range(n_repeats):
            cfgs.append(
                {
                    "config": config,
                    "pulse_t": float(pulse_t),
                    "phase_idx": phase_idx,
                    "trial_idx": trial_idx,
                    "target_E": target_E,
                    "base_seed": base_seed,
                }
            )

    print(
        f"Running GPP-1 phase-precision experiment with "
        f"{n_phases} phases × {n_repeats} repeats = {len(cfgs)} trials"
    )

    results = parallel(inner, cfgs)

    # -------------------------------------------------------------------------
    # 3. Aggregate trial-level metrics: gain, Fano, jitter
    # -------------------------------------------------------------------------
    dt = config.base.dt
    pre_ms = config.pulse.pre_window_ms
    post_ms = config.pulse.post_window_ms

    delta_mat = np.zeros((n_phases, n_repeats), dtype=float)
    spike_mat = np.zeros((n_phases, n_repeats), dtype=float)
    latencies_by_phase: list[list[float]] = [[] for _ in range(n_phases)]

    # Optional: sanity rasters for a small subset of trials
    raster_every = max(1, len(results) // 10)

    for i, result in enumerate(results):
        spikes = result["spikes"]
        phase_idx = int(result["phase_idx"])
        trial_idx = int(result["trial_idx"])
        pulse_t = float(result["pulse_t"])

        # Precision metrics for this trial
        delta_spikes, spike_count_post, latencies = compute_precision_metrics_for_trial(
            spikes=spikes,
            target_E=target_E,
            pulse_t_ms=pulse_t,
            pre_window_ms=pre_ms,
            post_window_ms=post_ms,
            dt_ms=dt,
        )

        delta_mat[phase_idx, trial_idx] = delta_spikes
        spike_mat[phase_idx, trial_idx] = spike_count_post
        if latencies.size > 0:
            latencies_by_phase[phase_idx].extend(latencies.tolist())

        # Save a few example rasters for QC
        if i % raster_every == 0:
            sliced_spikes = slice_spikes(
                spikes,
                start_time=config.plotting.raster.start_time,
                stop_time=config.plotting.raster.stop_time,
            )
            save_raster(
                sliced_spikes,
                dt=dt,
                # external_input is optional here; using None keeps file light.
                external_input=None,
                path=data_path / f"raster_with_pulse_trial{i + 1}.png",
                label=f"phase={phase_idx}, trial={trial_idx}, pulse_t={pulse_t:.2f}",
            )

    # Phase-wise aggregates
    mean_delta = delta_mat.mean(axis=1)

    # Fano factor vs phase
    mean_spikes = spike_mat.mean(axis=1)
    var_spikes = spike_mat.var(axis=1, ddof=1)
    # Avoid division by zero
    fano = np.full_like(mean_spikes, np.nan)
    nonzero = mean_spikes > 0
    fano[nonzero] = var_spikes[nonzero] / mean_spikes[nonzero]

    # Latency jitter vs phase
    jitter = np.full(n_phases, np.nan)
    for phase_idx in range(n_phases):
        lats = np.asarray(latencies_by_phase[phase_idx], dtype=float)
        if lats.size > 1:
            jitter[phase_idx] = float(lats.std(ddof=1))

    # Save raw metrics for later analysis
    np.savez(
        data_path / "gpp1_metrics.npz",
        pulses=pulses,
        mean_delta=mean_delta,
        delta_mat=delta_mat,
        spike_mat=spike_mat,
        fano=fano,
        jitter=jitter,
        time=time,
        lfp_proxy=lfp_proxy,
    )

    # -------------------------------------------------------------------------
    # 4. Plot: gain, Fano, jitter vs phase, with LFP proxy
    # -------------------------------------------------------------------------
    def plot_fn():
        prop_cycle = mpl.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        fig, axes = plt.subplots(
            3, 1, figsize=(8, 10), sharex=True, constrained_layout=True
        )

        # Panel 1: Δ-spikes (gain) and LFP proxy
        ax1 = axes[0]
        ax1.set_xlim(float(np.min(pulses)), float(np.max(pulses)))
        ax1.plot(
            pulses,
            mean_delta,
            "o-",
            label="Δ-spikes (target E, mean over trials)",
            color=colors[0],
        )
        ax1.set_ylabel("Δ-spikes")
        ax1.set_title("Gamma phase–dependent precision (GPP-1)")

        ax1b = ax1.twinx()
        ax1b.plot(
            time,
            lfp_proxy,
            "--",
            alpha=0.6,
            label="g_i_mean_E (LFP proxy)",
            color=colors[1],
        )
        ax1b.set_ylabel("Inhibitory conductance (LFP proxy)")

        # Hacky shared legend for panel 1
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")

        # Panel 2: Fano factor vs phase
        ax2 = axes[1]
        ax2.plot(
            pulses,
            fano,
            "o-",
            color=colors[2],
        )
        ax2.set_ylabel("Fano factor\n(post-window spike count)")
        ax2.axhline(1.0, linestyle="--", alpha=0.3)
        ax2.set_title("Reliability (count variability) vs phase")

        # Panel 3: latency jitter vs phase
        ax3 = axes[2]
        ax3.plot(
            pulses,
            jitter,
            "o-",
            color=colors[3] if len(colors) > 3 else colors[0],
        )
        ax3.set_ylabel("Latency jitter (ms)")
        ax3.set_xlabel("Pulse time (ms)")
        ax3.set_title("Temporal precision (first-spike jitter) vs phase")

    plot_fn()
    save_both(
        data_path / "phase_precision_gpp1.png",
        plot_fn,
    )

    print(f"GPP-1 results saved to {data_path}")
