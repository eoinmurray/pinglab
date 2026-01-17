
from matplotlib import pyplot as plt
import matplotlib as mpl
from pathlib import Path
import numpy as np

from pinglab.plots.styles import save_both
from pinglab.plots import save_raster, save_instrument_traces
from pinglab.inputs import tonic, add_pulse_to_input, compute_spike_delta
from pinglab.utils import slice_spikes
from pinglab.multiprocessing import parallel
from pinglab.lib.weights_builder import build_adjacency_matrices
from pinglab.run import run_network, build_model_from_config

from .model import LocalConfig


def hotloop(cfg: dict):
    config: LocalConfig = cfg["config"]
    baseline_input = cfg["baseline_input"]
    pulse_t = cfg["pulse_t"]
    target_E = cfg["target_E"]
    weights = cfg["weights"]

    pulse_input = add_pulse_to_input(# copies
        baseline_input,
        target_E,
        pulse_t,
        config.pulse.width_ms,
        config.pulse.amp,
        config.base.dt,
        num_steps=int(np.ceil(config.base.T / config.base.dt)),
    )

    model = build_model_from_config(config.base)
    result = run_network(config.base, pulse_input, model=model, weights=weights)
    
    return {
        "spikes": result.spikes,
        "input": pulse_input,
    }


def experiment_1(config: LocalConfig, data_path: Path) -> None:
    num_steps = int(np.ceil(config.base.T / config.base.dt))

    baseline_input = tonic(
        N_E=config.base.N_E,
        N_I=config.base.N_I,
        I_E=config.default_inputs.I_E,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
        num_steps=num_steps,
        seed=config.base.seed or 0,
    )

    if config.weights is None:
        raise RuntimeError("weights must be provided for adjacency-only runs.")
    weights = build_adjacency_matrices(
        N_E=config.base.N_E,
        N_I=config.base.N_I,
        mean_ee=config.weights.mean_ee,
        mean_ei=config.weights.mean_ei,
        mean_ie=config.weights.mean_ie,
        mean_ii=config.weights.mean_ii,
        std_ee=config.weights.std_ee,
        std_ei=config.weights.std_ei,
        std_ie=config.weights.std_ie,
        std_ii=config.weights.std_ii,
        p_ee=config.weights.p_ee,
        p_ei=config.weights.p_ei,
        p_ie=config.weights.p_ie,
        p_ii=config.weights.p_ii,
        clamp_min=config.weights.clamp_min,
        seed=config.base.seed,
    )

    print("Run baseline")
    baseline_model = build_model_from_config(config.base)
    baseline_results = run_network(
        config.base,
        baseline_input,
        model=baseline_model,
        weights=weights.W,
    )

    save_instrument_traces(baseline_results.instruments, data_path)
    save_raster(
        baseline_results.spikes,
        path=data_path / "raster_baseline.png",
        label="baseline",
        dt=config.base.dt,
        external_input=baseline_input,
    )
    
    rng = np.random.default_rng(config.base.seed or 0)
    target_E = rng.choice(config.base.N_E, size=max(1, int(0.1 * config.base.N_E)), replace=False)

    # # Pulse sweep across gamma phases
    pulses = np.linspace(
        config.pulse.linspace.start,
        config.pulse.linspace.stop,
        config.pulse.linspace.num,
    )
    deltas: list[float] = []

    cfgs = [{
        "config": config,
        "pulse_t": value,
        "baseline_input": baseline_input,
        "target_E": target_E,
        "weights": weights.W,
    } for value in pulses]

    results = parallel(hotloop, cfgs)

    for i, result in enumerate(results):
        pulsed_input = result["input"]
        spikes = result["spikes"]

        if i % max(1, len(results) // 10) == 0:
            spikes_to_plot = spikes
            if config.plotting is not None and config.plotting.raster is not None:
                spikes_to_plot = slice_spikes(
                    spikes,
                    start_time=config.plotting.raster.start_time,
                    stop_time=config.plotting.raster.stop_time,
                )

            save_raster(
                spikes_to_plot,
                dt=config.base.dt,
                external_input=pulsed_input[:, target_E],
                path=data_path / f"raster_with_pulse{i + 1}.png",
                label=f"pulse_t={cfgs[i]['pulse_t']:.2f}",
            )

        pulse_t = cfgs[i]["pulse_t"]
        delta = compute_spike_delta(
           spikes,
            target_E,
            pulse_t,
            config.pulse.pre_window_ms,
            config.pulse.post_window_ms,
        )

        deltas.append(delta)

    lfp_proxy = np.array(baseline_results.instruments.g_i_mean_E)
    time = np.arange(len(lfp_proxy)) * config.base.dt

    def plot_fn():
        prop_cycle = mpl.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        _, ax1 = plt.subplots(figsize=(8, 8))
        ax1.set_xlim((float(np.min(pulses)), float(np.max(pulses))))
        
        ax1.plot(pulses, deltas, "o-", label="Spike count delta", color=colors[0])
        ax1.set_ylabel("Δ-spikes (target E)")
        ax1.set_xlabel("Pulse time (ms)")

        ax2 = ax1.twinx()
        ax2.plot(time, lfp_proxy, "--", alpha=0.6, label="g_i_mean_E", color=colors[1])
        ax2.set_ylabel("Normalized inhibitory conductance (LFP proxy)")
        ax2.legend(loc="upper right")

        plt.title("Phase-dependent gain (Δ-spikes) in probed phase window")
        plt.tight_layout()

    plot_fn()
    save_both(
        data_path / "phase_gain_curve.png", 
        plot_fn
    )
