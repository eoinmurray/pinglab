
from matplotlib import pyplot as plt
import matplotlib as mpl
from pathlib import Path
import numpy as np
import shutil
import sys

from pinglab.plots.styles import save_both
from pinglab.plots import save_raster, save_instrument_traces
from pinglab.inputs import tonic
from pinglab.utils import load_config, slice_spikes
from pinglab.multiprocessing import parallel
from pinglab import run_network

# Add the experiment directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from local import add_pulse_to_input, compute_spike_delta


def inner(cfg: dict):
    config = cfg["config"]
    baseline_input = cfg["baseline_input"]
    pulse_t = cfg["pulse_t"]
    target_E = cfg["target_E"]

    pulse_input = add_pulse_to_input(# copies
        baseline_input,
        target_E,
        pulse_t,
        config.pulse.width_ms,
        config.pulse.amp,
        config.base.dt,
        num_steps=int(np.ceil(config.base.T / config.base.dt)),
    )

    result = run_network(config.base, pulse_input)
    
    return {
        "spikes": result.spikes,
        "input": pulse_input,
    }


def main() -> None:
    root = Path(__file__).parent
    data_path = root / "data"
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    config = load_config(root / "config.yaml")

    num_steps = int(np.ceil(config.base.T / config.base.dt))

    baseline_input = tonic(
        N_E=config.base.N_E,
        N_I=config.base.N_I,
        I_E=config.inputs.I_E,
        I_I=config.inputs.I_I,
        noise_std=config.inputs.noise,
        num_steps=num_steps,
        seed=config.base.seed or 0,
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
    } for value in pulses]

    results = parallel(inner, cfgs)

    for i, result in enumerate(results):
        pulsed_input = result["input"]
        spikes = result["spikes"]

        if i % max(1, len(results) // 10) == 0:
            sliced_spikes = slice_spikes(
                spikes,
                start_time=config.plotting.raster.start_time,
                stop_time=config.plotting.raster.stop_time,
            )

            save_raster(
                sliced_spikes,
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
        fig, ax1 = plt.subplots(figsize=(8, 8))
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

if __name__ == "__main__":
    main()
