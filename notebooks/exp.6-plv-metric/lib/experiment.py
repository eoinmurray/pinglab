from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from pinglab.analysis import plv_from_phase_series, plv_phase_series, population_plv
from pinglab.inputs import tonic
from pinglab.lib.weights_builder import build_adjacency_matrices
from pinglab.plots.raster import save_raster
from pinglab.plots.styles import save_both, figsize
from pinglab.run import build_model_from_config, run_network
from pinglab.utils import slice_spikes

from .model import LocalConfig


def _shift_spikes(spikes, offset_ms: float):
    return spikes.__class__(
        times=spikes.times - offset_ms,
        ids=spikes.ids,
        types=spikes.types,
        populations=spikes.populations,
    )


def run_experiment(config: LocalConfig, data_path: Path) -> None:
    sweep = config.sweep.I_E
    values = np.linspace(sweep.start, sweep.stop, sweep.num)

    run_cfg = config.base
    if config.weights is None:
        raise ValueError("weights must be provided for PLV experiment.")

    matrices = build_adjacency_matrices(
        N_E=run_cfg.N_E,
        N_I=run_cfg.N_I,
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
        seed=run_cfg.seed,
    )

    plv_cfg = config.plv
    burn_in_ms = float(plv_cfg.burn_in_ms)
    analysis_T = float(run_cfg.T) - burn_in_ms
    if analysis_T <= 0:
        raise ValueError("burn_in_ms must be < total simulation time.")

    plv_values = []
    debug_idx = int(values.size // 2)
    debug_I_E = float(values[debug_idx]) if values.size > 0 else None
    debug_series = None
    for I_E in values:
        external_input = tonic(
            N_E=int(run_cfg.N_E),
            N_I=int(run_cfg.N_I),
            I_E=float(I_E),
            I_I=float(config.default_inputs.I_I),
            noise_std=float(config.default_inputs.noise),
            num_steps=int(np.ceil(run_cfg.T / run_cfg.dt)),
            seed=run_cfg.seed if run_cfg.seed is not None else 0,
        )

        model = build_model_from_config(run_cfg)
        result = run_network(
            run_cfg,
            external_input=external_input,
            model=model,
            weights=matrices.W,
        )

        sliced = slice_spikes(result.spikes, burn_in_ms, float(run_cfg.T))
        shifted = _shift_spikes(sliced, burn_in_ms)
        plv = population_plv(
            spikes=shifted,
            T_ms=analysis_T,
            dt_ms=float(plv_cfg.bin_ms),
            fmin=float(plv_cfg.fmin),
            fmax=float(plv_cfg.fmax),
            pop="E",
            N_E=int(run_cfg.N_E),
            N_I=int(run_cfg.N_I),
        )
        plv_values.append(float(plv))

        save_raster(
            shifted,
            data_path / f"raster_I_E_{I_E:.2f}.png",
            label=f"I_E={I_E:.2f}",
            xlim=(0.0, analysis_T),
            ylim=(0.0, float(run_cfg.N_E + run_cfg.N_I)),
        )

        if debug_I_E is not None and np.isclose(I_E, debug_I_E):
            debug_series = plv_phase_series(
                spikes=shifted,
                T_ms=analysis_T,
                dt_ms=float(plv_cfg.bin_ms),
                fmin=float(plv_cfg.fmin),
                fmax=float(plv_cfg.fmax),
                pop="E",
                N_E=int(run_cfg.N_E),
                N_I=int(run_cfg.N_I),
            )

    summary = {
        "status": "ok",
        "I_E": values.tolist(),
        "plv": plv_values,
        "bin_ms": float(plv_cfg.bin_ms),
        "burn_in_ms": burn_in_ms,
        "fmin": float(plv_cfg.fmin),
        "fmax": float(plv_cfg.fmax),
        "debug_I_E": debug_I_E,
    }

    data_path.mkdir(parents=True, exist_ok=True)
    with (data_path / "plv_summary.yaml").open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    def plot_plv_curve():
        plt.figure(figsize=figsize)
        plt.plot(values, plv_values, marker="o")
        plt.xlabel("I_E")
        plt.ylabel("PLV")
        plt.ylim(0.0, 1.0)
        plt.title("PLV vs Input Current")
        plt.grid(True, alpha=0.3)

    save_both(data_path / "plv_vs_I_E.png", plot_plv_curve)

    def plot_plv_summary():
        plt.figure(figsize=figsize)
        plt.plot(values, plv_values, marker="o")
        plt.xlabel("I_E")
        plt.ylabel("PLV")
        plt.ylim(0.0, 1.0)
        plt.title("PLV Summary")
        plt.grid(True, alpha=0.3)

    save_both(data_path / "plv_summary.png", plot_plv_summary)

    if debug_series is not None:
        t_ms, rate_hz, filtered, phase, spike_times_ms = debug_series
        plv_dbg = plv_from_phase_series(t_ms, phase, spike_times_ms)
        label = f"I_E={debug_I_E:.2f}" if debug_I_E is not None else "debug"

        def plot_rate():
            plt.figure(figsize=figsize)
            plt.plot(t_ms, rate_hz)
            plt.xlabel("Time (ms)")
            plt.ylabel("Rate (Hz)")
            plt.title(f"Population rate ({label})")
            plt.grid(True, alpha=0.3)

        save_both(data_path / f"plv_steps_rate_I_E_{debug_I_E:.2f}.png", plot_rate)

        def plot_filtered():
            plt.figure(figsize=figsize)
            plt.plot(t_ms, filtered)
            plt.xlabel("Time (ms)")
            plt.ylabel("Filtered rate (a.u.)")
            plt.title(f"Bandpass filtered rate ({label})")
            plt.grid(True, alpha=0.3)

        save_both(
            data_path / f"plv_steps_filtered_I_E_{debug_I_E:.2f}.png",
            plot_filtered,
        )

        def plot_phase():
            plt.figure(figsize=figsize)
            plt.plot(t_ms, phase)
            plt.xlabel("Time (ms)")
            plt.ylabel("Phase (rad)")
            plt.title(f"Instantaneous phase ({label})")
            plt.grid(True, alpha=0.3)

        save_both(data_path / f"plv_steps_phase_I_E_{debug_I_E:.2f}.png", plot_phase)

        def plot_spike_phase_hist():
            plt.figure(figsize=figsize)
            if spike_times_ms.size > 0:
                spike_phase = np.interp(spike_times_ms, t_ms, phase)
                bins = np.linspace(-np.pi, np.pi, 24)
                plt.hist(spike_phase, bins=bins, density=True, alpha=0.7)
            plt.xlabel("Spike phase (rad)")
            plt.ylabel("Density")
            plt.title(f"Spike phases (PLV={plv_dbg:.2f}) ({label})")
            plt.grid(True, alpha=0.3)

        save_both(
            data_path / f"plv_steps_spike_phases_I_E_{debug_I_E:.2f}.png",
            plot_spike_phase_hist,
        )
