from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, ticker

from pinglab.inputs import tonic
from pinglab.analysis import population_rate
from pinglab.lib.weights_builder import build_adjacency_matrices
from pinglab.plots.raster import save_raster
from pinglab.plots.styles import figsize, save_both
from pinglab.run import run_network, build_model_from_config
from pinglab.types import InstrumentsConfig, Spikes
from pinglab.utils import slice_spikes

from .model import LocalConfig


def run_experiment(config: LocalConfig, data_path: Path) -> None:
    data_path.mkdir(parents=True, exist_ok=True)

    run_cfg = config.base.model_copy(
        update={
            "instruments": InstrumentsConfig(
                variables=["V"],
                neuron_ids=list(range(config.base.N_E)),
            )
        }
    )

    external_input = tonic(
        N_E=run_cfg.N_E,
        N_I=run_cfg.N_I,
        I_E=config.default_inputs.I_E,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
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
    V = result.instruments.V
    times = result.instruments.times
    if V is None:
        raise ValueError("Voltage recording missing from instruments output.")
    if V.size == 0 or times.size == 0:
        raise ValueError("Voltage recording is empty.")

    def plot_voltage_map() -> None:
        plt.figure(figsize=figsize)
        extent = (float(times[0]), float(times[-1]), 0.0, float(run_cfg.N_E))
        norm = colors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=float(np.min(V)), vmax=float(np.max(V)))
        img = plt.imshow(
            V.T,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
            norm=norm,
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron ID (E)")
        plt.title("Voltage Map (E Population)")
        cbar = plt.colorbar(img, label="Membrane potential (mV)")
        vmin = float(np.min(V))
        vmax = float(np.max(V))
        candidate_ticks = [-100, -50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50, 100]
        ticks = [t for t in candidate_ticks if vmin <= t <= vmax]
        if ticks:
            cbar.set_ticks(ticks)
        else:
            cbar.locator = ticker.SymmetricalLogLocator(
                base=10,
                linthresh=1.0,
                subs=[1.0, 2.0, 5.0],
            )
            cbar.update_ticks()
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        plt.tight_layout()

    save_both(data_path / "voltage_map_E.png", plot_voltage_map)

    def plot_single_neuron() -> None:
        plt.figure(figsize=figsize)
        plt.plot(times, V[:, 0], linewidth=1.2)
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane potential (mV)")
        plt.title("Voltage Trace (E neuron 0)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    save_both(data_path / "voltage_trace_E0.png", plot_single_neuron)

    # Skip baseline raster; only plot sweep rasters for now.

    if config.weights is None:
        raise ValueError("weights must be provided in config for adjacency-only runs.")

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

    def plot_adjacency_matrix() -> None:
        plt.figure(figsize=figsize)
        plt.imshow(matrices.W, aspect="auto", origin="lower", cmap="Greys")
        plt.xlabel("Source neuron")
        plt.ylabel("Target neuron")
        plt.title("Adjacency Matrix (Weights)")
        plt.colorbar(label="Weight")
        plt.tight_layout()

    save_both(data_path / "adjacency_matrix.png", plot_adjacency_matrix)

    def _autocorr_rhythmicity(
        rate_hz: np.ndarray,
        dt_ms: float,
        tau_min_ms: float,
        tau_max_ms: float,
    ) -> float:
        if rate_hz.size == 0:
            return 0.0

        mean = float(np.mean(rate_hz))
        std = float(np.std(rate_hz))
        if std == 0.0:
            return 0.0

        x = (rate_hz - mean) / std
        n = x.size
        corr = np.correlate(x, x, mode="full")[n - 1 :]
        norm = np.arange(n, 0, -1, dtype=float)
        C = corr / norm

        lag_min = max(1, int(np.ceil(tau_min_ms / dt_ms)))
        lag_max = min(n - 1, int(np.floor(tau_max_ms / dt_ms)))
        if lag_max < lag_min:
            return 0.0

        window = C[lag_min : lag_max + 1]
        peak_idx = int(np.argmax(window))
        lag_idx = lag_min + peak_idx
        return float(C[lag_idx])

    def _shift_spikes(spikes: Spikes, offset_ms: float) -> Spikes:
        return Spikes(
            times=spikes.times - offset_ms,
            ids=spikes.ids,
            types=spikes.types,
            populations=spikes.populations,
        )

    mean_scales = np.linspace(
        config.heatmap.mean_scale.start,
        config.heatmap.mean_scale.stop,
        config.heatmap.mean_scale.num,
    )
    std_scales = np.linspace(
        config.heatmap.std_scale.start,
        config.heatmap.std_scale.stop,
        config.heatmap.std_scale.num,
    )

    baseline_means = {
        "gee": config.weights.mean_ee,
        "gei": config.weights.mean_ei,
        "gie": config.weights.mean_ie,
        "gii": config.weights.mean_ii,
    }
    baseline_stds = {
        "gee": config.weights.std_ee,
        "gei": config.weights.std_ei,
        "gie": config.weights.std_ie,
        "gii": config.weights.std_ii,
    }

    num_steps = int(np.ceil(run_cfg.T / run_cfg.dt))
    base_external_input = tonic(
        N_E=run_cfg.N_E,
        N_I=run_cfg.N_I,
        I_E=config.default_inputs.I_E,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
        num_steps=num_steps,
        seed=run_cfg.seed if run_cfg.seed is not None else 0,
    )

    sweeps = [
        ("g_ei", config.gei_sweep.g_ei),
        ("g_ie", config.gie_sweep.g_ie),
        ("g_ee", config.gee_sweep.g_ee),
        ("g_ii", config.gii_sweep.g_ii),
    ]

    for sweep_name, sweep_cfg in sweeps:
        values = np.linspace(sweep_cfg.start, sweep_cfg.stop, sweep_cfg.num)
        print(f"[exp.7] {sweep_name} sweep {values[0]:.2f}..{values[-1]:.2f} ({len(values)})")
        for idx, value in enumerate(values):
            means = baseline_means.copy()
            means[sweep_name.replace("g_", "g")] = float(value)
            matrices = build_adjacency_matrices(
                N_E=run_cfg.N_E,
                N_I=run_cfg.N_I,
                mean_ee=means["gee"],
                mean_ei=means["gei"],
                mean_ie=means["gie"],
                mean_ii=means["gii"],
                std_ee=baseline_stds["gee"],
                std_ei=baseline_stds["gei"],
                std_ie=baseline_stds["gie"],
                std_ii=baseline_stds["gii"],
                p_ee=config.weights.p_ee,
                p_ei=config.weights.p_ei,
                p_ie=config.weights.p_ie,
                p_ii=config.weights.p_ii,
                clamp_min=config.weights.clamp_min,
                seed=run_cfg.seed,
            )
            model = build_model_from_config(run_cfg)
            result = run_network(
                run_cfg,
                external_input=base_external_input,
                model=model,
                weights=matrices.W,
            )
            raster_label = f"{sweep_name}={value:.2f} ({idx + 1}/{len(values)})"
            save_raster(
                slice_spikes(
                    result.spikes,
                    config.plotting.raster.start_time,
                    config.plotting.raster.stop_time,
                ),
                data_path / f"raster_{sweep_name}_{idx:03d}_val_{value:.2f}.png",
                label=raster_label,
                xlim=(config.plotting.raster.start_time, config.plotting.raster.stop_time),
                ylim=(0.0, float(run_cfg.N_E + run_cfg.N_I)),
            )
            print(f"[exp.7] {sweep_name} {value:.2f} ({idx + 1}/{len(values)}) done")

    # TODO: re-enable full heatmap sweep once g_ei scans look right.
    return

    analysis_start = min(config.heatmap.burn_in_ms, max(0.0, run_cfg.T - run_cfg.dt))
    analysis_stop = run_cfg.T
    analysis_T = analysis_stop - analysis_start

    for weight_name in ("gee", "gei", "gie", "gii"):
        print(f"[exp.7] heatmap {weight_name}: {len(mean_scales)}x{len(std_scales)}")
        heatmap = np.zeros((len(mean_scales), len(std_scales)), dtype=float)
        for i, mean_scale in enumerate(mean_scales):
            print(
                f"[exp.7] {weight_name} mean_scale {mean_scale:.2f} "
                f"({i + 1}/{len(mean_scales)})"
            )
            for j, std_scale in enumerate(std_scales):
                means = baseline_means.copy()
                stds = baseline_stds.copy()
                means[weight_name] = baseline_means[weight_name] * float(mean_scale)
                stds[weight_name] = baseline_means[weight_name] * float(std_scale)

                matrices = build_adjacency_matrices(
                    N_E=run_cfg.N_E,
                    N_I=run_cfg.N_I,
                    mean_ee=means["gee"],
                    mean_ei=means["gei"],
                    mean_ie=means["gie"],
                    mean_ii=means["gii"],
                    std_ee=stds["gee"],
                    std_ei=stds["gei"],
                    std_ie=stds["gie"],
                    std_ii=stds["gii"],
                    p_ee=config.weights.p_ee,
                    p_ei=config.weights.p_ei,
                    p_ie=config.weights.p_ie,
                    p_ii=config.weights.p_ii,
                    clamp_min=config.weights.clamp_min,
                    seed=run_cfg.seed,
                )

                model = build_model_from_config(run_cfg)
                result = run_network(
                    run_cfg,
                    external_input=base_external_input,
                    model=model,
                    weights=matrices.W,
                )

                sliced = slice_spikes(result.spikes, analysis_start, analysis_stop)
                shifted = _shift_spikes(sliced, analysis_start)
                t_ms, rate_hz = population_rate(
                    shifted,
                    T_ms=analysis_T,
                    dt_ms=config.heatmap.bin_ms,
                    pop="E",
                    N_E=run_cfg.N_E,
                    N_I=run_cfg.N_I,
                )
                rho = _autocorr_rhythmicity(
                    rate_hz,
                    dt_ms=config.heatmap.bin_ms,
                    tau_min_ms=config.heatmap.tau_min_ms,
                    tau_max_ms=config.heatmap.tau_max_ms,
                )
                heatmap[i, j] = rho
                print(
                    f"[exp.7] {weight_name} std_scale {std_scale:.2f} "
                    f"({j + 1}/{len(std_scales)}) rho={rho:.3f}"
                )

                raster_label = (
                    f"{weight_name} mean={means[weight_name]:.3f} std={stds[weight_name]:.3f}"
                )
                save_raster(
                    slice_spikes(
                        result.spikes,
                        config.plotting.raster.start_time,
                        config.plotting.raster.stop_time,
                    ),
                    data_path / f"raster_{weight_name}_m{i:02d}_s{j:02d}.png",
                    label=raster_label,
                    xlim=(config.plotting.raster.start_time, config.plotting.raster.stop_time),
                    ylim=(0.0, float(run_cfg.N_E + run_cfg.N_I)),
                )

        def plot_heatmap() -> None:
            plt.figure(figsize=figsize)
            extent = (
                float(std_scales[0] * baseline_means[weight_name]),
                float(std_scales[-1] * baseline_means[weight_name]),
                float(mean_scales[0] * baseline_means[weight_name]),
                float(mean_scales[-1] * baseline_means[weight_name]),
            )
            plt.imshow(
                heatmap,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="Greys",
            )
            plt.xlabel("Weight std")
            plt.ylabel("Weight mean")
            plt.title(f"Rhythmicity heatmap ({weight_name})")
            plt.colorbar(label="Rhythmicity")
            plt.tight_layout()

        save_both(data_path / f"rhythmicity_heatmap_{weight_name}.png", plot_heatmap)
