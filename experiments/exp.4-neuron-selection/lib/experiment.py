from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from pinglab.analysis import synchrony_index
from pinglab.inputs import tonic
from pinglab.plots.raster import save_raster
from pinglab.plots.styles import LIGHT, DARK, apply_style, save_both
from pinglab.run import run_network
from pinglab.types import InstrumentsConfig, Spikes
from pinglab.utils import slice_spikes

from .model import LocalConfig, ModelConfig


def _shift_spikes(spikes: Spikes, offset_ms: float) -> Spikes:
    return Spikes(
        times=spikes.times - offset_ms,
        ids=spikes.ids,
        types=spikes.types,
    )


def _rate_e(spikes: Spikes, start_ms: float, stop_ms: float, n_e: int) -> float:
    if stop_ms <= start_ms or n_e <= 0:
        return 0.0
    mask = (spikes.times >= start_ms) & (spikes.times < stop_ms) & (spikes.ids < n_e)
    count = int(np.sum(mask))
    duration_s = (stop_ms - start_ms) / 1000.0
    return count / (n_e * duration_s) if duration_s > 0 else 0.0


def _rhythmicity(
    spikes: Spikes,
    start_ms: float,
    stop_ms: float,
    bin_ms: float,
    n_e: int,
    n_i: int,
) -> float:
    if stop_ms <= start_ms:
        return 0.0
    sliced = slice_spikes(spikes, start_ms, stop_ms)
    shifted = _shift_spikes(sliced, start_ms)
    return synchrony_index(shifted, T=stop_ms - start_ms, bin_ms=bin_ms, N_E=n_e, N_I=n_i)


def _plot_voltage_trace(
    times: np.ndarray,
    voltage: np.ndarray,
    label: str,
    path: Path,
    style: dict,
) -> None:
    apply_style(style)
    plt.figure(figsize=(8, 8))
    plt.plot(times, voltage, linewidth=1.2)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title(label)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _run_voltage_gif(
    config: LocalConfig,
    model: ModelConfig,
    run_cfg,
    data_path: Path,
) -> None:
    frame_paths_light: list[Path] = []
    frame_paths_dark: list[Path] = []
    num_steps = int(np.ceil(run_cfg.T / run_cfg.dt))

    instruments = InstrumentsConfig(
        variables=["V"],
        neuron_ids=[0],
        downsample=1,
    )
    run_cfg = run_cfg.model_copy(update={"instruments": instruments})

    for idx, I_E in enumerate(config.sweep.gif_levels):
        external_input = tonic(
            N_E=run_cfg.N_E,
            N_I=run_cfg.N_I,
            I_E=float(I_E) * model.input_scale_E,
            I_I=config.default_inputs.I_I * model.input_scale_I,
            noise_std=config.default_inputs.noise,
            num_steps=num_steps,
            seed=run_cfg.seed or 0,
        )
        result = run_network(run_cfg, external_input=external_input)
        if result.instruments is None or result.instruments.V is None:
            raise RuntimeError("Missing instrumented voltage for GIF generation.")

        label = f"{model.name} voltage (I_E={I_E:.2f})"
        frame_light = data_path / f"voltage_{model.name}_frame_{idx:02d}_light.png"
        frame_dark = data_path / f"voltage_{model.name}_frame_{idx:02d}_dark.png"
        _plot_voltage_trace(
            result.instruments.times,
            result.instruments.V[:, 0],
            label,
            frame_light,
            LIGHT,
        )
        _plot_voltage_trace(
            result.instruments.times,
            result.instruments.V[:, 0],
            label,
            frame_dark,
            DARK,
        )
        frame_paths_light.append(frame_light)
        frame_paths_dark.append(frame_dark)

    gif_light = data_path / f"voltage_{model.name}_light.gif"
    gif_dark = data_path / f"voltage_{model.name}_dark.gif"
    frames_light = [imageio.imread(path) for path in frame_paths_light]
    frames_dark = [imageio.imread(path) for path in frame_paths_dark]
    imageio.mimsave(
        gif_light,
        frames_light,
        duration=config.sweep.gif_frame_duration_s,
        loop=0,
    )
    imageio.mimsave(
        gif_dark,
        frames_dark,
        duration=config.sweep.gif_frame_duration_s,
        loop=0,
    )

    for path in frame_paths_light + frame_paths_dark:
        path.unlink(missing_ok=True)


def _run_sweep(
    config: LocalConfig,
    model: ModelConfig,
    run_cfg,
    data_path: Path,
) -> list[tuple[float, Spikes, np.ndarray]]:
    num_steps = int(np.ceil(run_cfg.T / run_cfg.dt))
    values = np.linspace(
        config.sweep.I_E.start,
        config.sweep.I_E.stop,
        config.sweep.I_E.num,
    )
    results = []

    for I_E in values:
        external_input = tonic(
            N_E=run_cfg.N_E,
            N_I=run_cfg.N_I,
            I_E=float(I_E) * model.input_scale_E,
            I_I=config.default_inputs.I_I * model.input_scale_I,
            noise_std=config.default_inputs.noise,
            num_steps=num_steps,
            seed=run_cfg.seed or 0,
        )
        result = run_network(run_cfg, external_input=external_input)
        results.append((float(I_E), result.spikes, external_input))

    return results


def run_model_experiment(config: LocalConfig, model: ModelConfig, data_path: Path) -> None:
    model_path = data_path / model.name
    model_path.mkdir(parents=True, exist_ok=True)

    run_cfg = config.base.model_copy(update={"neuron_model": model.neuron_model, **model.overrides})
    sweep_results = _run_sweep(config, model, run_cfg, model_path)

    I_E_values = [value for value, _, _ in sweep_results]
    stop_ms = run_cfg.T
    burn_in_ms = min(config.sweep.burn_in_ms, max(0.0, stop_ms - run_cfg.dt))

    rates = [
        _rate_e(spikes, burn_in_ms, stop_ms, run_cfg.N_E)
        for _, spikes, _ in sweep_results
    ]
    rhythmicity = [
        _rhythmicity(spikes, burn_in_ms, stop_ms, config.sweep.bin_ms, run_cfg.N_E, run_cfg.N_I)
        for _, spikes, _ in sweep_results
    ]

    def plot_if_curve() -> None:
        plt.figure(figsize=(8, 8))
        plt.plot(I_E_values, rates, "o-", linewidth=1.5)
        plt.xlabel("Tonic Input $I_E$")
        plt.ylabel("E Population Firing Rate (Hz)")
        plt.title(f"IF curve ({model.name})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    save_both(model_path / f"if_curve_{model.name}.png", plot_if_curve)

    def plot_rhythmicity() -> None:
        plt.figure(figsize=(8, 8))
        plt.plot(I_E_values, rhythmicity, "o-", linewidth=1.5)
        plt.xlabel("Tonic Input $I_E$")
        plt.ylabel("Synchrony Index")
        plt.title(f"Rhythmicity vs Input ({model.name})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    save_both(model_path / f"rhythmicity_{model.name}.png", plot_rhythmicity)

    for I_E in config.sweep.raster_levels:
        idx = int(np.argmin(np.abs(np.array(I_E_values) - I_E)))
        value, spikes, _ = sweep_results[idx]
        raster_start = min(config.plotting.raster.start_time, stop_ms)
        raster_stop = min(config.plotting.raster.stop_time, stop_ms)
        if raster_start >= raster_stop:
            continue
        sliced = slice_spikes(spikes, start_time=raster_start, stop_time=raster_stop)
        save_raster(
            sliced,
            path=model_path / f"raster_{model.name}_I_E_{value:.2f}.png",
            label=f"I_E={value:.2f}",
        )

    _run_voltage_gif(config, model, run_cfg, model_path)
