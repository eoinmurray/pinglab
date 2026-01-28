from __future__ import annotations

from pathlib import Path

from pinglab.inputs.tonic import tonic
from pinglab.lib.weights_builder import build_adjacency_matrices
from pinglab.plots.raster import save_raster
from pinglab.run import build_model_from_config, run_network
from pinglab.utils import slice_spikes

from .model import LocalConfig
from .plots import plot_weight_histograms


def run_experiment(
    config: LocalConfig,
    data_path: Path,
    *,
    config_name: str = "config-0",
) -> None:
    run_cfg = config.base
    plot_cfg = config.plotting.raster if config.plotting else None

    num_steps = int(run_cfg.T / run_cfg.dt)
    matrices = build_adjacency_matrices(
        N_E=run_cfg.N_E,
        N_I=run_cfg.N_I,
        ee=config.weights.ee,
        ei=config.weights.ei,
        ie=config.weights.ie,
        ii=config.weights.ii,
        clamp_min=config.weights.clamp_min,
        seed=int(run_cfg.seed or 0),
    )

    external_input = tonic(
        N_E=run_cfg.N_E,
        N_I=run_cfg.N_I,
        I_E=float(config.default_inputs.I_E),
        I_I=float(config.default_inputs.I_I),
        noise_std=float(config.default_inputs.noise),
        num_steps=num_steps,
        seed=int(run_cfg.seed or 0),
    )
    neuron_model = build_model_from_config(run_cfg)
    result = run_network(
        run_cfg,
        external_input=external_input,
        model=neuron_model,
        weights=matrices.W,
    )

    if plot_cfg:
        raster_start = min(plot_cfg.start_time, run_cfg.T)
        raster_stop = min(plot_cfg.stop_time, run_cfg.T)
    else:
        raster_start = 0.0
        raster_stop = run_cfg.T

    raster_slice = slice_spikes(result.spikes, raster_start, raster_stop)
    save_raster(
        raster_slice,
        data_path / f"raster_{config_name}.png",
        label=f"{config_name}",
        xlim=(raster_start, raster_stop),
        ylim=(0.0, float(run_cfg.N_E + run_cfg.N_I)),
    )
    plot_weight_histograms(
        matrices.W_ee,
        matrices.W_ei,
        matrices.W_ie,
        matrices.W_ii,
        data_path / f"weights_hist_blocks_{config_name}.png",
        title="",
    )
