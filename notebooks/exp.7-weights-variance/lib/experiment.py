from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, ticker

from pinglab.inputs import tonic
from pinglab.plots.raster import save_raster
from pinglab.plots.styles import figsize, save_both
from pinglab.run import run_network
from pinglab.types import InstrumentsConfig
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

    result = run_network(run_cfg, external_input=external_input)
    if result.instruments is None or result.instruments.V is None:
        raise ValueError("Voltage recording missing from instruments output.")

    V = result.instruments.V
    times = result.instruments.times
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

    if config.plotting is not None:
        raster_start = config.plotting.raster.start_time
        raster_stop = config.plotting.raster.stop_time
        if raster_start < raster_stop:
            raster_slice = slice_spikes(result.spikes, raster_start, raster_stop)
            save_raster(
                raster_slice,
                data_path / "raster.png",
                xlim=(raster_start, raster_stop),
                ylim=(0.0, float(run_cfg.N_E + run_cfg.N_I)),
            )
