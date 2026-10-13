"""Inhibitory-decay overview using experiment-local measurements and rasters.

Consumes explicitly validated upstream measurements and spike recordings.
"""

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from experiments.exp123 import recipe


def plot_inhibitory_overview(raw, raster_compute, destination, save):
    sweep = "tau"
    cfg = raw["design"]
    conditions = [c for c in cfg["conditions"] if c["sweep"] == sweep]
    values = [c["axis_value"] for c in conditions]
    summaries = [r for r in raw["summaries"] if r["sweep"] == sweep]
    fig = plt.figure(figsize=(180 / 25.4, 105 / 25.4), layout="constrained")
    groups = fig.add_gridspec(1, 2)
    metric_grid = groups[0].subgridspec(2, 2)
    raster_grid = groups[1].subgridspec(3, 1)
    axes = np.array([fig.add_subplot(metric_grid[i, j]) for i in range(2) for j in range(2)])
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.06)
    specs = [
        ("psd_frequency_hz", "PSD frequency", "Spectral peak (Hz)"),
        ("participation_e_pct", "E participation", "Participation (%)"),
        ("e_spikes_per_second", "E spike rate", "E spikes/s"),
        ("rhythmicity_contrast", "Rhythmicity", "Lobe–trough contrast"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, specs):
        ax.plot(
            values,
            [r[key] for r in summaries],
            "o-",
            color=theme.INK_BLACK,
            linewidth=1.3,
            markersize=3,
            label="Peak of mean PSD" if key == "psd_frequency_hz" else "Encoding mean",
        )
        ax.axvline(6.0 if sweep == "tau" else 1.0, color=theme.FAINT, ls="--", lw=0.7)
        ax.set(
            xscale="log",
            xlabel={
                "tau": "GABA decay (ms)",
                "leak": "Leak multiplier",
                "cap": "Capacitance multiplier",
            }[sweep],
            ylabel=ylabel,
            title=title,
        )
        ax.set_xticks(
            [3, 6, 10, 20, 30] if sweep == "tau" else [0.5, 1, 2],
            labels=["3", "6", "10", "20", "30"]
            if sweep == "tau"
            else ["0.5", "1", "2"],
        )
        ax.minorticks_off()
        if key == "participation_e_pct":
            ax.set_ylim(0, 30)
        if key == "rhythmicity_contrast":
            ax.set_ylim(0, 1.05)
        if key == "e_spikes_per_second":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    theme.label_panels(axes.flat)
    axes = [fig.add_subplot(raster_grid[i, 0]) for i in range(3)]
    raster_cfg = raster_compute.record["execution"]["configuration"]["protocols"]["trajectories"]
    raster_unit = raster_compute.unit(recipe.recording_unit(0, recipe.RATE_HZ, "trajectories"))
    for ax, tau in zip(axes, recipe.TAUS_MS):
        with np.load(
            raster_unit / f"tau{tau:g}--spikes.npz",
            allow_pickle=False,
        ) as data:
            for key, colour, offset in [
                ("spk_e", theme.INK_BLACK, 0),
                ("spk_i", theme.DEEP_RED, 1024),
            ]:
                k, neuron = np.nonzero(data[key][:, 0])
                ax.scatter(
                    k * raster_cfg["dt_ms"],
                    neuron + offset,
                    s=0.22,
                    color=colour,
                    linewidths=0,
                    rasterized=True,
                )
        ax.axhline(1023.5, color=theme.FAINT, lw=0.5)
        ax.set(
            xlim=(0, 200),
            ylim=(-1, 1280),
            xlabel="Time (ms)",
            ylabel="Neuron",
            title=f"τ = {tau:g} ms",
        )
        ax.set_yticks([0, 1024, 1280])
        ax.set_xticks([0, 100, 200])
    theme.label_panels(axes, labels=["E", "F", "G"])
    save(fig, destination)

