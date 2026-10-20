"""Render a fresh structural diagram and paired all-neuron rasters.

This experiment-local helper is called by the independent stages. It consumes
explicit inputs and resolved settings; importing it does not launch a run or
select scientific evidence.
"""

from importlib.metadata import version

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from snnlab.viz import Theme

from experiments.exp130 import recipe, dataset


def network_diagram(path, bundle):
    bundle.visualise(path, view="circuit")
    return dict(
        library="snnlab",
        version=version("snnlab"),
        view="circuit",
        construction="lang.Bundle.visualise",
        renderer="lang.Bundle.visualise",
    )


def render_raster(recordings, cfg, path):
    theme = Theme()
    theme.apply()
    fig, axes = plt.subplots(
        3, 2, figsize=(16, 8), dpi=180, gridspec_kw={"height_ratios": [1, 3, 1.5]}
    )
    origin = cfg["burn_in_ms"]
    start, stop = cfg["view_start_ms"] - origin, cfg["view_end_ms"] - origin
    t = np.arange(round(cfg["t_ms"] / cfg["dt_ms"])) * cfg["dt_ms"] - origin
    colours = dict(e=theme.ink, i=theme.accent)
    for column, condition in enumerate(recipe.CONDITIONS):
        drive, raster, counts_ax = axes[:, column]
        rates = dataset.source_rates(
            t + origin, recipe.condition_configuration(cfg, condition)
        )
        for pop, rate, linestyle in zip(("e", "i"), rates, ("-", "--")):
            drive.plot(
                t,
                rate,
                color=colours[pop],
                ls=linestyle,
                lw=1.5,
                label=f"{pop.upper()}-targeted",
            )
        title = (
            f"THETA · {cfg['theta_hz']:g} Hz"
            if condition == "theta"
            else f"CONSTANT · {cfg['baseline_e_hz']:g} Hz"
        )
        drive.set_title(f"{'A' if column == 0 else 'E'} · {title}", loc="left")
        drive.set(
            ylabel="Hz / afferent",
            ylim=(
                cfg["baseline_e_hz"] - cfg["modulation_hz"] - 0.07,
                cfg["baseline_e_hz"] + cfg["modulation_hz"] + 0.07,
            ),
        )
        drive.legend(
            loc="lower right",
            bbox_to_anchor=(1, 1.02),
            frameon=False,
            fontsize=8,
            ncol=2,
        )
        edges = np.arange(start, stop, 1.0)
        edges = np.append(edges, stop)
        for pop, offset in (("e", 0), ("i", cfg["n_e"])):
            step, cell = np.nonzero(recordings[condition][f"spk_{pop}"])
            times = t[step]
            selected = (times >= start) & (times < stop)
            raster.scatter(
                times[selected],
                cell[selected] + offset,
                marker="|",
                s=1,
                linewidths=0.35,
                color=colours[pop],
                rasterized=True,
            )
            selected = times[(times >= start) & (times < stop)]
            counts, _ = np.histogram(selected, bins=edges)
            if counts.sum() != len(selected):
                raise ValueError("histogram lost spikes")
            # Final partial bin is divided by its width, preserving comparable units.
            counts_ax.stairs(
                counts / np.diff(edges),
                edges,
                color=colours[pop],
                lw=1.0,
                label=pop.upper(),
            )
        for ax in (raster,):
            ax.axhline(cfg["n_e"] - 0.5, color=theme.muted, lw=0.5)
            ax.set(
                ylim=(-0.5, cfg["n_e"] + cfg["n_i"] - 0.5),
                ylabel="Neuron index",
                yticks=[0, 800, 1600, 2000],
            )
        raster.set_title(
            f"{'B' if column == 0 else 'F'} · ALL-NEURON RASTER", loc="left"
        )
        counts_ax.set_title(
            f"{'D' if column == 0 else 'H'} · POPULATION COUNTS", loc="left"
        )
        for ax in (drive, raster, counts_ax):
            ax.set_xlim(start, stop)
            ax.set_xticks(np.linspace(start, stop, 6))
            ax.set_xlabel("Time (ms)")
        counts_ax.set(xlabel="Time (ms)", ylabel="Spikes / ms", ylim=(0, None))
        counts_ax.legend(frameon=False, fontsize=8)
    for row in (0, 1):
        limits = [ax.get_ylim() for ax in axes[row]]
        for ax in axes[row]:
            ax.set_ylim(min(x[0] for x in limits), max(x[1] for x in limits))
    ceiling = max(ax.get_ylim()[1] for ax in axes[2])
    for ax in axes[2]:
        ax.set_ylim(0, ceiling)
    fig.subplots_adjust(
        left=0.065, right=0.98, top=0.94, bottom=0.09, hspace=0.65, wspace=0.24
    )
    fig.text(
        0.065,
        0.025,
        f"1,600 E (black) / 400 I (red) · seed {cfg['seed']} · identical circuit · private Poisson inputs · counts unsmoothed; E has four times as many cells",
        fontsize=9,
        color=theme.muted,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
