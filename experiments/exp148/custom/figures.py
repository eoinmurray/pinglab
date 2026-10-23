"""Draw the protocol raster and control schedule in Pinglab house style.

All displayed spike times, windows and drive gains come from one explicitly
selected compute run. This module creates figures without reading run storage.
"""

import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp148.recipe import drive_gain


def transition_figure(events, cfg):
    theme.apply()
    plt.rcParams["svg.fonttype"] = "none"
    fig, (raster, control) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1.15]},
        layout="constrained",
    )
    for p, color, offset in (
        ("E", theme.INK_BLACK, 0),
        ("I", theme.DEEP_RED, cfg["n_e"]),
    ):
        times, cells = events[p]
        selected = times >= cfg["burn_in_ms"]
        raster.scatter(
            times[selected] - cfg["burn_in_ms"],
            cells[selected] + offset,
            s=1,
            color=color,
            linewidths=0,
            rasterized=True,
            label=p,
        )
    raster.axhline(cfg["n_e"], color=theme.GREY_LIGHT, lw=0.7)
    raster.set(
        ylim=(0, cfg["n_e"] + cfg["n_i"]),
        ylabel="Neuron index",
        title="Reducing external drive to I",
    )

    raster.legend(loc="upper left", ncol=2, markerscale=3)
    schedule = cfg["control"]
    burn = cfg["burn_in_ms"]
    start = schedule["ramp_start_ms"] - burn
    end = schedule["ramp_end_ms"] - burn
    duration = cfg["duration_ms"] - burn
    for left, right, label in (
        (0, start, "BASELINE"),
        (start, end, "RAMP I DRIVE"),
        (end, duration, "HOLD: REDUCED I DRIVE"),
    ):
        raster.text(
            (left + right) / 2,
            0.985,
            label,
            transform=raster.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=theme.SIZE_ANNOTATION,
            fontweight="bold",
            bbox=dict(facecolor=theme.PAPER, edgecolor="none", pad=3),
        )
    times = np.r_[0, np.arange(start, end, schedule["update_ms"]), end, duration]
    gains = drive_gain(times + burn, cfg)
    control.step(
        times,
        100 * gains,
        where="post",
        color=theme.DEEP_RED,
        label="X→I",
    )
    control.plot(
        [0, duration],
        [100, 100],
        color=theme.INK_BLACK,
        ls="--",
        label="X→E",
    )
    control.set(
        ylim=(-5, 115),
        yticks=[0, 50, 100],
        ylabel="Input weight\n(% baseline)",
        xlabel="Time after burn-in (ms)",
        title="External input strength",
    )
    control.legend(loc="center right", ncol=2)
    for axis in (raster, control):
        axis.set_xlim(0, duration)
        axis.set_xticks(np.linspace(0, duration, 7))
        for boundary in (start, end):
            axis.axvline(boundary, color=theme.GREY_MID, ls="--", lw=1)
    theme.label_panels([raster, control])
    fig.supxlabel(
        f"{cfg['n_e']:,} E / {cfg['n_i']:,} I · {cfg['n_external']:,} shared Poisson inputs · seed {cfg['seed']}\n"
        f"{burn:g} ms burn-in excluded · {cfg['baseline_ms']:g} / {cfg['ramp_ms']:g} / {cfg['hold_ms']:g} ms baseline / ramp / hold · "
        f"X→I updated every {schedule['update_ms']:g} ms",
        fontsize=theme.SIZE_CAPTION,
    )
    return fig
