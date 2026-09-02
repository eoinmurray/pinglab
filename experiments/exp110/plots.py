"""Compose manuscript-owned figures from validated upstream measurements."""

import matplotlib.pyplot as plt
import numpy as np
from experiments.exp054 import plots as exp054_plots
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure
from matplotlib.gridspec import GridSpec


def _label_panel(axis, label, *, y=1.04):
    return theme.label_panel(axis, label, x=-0.12, y=y)


def build_onset_super_compound(grid, results, hopf, sweep, mf, meas, out_path):
    """Combine the exp054 coupling map with its retained onset comparison."""
    previous_paper_mode = theme.PAPER_MODE
    theme.set_paper_mode(True)
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    fig = plt.figure(figsize=(6.9, 6.13), dpi=150)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[1.25, 0.92, 1.05],
        hspace=0.5,
        wspace=0.32,
        top=0.95,
        bottom=0.06,
        left=0.07,
        right=0.95,
    )

    raster_letters = ("D", "E", "F")
    for index, (values, title, vmax, fmt, marked) in enumerate(
        exp054_plots.turnon_map_panels(grid)
    ):
        axis = fig.add_subplot(gs[0, index])
        exp054_plots.draw_turnon_map(
            axis,
            values,
            title=title,
            vmax_color=vmax,
            fmt=fmt,
            mark=marked,
            show_y=index == 0,
            mark_labels=raster_letters if marked else None,
            cell_fontsize=3.0,
        )
        _label_panel(axis, "ABC"[index])

    for index, (label, wei_index, wie_index) in enumerate(exp054_plots.TURNON_POINTS):
        axis = fig.add_subplot(gs[1, index])
        exp054_plots.draw_turnon_raster(
            axis,
            grid[wie_index][wei_index],
            label=label,
            wei_i=wei_index,
            wie_i=wie_index,
            show_label=False,
        )
        _label_panel(axis, raster_letters[index])

    eigen_axis = fig.add_subplot(gs[2, 0])
    drives = np.array([row["I_ext"] for row in results])
    eigen_real = np.array([[value[0] for value in row["eigs"]] for row in results])
    eigen_imag = np.array([[value[1] for value in row["eigs"]] for row in results])
    scatter = None
    for index in range(eigen_real.shape[1]):
        scatter = eigen_axis.scatter(
            eigen_real[:, index],
            eigen_imag[:, index],
            c=drives,
            cmap="magma",
            s=4,
            linewidths=0,
        )
    eigen_axis.axvline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    if hopf:
        omega = hopf["omega_star"]
        eigen_axis.scatter(
            [0, 0],
            [omega, -omega],
            facecolors="none",
            edgecolors=theme.ELECTRIC_CYAN,
            s=60,
            lw=1.4,
            zorder=5,
        )
    assert scatter is not None
    color_axis = eigen_axis.inset_axes((0.06, 0.56, 0.035, 0.38))
    colorbar = fig.colorbar(scatter, cax=color_axis)
    colorbar.set_label("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_TICK - 2, labelpad=2)
    colorbar.ax.tick_params(labelsize=theme.SIZE_TICK - 2)
    colorbar.ax.yaxis.set_ticks_position("right")
    colorbar.ax.yaxis.set_label_position("right")
    eigen_axis.set_xlabel("Re$(\\lambda)$ (ms$^{-1}$)", fontsize=theme.SIZE_LABEL)
    eigen_axis.set_ylabel("Im$(\\lambda)$ (ms$^{-1}$)", fontsize=theme.SIZE_LABEL)
    eigen_axis.set_title(
        f"Hopf crossing at $I^\\star$ = {hopf['I_ext_star']:.2f} nA",
        loc="left",
        fontsize=theme.SIZE_LABEL,
        fontweight="semibold",
    )
    _label_panel(eigen_axis, "G", y=1.12)
    exp054_plots._despine(eigen_axis)

    amplitude_axis = fig.add_subplot(gs[2, 1])
    amplitude_axis.plot(
        [row["I_ext"] for row in sweep["up"]],
        [row["amp"] for row in sweep["up"]],
        "o-",
        color=theme.INK_BLACK,
        lw=1.2,
        ms=4,
        label="drive ↑",
    )
    amplitude_axis.plot(
        [row["I_ext"] for row in sweep["down"]],
        [row["amp"] for row in sweep["down"]],
        "s--",
        color=theme.DEEP_RED,
        lw=1.0,
        ms=4,
        markerfacecolor="none",
        label="drive ↓",
    )
    amplitude_axis.axvline(hopf["I_ext_star"], color=theme.AMBER, lw=0.6, ls=":")
    amplitude_axis.set_xlabel("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_LABEL)
    amplitude_axis.set_ylabel("E amplitude (ms$^{-1}$)", fontsize=theme.SIZE_LABEL)
    amplitude_axis.set_title(
        "Mean-field amplitude",
        loc="left",
        fontsize=theme.SIZE_LABEL,
        fontweight="semibold",
    )
    amplitude_axis.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    _label_panel(amplitude_axis, "H", y=1.12)
    exp054_plots._despine(amplitude_axis)

    frequency_axis = fig.add_subplot(gs[2, 2])
    decay = [row["tau_gaba_ms"] for row in mf if row["f_star_Hz"] is not None]
    frequency = [row["f_star_Hz"] for row in mf if row["f_star_Hz"] is not None]
    frequency_axis.plot(
        decay,
        frequency,
        "o-",
        color=theme.INK_BLACK,
        lw=1.4,
        label="mean-field $f^\\star$",
    )
    if meas:
        measured_decay = sorted(meas)
        frequency_axis.plot(
            measured_decay,
            [meas[value] for value in measured_decay],
            "s--",
            color=theme.DEEP_RED,
            lw=1.3,
            label="spiking median $f_\\gamma$",
        )
    frequency_axis.set_xlabel("$\\tau_\\text{GABA}$ (ms)", fontsize=theme.SIZE_LABEL)
    frequency_axis.set_ylabel("gamma frequency (Hz)", fontsize=theme.SIZE_LABEL)
    frequency_axis.set_title(
        "Frequency comparison",
        loc="left",
        fontsize=theme.SIZE_LABEL,
        fontweight="semibold",
    )
    frequency_axis.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    _label_panel(frequency_axis, "I", y=1.12)
    exp054_plots._despine(frequency_axis)

    save_figure(fig, out_path, formats=("png", "pdf"))
    plt.close(fig)
    theme.set_paper_mode(previous_paper_mode)
    theme.apply()
