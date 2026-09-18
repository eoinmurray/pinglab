"""Compose manuscript-owned figures from validated upstream measurements."""

import matplotlib.pyplot as plt
from experiments.exp054 import plots as exp054_plots
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure
from matplotlib.gridspec import GridSpec


def _label_panel(axis, label, *, y=1.04):
    return theme.label_panel(axis, label, x=-0.08, y=y)


def build_onset_super_compound(grid, branch, reference, mf, meas, out_path):
    """Combine the exp054 coupling map with its retained onset comparison."""
    previous_paper_mode = theme.PAPER_MODE
    theme.set_paper_mode(True)
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    fig = plt.figure(figsize=(180 / 25.4, 6.4), dpi=150)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[1.25, 0.92, 1.05],
        hspace=0.65,
        wspace=0.60,
        top=0.95,
        bottom=0.06,
        left=0.07,
        right=0.90,
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
        for text in list(axis.texts):
            if text.get_text() not in raster_letters:
                text.remove()
        axis.set_title("", loc="center")
        axis.set_title(("E rate (Hz)", "I rate (Hz; clipped)", "Lobe–trough contrast")[index], fontsize=theme.SIZE_LABEL)
        axis.set_xlabel(r"$W_{EI}$ (µS)")
        if index == 0:
            axis.set_ylabel(r"$W_{IE}$ (µS)")
        color_axis = axis.inset_axes((1.04, 0, 0.045, 1))
        colorbar = fig.colorbar(axis.images[0], cax=color_axis)
        colorbar.ax.tick_params(labelsize=theme.SIZE_TICK)
        colorbar.locator = plt.MaxNLocator(3)
        colorbar.update_ticks()
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
        axis.set_title(("Loop off", "Intermediate coupling", "Strong coupling")[index], fontsize=theme.SIZE_LABEL)
        cell = grid[wie_index][wei_index]
        n_e = cell["e"].shape[1]
        n_i = 0 if cell["i"] is None else cell["i"].shape[1]
        axis.set_yticks((n_e / 2, n_e + n_i / 2), ("E", "I"))
        _label_panel(axis, raster_letters[index])

    eigen_axis = fig.add_subplot(gs[2, 0])
    onset = reference["onset"]
    eigen_axis.plot(
        branch["drive_nA"],
        branch["leading_real_per_ms"],
        color=theme.INK_BLACK,
        lw=1.2,
    )
    eigen_axis.axhline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    eigen_axis.axvline(onset["drive_nA"], color=theme.AMBER, lw=0.6, ls=":")
    eigen_axis.scatter(
        [onset["drive_nA"]],
        [0],
        facecolors="none",
        edgecolors=theme.ELECTRIC_CYAN,
        s=45,
        lw=1.2,
        zorder=5,
    )
    eigen_axis.set_xlabel("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_LABEL)
    eigen_axis.set_ylabel("leading Re$(\\lambda)$ (ms$^{-1}$)", fontsize=theme.SIZE_LABEL)
    eigen_axis.set_title(
        "Hopf crossing",
        loc="left",
        fontsize=theme.SIZE_LABEL,
        fontweight="semibold",
    )
    _label_panel(eigen_axis, "G")
    exp054_plots._despine(eigen_axis)

    amplitude_axis = fig.add_subplot(gs[2, 1])
    amplitude_axis.plot(
        branch["ramp_drive_nA"],
        branch["ramp_up_amplitude_per_ms"],
        "o-",
        color=theme.INK_BLACK,
        lw=1.2,
        ms=4,
        label="drive ↑",
    )
    amplitude_axis.plot(
        branch["ramp_drive_nA"],
        branch["ramp_down_amplitude_per_ms"],
        "s--",
        color=theme.DEEP_RED,
        lw=1.0,
        ms=4,
        markerfacecolor="none",
        label="drive ↓",
    )
    amplitude_axis.axvline(onset["drive_nA"], color=theme.AMBER, lw=0.6, ls=":")
    amplitude_axis.set_xlabel("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_LABEL)
    amplitude_axis.set_ylabel("E amplitude (ms$^{-1}$)", fontsize=theme.SIZE_LABEL)
    amplitude_axis.set_title(
        "Mean-field amplitude",
        loc="left",
        fontsize=theme.SIZE_LABEL,
        fontweight="semibold",
    )
    amplitude_axis.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    _label_panel(amplitude_axis, "H")
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
        label="mean-field",
    )
    if meas:
        measured_decay = sorted(meas)
        frequency_axis.plot(
            measured_decay,
            [meas[value] for value in measured_decay],
            "s--",
            color=theme.DEEP_RED,
            lw=1.3,
            label="spiking median",
        )
    frequency_axis.set_xlabel("$\\tau_\\text{GABA}$ (ms)", fontsize=theme.SIZE_LABEL)
    frequency_axis.set_ylabel("frequency (Hz)", fontsize=theme.SIZE_LABEL)
    frequency_axis.set_title(
        "Frequency comparison",
        loc="left",
        fontsize=theme.SIZE_LABEL,
        fontweight="semibold",
    )
    frequency_axis.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    _label_panel(frequency_axis, "I")
    exp054_plots._despine(frequency_axis)

    save_figure(fig, out_path, formats=("png", "pdf"))
    plt.close(fig)
    theme.set_paper_mode(previous_paper_mode)
    theme.apply()
