"""House-style views of measured deadline trade-offs and empirical frontiers."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from experiments.helpers import theme


def conditions(report, rate, tau):
    return sorted(
        (
            r
            for r in report["rows"]
            if r["input_rate_hz"] == rate and r["tau_gaba_ms"] == tau
        ),
        key=lambda r: r["deadline_ms"],
    )


def plot_tradeoffs(report, destination, save):
    fig, axes = plt.subplots(
        3, 3, figsize=(180 / 25.4, 165 / 25.4), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.07)
    colours = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    for ri, rate in enumerate(report["design"]["input_rates_hz"]):
        for tau, colour in zip(report["design"]["tau_gaba_ms"], colours):
            rows = conditions(report, rate, tau)
            t = np.array([r["deadline_ms"] for r in rows])
            a = np.array([r["accuracy"] for r in rows]) * 100
            c = np.array([r["mean_network_spikes"] for r in rows]) / 1000
            low, high = np.array([r["accuracy_ci95"] for r in rows]).T * 100
            for ax, x, y in zip(axes[ri], (t, t, c), (a, c, a)):
                ax.plot(
                    x,
                    y,
                    color=colour,
                    lw=1.3,
                    marker="o",
                    markersize=3,
                    label=f"GABA {tau:g} ms",
                )
            axes[ri, 0].fill_between(t, low, high, color=colour, alpha=0.12)
        for j, (xlabel, ylabel, title) in enumerate(
            (
                ("Deadline (ms)", "Accuracy (%)", "Accuracy"),
                ("Deadline (ms)", "Network spikes (×10³)", "Spike cost"),
                ("Network spikes (×10³)", "Accuracy (%)", "Accuracy / cost"),
            )
        ):
            ax = axes[ri, j]
            ax.set_title(f"{rate:g} Hz · {title}", fontsize=theme.SIZE_ANNOTATION)
            ax.set_xlabel(xlabel, fontsize=theme.SIZE_LABEL)
            ax.set_ylabel(ylabel, fontsize=theme.SIZE_LABEL)
            if j != 1:
                ax.set_ylim(0, 100)
                ax.set_yticks([0, 25, 50, 75, 100])
            else:
                ax.set_ylim(bottom=0)
            if j < 2:
                ax.set_xlim(0, 205)
                ax.set_xticks([0, 100, 200])
            else:
                ax.set_xlim(left=0)
    theme.label_panels(axes.flat)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncols=3, fontsize=theme.SIZE_LEGEND
    )
    save(fig, destination)


def plot_frontiers(report, destination, save):
    fig, axes = plt.subplots(
        2, 3, figsize=(180 / 25.4, 110 / 25.4), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.07)
    colours = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    maximum = max(r["mean_network_spikes"] for r in report["rows"])
    area_scale = 100 / maximum
    for ri, rate in enumerate(report["design"]["input_rates_hz"]):
        ax = axes[0, ri]
        for tau, colour in zip(report["design"]["tau_gaba_ms"], colours):
            rows = conditions(report, rate, tau)
            t = np.array([r["deadline_ms"] for r in rows])
            a = np.array([r["accuracy"] for r in rows]) * 100
            area = np.array([r["mean_network_spikes"] for r in rows]) * area_scale
            front = np.array([r["nondominated"] for r in rows])
            ax.plot(t, a, color=colour, lw=0.9, alpha=0.65, label=f"GABA {tau:g} ms")
            ax.scatter(
                t[~front],
                a[~front],
                s=area[~front],
                color=colour,
                alpha=0.18,
                edgecolors="none",
                zorder=3,
            )
            ax.scatter(
                t[front],
                a[front],
                s=area[front],
                color=colour,
                edgecolors=theme.INK_BLACK,
                linewidths=0.5,
                zorder=4,
            )
        ax.set_title(f"Encoding {rate:g} Hz", fontsize=theme.SIZE_TITLE)
        ax.set(xlim=(0, 220), ylim=(0, 100), xlabel="Deadline (ms)")
        ax.set_xticks([0, 100, 200])
        ax.set_yticks([0, 25, 50, 75, 100])
        frequency = [r for r in report["frequencies"] if r["input_rate_hz"] == rate]
        axes[1, ri].plot(
            [r["tau_gaba_ms"] for r in frequency],
            [r["peak_of_mean_psd_hz"] for r in frequency],
            color=theme.INK_BLACK,
            marker="o",
            markersize=3,
            lw=1.3,
        )
        axes[1, ri].set_title("E spectral peak", fontsize=theme.SIZE_TITLE)
        axes[1, ri].set(xlabel="GABA decay (ms)", ylim=(0, 155), xlim=(2, 13))
        axes[1, ri].set_xticks([3, 6, 12])
        axes[1, ri].set_yticks([0, 50, 100, 150])
        if ri == 0:
            ax.set_ylabel("Accuracy (%)")
            axes[1, ri].set_ylabel("PSD peak (Hz)")
        else:
            ax.tick_params(labelleft=False)
            axes[1, ri].tick_params(labelleft=False)
    theme.label_panels(axes.flat)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncols=3, fontsize=theme.SIZE_LEGEND
    )
    costs = [5000, 15000, 30000]
    size_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            color=theme.INK_BLACK,
            markerfacecolor="none",
            markersize=np.sqrt(c * area_scale),
            label=f"{c:,} spikes",
        )
        for c in costs
    ]
    fig.legend(
        handles=size_handles,
        loc="outside lower center",
        ncols=3,
        fontsize=theme.SIZE_LEGEND,
    )
    save(fig, destination)
