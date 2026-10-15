"""House-style views of measured deadline trade-offs and empirical frontiers."""

import matplotlib.pyplot as plt
import numpy as np
from experiments.exp123 import recipe
from experiments.helpers import theme
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


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
    rate = recipe.RATE_HZ
    fig, ax = plt.subplots(
        figsize=(recipe.FIGURE_SIZE_MM[0] / 25.4, 120 / 25.4),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.07)
    colours = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    for tau, colour in zip(report["design"]["tau_gaba_ms"], colours):
        rows = conditions(report, rate, tau)
        a = np.array([r["accuracy"] for r in rows]) * 100
        c = np.array([r["mean_network_spikes"] for r in rows]) / 1000
        ax.plot(
            c, a, color=colour, lw=1.3, marker="o", markersize=4,
            label=f"GABA {tau:g} ms", zorder=3,
        )
    for deadline in report["design"]["deadlines_ms"]:
        rows = sorted(
            (r for r in report["rows"]
             if r["input_rate_hz"] == rate and r["deadline_ms"] == deadline),
            key=lambda r: r["mean_network_spikes"],
        )
        cost = np.array([r["mean_network_spikes"] for r in rows]) / 1000
        accuracy = np.array([r["accuracy"] for r in rows]) * 100
        ax.plot(cost, accuracy, color="0.55", ls="--", lw=0.8, zorder=2)
        ax.annotate(
            f"{deadline:g} ms", (cost[-1], accuracy[-1]),
            xytext=(6, 8), textcoords="offset points", va="bottom",
            fontsize=theme.SIZE_ANNOTATION, color="0.35",
        )
    ax.set_xlabel("Mean cumulative network spikes (×10³)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(0, max(r["mean_network_spikes"] for r in report["rows"]
                       if r["input_rate_hz"] == rate) / 1000 * 1.15)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncols=3, fontsize=theme.SIZE_LEGEND
    )
    save(fig, destination)


def plot_frontiers(report, destination, save):
    fig, axes = plt.subplots(
        2, 3,
        figsize=tuple(mm / 25.4 for mm in recipe.FIGURE_SIZE_MM),
        layout="constrained",
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
        ax.set(xlim=(0, 220), ylim=(0, 100), xlabel="Elapsed time (ms)")
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
