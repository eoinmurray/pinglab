"""Render saved exp046 analysis; scientific-reference review remains deferred.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from experiments import theme

from experiments.exp110.custom.source_exp046_custom_figures import save_figure


def plot_equal_network_distribution(
    per_tau: dict, out_path: Path, *, axes=None, panel_labels=None
) -> None:
    """Plot equal-network means in one monochrome row."""
    theme.apply()
    taus_sorted = sorted(float(k.removeprefix("tau_")) for k in per_tau)
    standalone = axes is None
    if standalone:
        fig, grid = plt.subplots(
            1,
            len(taus_sorted),
            figsize=(6.9, 2.15),
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        axes = grid.ravel()
    axes = list(axes)
    labels_for_panels = (
        list(panel_labels)
        if panel_labels is not None
        else [chr(ord("A") + i) for i in range(len(taus_sorted))]
    )
    if len(axes) != len(taus_sorted) or len(labels_for_panels) != len(taus_sorted):
        raise ValueError("one axis and panel label are required per decay condition")
    fig = axes[0].figure
    labels = ["0", "1", "2", "≥3"]
    keys = ("frac_zero", "frac_one", "frac_two", "frac_three_plus")
    for i, tau in enumerate(taus_sorted):
        ax = axes[i]
        means = [per_tau[f"tau_{tau:g}"][key] for key in keys]
        ax.bar(labels, means, color="black", edgecolor="black", lw=0.5)
        ax.set_title(
            "$\\tau_{\\mathrm{GABA}}$\n" + f"{tau:g} ms",
            fontsize=theme.SIZE_TITLE,
            color="black",
            loc="center",
            pad=8,
        )
        ax.text(
            0.03,
            0.97,
            labels_for_panels[i],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=theme.SIZE_PANEL,
            fontweight="bold",
            color="black",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("black")
        ax.tick_params(colors="black", labelsize=theme.SIZE_LABEL)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        ax.grid(False)
    if standalone:
        fig.supxlabel(
            "Spikes per neuron–cycle", fontsize=theme.SIZE_LABEL, color="black"
        )
        fig.supylabel(
            "Equal-network\nmean fraction", fontsize=theme.SIZE_LABEL, color="black"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        plt.close(fig)
