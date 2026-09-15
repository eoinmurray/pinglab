"""Publication-sized rendering from retained exp114 measurements."""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from tools.snnviz import DiagramGroup, render_diagram  # noqa: TID251


def _style():
    theme.apply()
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "figure.dpi": 180, "savefig.dpi": 300})


def network(bundle, cfg, path):
    from tools import snnlang as snn  # noqa: TID251

    diagram = snn.diagram(bundle, view="expanded")
    nodes = tuple(replace(node, title=node.id.replace("private_", "DRIVE ").upper()) for node in diagram.nodes)
    edges = tuple(replace(edge, label="", constraint=not edge.id.endswith("_cross")) for edge in diagram.edges)
    groups = (
        DiagramGroup("module1", f"PING MODULE 1 · {cfg['n_e_per_module']} E / {cfg['n_i_per_module']} I", ("E1", "I1"), same_rank=True),
        DiagramGroup("module2", f"PING MODULE 2 · {cfg['n_e_per_module']} E / {cfg['n_i_per_module']} I", ("E2", "I2"), same_rank=True),
    )
    render_diagram(replace(diagram, nodes=nodes, edges=edges, groups=groups, title="COUPLED TWO-MODULE PING CIRCUIT"), path, height_to_width_ratio=0.58)


def locking_map(aggregate, cfg, path):
    _style()
    detunings = np.array(cfg["signed_drive_differences_hz"])
    couplings = np.array(cfg["cross_e_weight_us"]) * 1000
    shape = (len(couplings), len(detunings))
    concentration = np.empty(shape)
    difference = np.empty(shape)
    locked = np.empty(shape, dtype=bool)
    for row in aggregate:
        idx = row["coupling_index"], row["detuning_index"]
        concentration[idx] = row["phase_concentration"]
        difference[idx] = row["absolute_peak_difference_hz"]
        locked[idx] = row["locked"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), constrained_layout=True)
    panels = ((concentration, "Phase concentration", "viridis", 0, 1), (difference, "|Peak-frequency difference| (Hz)", "magma", 0, max(2, difference.max())))
    for label, (ax, (values, title, cmap, vmin, vmax)) in zip("AB", zip(axes, panels)):
        image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(detunings)), [f"{x:+.2f}" for x in detunings])
        ax.set_yticks(range(len(couplings)), [f"{x:g}" for x in couplings])
        ax.set_xlabel("Drive 2 − drive 1 (Hz per afferent)")
        ax.set_ylabel("Cross-module event weight (nS)")
        ax.set_title(title)
        ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=10)
        for ci in range(len(couplings)):
            for di in range(len(detunings)):
                if locked[ci, di]:
                    ax.add_patch(plt.Rectangle((di - 0.45, ci - 0.45), 0.9, 0.9, fill=False, edgecolor="white", linewidth=1.4))
        fig.colorbar(image, ax=ax, shrink=0.82)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def phase_and_examples(aggregate, traces, cfg, path):
    _style()
    detunings = np.array(cfg["signed_drive_differences_hz"])
    couplings = np.array(cfg["cross_e_weight_us"]) * 1000
    fig = plt.figure(figsize=(7.1, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.15))
    ax_phase = fig.add_subplot(grid[0, :])
    for ci, coupling in enumerate(couplings):
        rows = sorted((row for row in aggregate if row["coupling_index"] == ci), key=lambda row: row["detuning_index"])
        values = [row["phase_offset_cycles"] if row["locked"] else np.nan for row in rows]
        ax_phase.plot(detunings, values, marker="o", label=f"{coupling:g} nS")
    ax_phase.axhline(0, color="0.65", linewidth=0.8)
    ax_phase.set(xlabel="Drive 2 − drive 1 (Hz per afferent)", ylabel="Phase 2 − phase 1 (cycles)", title="Signed relative phase")
    ax_phase.legend(title="Cross coupling", ncol=4, frameon=False)
    ax_phase.text(-0.07, 1.05, "A", transform=ax_phase.transAxes, fontweight="bold", fontsize=10)
    selected_detuning = len(detunings) - 1
    selected_seed = cfg["seeds"][0]
    for panel, ci, ax in zip("BC", (0, len(couplings) - 1), (fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]))):
        cid = f"d{selected_detuning}_c{ci}_s{selected_seed}"
        e1, e2 = traces[f"{cid}__e1"], traces[f"{cid}__e2"]
        time = np.arange(len(e1))
        smooth = np.ones(9) / 9
        ax.plot(time, np.convolve(e1, smooth, mode="same"), label="Module 1", color="#111111")
        ax.plot(time, np.convolve(e2, smooth, mode="same"), label="Module 2", color=theme.DEEP_RED, alpha=0.85)
        ax.set_xlim(250, 500)
        ax.set(xlabel="Time after burn-in (ms)", ylabel="E spikes / ms", title=f"{couplings[ci]:g} nS cross coupling")
        ax.text(-0.16, 1.05, panel, transform=ax.transAxes, fontweight="bold", fontsize=10)
        ax.legend(frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
