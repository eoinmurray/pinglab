"""Render matched TR02 (no rate penalty) rasters and single-trial E/I/total population spike-rate curves."""

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp173 import recipe
from matplotlib.font_manager import FontProperties
from pingstore.library import source_run, stage_run
from snnlab import lang, viz

FONT_SCALE = 1.75


def circuit(bundle, path):
    native = lang.diagram(bundle, view="circuit")
    symbols = {"drive": "X", "E": "E", "I": "I", "readout": "R"}
    diagram = replace(
        native,
        title="Frozen TR02 (no rate penalty) PING",
        groups=(),
        nodes=tuple(
            replace(node, detail=node.detail.replace("1 neurons", "1 neuron"))
            for node in native.nodes
            if node.id in symbols
        ),
        edges=tuple(
            edge
            for edge in native.edges
            if edge.source in symbols and edge.target in symbols
        ),
    )
    positions = {"drive": (-380, 0), "E": (0, 0), "I": (0, -320), "readout": (380, 0)}
    ports = {
        "upper_left": (0.3, 0),
        "upper_right": (0.7, 0),
        "lower_left": (0.3, 1),
        "lower_right": (0.7, 1),
    }
    routes = {
        "input_to_E": viz.EdgeLayout(source_port="right", target_port="left"),
        "readout_projection": viz.EdgeLayout(source_port="right", target_port="left"),
        "E_to_I": viz.EdgeLayout(source_port="upper_left", target_port="lower_left"),
        "I_to_E": viz.EdgeLayout(source_port="lower_right", target_port="upper_right"),
    }
    layout = viz.DiagramLayout(
        nodes={
            key: viz.NodeLayout(
                position=position,
                label=f"${symbols[key]}$",
                min_size=(220, 150),
                ports=ports,
            )
            for key, position in positions.items()
        },
        edges={
            edge.id: replace(
                routes[edge.id],
                label=viz.DiagramLabel(
                    rf"$W_{{{symbols[edge.target]},{symbols[edge.source]}}}$"
                ),
            )
            for edge in diagram.edges
        },
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=16), gap=30
        ),
    )
    diagram_theme = viz.DiagramTheme(
        font_name=FontProperties(family="monospace").get_name(),
        background=theme.PAPER,
        ink=theme.INK_BLACK,
        muted=theme.MUTED,
        line=theme.RULE,
        neutral=theme.PAPER,
        signal=theme.INK_BLACK,
        inhibitory=theme.DEEP_RED,
        input_fill=theme.PAPER,
        input_border=theme.INK_BLACK,
        population_fill=theme.PAPER,
        population_border=theme.INK_BLACK,
        output=theme.PAPER,
        output_border=theme.INK_BLACK,
        title_size=theme.SIZE_PANEL * FONT_SCALE,
        label_size=theme.SIZE_TITLE * FONT_SCALE,
        secondary_size=theme.SIZE_LABEL * FONT_SCALE,
    )
    resolved = viz.resolve_diagram(diagram, layout=layout, theme=diagram_theme)
    for target in (path, path.with_suffix(".png")):
        viz.render_diagram(resolved, target)


def present(identity):
    started = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="analyse"
    )
    numbers = json.loads(source.file("numbers.json").read_text())
    cfg = numbers["configuration"]
    pin = source.record["inputs"]["compute"]
    compute = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        reference=pin,
        experiment=recipe.SLUG,
        stage="compute",
    )
    theme.set_paper_mode(True)
    theme.apply()
    plt.rcParams.update(
        {
            "font.size": theme.SIZE_BASE * FONT_SCALE,
            "axes.titlesize": theme.SIZE_TITLE * FONT_SCALE,
            "axes.labelsize": theme.SIZE_LABEL * FONT_SCALE,
            "xtick.labelsize": theme.SIZE_TICK * FONT_SCALE,
            "ytick.labelsize": theme.SIZE_TICK * FONT_SCALE,
            "legend.fontsize": theme.SIZE_LEGEND * FONT_SCALE,
        }
    )
    with stage_run(
        ROOT,
        recipe.SLUG,
        "present",
        inputs={"analysis": source},
        configuration={
            "style": "experiments.theme paper; enlarged article typography",
            "font_scale": FONT_SCALE,
            "pool_size_axis": "log2",
            "gain_axis": "log2",
            "diagram_style": "house monospace and palette from experiments.theme",
            "raster_probe_index": cfg["raster_probe_index"],
            "rate_measure": "population spikes per second, summed across neurons",
            "rate_series": ["E", "I", "Total"],
            "accuracy_measure": "percent correct on the manuscript fixed 1000-image official-test subset",
            "panels": [
                "population spike rates",
                "test accuracy",
                "connectivity gain multipliers",
            ],
        },
    ) as run:
        shutil.copyfile(source.file("numbers.json"), run.export / "numbers.json")
        rows = (len(cfg["pool_sizes"]) + 1) // 2
        fig, grid = plt.subplots(
            rows, 2, figsize=(12, 3.8 * rows), sharex=True, sharey=True, squeeze=False
        )
        axes = list(grid.flat)[: len(cfg["pool_sizes"])]
        for unused in list(grid.flat)[len(cfg["pool_sizes"]) :]:
            fig.delaxes(unused)
        for ax, n_i in zip(axes, cfg["pool_sizes"], strict=True):
            with np.load(
                compute.file(f"i{n_i}", "raster.npz"), allow_pickle=False
            ) as events:
                for p, color, offset in (
                    ("E", theme.INK_BLACK, 0),
                    ("I", theme.DEEP_RED, cfg["n_e"]),
                ):
                    ax.scatter(
                        events[p + "_time_ms"],
                        events[p + "_cell"] + offset,
                        s=1.2,
                        c=color,
                        linewidths=0,
                        rasterized=True,
                    )
            ax.axhline(cfg["n_e"] - 0.5, color=theme.MUTED, lw=0.5)
            ax.set(
                xlim=(0, cfg["duration_ms"]),
                ylim=(-0.5, cfg["n_e"] + cfg["original_n_i"] - 0.5),
                title=f"{n_i} I "
                + ("neuron" if n_i == 1 else "neurons")
                + (" (original)" if n_i == cfg["original_n_i"] else ""),
                xlabel="Time (ms)",
                ylabel="Neuron index (E below I)",
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", labelbottom=True)
            bundle_path = compute.restore_bundle(
                f"i{n_i}/network.bundle", run.scratch / f"i{n_i}.bundle"
            )
            circuit(lang.load_bundle(bundle_path), run.export / f"circuit-i{n_i}.svg")
        for letter in theme.label_panels(axes):
            letter.set_fontsize(theme.SIZE_PANEL * FONT_SCALE)
        fig.suptitle(
            "TR02 PING · no rate penalty",
            fontsize=theme.SIZE_TITLE * FONT_SCALE,
        )
        fig.tight_layout()
        for suffix in ("svg", "png"):
            fig.savefig(
                run.export / f"rasters.{suffix}", dpi=plt.rcParams["savefig.dpi"]
            )
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
        ax = axes[0]
        population_rates = {
            p: [
                row[p]["total_spikes"]
                / (row[p]["images"] * row[p]["draws"] * cfg["duration_ms"] / 1000)
                for row in numbers["rates"]
            ]
            for p in ("E", "I")
        }
        population_rates["Total"] = [
            e + i
            for e, i in zip(population_rates["E"], population_rates["I"], strict=True)
        ]
        for label, color in (
            ("E", theme.INK_BLACK),
            ("I", theme.DEEP_RED),
            ("Total", theme.ELECTRIC_CYAN),
        ):
            ax.plot(
                cfg["pool_sizes"],
                population_rates[label],
                "o-",
                color=color,
                label=label,
            )
        ax.set(
            ylabel="Population spike rate\n(spikes/s)",
            ylim=(0, None),
            title="Firing rates",
        )
        ax.legend(frameon=False, loc="upper left")
        accuracy_axis = axes[1]
        accuracy = [row["test_accuracy"]["percent"] for row in numbers["rates"]]
        accuracy_axis.plot(cfg["pool_sizes"], accuracy, "o-", color=theme.INK_BLACK)
        accuracy_axis.set(ylabel="Test accuracy (%)", ylim=(0, 100), title="Accuracy")
        for index, (size, percent) in enumerate(
            zip(cfg["pool_sizes"], accuracy, strict=True)
        ):
            accuracy_axis.annotate(
                f"{percent:g}%",
                (size, percent),
                xytext=(0, -20 if index % 2 else 8),
                textcoords="offset points",
                ha="left" if index == 0 else "center",
                fontsize=theme.SIZE_ANNOTATION * FONT_SCALE,
                color=theme.INK_BLACK,
            )
        gain_axis = axes[2]
        for key, label, color in (
            ("ei_multiplier", "E→I", theme.INK_BLACK),
            ("ie_multiplier", "I→E", theme.DEEP_RED),
        ):
            gain_axis.plot(
                cfg["pool_sizes"],
                [row[key] for row in numbers["rates"]],
                "o-",
                color=color,
                label=label,
            )
        gain_axis.set(
            ylabel="Weight multiplier (×)",
            ylim=(0.75, max(row["ie_multiplier"] for row in numbers["rates"]) * 1.25),
            title="Connectivity gains",
        )
        gain_axis.set_yscale("log", base=2)
        gain_ticks = sorted(
            {
                row[key]
                for row in numbers["rates"]
                for key in ("ei_multiplier", "ie_multiplier")
            }
        )
        gain_axis.set_yticks(gain_ticks, labels=[f"{value:g}" for value in gain_ticks])
        gain_axis.legend(frameon=False, loc="upper right")
        for panel in axes:
            panel.set_xscale("log", base=2)
            panel.set(xlabel="I-pool size (neurons)")
            panel.set_xticks(
                cfg["pool_sizes"], labels=[str(size) for size in cfg["pool_sizes"]]
            )
            panel.spines[["top", "right"]].set_visible(False)
        for letter in theme.label_panels(axes):
            letter.set_fontsize(theme.SIZE_PANEL * FONT_SCALE)
        fig.tight_layout(w_pad=2.4)
        for suffix in ("svg", "png"):
            fig.savefig(
                run.export / f"firing-rates.{suffix}", dpi=plt.rcParams["savefig.dpi"]
            )
        plt.close(fig)
        run.record["execution"]["elapsed_s"] = time.perf_counter() - started
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
