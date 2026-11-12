"""Render latency/spike-budget and supporting accuracy/cost curves."""

import argparse
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp174 import recipe
from pingstore.library import source_run, stage_run
from snnlab import lang, viz


def circuit(bundle, path, kind):
    """Lay out the actual compiled pathways, omitting reduction/output plumbing."""
    native = lang.diagram(bundle, view="circuit")
    symbols = {"drive": "X", "artificial": "A", "E": "E", "I": "I", "readout": "R"}
    nodes = tuple(node for node in native.nodes if node.id in symbols)
    ids = {node.id for node in nodes}
    diagram = replace(
        native,
        title={
            "coba": "COBA (feedback weights zero)",
            "ping": "PING",
            "artificial": "Artificial inhibition (biological feedback zero)",
        }[kind],
        groups=(),
        nodes=nodes,
        edges=tuple(
            edge for edge in native.edges if edge.source in ids and edge.target in ids
        ),
    )
    positions = {
        "drive": (-400, 0),
        "artificial": (-400, -300),
        "E": (0, 0),
        "I": (0, -330),
        "readout": (400, 0),
    }
    ports = {
        "upper_left": (0.3, 0),
        "upper_right": (0.7, 0),
        "lower_left": (0.3, 1),
        "lower_right": (0.7, 1),
    }
    routes = {
        "input_to_E": viz.EdgeLayout(source_port="right", target_port="left"),
        "readout_projection": viz.EdgeLayout(source_port="right", target_port="left"),
        "E_to_I": viz.EdgeLayout(source_port="upper_right", target_port="lower_right"),
        "I_to_E": viz.EdgeLayout(source_port="lower_left", target_port="upper_left"),
        "artificial_to_E": viz.EdgeLayout(source_port="right", target_port="left"),
    }

    def label(edge):
        if edge.id == "artificial_to_E":
            return r"$1$ (one-to-one)"
        if kind != "ping" and edge.id in ("E_to_I", "I_to_E"):
            return "$0$"
        return rf"$W_{{{symbols[edge.target]}{symbols[edge.source]}}}$"

    layout = viz.DiagramLayout(
        nodes={
            node.id: viz.NodeLayout(
                position=positions[node.id],
                label=f"${symbols[node.id]}$",
                min_size=(220, 150),
                ports=ports,
            )
            for node in nodes
        },
        edges={
            edge.id: replace(routes[edge.id], label=viz.DiagramLabel(label(edge)))
            for edge in diagram.edges
        },
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=16), gap=30
        ),
    )
    style = viz.DiagramTheme(
        font_name="monospace",
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
        title_size=theme.SIZE_PANEL * 1.5,
        label_size=theme.SIZE_TITLE * 1.5,
        secondary_size=theme.SIZE_LABEL * 1.5,
    )
    resolved = viz.resolve_diagram(diagram, layout=layout, theme=style)
    for destination in (path, path.with_suffix(".png")):
        viz.render_diagram(resolved, destination)


def present(identity):
    started = time.perf_counter()
    source = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="analyse"
    )
    numbers = json.loads(source.file("numbers.json").read_text())
    pin = source.record["inputs"]["compute"]
    compute = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        reference=pin,
        experiment=recipe.SLUG,
        stage="compute",
    )
    theme.apply()
    with stage_run(
        ROOT,
        recipe.SLUG,
        "present",
        inputs={"analysis": source},
        configuration={"plot": "decision-budget/v2"},
    ) as run:
        shutil.copyfile(source.file("numbers.json"), run.export / "numbers.json")
        with np.load(source.file("curves.npz"), allow_pickle=False) as curves:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
            a, b, c = axes
            for row, color, marker in zip(
                numbers["conditions"],
                (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN),
                ("o", "s", "^"),
                strict=True,
            ):
                kind, label = row["kind"], row["label"]
                times = curves["time_ms"]
                accuracy = 100 * curves[kind + "_accuracy"]
                spikes = curves[kind + "_e_spikes"]
                b.plot(times, accuracy, color=color, label=label)
                c.plot(spikes, accuracy, color=color, label=label)
                if row["latency_ms"] is not None:
                    t, n = row["latency_ms"], row["e_spikes_at_threshold"]
                    a.scatter(
                        n, t, color=color, marker=marker, s=65, zorder=4, label=label
                    )
                    a.annotate(
                        label,
                        (n, t),
                        xytext=(7, 8),
                        textcoords="offset points",
                        color=color,
                        fontsize=theme.SIZE_ANNOTATION,
                    )
                    idx = int(round(t / numbers["configuration"]["dt_ms"])) - 1
                    b.scatter(
                        t, accuracy[idx], color=color, marker=marker, s=35, zorder=5
                    )
                    c.scatter(
                        n, accuracy[idx], color=color, marker=marker, s=35, zorder=5
                    )
                else:
                    a.plot(
                        [],
                        [],
                        marker=marker,
                        color=color,
                        linestyle="none",
                        label=label + ": 80% unreached",
                    )
            for ax in (b, c):
                ax.axhline(80, color=theme.MUTED, linestyle="--", linewidth=0.8)
                ax.set_ylim(0, 102)
                ax.set_ylabel("Decision accuracy (%)")
            a.set(
                xlabel="Mean cumulative E spikes per image",
                ylabel="Latency to sustained 80% accuracy (ms)",
                title="Spike budget and decision latency",
            )
            reached = [r for r in numbers["conditions"] if r["latency_ms"] is not None]
            a.set_xlim(
                0, max((r["e_spikes_at_threshold"] for r in reached), default=1) * 1.5
            )
            a.set_ylim(0, max((r["latency_ms"] for r in reached), default=200) * 1.2)
            b.set(
                xlabel="Elapsed stimulus time (ms)",
                xlim=(0, 200),
                title="Accuracy over time",
            )
            c.set(
                xlabel="Mean cumulative E spikes per image",
                xlim=(0, None),
                title="Accuracy per E spike budget",
            )
            b.legend(loc="lower right", frameon=False)
            if any(r["latency_ms"] is None for r in numbers["conditions"]):
                a.legend(loc="best", frameon=False)
            theme.label_panels(axes.flat)
            fig.suptitle(
                f"No rate penalty · {numbers['configuration']['samples']} matched test images · seed 42",
                fontsize=theme.SIZE_TITLE,
            )
            for extension in ("svg", "png", "pdf"):
                fig.savefig(run.export / ("decision-budget." + extension), dpi=180)
            plt.close(fig)
        for kind in recipe.KINDS:
            bundle = lang.load_bundle(
                compute.restore_bundle(
                    kind + "/network.bundle", run.scratch / (kind + ".bundle")
                )
            )
            circuit(bundle, run.export / ("circuit-" + kind + ".svg"), kind)
        run.record["execution"]["elapsed_seconds"] = time.perf_counter() - started
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
