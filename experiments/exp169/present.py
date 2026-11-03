"""Render the three measured rasters and compiled native architectures."""

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp169 import recipe
from experiments.exp169.analyse import read_evidence
from pingstore.library import source_run, stage_run
from snnlab import lang, viz


def raster_figure(rows, cfg):
    """Draw unchanged spike coordinates using the shared Pinglab figure style."""
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 10.5), sharex=True, layout="constrained"
    )
    seconds = cfg["duration_ms"] / 1000
    for index, (ax, row) in enumerate(zip(axes, rows, strict=True)):
        e, i = row["e"], row["i"]
        for spikes, offset, colour in [
            (e, 0, theme.INK_BLACK),
            (i, e.shape[1], theme.DEEP_RED),
        ]:
            t, c = np.nonzero(spikes)
            ax.scatter(
                t * cfg["dt_ms"],
                c + offset,
                marker="|",
                s=9,
                color=colour,
                linewidths=0.8,
                rasterized=True,
            )
        ax.axhline(e.shape[1] - 0.5, color=theme.MUTED, lw=0.5)
        note = (
            f"I {i.sum() / i.shape[1] / seconds:.1f} Hz"
            if index < 2
            else f"{i.shape[1]:,} private inhibitory input streams"
        )
        ax.set(
            xlim=(0, cfg["duration_ms"]),
            ylim=(-0.5, e.shape[1] + i.shape[1] - 0.5),
            ylabel="Neuron index" if index < 2 else "Neuron / input index",
            title=f"{row['name']}  |  E {e.sum() / e.shape[1] / seconds:.1f} Hz  |  {note}",
        )
        ax.set_yticks([0, 512, e.shape[1] - 1, e.shape[1] + i.shape[1] - 1])
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (ms)")
    theme.label_panels(axes)
    return fig


def circuit(bundle, destination, *, reference=False):
    native = lang.diagram(bundle, view="circuit")
    positions = {"drive": (-440, 0), "E": (0, 0), "I": (0, 360), "readout": (440, 0)}
    if not reference:
        positions["artificial"] = (-440, 360)
    native = replace(
        native,
        title="TR02 reference topology"
        if reference
        else "Frozen COBA with private GABA inputs",
        groups=(),
        nodes=tuple(
            replace(
                node,
                detail=f"{next(row for row in bundle.graph['inputs'] if row['id'] == node.id)['shape'][-1]:,} spike channels",
            )
            if node.kind == "input"
            else node
            for node in native.nodes
            if node.id in positions
        ),
        edges=tuple(
            edge
            for edge in native.edges
            if edge.source in positions and edge.target in positions
        ),
    )
    labels = {
        "input_to_E": r"$W_{EX}^{C/P}$" if reference else r"$W_{EX}^{C}$",
        "E_to_I": r"$W_{IE}^{C/P}$" if reference else "$0$",
        "I_to_E": r"$W_{EI}^{C/P}$" if reference else "$0$",
        "readout_projection": r"$W_{RE}^{C/P}$" if reference else r"$W_{RE}^{C}$",
        "artificial_to_E": r"$q_j\delta_{jk}$",
    }
    routes = {
        "input_to_E": dict(source_port="right", target_port="left"),
        "E_to_I": dict(source_port="bottom_right", target_port="top_right"),
        "I_to_E": dict(source_port="top_left", target_port="bottom_left"),
        "readout_projection": dict(source_port="right", target_port="left"),
        "artificial_to_E": dict(
            source_port="top",
            target_port="lower_left",
            routing="polyline",
            waypoints=((-300, 180), (-190, 180)),
        ),
    }
    ports = {
        "top_left": (0.25, 0),
        "top_right": (0.75, 0),
        "bottom_left": (0.25, 1),
        "bottom_right": (0.75, 1),
        "lower_left": (0, 0.8),
    }
    layout = viz.DiagramLayout(
        nodes={
            key: viz.NodeLayout(position=pos, min_size=(200, 130), ports=ports)
            for key, pos in positions.items()
        },
        edges={
            edge.id: viz.EdgeLayout(
                **routes[edge.id], label=viz.DiagramLabel(labels[edge.id])
            )
            for edge in native.edges
        },
    )
    viz.render_diagram(native, destination, layout=layout)


def present(identity, *, run_id=None):
    started = perf_counter()
    analysis = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="analyse"
    )
    pin = analysis.record["inputs"]["compute"]
    computed = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        experiment=recipe.SLUG,
        stage="compute",
        reference=pin,
    )
    results = json.loads(analysis.file("results.json").read_text())
    cfg, bank, rows, _, _, _ = read_evidence(computed)
    with stage_run(
        ROOT,
        recipe.SLUG,
        "present",
        inputs={"analyse": analysis, "compute": computed},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        with matplotlib.rc_context():
            theme.apply()
            fig = raster_figure(rows, cfg)
            for ext in ("png", "svg"):
                fig.savefig(run.export / ("rasters." + ext), dpi=200)
            plt.close(fig)
        bundle = lang.load_bundle(
            computed.restore_bundle("network.bundle", run.scratch / "control.bundle")
        )
        for ext in ("svg", "png"):
            circuit(bundle, run.export / ("circuit." + ext))
        authored = recipe.author_network(recipe.configuration())
        if authored.graph != bundle.graph:
            raise ValueError("Current recipe differs from retained native architecture")
        reference = recipe.author_network(recipe.configuration(), include_private=False)
        reference.write(run.scratch / "reference.bundle", visualise=False)
        shutil.make_archive(
            str(run.export / "reference.bundle"),
            "zip",
            run.scratch / "reference.bundle",
        )
        for ext in ("svg", "png"):
            circuit(
                reference, run.export / ("reference-circuit." + ext), reference=True
            )
        (run.export / "numbers.json").write_text(json.dumps(results, indent=2) + "\n")
        run.record["execution"].update(
            no_simulation=True, elapsed_seconds=perf_counter() - started
        )
        for source in (analysis, computed, bank):
            source.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)
