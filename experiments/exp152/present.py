"""Export the callback replay and native controller circuit.

Consumes an explicit analysis run without executing the model. Full traces remain
in its ancestry; replay states appear only at recorded callback completion times.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from snnlab import lang, viz

NODES = ("flexor", "extensor", "plant")


def network_diagram(bundle):
    """Resolve the saved circuit with explicit routes and target-first weight labels."""
    native = lang.diagram(bundle, view="circuit")
    symbols = {"peer": "P", "sensory": "S", "tonic": "T", "motor": "M"}
    diagram = replace(
        native,
        title="Single-cell controller inputs",
        nodes=tuple(
            replace(
                node,
                detail=node.detail.splitlines()[0].replace("1 neurons", "1 neuron"),
            )
            for node in native.nodes
            if node.id in symbols
        ),
        edges=tuple(edge for edge in native.edges if edge.role != "output"),
        groups=(),
    )
    positions = {
        "peer": (0, -250),
        "sensory": (0, 0),
        "tonic": (0, 250),
        "motor": (520, 0),
    }
    routes = dict(
        peer_inhibition=viz.EdgeLayout(
            source_port="right", target_port="top", waypoints=((520, -250),)
        ),
        sensory_drive=viz.EdgeLayout(source_port="right", target_port="left"),
        local_bias=viz.EdgeLayout(
            source_port="right", target_port="bottom", waypoints=((520, 250),)
        ),
    )
    ports = dict(
        upper_left=(0, 0.3),
        lower_left=(0, 0.7),
        upper_right=(1, 0.3),
        lower_right=(1, 0.7),
        top_left=(0.25, 0),
        top_right=(0.75, 0),
        bottom_left=(0.25, 1),
        bottom_right=(0.75, 1),
    )
    layout = viz.DiagramLayout(
        nodes={
            identifier: viz.NodeLayout(
                position=position,
                label=f"${symbols[identifier]}$",
                min_size=(240, 160),
                ports=ports,
            )
            for identifier, position in positions.items()
        },
        edges={
            edge.id: replace(
                routes[edge.id],
                label=viz.DiagramLabel(
                    rf"$W_{{{symbols[edge.target]} , {symbols[edge.source]}}}$"
                ),
            )
            for edge in diagram.edges
        },
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=16), gap=30
        ),
    )
    # Embedded math must keep its measured size independently of plot cropping.
    with matplotlib.rc_context({"savefig.bbox": None}):
        return viz.resolve_diagram(
            diagram, layout=layout, theme=viz.DiagramTheme(secondary_size=13)
        )


def recording(values, summary):
    """Keep every timing sample and expose controller states when callbacks finish."""
    cfg = summary["configuration"]
    populations = cfg["controller_populations"]
    if (
        cfg["neuron_model"] != "elevate"
        or populations != ["motor"]
        or cfg["neurons"] != 1
        or cfg["spike_transport"] != "logical"
        or cfg["joint"]["angle_convention"] != "zero_downward_positive_right"
    ):
        raise ValueError("Replay requires the single-cell Elevate logical joint model")
    nodes = {}
    for name in NODES:
        prefix = name + "_"
        times = np.asarray(values[prefix + "time_s"]) * 1000
        callback = np.asarray(values[prefix + "callback_ms"])
        row = dict(
            time_ms=times.tolist(),
            callback_ms=callback.tolist(),
            statistics=summary["full_callback_ms"][name],
        )
        if len(times) != len(callback) or np.any(np.diff(times) < 0):
            raise ValueError(f"Invalid callback timeline for {name}")
        if name != "plant":
            row.update(
                voltage=np.asarray(values[prefix + "voltage"]).tolist(),
                spikes=np.asarray(values[prefix + "spikes"]).tolist(),
            )
        else:
            row.update(
                angle=np.asarray(values["angle"]).tolist(),
                velocity=np.asarray(values["velocity"]).tolist(),
                activation=np.asarray(values["activation"]).tolist(),
            )
        for field, samples in row.items():
            if not isinstance(samples, list):
                continue
            array = np.asarray(samples)
            if len(array) != len(times) or not np.isfinite(array).all():
                raise ValueError(f"Invalid callback field for {name}: {field}")
        if (
            name != "plant"
            and len(times)
            and (
                np.asarray(row["voltage"]).shape != (len(times), len(populations))
                or np.asarray(row["spikes"]).shape != (len(times), len(populations))
            )
        ):
            raise ValueError(f"Invalid population channels for {name}")
        if name != "plant" and (np.asarray(row["spikes"]) < 0).any():
            raise ValueError(f"Negative spike count for {name}")
        nodes[name] = row
    return dict(
        schema="pinglab.exp152.replay/v2",
        controller_populations=populations,
        mechanics=cfg["joint"],
        duration_ms=summary["duration_seconds"] * 1000,
        period_ms=summary["timer_ms"],
        neuron_model=cfg["neuron_model"],
        voltage_range_mv=[-80, 40],
        neurons_per_population=summary["neurons_per_population"],
        clock="Monotonic callback completion time relative to each node's mapped shared start",
        sensory_time_ms={
            side: (np.asarray(values["sensory_" + side + "_time_s"]) * 1000).tolist()
            for side in ("flexor", "extensor")
        },
        nodes=nodes,
    )


def present(source, *, run_id=None):
    upstream = source_run(
        REPO / ".pingstore", source, stage="analyse", experiment="exp152"
    )
    numbers = load_json(upstream.file("numbers.json"))
    cfg = numbers["configuration"]
    with np.load(upstream.file("traces.npz"), allow_pickle=False) as data:
        values = dict(data)
    payload = recording(values, numbers)
    reference = upstream.record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore",
        reference["run_id"],
        stage="compute",
        experiment="exp152",
        reference=reference,
    )
    with stage_run(
        REPO,
        "exp152",
        "present",
        inputs={"analyse": upstream},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        write_json_atomic(run.export / "circuit.json", payload)
        bundle = lang.load_bundle(
            compute.restore_bundle(
                "flexor/network.bundle", run.scratch / "diagram.bundle"
            )
        )
        circuit = network_diagram(bundle)
        with matplotlib.rc_context({"savefig.bbox": None}):
            for extension in ("svg", "png"):
                viz.render_diagram(circuit, run.export / f"circuit.{extension}")
        run.record["execution"]["snnlab_version"] = version("snnlab")
        write_json_atomic(run.export / "numbers.json", numbers)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    print(present(**vars(parser.parse_args())), flush=True)
