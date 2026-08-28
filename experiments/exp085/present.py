"""Render saved exp085 analysis; never simulate, remeasure or publish."""

import argparse
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp085 import evidence, inputs, plots, recipe
from experiments.helpers.fmt import format_duration
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from tools import snnlang as snn  # noqa: TID251


def render_schematic(path, graph, results):
    """Relabel only SVG text and tooltips; retain the graph and drawing geometry."""
    snn.Bundle(graph=graph, training=None, manifest={}, diagnostics=[]).visualise(
        path, view="circuit", expand_groups=recipe.PING_GROUPS
    )
    network = results["network"]
    labels = {
        f"time × batch × {recipe.N_INPUT}": f"{recipe.N_INPUT} spike channels",
        "SPIKES": "SPIKES",
        "SPIKING POPULATION": "CONDUCTANCE LIF",
    }
    titles = {"snnlang": "Two coupled PING networks"}
    for name in ("A", "B"):
        rate = network["detuning_input_rates_hz"][f"PING_{name}"]
        drive = f"Drive {name}: {rate:g} Hz"
        labels[f"drive_{name}_{rate:g}_Hz"] = drive
        titles[f"drive_{name}_{rate:g}_Hz"] = drive
        labels[f"PING {name}"] = f"PING {name}"
        titles[f"cluster_n_PING_{name}"] = f"Network {name}"
        for population, kind in (("E", "excitatory"), ("I", "inhibitory")):
            size = network["populations_per_network"][population]
            labels[f"PING {name} {population}"] = f"{name} {kind}"
            labels[f"{size} units · coba_lif"] = f"{size} neurons"
            titles[f"PING_{name}_{population}"] = f"Network {name} {kind} neurons"
    svg = ElementTree.parse(path)
    for node in svg.iter():
        kind = node.tag.rsplit("}", 1)[-1]
        if kind == "text":
            if node.text not in labels:
                raise PingstoreError(f"unrecognized schematic label: {node.text!r}")
            node.text = labels[node.text]
        elif kind == "title":
            parts = (node.text or "").split("->")
            if any(part not in titles for part in parts):
                raise PingstoreError(f"unrecognized schematic tooltip: {node.text!r}")
            node.text = " → ".join(titles[part] for part in parts)
    ElementTree.register_namespace("", "http://www.w3.org/2000/svg")
    svg.write(path, encoding="utf-8", xml_declaration=True)


def render(root, results, data, graph):
    render_schematic(root / "network.svg", graph, results)
    plots.plot_uncoupled(data["uncoupled"], root / "uncoupled.png")
    plots.plot_phase_response_examples(
        data["phase_response_examples"], root / "phase_response_examples.png"
    )
    plots.plot_phase_response(
        results["phase_response"],
        data["phase_response_examples"],
        root / "phase_response.png",
    )
    plots.plot_pathway_comparison(
        results["pathway_comparison"],
        data["pathway_traces"],
        root / "pathway_comparison.png",
    )
    plots.plot_event_aligned_mechanism(
        results["event_aligned_mechanism"],
        data["mechanism_traces"],
        root / "event_aligned_mechanism.png",
    )


def present(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "analyse")
    pins = source.record["inputs"]
    if set(pins) != {"compute"}:
        raise PingstoreError("exp085 analysis must pin exactly one compute input")
    compute = inputs.source(
        REPO, pins["compute"]["run_id"], "compute", reference=pins["compute"]
    )
    if compute.record["inputs"]:
        raise PingstoreError("standalone exp085 compute must have no upstream inputs")
    result = load_json(source.export / "results.json")
    if (
        result.get("schema") != "exp085.analysis/v1"
        or result.get("recipe") != recipe.configuration()
    ):
        raise PingstoreError("inconsistent exp085 analysis payload")
    graph = load_json(source.export / "network.json")
    if recipe.graph_digest(graph) != result["recipe"]["graph_hashes"]["both"]:
        raise PingstoreError("exp085 presentation graph differs from the recipe")
    data = evidence.load_plot_data(source.export)
    started = time.monotonic()
    with inputs.execution(
        REPO, "present", sources={"analysis": source}, run_id=run_id
    ) as run:
        render(run.export, result["results"], data, graph)
        duration = time.monotonic() - started
        write_json_atomic(run.export / "protocol.json", result["results"])
        write_json_atomic(
            run.export / "numbers.json",
            {
                "run_id": run.run_id,
                "git_sha": run.record["provenance"]["git_commit"],
                "duration_s": round(duration, 1),
                "duration": format_duration(duration),
                **result["results"],
            },
        )
        if any(not (run.export / name).is_file() for name in recipe.FIGURES):
            raise PingstoreError("incomplete exp085 figure export")
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp085 v3 analyse ID"
    )
    parser.add_argument("--run-id", help="unused source-neutral v3 reservation")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"exp085 present: {exc}\n")


if __name__ == "__main__":
    main()
