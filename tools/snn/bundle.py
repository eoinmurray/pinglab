"""Translate the first supported snnlang graph subset into legacy CLI settings.

This module deliberately does not import snnlang.  Archived data-only bundles
remain executable without the authoring compiler installed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BundleCompatibilityError(ValueError):
    """The bundle is valid data, but outside this backend adapter's subset."""


@dataclass(frozen=True)
class LegacySettings:
    dt: float
    hidden_size: int
    input_size: int
    output_size: int
    w_in: tuple[float, float]
    w_ei: tuple[float, float]
    w_ie: tuple[float, float]
    tau_gaba: float
    readout_mode: str


def _canonical_json(data: Any) -> bytes:
    return (
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def _digest(data: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(data)).hexdigest()


def load_graph_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and authenticate manifest.json + graph.json from a bundle directory."""
    root = Path(path)
    if root.is_file():
        if root.name != "manifest.json":
            raise BundleCompatibilityError(
                "--bundle must name a bundle directory or its manifest.json"
            )
        root = root.parent
    manifest_path = root / "manifest.json"
    graph_path = root / "graph.json"
    if not manifest_path.is_file() or not graph_path.is_file():
        raise BundleCompatibilityError(
            f"{root} is not a bundle: manifest.json and graph.json are required"
        )
    manifest = json.loads(manifest_path.read_text())
    graph = json.loads(graph_path.read_text())
    if manifest.get("schema") != "snnlang.bundle/v1":
        raise BundleCompatibilityError(
            f"unsupported bundle schema: {manifest.get('schema')!r}"
        )
    if graph.get("schema") != "snnlang.graph/v1":
        raise BundleCompatibilityError(
            f"unsupported graph schema: {graph.get('schema')!r}"
        )
    actual = _digest(graph)
    if actual != manifest.get("graph_digest"):
        raise BundleCompatibilityError("graph.json digest does not match manifest.json")
    return manifest, graph


def _normal(
    projection: dict[str, Any], parameters: dict[str, Any]
) -> tuple[float, float]:
    ids = projection.get("parameters", [])
    if len(ids) != 1 or ids[0] not in parameters:
        raise BundleCompatibilityError(
            f"projection {projection.get('id')} must own exactly one weight parameter"
        )
    initializer = parameters[ids[0]].get("initializer", {})
    if initializer.get("kind") != "normal":
        raise BundleCompatibilityError(
            f"projection {projection.get('id')} requires a normal initializer"
        )
    return float(initializer["mean"]), float(initializer["std"])


def translate_cobanet_v1(graph: dict[str, Any]) -> LegacySettings:
    """Recognise the exact one-layer PING + mean-voltage COBANet subset."""
    populations = {row["id"]: row for row in graph.get("populations", [])}
    parameters = {row["id"]: row for row in graph.get("parameters", [])}
    projections = graph.get("projections", [])
    operations = {row["id"]: row for row in graph.get("operations", [])}
    inputs = graph.get("inputs", [])
    outputs = graph.get("outputs", [])

    spiking = [row for row in populations.values() if row.get("spiking")]
    analogue = [row for row in populations.values() if not row.get("spiking")]
    if len(spiking) != 2 or len(analogue) != 1:
        raise BundleCompatibilityError(
            "COBANet v1 requires exactly two spiking E/I populations and one "
            "non-spiking readout population"
        )
    if len(inputs) != 1 or len(outputs) != 1:
        raise BundleCompatibilityError(
            "COBANet v1 requires exactly one input and one named output"
        )

    inhibitory = [row for row in projections if row.get("polarity") == "inhibitory"]
    recurrent_exc = [
        row
        for row in projections
        if row.get("connection") == "recurrent" and row.get("polarity") == "excitatory"
    ]
    if len(inhibitory) != 1 or len(recurrent_exc) != 1:
        raise BundleCompatibilityError(
            "COBANet v1 requires one recurrent E→I and one recurrent I→E projection"
        )
    e_id = inhibitory[0]["target"].partition(".")[0]
    i_id = inhibitory[0]["source"].partition(".")[0]
    if (
        recurrent_exc[0]["source"].partition(".")[0] != e_id
        or recurrent_exc[0]["target"].partition(".")[0] != i_id
    ):
        raise BundleCompatibilityError("recurrent projections do not form one E↔I loop")
    e_pop, i_pop = populations[e_id], populations[i_id]
    if i_pop["size"] != e_pop["size"] // 4:
        raise BundleCompatibilityError(
            "current COBANet backend requires inhibitory size = excitatory size / 4"
        )
    if e_pop.get("neuron") != {
        "kind": "coba_lif",
        "tau_mem": {"value": 20.0, "unit": "ms"},
    } or i_pop.get("neuron") != {
        "kind": "coba_lif",
        "tau_mem": {"value": 10.0, "unit": "ms"},
    }:
        raise BundleCompatibilityError(
            "current COBANet backend requires 20 ms E and 10 ms I COBA-LIF neurons"
        )

    input_id = inputs[0]["id"]
    input_projection = [
        row
        for row in projections
        if row["source"] == f"{input_id}.value"
        and row["target"] == f"{e_id}.excitatory"
    ]
    readout_id = analogue[0]["id"]
    readout_projection = [
        row
        for row in projections
        if row["source"] == f"{e_id}.spikes"
        and row["target"] == f"{readout_id}.excitatory"
    ]
    if len(input_projection) != 1 or len(readout_projection) != 1:
        raise BundleCompatibilityError(
            "COBANet v1 requires direct input→E and E→readout projections"
        )
    ampa = {"kind": "ampa", "tau": {"value": 5.0, "unit": "ms"}}
    if (
        input_projection[0].get("synapse") != ampa
        or recurrent_exc[0].get("synapse") != ampa
    ):
        raise BundleCompatibilityError(
            "current COBANet backend requires 5 ms AMPA input and E→I synapses"
        )
    supported_projection_ids = {
        input_projection[0]["id"],
        readout_projection[0]["id"],
        recurrent_exc[0]["id"],
        inhibitory[0]["id"],
    }
    extras = {row["id"] for row in projections} - supported_projection_ids
    if extras:
        raise BundleCompatibilityError(
            f"unsupported projections for COBANet v1: {', '.join(sorted(extras))}"
        )

    output_signal = outputs[0]["signal"]
    output_op = operations.get(output_signal.partition(".")[0])
    if (
        not output_op
        or output_op.get("kind") != "reduce_mean"
        or output_op.get("sources") != [f"{readout_id}.voltage"]
    ):
        raise BundleCompatibilityError(
            "COBANet v1 currently supports only MeanVoltage named outputs"
        )
    if set(operations) != {output_op["id"]}:
        raise BundleCompatibilityError(
            "COBANet v1 does not support additional graph operations"
        )

    dt = graph.get("timebase", {}).get("dt", {})
    if dt.get("unit") != "ms" or float(dt.get("value", 0)) <= 0:
        raise BundleCompatibilityError(
            "graph timebase must declare a positive dt in ms"
        )
    input_shape = inputs[0].get("shape", [])
    if len(input_shape) < 1 or not isinstance(input_shape[-1], int):
        raise BundleCompatibilityError(
            "input shape must end in a concrete channel count"
        )
    tau_gaba = inhibitory[0].get("synapse", {}).get("tau", {})
    if (
        inhibitory[0].get("synapse", {}).get("kind") != "gaba"
        or tau_gaba.get("unit") != "ms"
    ):
        raise BundleCompatibilityError("I→E GABA projection must declare tau in ms")
    if readout_projection[0].get("parameters", []) != [
        f"{readout_projection[0]['id']}.weight"
    ]:
        raise BundleCompatibilityError("readout projection weight is malformed")
    readout_weight = _normal(readout_projection[0], parameters)
    if readout_weight != (5.1, 3.8):
        raise BundleCompatibilityError(
            "current COBANet backend requires readout Normal(5.1, 3.8)"
        )
    readout_tau = readout_projection[0].get("synapse", {}).get("tau", {})
    if readout_tau != {"value": 2.0, "unit": "ms"}:
        raise BundleCompatibilityError(
            "current COBANet backend requires a 2 ms MeanVoltage integrator"
        )

    return LegacySettings(
        dt=float(dt["value"]),
        hidden_size=int(e_pop["size"]),
        input_size=int(input_shape[-1]),
        output_size=int(analogue[0]["size"]),
        w_in=_normal(input_projection[0], parameters),
        w_ei=_normal(recurrent_exc[0], parameters),
        w_ie=_normal(inhibitory[0], parameters),
        tau_gaba=float(tau_gaba["value"]),
        readout_mode="mem-mean",
    )


_STRUCTURAL_FLAGS = {
    "--model",
    "--n-hidden",
    "--readout",
    "--dt",
    "--w-in",
    "--w-in-sparsity",
    "--ei-strength",
    "--ei-ratio",
    "--w-ei",
    "--w-ie",
    "--ei-sparsity",
    "--tau-gaba",
    "--n-in",
    "--dataset",
    "--load-config",
}


def apply_bundle_to_args(args, argv: list[str]):
    """Apply bundle structure while preserving execution-only CLI overrides."""
    if not getattr(args, "bundle", None):
        return args
    if args.mode != "sim":
        raise BundleCompatibilityError(
            "--bundle currently supports sim only; bundle-driven training is the next stage"
        )
    explicit = {item.split("=", 1)[0] for item in argv if item.startswith("--")}
    conflicts = sorted(explicit & _STRUCTURAL_FLAGS)
    if conflicts:
        raise BundleCompatibilityError(
            "bundle owns structural settings; remove conflicting flags: "
            + ", ".join(conflicts)
        )
    _, graph = load_graph_bundle(args.bundle)
    settings = translate_cobanet_v1(graph)
    if settings.output_size != 10 or settings.input_size != 784:
        raise BundleCompatibilityError(
            "first COBANet bundle route is intentionally limited to MNIST "
            "(784 inputs, 10 outputs)"
        )
    args.model = "ping"
    args.n_hidden = [settings.hidden_size]
    args.n_in = settings.input_size
    args.dataset = "mnist"
    args.dt = settings.dt
    args.readout_mode = settings.readout_mode
    args.w_in = list(settings.w_in)
    args.w_in_sparsity = 0.0
    args.ei_strength = settings.w_ei[0]
    args.ei_ratio = settings.w_ie[0] / settings.w_ei[0]
    args.w_ei = list(settings.w_ei)
    args.w_ie = list(settings.w_ie)
    args.ei_sparsity = 0.0
    args.tau_gaba = settings.tau_gaba
    return args
