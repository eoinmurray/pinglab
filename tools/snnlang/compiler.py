"""Validation, canonical serialisation, bundle I/O, and reports."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .core import Network
from .training import TrainSpec

SCHEMA = "snnlang.graph/v1"
BUNDLE_SCHEMA = "snnlang.bundle/v1"
TRAINING_SCHEMA = "snnlang.training/v1"
CAPABILITY_SCHEMA = "snnlang.capabilities/v1"


def canonical_json(data: Any) -> bytes:
    return (
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def digest(data: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(data)).hexdigest()


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    code: str
    message: str
    subject: str | None = None

    def line(self) -> str:
        where = f" [{self.subject}]" if self.subject else ""
        return f"{self.severity.upper()} {self.code}{where}: {self.message}"


@dataclass
class ValidationResult:
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "error"]

    @property
    def warnings(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity == "warning"]

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("\n".join(d.line() for d in self.errors))


def graph_dict(net: Network) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "name": net.name,
        "timebase": {"dt": net.dt.json()},
        "inputs": sorted(net.inputs, key=lambda x: x["id"]),
        "populations": sorted(net.populations, key=lambda x: x["id"]),
        "projections": sorted(net.projections, key=lambda x: x["id"]),
        "operations": sorted(net.operations, key=lambda x: x["id"]),
        "parameters": sorted(net.parameters, key=lambda x: x["id"]),
        "constants": sorted(net.constants, key=lambda x: x["id"]),
        "outputs": sorted(net.outputs, key=lambda x: x["id"]),
        "observables": sorted(net.observables, key=lambda x: x["id"]),
        "assets": sorted(net.assets, key=lambda x: x["id"]),
        "groups": [
            {"id": g.name, "members": sorted(g.members), "parent": g.parent}
            for g in sorted(net.groups.values(), key=lambda x: x.name)
        ],
    }


def _training_dict(spec: TrainSpec, graph_digest: str) -> dict[str, Any]:
    return {
        "schema": TRAINING_SCHEMA,
        "graph_digest": graph_digest,
        "objectives": [
            {
                "kind": o.kind,
                "prediction": o.prediction,
                "target": o.target,
                "weight": o.weight,
            }
            for o in spec.objectives
        ],
        "parameter_groups": [
            {
                "id": g.name,
                "parameters": sorted(g.ids()),
                "lr": g.lr,
                "frozen": g.frozen,
            }
            for g in spec.parameter_groups
        ],
        "regularizers": [
            {
                "kind": r.kind,
                "signal": r.signal,
                "strength": r.strength,
                "config": r.config,
            }
            for r in spec.regularizers
        ],
        "stop_gradients": sorted(s.signal for s in spec.stop_gradients),
        "optimizer": {"kind": spec.optimizer.kind, "config": spec.optimizer.config},
        "epochs": spec.epochs,
        "gradient_clip": spec.gradient_clip,
        "surrogate": spec.surrogate.json() if spec.surrogate else None,
    }


def validate_graph(graph: Mapping[str, Any]) -> ValidationResult:
    out = ValidationResult()
    collections = (
        "inputs",
        "populations",
        "projections",
        "operations",
        "parameters",
        "constants",
        "outputs",
        "observables",
        "assets",
        "groups",
    )
    seen: dict[str, str] = {}
    for collection in collections:
        for row in graph.get(collection, []):
            name = row.get("id")
            if not name:
                out.diagnostics.append(
                    Diagnostic("error", "E001", "missing identifier", collection)
                )
            elif name in seen:
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E002",
                        f"duplicate identifier; first used by {seen[name]}",
                        name,
                    )
                )
            else:
                seen[name] = collection

    signals: dict[str, dict[str, Any]] = {}
    for row in graph.get("inputs", []):
        signals[f"{row['id']}.value"] = row
    for row in graph.get("populations", []):
        signals[f"{row['id']}.voltage"] = {
            "shape": ["time", "batch", row["size"]],
            "unit": "mV",
        }
        if row["spiking"]:
            signals[f"{row['id']}.spikes"] = {
                "shape": ["time", "batch", row["size"]],
                "unit": "spike",
            }
    for row in graph.get("operations", []):
        signals[f"{row['id']}.value"] = row

    parameter_rows = {p["id"]: p for p in graph.get("parameters", [])}
    initializer_fields = {
        "normal": {"mean", "std"},
        "lower_clamped_normal": {"mean", "std", "initial_zero_fraction", "zeroing"},
        "signed_normal": {"mean", "std"},
        "uniform": {"low", "high"},
        "constant": {"value"},
        "zeros": set(),
    }
    for row in parameter_rows.values():
        if not row.get("unit"):
            out.diagnostics.append(
                Diagnostic(
                    "error", "E109", "parameter requires an explicit unit", row["id"]
                )
            )
        initializer = row.get("initializer", {})
        kind = initializer.get("kind")
        if kind not in initializer_fields:
            out.diagnostics.append(
                Diagnostic(
                    "error", "E110", f"unsupported initializer {kind}", row["id"]
                )
            )
            continue
        missing = initializer_fields[kind] - set(initializer)
        if missing:
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E111",
                    f"initializer missing fields {sorted(missing)}",
                    row["id"],
                )
            )
        constraint = row.get("constraint")
        if constraint is not None and constraint.get("kind") != "non_negative":
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E112",
                    f"unsupported constraint {constraint.get('kind')}",
                    row["id"],
                )
            )
    parameter_ids = set(parameter_rows)
    population_ids = {p["id"] for p in graph.get("populations", [])}
    consumers: set[str] = set()
    adjacency: dict[str, set[str]] = {p: set() for p in population_ids}
    for row in graph.get("projections", []):
        if not isinstance(row.get("enabled", True), bool):
            out.diagnostics.append(
                Diagnostic(
                    "error", "E108", "projection enabled must be boolean", row["id"]
                )
            )
        source = row["source"]
        target_pop, _, target_port = row["target"].partition(".")
        if source not in signals:
            out.diagnostics.append(
                Diagnostic("error", "E101", f"unresolved source {source}", row["id"])
            )
        if target_pop not in population_ids:
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E102",
                    f"unresolved target population {target_pop}",
                    row["id"],
                )
            )
        if target_port not in {"excitatory", "inhibitory", "modulatory"}:
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E103",
                    f"incompatible target port {target_port}",
                    row["id"],
                )
            )
        expected = {
            "excitatory": "excitatory",
            "inhibitory": "inhibitory",
            "modulatory": "modulatory",
        }
        if expected.get(target_port) != row.get("polarity"):
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E104",
                    "projection polarity and target port disagree",
                    row["id"],
                )
            )
        delay = row.get("delay")
        if row.get("connection") == "feedback" and (
            not delay or not isinstance(delay, dict) or delay.get("value", 0) <= 0
        ):
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E105",
                    "feedback requires an explicit non-zero delay",
                    row["id"],
                )
            )
        for pid in row.get("parameters", []):
            if pid not in parameter_ids:
                out.diagnostics.append(
                    Diagnostic(
                        "error", "E106", f"unresolved parameter {pid}", row["id"]
                    )
                )
            elif source in signals and target_pop in population_ids:
                expected_shape = [
                    next(
                        p["size"] for p in graph["populations"] if p["id"] == target_pop
                    ),
                    signals[source]["shape"][-1],
                ]
                if parameter_rows[pid]["shape"] != expected_shape:
                    out.diagnostics.append(
                        Diagnostic(
                            "error",
                            "E107",
                            f"projection parameter shape {parameter_rows[pid]['shape']} does not match {expected_shape}",
                            row["id"],
                        )
                    )
        consumers.add(source)
        source_owner = source.partition(".")[0]
        if (
            row.get("enabled", True)
            and source_owner in adjacency
            and target_pop in adjacency
        ):
            adjacency[source_owner].add(target_pop)

    for row in graph.get("operations", []):
        for source in row["sources"]:
            if source not in signals:
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E201",
                        f"unresolved operation source {source}",
                        row["id"],
                    )
                )
            consumers.add(source)
        for pid in row.get("parameters", []):
            if pid not in parameter_ids:
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E202",
                        f"unresolved operation parameter {pid}",
                        row["id"],
                    )
                )
        if (
            row["kind"] == "duration_normalise"
            and not row["config"].get("duration")
            and not row["config"].get("mask")
        ):
            out.diagnostics.append(
                Diagnostic(
                    "error", "E203", "spike-rate duration is ambiguous", row["id"]
                )
            )
        if (
            row["kind"] == "duration_normalise"
            and row["config"].get("duration") is not None
        ):
            try:
                duration = float(row["config"]["duration"])
            except (TypeError, ValueError):
                duration = -1
            if duration <= 0:
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E208",
                        "spike-rate duration must be positive seconds",
                        row["id"],
                    )
                )
        if not row.get("shape") or not row.get("unit"):
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E204",
                    "operation requires explicit shape and unit",
                    row["id"],
                )
            )
        primary = signals.get(row["sources"][0]) if row.get("sources") else None
        if primary:
            primary_shape = list(primary.get("shape", []))
            expected_shape: list[Any] | None = None
            if row["kind"] == "linear":
                expected_shape = [
                    *primary_shape[:-1],
                    row.get("config", {}).get("size"),
                ]
                for pid in row.get("parameters", []):
                    if pid in parameter_rows:
                        expected_parameter = [
                            row.get("config", {}).get("size"),
                            primary_shape[-1],
                        ]
                        if parameter_rows[pid]["shape"] != expected_parameter:
                            out.diagnostics.append(
                                Diagnostic(
                                    "error",
                                    "E209",
                                    f"linear parameter shape {parameter_rows[pid]['shape']} does not match {expected_parameter}",
                                    row["id"],
                                )
                            )
            elif row["kind"] in {"reduce_mean", "reduce_sum"}:
                if row.get("config", {}).get("window", "full") != "full":
                    out.diagnostics.append(
                        Diagnostic(
                            "error",
                            "E210",
                            "only full-window reductions are supported",
                            row["id"],
                        )
                    )
                if "time" not in primary_shape:
                    out.diagnostics.append(
                        Diagnostic(
                            "error",
                            "E211",
                            "time reduction requires a time axis",
                            row["id"],
                        )
                    )
                else:
                    time_axis = primary_shape.index("time")
                    expected_shape = [
                        dimension
                        for index, dimension in enumerate(primary_shape)
                        if index != time_axis
                    ]
            elif row["kind"] == "select_final":
                if not primary_shape or primary_shape[0] != "time":
                    out.diagnostics.append(
                        Diagnostic(
                            "error",
                            "E212",
                            "final selection requires a leading time axis",
                            row["id"],
                        )
                    )
                else:
                    expected_shape = primary_shape[1:]
            elif row["kind"] == "cumulative_sum":
                expected_shape = primary_shape
            elif row["kind"] == "duration_normalise":
                expected_shape = primary_shape
            if expected_shape is not None and row.get("shape") != expected_shape:
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E213",
                        f"operation shape {row.get('shape')} does not match inferred {expected_shape}",
                        row["id"],
                    )
                )
        if (
            primary
            and row["kind"]
            in {"linear", "reduce_mean", "reduce_sum", "select_final", "cumulative_sum"}
            and row["unit"] != primary.get("unit")
        ):
            out.diagnostics.append(
                Diagnostic(
                    "error",
                    "E205",
                    f"operation unit {row['unit']} is incompatible with source unit {primary.get('unit')}",
                    row["id"],
                )
            )
        if row["kind"] == "duration_normalise":
            mask_id = row["config"].get("mask")
            if primary and primary.get("unit") != "spike":
                out.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E206",
                        "spike-rate numerator must have unit spike",
                        row["id"],
                    )
                )
            if mask_id:
                mask = signals.get(mask_id)
                if mask and (
                    mask.get("signal_type") != "mask"
                    or mask.get("shape", []) != ["time", "batch"]
                ):
                    out.diagnostics.append(
                        Diagnostic(
                            "error",
                            "E207",
                            "valid-duration mask must have type mask and shape (time, batch)",
                            row["id"],
                        )
                    )

    roots = {s.partition(".")[0] for s in consumers}
    output_signals = {o["signal"] for o in graph.get("outputs", [])}
    observable_signals = {o["signal"] for o in graph.get("observables", [])}
    for row in graph.get("outputs", []) + graph.get("observables", []):
        if row["signal"] not in signals:
            out.diagnostics.append(
                Diagnostic(
                    "error", "E301", f"unresolved signal {row['signal']}", row["id"]
                )
            )
    for pop in population_ids:
        incoming = any(pop in targets for targets in adjacency.values())
        outgoing = bool(adjacency[pop]) or pop in roots
        if not incoming and not outgoing:
            out.diagnostics.append(
                Diagnostic("warning", "W101", "disconnected population", pop)
            )
    for row in graph.get("projections", []):
        target = row["target"].partition(".")[0]
        target_used = (
            target in roots
            or bool(adjacency.get(target))
            or any(
                s.startswith(target + ".") for s in output_signals | observable_signals
            )
        )
        if not target_used:
            out.diagnostics.append(
                Diagnostic(
                    "warning",
                    "W102",
                    "projection target has no downstream consumer or observation",
                    row["id"],
                )
            )
    return out


def validate_training(
    graph: Mapping[str, Any], training: Mapping[str, Any]
) -> ValidationResult:
    result = ValidationResult()
    parameters = {p["id"] for p in graph["parameters"]}
    output_signals = {o["signal"] for o in graph["outputs"]}
    signals = {f"{x['id']}.value" for x in graph["operations"]} | output_signals
    selected: dict[str, str] = {}
    for group in training.get("parameter_groups", []):
        for pid in group["parameters"]:
            if pid not in parameters:
                result.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E401",
                        f"unknown training parameter {pid}",
                        group["id"],
                    )
                )
            if pid in selected:
                result.diagnostics.append(
                    Diagnostic(
                        "error",
                        "E402",
                        f"parameter already selected by {selected[pid]}",
                        pid,
                    )
                )
            selected[pid] = group["id"]
    for objective in training.get("objectives", []):
        if objective["prediction"] not in output_signals:
            result.diagnostics.append(
                Diagnostic(
                    "error",
                    "E403",
                    f"objective prediction is not a named reachable output: {objective['prediction']}",
                )
            )
        if not objective.get("target"):
            result.diagnostics.append(
                Diagnostic("error", "E404", "objective target is empty")
            )
    for regularizer in training.get("regularizers", []):
        if regularizer["signal"] not in signals and not any(
            regularizer["signal"].startswith(p["id"] + ".")
            for p in graph["populations"]
        ):
            result.diagnostics.append(
                Diagnostic(
                    "error",
                    "E405",
                    f"regularizer signal is unresolved: {regularizer['signal']}",
                )
            )
    for signal in training.get("stop_gradients", []):
        if signal not in signals and not any(
            signal.startswith(p["id"] + ".") for p in graph["populations"]
        ):
            result.diagnostics.append(
                Diagnostic(
                    "error", "E406", f"stop-gradient signal is unresolved: {signal}"
                )
            )
    return result


def capability_report(graph: Mapping[str, Any], target: str | None) -> list[Diagnostic]:
    if target is None:
        return []
    supported = {
        "linear",
        "reduce_mean",
        "reduce_sum",
        "select_final",
        "duration_normalise",
        "cumulative_sum",
        "divide",
    }
    diagnostics = []
    vocabulary = "snnlang.capabilities/v1"
    neuron_support = {"coba_lif", "leaky_integrator"}
    synapse_support = {"ampa", "gaba", "leaky_integrator"}
    connection_support = {"feedforward", "recurrent", "feedback"}
    for population in graph["populations"]:
        kind = population["neuron"]["kind"]
        if kind not in neuron_support:
            diagnostics.append(
                Diagnostic(
                    "warning",
                    "C102",
                    f"{vocabulary}: {target} lacks neuron:{kind}",
                    population["id"],
                )
            )
    for projection in graph["projections"]:
        synapse = projection["synapse"]["kind"]
        connection = projection["connection"]
        if synapse not in synapse_support:
            diagnostics.append(
                Diagnostic(
                    "warning",
                    "C103",
                    f"{vocabulary}: {target} lacks synapse:{synapse}",
                    projection["id"],
                )
            )
        if connection not in connection_support:
            diagnostics.append(
                Diagnostic(
                    "warning",
                    "C104",
                    f"{vocabulary}: {target} lacks connection:{connection}",
                    projection["id"],
                )
            )
    for op in graph["operations"]:
        if op["kind"] not in supported:
            diagnostics.append(
                Diagnostic(
                    "warning",
                    "C101",
                    f"{target} capability for operation is unknown",
                    op["id"],
                )
            )
    return diagnostics


def capability_requirements(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical element-level requirements archived with every bundle."""
    elements = []
    for population in graph["populations"]:
        elements.append(
            {
                "element": population["id"],
                "features": [f"neuron:{population['neuron']['kind']}"],
            }
        )
    for projection in graph["projections"]:
        delay = projection.get("delay")
        elements.append(
            {
                "element": projection["id"],
                "features": [
                    f"synapse:{projection['synapse']['kind']}",
                    f"connection:{projection['connection']}",
                    "delay:none" if delay is None else "delay:explicit",
                ],
            }
        )
    for operation in graph["operations"]:
        elements.append(
            {"element": operation["id"], "features": [f"operation:{operation['kind']}"]}
        )
    for observable in graph["observables"]:
        elements.append(
            {
                "element": observable["id"],
                "features": [f"recording:{observable['signal'].partition('.')[2]}"],
            }
        )
    return {
        "schema": CAPABILITY_SCHEMA,
        "elements": sorted(elements, key=lambda row: row["element"]),
    }


def text_report(
    graph: Mapping[str, Any],
    training: Mapping[str, Any] | None,
    diagnostics: list[Diagnostic],
) -> str:
    params = graph["parameters"]
    count = sum(_shape_product(p["shape"]) for p in params)
    state_scalars = sum(p["size"] for p in graph["populations"])
    projection_edges = sum(
        _shape_product(
            next(p["shape"] for p in params if p["id"] == projection["parameters"][0])
        )
        for projection in graph["projections"]
    )
    selected = {
        p
        for group in (training or {}).get("parameter_groups", [])
        if not group["frozen"]
        for p in group["parameters"]
    }
    recurrent = sorted(
        p["id"]
        for p in graph["projections"]
        if p["connection"] in {"recurrent", "feedback"}
    )
    lines = [
        f"# snnlang report — {graph['name']}",
        "",
        f"Populations: {len(graph['populations'])} ({sum(p['size'] for p in graph['populations']):,} units)",
        f"Projections: {len(graph['projections'])}",
        f"Operations: {len(graph['operations'])}",
        f"Parameters: {len(params)} tensors / {count:,} scalars",
        f"Estimated state: {state_scalars:,} scalars per sample and timestep",
        f"Estimated dense projection edges: {projection_edges:,}",
        f"Trainable this recipe: {len(selected)} tensors",
        f"Outputs: {', '.join(o['id'] for o in graph['outputs']) or 'none'}",
        f"Recurrent paths: {', '.join(recurrent) or 'none'}",
        f"Diagnostics: {sum(d.severity == 'error' for d in diagnostics)} errors, {sum(d.severity == 'warning' for d in diagnostics)} warnings",
        "",
        "## Populations",
    ]
    lines += [
        f"- {p['id']}: {p['size']} × {p['neuron']['kind']} ({'spiking' if p['spiking'] else 'non-spiking'})"
        for p in graph["populations"]
    ]
    lines += ["", "## Projections"]
    lines += [
        f"- {p['id']}: {p['source']} → {p['target']} [{p['connection']}, {p['polarity']}]"
        for p in graph["projections"]
    ]
    lines += ["", "## Parameters"]
    lines += [
        f"- {p['id']}: {p['shape']} {p['unit']} ({'selected' if p['id'] in selected else 'frozen/unselected'})"
        for p in params
    ]
    if diagnostics:
        lines += ["", "## Diagnostics"] + [f"- {d.line()}" for d in diagnostics]
    return "\n".join(lines) + "\n"


def _shape_product(shape: list[Any]) -> int:
    result = 1
    for value in shape:
        if isinstance(value, int):
            result *= value
    return result


@dataclass
class Bundle:
    graph: dict[str, Any]
    training: dict[str, Any] | None
    manifest: dict[str, Any]
    diagnostics: list[Diagnostic]
    asset_sources: dict[str, Path] = field(default_factory=dict)

    def write(self, path: str | Path, *, visualise: bool = False) -> Path:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "graph.json").write_bytes(canonical_json(self.graph))
        if self.training:
            (root / "training.json").write_bytes(canonical_json(self.training))
        assets_dir = root / "assets"
        for name, source in sorted(self.asset_sources.items()):
            assets_dir.mkdir(exist_ok=True)
            shutil.copyfile(source, assets_dir / name)
        (root / "manifest.json").write_bytes(canonical_json(self.manifest))
        reports = root / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "summary.md").write_text(
            text_report(self.graph, self.training, self.diagnostics)
        )
        if visualise:
            from .visualize import visualise_bundle

            for view in ("circuit", "training", "expanded"):
                visualise_bundle(self, reports / f"{view}.svg", view=view)
                visualise_bundle(self, reports / f"{view}.png", view=view, scale=2)
        return root

    def visualise(
        self, path: str | Path, *, view: str = "circuit", scale: int = 1
    ) -> Path:
        from .visualize import visualise_bundle

        return visualise_bundle(self, Path(path), view=view, scale=scale)


def compile(
    network: Network,
    *,
    training: TrainSpec | None = None,
    target: str | None = None,
    assets: Mapping[str, str | Path] | None = None,
) -> Bundle:
    graph = graph_dict(network)
    graph_validation = validate_graph(graph)
    graph_validation.raise_for_errors()
    graph_digest = digest(graph)
    training_data = _training_dict(training, graph_digest) if training else None
    training_validation = (
        validate_training(graph, training_data) if training_data else ValidationResult()
    )
    training_validation.raise_for_errors()
    diagnostics = (
        graph_validation.diagnostics
        + training_validation.diagnostics
        + capability_report(graph, target)
    )
    asset_sources: dict[str, Path] = {}
    manifest_assets = []
    declarations = {a["id"]: a for a in graph["assets"]}
    for logical, source_value in sorted((assets or {}).items()):
        if logical not in declarations:
            raise ValueError(
                f"physical asset supplied for undeclared logical asset: {logical}"
            )
        source = Path(source_value)
        if not source.is_file():
            raise ValueError(f"asset does not exist: {source}")
        suffix = source.suffix
        bundled_name = logical + suffix
        content_digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
        asset_sources[bundled_name] = source
        manifest_assets.append(
            {"id": logical, "path": f"assets/{bundled_name}", "digest": content_digest}
        )
    missing = set(declarations) - set(assets or {})
    if missing:
        raise ValueError(
            f"no physical path supplied for assets: {', '.join(sorted(missing))}"
        )
    files = [{"path": "graph.json", "digest": graph_digest}]
    if training_data:
        files.append({"path": "training.json", "digest": digest(training_data)})
    files.extend({"path": x["path"], "digest": x["digest"]} for x in manifest_assets)
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "compiler": {"name": "snnlang", "version": "0.1.0"},
        "graph_digest": graph_digest,
        "target": target,
        "required_capabilities": capability_requirements(graph),
        "files": files,
        "assets": manifest_assets,
    }
    return Bundle(graph, training_data, manifest, diagnostics, asset_sources)


def load_bundle(path: str | Path) -> Bundle:
    root = Path(path)
    graph = json.loads((root / "graph.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    training_path = root / "training.json"
    training = json.loads(training_path.read_text()) if training_path.exists() else None
    if digest(graph) != manifest["graph_digest"]:
        raise ValueError("graph digest does not match manifest")
    asset_sources: dict[str, Path] = {}
    for file_entry in manifest.get("files", []):
        file_path = root / file_entry["path"]
        if not file_path.is_file():
            raise ValueError(f"bundle file is missing: {file_entry['path']}")
        if file_entry["path"].endswith(".json"):
            actual_digest = digest(json.loads(file_path.read_text()))
        else:
            actual_digest = (
                "sha256:" + hashlib.sha256(file_path.read_bytes()).hexdigest()
            )
        if actual_digest != file_entry["digest"]:
            raise ValueError(f"bundle file digest mismatch: {file_entry['path']}")
    for asset in manifest.get("assets", []):
        asset_sources[Path(asset["path"]).name] = root / asset["path"]
    validation = validate_graph(graph)
    if training:
        if training["graph_digest"] != manifest["graph_digest"]:
            raise ValueError("training specification targets a different graph")
        validation.diagnostics.extend(validate_training(graph, training).diagnostics)
    validation.raise_for_errors()
    return Bundle(graph, training, manifest, validation.diagnostics, asset_sources)
