"""Native SNNLab execution evidence for exp022.

Retain only the bundle, snapshot lifecycle and parameter readers used by this
experiment. SNNLab owns snapshot metadata and parameter serialization."""

from contextlib import contextmanager
from pathlib import Path

from snnlab import lang
from snnlab.sim import SnapshotSpec
from snnlab.sim.artifacts.snapshots import SnapshotSession


def save_bundle(bundle, directory, *, visualise=False):
    """Write and authenticate the complete authoring product, including assets."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"network bundle already exists: {directory}")
    bundle.write(directory, visualise=visualise)
    lang.load_bundle(directory)
    return directory


@contextmanager
def model_execution(
    model,
    bundle,
    directory,
    *,
    kind="simulate",
    seed=0,
    protocol=None,
    settings=None,
    snapshot_stages=("start",),
):
    """Snapshot one scientific execution, including all its batches or time chunks.

    Use SNNLab's native lifecycle for this experiment's custom loop. Simulation
    retains start evidence only; training explicitly requests end evidence.
    SNNLab owns metadata and parameter serialization.
    """
    from snnlab.sim.requests.spec import ExecutionSpec

    if snapshot_stages not in (("start",), ("end",), ("start", "end")):
        raise ValueError("snapshot_stages must be start, end, or start/end")
    stage = "both" if len(snapshot_stages) == 2 else snapshot_stages[0]
    directory = Path(directory)
    save_bundle(bundle, directory / "network.bundle")
    device = str(next(iter(model.parameter_map().values())).device)
    spec = ExecutionSpec(
        kind=kind,
        graph=bundle.graph,
        training=bundle.training,
        seed=seed,
        device=device,
        protocol=protocol or {},
        options=settings or {},
        snapshot=SnapshotSpec(
            directory if stage == "both" else directory / stage,
            stage=stage,
            parameters=True,
        ),
    )
    session = SnapshotSession(
        spec,
        model,
        kind=kind,
        protocol=spec.protocol,
        training=bundle.training,
        device_resolved=device,
    )
    if "start" in snapshot_stages:
        session.capture("start")
    yield session
    if "end" in snapshot_stages:
        session.capture("end")


def record_snapshot_policy(run):
    """Declare native retention before Pingstore validates this stage's export."""
    import json

    policies = run.record.setdefault("scientific_artifacts", {}).setdefault(
        "snapshot_stages", {}
    )
    for path in run.export.rglob("snapshot.json"):
        metadata = json.loads(path.read_text())
        if metadata.get("schema") != "snnlab.run-snapshot/v1":
            continue
        root = path.parent.parent.relative_to(run.export).as_posix()
        policies[root] = ["end"] if metadata["kind"] == "train" else ["start"]
    save_circuits(run)


def copy_circuits(run):
    """Copy only circuit assets from this presentation's explicit compute ancestry."""
    import shutil
    from pingstore.stages import source_run

    seen = set()
    files = set()

    def visit(reference):
        identity = reference["run_id"]
        if identity in seen:
            return
        seen.add(identity)
        source = source_run(run.directory.parents[1], identity, reference=reference)
        if source.record["experiment"] != run.record["experiment"]:
            return
        if source.record["stage"] == "compute":
            for path in source.export.glob("circuit*.svg"):
                target = run.export / path.name
                if target.exists() and target.read_bytes() != path.read_bytes():
                    raise ValueError(f"conflicting pinned circuit: {path.name}")
                shutil.copyfile(path, target)
                files.add(path.name)
        else:
            for parent in source.record["inputs"].values():
                visit(parent)

    for reference in run.record["inputs"].values():
        visit(reference)
    return sorted(files)


def save_circuits(run):
    """Render once per topology from the saved compiled execution bundles."""
    import hashlib
    import json

    layouts = set()
    for path in sorted(run.export.rglob("manifest.json")):
        metadata = json.loads(path.read_text())
        if metadata.get("schema") != "snnlang.bundle/v1":
            continue
        bundle = lang.load_bundle(path.parent)
        graph = bundle.graph
        topology = dict(
            inputs=[(row["id"], row["shape"]) for row in graph["inputs"]],
            populations=[
                (row["id"], row["size"], row["neuron"]["kind"])
                for row in graph["populations"]
            ],
            projections=[
                (
                    row["source"],
                    row["target"],
                    row["synapse"]["kind"],
                    row["connection"],
                )
                for row in graph["projections"]
            ],
        )
        identity = hashlib.sha256(
            json.dumps(topology, sort_keys=True).encode()
        ).hexdigest()[:12]
        if identity not in layouts:
            bundle.visualise(run.export / f"circuit-{identity}.svg", view="circuit")
            layouts.add(identity)
