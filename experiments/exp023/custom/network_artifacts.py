"""Native SNNLab execution evidence for exp023.

Retain only the bundle, snapshot lifecycle and parameter readers used by this
experiment. SNNLab owns snapshot metadata and parameter serialization."""

from dataclasses import replace
from pathlib import Path

from snnlab import lang
from snnlab.sim import SnapshotSpec


def save_bundle(bundle, directory, *, visualise=False):
    """Write and authenticate the complete authoring product, including assets."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"network bundle already exists: {directory}")
    bundle.write(directory, visualise=visualise)
    lang.load_bundle(directory)
    return directory


def snapshot_request(spec, bundle, directory, *, visualise=False):
    """Attach native snapshots to public execution kwargs or a legacy request."""
    directory = Path(directory)
    if not isinstance(spec, dict) and spec.kind == "build":
        raise ValueError("build is not a simulation, training or inference execution")
    save_bundle(bundle, directory / "network.bundle", visualise=visualise)
    snapshot = SnapshotSpec(directory, stage="both", parameters=True)
    if isinstance(spec, dict):
        return {**spec, "snapshot": snapshot}
    return replace(spec, snapshot=snapshot)
