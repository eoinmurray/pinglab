"""External-backend execution snapshots for exp080.

Capture the resolved configuration and numerical parameters before and after
one complete execution; these inspection records are not resumable checkpoints."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np
from pingstore.contracts import file_sha256, write_json_atomic

SCHEMA = "pinglab.external-run-snapshot/v1"


def _write(directory, stage, backend, configuration, parameters):
    directory = Path(directory) / stage
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {}
    for name, value in parameters.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arrays[name] = np.asarray(value).copy()
        if arrays[name].dtype.hasobject:
            raise ValueError(f"snapshot parameters must be numerical arrays: {name}")
    inventory = {
        name: {"key": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in arrays.items()
    }
    metadata = {
        "schema": SCHEMA,
        "backend": backend,
        "stage": stage,
        "configuration": configuration,
        "parameters_included": True,
        "parameters": inventory,
        "continuation": "inspection only; optimizer and RNG states are not retained",
    }
    if arrays:
        path = directory / "parameters.npz"
        np.savez_compressed(path, **arrays)
        metadata["parameter_file_sha256"] = file_sha256(path)
    write_json_atomic(directory / "snapshot.json", metadata)


@contextmanager
def external_execution(directory, *, backend, configuration, parameters=dict):
    """Save actual backend definitions/parameters around one complete execution."""
    _write(directory, "start", backend, configuration, parameters())
    yield
    _write(directory, "end", backend, configuration, parameters())
