"""Native SNNLab execution evidence for exp141.

Retain only the bundle, snapshot lifecycle and parameter readers used by this
experiment. SNNLab owns snapshot metadata and parameter serialization."""

import numpy as np
from pingstore.library import PingstoreError, file_sha256, load_json


def snapshot_parameters(source, *parts):
    """Read named arrays from a validated source's native snapshot inventory."""
    metadata = load_json(source.file(*parts, "snapshot.json"))
    if metadata.get("schema") != "snnlab.run-snapshot/v1" or not metadata.get(
        "parameters_included"
    ):
        raise PingstoreError("expected a native parameter snapshot")
    if not metadata["parameters"]:
        return {}
    path = source.file(*parts, "parameters.npz")
    if file_sha256(path) != metadata.get("parameter_file_sha256"):
        raise PingstoreError("snapshot parameter checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        values = {}
        for name, entry in metadata["parameters"].items():
            value = arrays[entry["key"]]
            if (
                list(value.shape) != entry["shape"]
                or str(value.dtype) != entry["storage_dtype"]
            ):
                raise PingstoreError(
                    f"snapshot parameter shape or dtype mismatch: {name}"
                )
            if entry["dtype"] == "bfloat16":
                raise PingstoreError("use SNNLab tensors to read bfloat16 parameters")
            values[name] = value.copy()
    return values
