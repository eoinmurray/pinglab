"""Lossless, non-pickled numerical arrays with a JSON structure index."""

import math

import numpy as np
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def write(directory, document):
    arrays = {}

    def pack(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise PingstoreError("exp033 arrays must be finite real numbers")
            name = f"a{len(arrays):04d}"
            arrays[name] = value
            return {"array": name}
        if isinstance(value, dict):
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [pack(v) for v in value]
        if isinstance(value, np.generic):
            return pack(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            raise PingstoreError("exp033 numerical scalars must be finite")
        return value

    index = pack(document)
    np.savez_compressed(directory / "arrays.npz", **arrays)
    write_json_atomic(directory / "evidence.json", index)


def read(directory):
    index = load_json(directory / "evidence.json")
    used = set()
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:

        def unpack(value):
            if isinstance(value, dict) and set(value) == {"array"}:
                name = value["array"]
                if name not in arrays:
                    raise PingstoreError("missing exp033 numerical array")
                result = arrays[name]
                if result.dtype.kind not in "biuf" or not np.isfinite(result).all():
                    raise PingstoreError("exp033 arrays must be finite real numbers")
                used.add(name)
                return result
            if isinstance(value, dict):
                return {k: unpack(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value):
                raise PingstoreError("exp033 numerical scalars must be finite")
            return value

        result = unpack(index)
        if used != set(arrays.files):
            raise PingstoreError("unreferenced exp033 numerical arrays")
    return result
