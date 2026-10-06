"""Lossless storage and validation of current and retained scientific evidence."""

import math

import numpy as np
from pingstore.contracts import PingstoreError, load_json, write_json_atomic

from . import recipe


def write(directory, document):
    arrays = {}

    def pack(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf" or np.isinf(value).any():
                raise PingstoreError("invalid exp054 numerical array")
            name = f"a{len(arrays):04d}"
            arrays[name] = value
            return {"__array__": name}
        if isinstance(value, dict):
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(v) for v in value]
        if isinstance(value, np.generic):
            return pack(value.item())
        if isinstance(value, float) and math.isnan(value):
            return {"__float__": "nan"}
        if isinstance(value, float) and not math.isfinite(value):
            raise PingstoreError("invalid exp054 scalar")
        return value

    index = pack(document)
    np.savez_compressed(directory / "arrays.npz", **arrays)
    write_json_atomic(directory / "evidence.json", index)


def read(directory):
    used = set()
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:

        def unpack(value):
            if isinstance(value, dict) and set(value) == {"__array__"}:
                name = value["__array__"]
                if name not in arrays:
                    raise PingstoreError("missing exp054 array")
                a = arrays[name]
                if a.dtype.kind not in "biuf" or np.isinf(a).any():
                    raise PingstoreError("invalid exp054 numerical array")
                used.add(name)
                return a
            if value == {"__float__": "nan"}:
                return float("nan")
            if isinstance(value, dict):
                return {k: unpack(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value):
                raise PingstoreError("invalid exp054 scalar")
            return value

        document = unpack(load_json(directory / "evidence.json"))
        if used != set(arrays.files):
            raise PingstoreError("unreferenced exp054 arrays")
    return document


def raster(path, cfg):
    with np.load(path, allow_pickle=False) as archive:
        fields = {"dt", "T", "n_trials", "n_e", "n_i"} | {
            f"{prefix}_{field}"
            for prefix in ("e", "i", "out")
            for field in ("trial", "t", "cell")
        }
        compact = "recording_start_step" in archive.files
        if compact:
            fields -= {f"out_{field}" for field in ("trial", "t", "cell")}
            fields.add("recording_start_step")
        if len(archive.files) != len(fields) or set(archive.files) != fields:
            raise PingstoreError("unexpected exp054 raster fields")
        data = {key: archive[key] for key in fields}
    expected = {
        "dt": cfg["dt_ms"],
        "T": int(cfg["sim_ms"] / cfg["dt_ms"]),
        "n_trials": 1,
        "n_e": cfg["n_e"],
        "n_i": cfg["n_i"],
    }
    if compact:
        expected["recording_start_step"] = int(cfg["burn_ms"] / cfg["dt_ms"])
    for key, value in expected.items():
        a = data[key]
        if key == "dt" and a.dtype.kind == "f" and a.dtype.itemsize in (4, 8):
            # Compare at the recording's precision: 0.1 is not exact in float32.
            value = np.asarray(value, dtype=a.dtype).item()
        if a.shape != () or a.dtype.kind not in "iuf" or a.item() != value:
            raise PingstoreError("exp054 raster dimensions differ from recipe")
    for prefix, width in (("e", cfg["n_e"]), ("i", cfg["n_i"]), ("out", None)):
        if compact and prefix == "out":
            continue
        trial, times, cells = (data[f"{prefix}_{k}"] for k in ("trial", "t", "cell"))
        if any(a.ndim != 1 or a.dtype.kind not in "iu" for a in (trial, times, cells)):
            raise PingstoreError("exp054 sparse indices must be integer vectors")
        if not (trial.shape == times.shape == cells.shape):
            raise PingstoreError("exp054 sparse indices have unequal lengths")
        if (
            np.any(trial != 0)
            or np.any(times < expected.get("recording_start_step", 0))
            or np.any(times >= expected["T"])
            or np.any(cells < 0)
            or (width is not None and np.any(cells >= width))
        ):
            raise PingstoreError("exp054 spike index outside recording")
        if len(set(zip(trial.tolist(), times.tolist(), cells.tolist()))) != len(times):
            raise PingstoreError("duplicate exp054 spike index")
    return data


def compute_contract(source):
    cfg = recipe.validate(source.record["execution"]["configuration"])
    index = load_json(source.export / "recordings.json")
    if index != {
        "schema": "exp054.recordings/v1",
        "recipe": cfg,
        "jobs": recipe.jobs(cfg),
    }:
        raise PingstoreError("incomplete exp054 probe inventory")
    expected = {item["id"] for item in recipe.jobs(cfg)}
    actual = {
        path.name.removeprefix("probe--").removesuffix("--rasters.npz")
        for path in source.outputs.glob("probe--*--rasters.npz")
    }
    if actual != expected:
        raise PingstoreError("exp054 probe files differ from recipe")
    for item in recipe.jobs(cfg):
        if not source.file("probe", item["id"], "rasters.npz").is_file():
            raise PingstoreError("unexpected exp054 probe payload")
    return cfg
