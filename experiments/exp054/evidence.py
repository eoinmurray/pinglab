"""Lossless storage and validation of current and retained scientific evidence."""

import json
import math
import zipfile

import numpy as np
from experiments.exp033 import evidence as mean_field_evidence
from experiments.exp033 import measurements as mean_field_measurements
from pingstore.contracts import PingstoreError, load_json, write_json_atomic

from . import recipe


def retained_mean_field(source, frequencies):
    """Validate theory carried by an existing immutable imported run."""
    if source.record["execution"]["operation"] != "historical-import":
        raise PingstoreError("retained exp054 theory requires imported exp033 evidence")
    old = load_json(source.export / "historical-numbers.json")
    with zipfile.ZipFile(source.export / "mean-field.zip") as archive:
        if archive.namelist() != ["numerical-evidence.json"]:
            raise PingstoreError("unexpected retained exp054 mean-field entries")
        subset = json.loads(archive.read("numerical-evidence.json"))
    mean_field_evidence.validate_summary(old, subset)
    info = source.record.get("historical_import", {})
    producer = info.get("cache_producer", {})
    if producer.get("experiment") != "exp054" or producer.get("job_id") != "33913631":
        raise PingstoreError("retained mean-field producer is not exp054")
    if source.record["inputs"].get("frequencies") != frequencies.reference:
        raise PingstoreError("retained exp054 theory has different frequency ancestry")
    medians = mean_field_measurements.spiking_medians(
        load_json(frequencies.export / "results.json")
    )
    deltas = {str(k): v - subset["spiking_exp041"][str(k)] for k, v in medians.items()}
    if deltas != info.get("frequency_deltas_hz"):
        raise PingstoreError("retained exp054 frequency delta differs")
    recomputed = mean_field_evidence.verify_amplitudes(
        subset["criticality"], subset["hopf"]["I_ext_star"]
    )
    return subset, {
        "source": source.reference,
        "original_producer": producer,
        "frequency_deltas_hz": deltas,
        "amplitude_recheck": recomputed,
        "policy": "retain scalars and overlay; no missing trajectories regenerated",
    }


def compare_retained_numbers(current, original):
    differences = {}

    def compare(a, b, path):
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                raise PingstoreError("exp054 retained fields differ: " + path)
            for key in a:
                compare(a[key], b[key], path + "/" + key)
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                raise PingstoreError("exp054 retained dimensions differ: " + path)
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                compare(x, y, path + "/" + str(i))
        elif a != b:
            is_contrast = "contrast" in path or path.endswith("null_max")
            if not (
                is_contrast
                and isinstance(a, (int, float))
                and isinstance(b, (int, float))
                and np.isfinite(a)
                and np.isfinite(b)
                and np.isclose(a, b, rtol=1e-12, atol=1e-15)
            ):
                raise PingstoreError(
                    "exp054 retained recordings do not reproduce " + path
                )
            differences[path] = a - b

    for key in ("config", "grid", "rate_invariance"):
        compare(current[key], original[key], key)
    return {
        "contrast_tolerance": {"rtol": 1e-12, "atol": 1e-15},
        "contrast_deltas": differences,
        "maximum_absolute_contrast_delta": max(
            map(abs, differences.values()), default=0.0
        ),
        "rates_and_configuration_exact": True,
        "policy": "preserve original scalars; retain independent numerical recheck",
    }


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


def simulation_config(record, cfg, item):
    expected = {
        "mode": "sim",
        "model": "ping",
        "input": "synthetic-spikes",
        "n_hidden": [cfg["n_e"]],
        "n_inh": cfg["n_i"],
        "n_batch": 1,
        "n_in": cfg["n_e"] if item["private"] else cfg["shared_n_in"],
        "t_ms": cfg["sim_ms"],
        "dt": cfg["dt_ms"],
        "seed": cfg["seed"],
        "spike_rate": item["rate_hz"],
        "w_ei_mean": item["wei"],
        "w_ie_mean": item["wie"],
        "private_w_in": item["private"],
        "w_in": [cfg["private_w_in"] if item["private"] else cfg["shared_w_in"]],
        "scale_w_in": 1.0,
        "scale_w_ei": 1.0,
        "scale_w_ie": 1.0,
        "dales_law": True,
        "recurrent_initial_zero_fraction": 0.0,
    }
    if not item["private"]:
        expected["w_in_initial_zero_fraction"] = cfg["shared_zero_fraction"]
    if any(record.get(k) != v for k, v in expected.items()):
        raise PingstoreError("exp054 simulation configuration differs from recipe")
    if (
        record.get("load_weights")
        or record.get("intervention")
        or record.get("scale_projection")
    ):
        raise PingstoreError("exp054 probes must be untrained and unperturbed")


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


def repack(source, destination):
    """Preserve every NPY member byte exactly; change ZIP compression only."""
    with (
        zipfile.ZipFile(source) as original,
        zipfile.ZipFile(
            destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9
        ) as packed,
    ):
        names = original.namelist()
        if len(names) != len(set(names)) or any(
            "/" in n or not n.endswith(".npy") for n in names
        ):
            raise PingstoreError("invalid exp054 NPZ members")
        for name in names:
            packed.writestr(name, original.read(name))
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination) as packed:
        if any(original.read(n) != packed.read(n) for n in original.namelist()):
            raise PingstoreError("exp054 repacking changed NPY bytes")


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
    if {path.name.removeprefix("probe--") for path in source.outputs.glob("probe--*")} != expected:
        raise PingstoreError("exp054 probe directory differs from recipe")
    for item in recipe.jobs(cfg):
        if {p.name for p in source.unit("probe", item["id"]).iterdir()} != {
            "rasters.npz"
        }:
            raise PingstoreError("unexpected exp054 probe payload")
    return cfg
