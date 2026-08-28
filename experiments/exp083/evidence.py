"""Experiment-specific payload validation, separate from checksum validation."""

import numpy as np
from pingstore.contracts import PingstoreError, load_json
from tools.snnlang.compiler import digest  # noqa: TID251

from . import recipe


def recording_arrays(arrays):
    sizes = {
        "input_spikes": recipe.N_INPUT,
        "e_spikes": recipe.N_E,
        "i_spikes": recipe.N_I,
    }
    if set(arrays) != set(sizes):
        raise PingstoreError("exp083 recording must retain input and both populations")
    for key, size in sizes.items():
        value = arrays[key]
        expected = (round(recipe.T_MS / recipe.DT_MS), len(recipe.TRIAL_SEEDS), size)
        if value.shape != expected or value.dtype != np.uint8 or np.any(value > 1):
            raise PingstoreError(f"invalid exp083 binary recording: {key}")


def recording(path):
    with np.load(path, allow_pickle=False) as raw:
        arrays = {key: raw[key] for key in raw.files}
    recording_arrays(arrays)
    return arrays


def graph_record(graph, manifest):
    if manifest.get("graph_digest") != digest(graph):
        raise PingstoreError("exp083 compiled graph digest differs")
    if graph.get("name") != "default_ping_drive_response":
        raise PingstoreError("exp083 graph name differs")
    return {"digest": manifest["graph_digest"], "name": graph["name"]}


def compute_payload(source):
    cfg = recipe.configuration()
    graph = load_json(source.export / "network.bundle/graph.json")
    manifest = load_json(source.export / "network.bundle/manifest.json")
    expected = {
        "schema": "exp083.compute/v1",
        "recipe": cfg,
        "graph": graph_record(graph, manifest),
        "conditions": recipe.conditions(),
    }
    if load_json(source.export / "evidence.json") != expected:
        raise PingstoreError("exp083 compute evidence grid, graph or recipe differs")
    if {p.name for p in (source.export / "conditions").iterdir()} != {
        f"rate-{rate:g}.npz" for rate in recipe.INPUT_RATES_HZ
    }:
        raise PingstoreError("exp083 recording grid is incomplete or unexpected")
    return expected, graph, manifest


def display_entries(kind, rates):
    return [
        {"input_rate_hz": rate, "file": f"{kind}/rate-{rate:g}.npz"} for rate in rates
    ]


def analysis_payload(source, compute):
    result = load_json(source.export / "results.json")
    original, _, _ = compute_payload(compute)
    graph = load_json(source.export / "network-graph.json")
    manifest = load_json(source.export / "network-manifest.json")
    expected = {
        "schema": "exp083.analysis/v1",
        "recipe": recipe.configuration(),
        "config": recipe.SCALE,
        "frequency_analysis": recipe.FREQUENCY_CONFIG.json(),
        "representative_rates_hz": list(recipe.REPRESENTATIVE_RATES_HZ),
        "graph": original["graph"],
        "rasters": display_entries("rasters", recipe.REPRESENTATIVE_RATES_HZ),
        "spectra": display_entries("spectra", recipe.INPUT_RATES_HZ),
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise PingstoreError("exp083 analysis recipe, graph or display grid differs")
    if graph_record(graph, manifest) != original["graph"]:
        raise PingstoreError("exp083 analysis graph differs from compute")
    conditions = result.get("conditions", [])
    if [row.get("input_rate_hz") for row in conditions] != list(recipe.INPUT_RATES_HZ):
        raise PingstoreError("exp083 analysis condition grid differs")
    for row in conditions:
        if [
            (r.get("trial"), r.get("seed"), r.get("input_rate_hz"))
            for r in row.get("trials", [])
        ] != [
            (index, seed, row["input_rate_hz"])
            for index, seed in enumerate(recipe.TRIAL_SEEDS)
        ]:
            raise PingstoreError("exp083 analysis trial grid differs")
    return result, graph, manifest


def display_arrays(source, entries, kind):
    values = {}
    for entry in entries:
        with np.load(source.export / entry["file"], allow_pickle=False) as raw:
            arrays = {key: raw[key] for key in raw.files}
        if kind == "spectra":
            if set(arrays) != {"frequencies_hz", "mean_psd"}:
                raise PingstoreError("exp083 spectrum keys differ")
            f, p = arrays["frequencies_hz"], arrays["mean_psd"]
            if f.ndim != 1 or p.shape != f.shape or np.any(np.diff(f) <= 0):
                raise PingstoreError("invalid exp083 spectrum dimensions")
            if (
                not np.all(np.isfinite(f))
                or not np.all(np.isfinite(p))
                or np.any(p < 0)
            ):
                raise PingstoreError("invalid exp083 spectrum values")
        else:
            if set(arrays) != {"e_t", "e_cells", "i_t", "i_cells"}:
                raise PingstoreError("exp083 raster coordinate keys differ")
            for population, count in (("e", recipe.N_E), ("i", recipe.N_I)):
                t, cells = arrays[f"{population}_t"], arrays[f"{population}_cells"]
                if t.ndim != 1 or t.shape != cells.shape:
                    raise PingstoreError("invalid exp083 raster coordinate dimensions")
                for value, maximum in (
                    (t, round(recipe.T_MS / recipe.DT_MS)),
                    (cells, count),
                ):
                    if (
                        value.dtype.kind not in "iu"
                        or np.any(value < 0)
                        or np.any(value >= maximum)
                    ):
                        raise PingstoreError("invalid exp083 raster coordinate bounds")
        values[entry["input_rate_hz"]] = arrays
    return values
