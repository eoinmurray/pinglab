"""Validate explicit grid coverage, native definition and binary acquisition."""

import hashlib

import numpy as np
from experiments.helpers.network_artifacts import snapshot_parameters
from pingstore.contracts import PingstoreError, load_json
from snnlab.sim import load_runtime_state

from . import inputs, recipe


def parameter_reference(source):
    """Read the fixed 128-channel comparison's full-strength native parameters."""
    cfg = source.record["execution"].get("configuration", {})
    expected = recipe.configuration()
    excluded = {"input_channels", "bundle_digest"}
    if {k: v for k, v in cfg.items() if k not in excluded} != {
        k: v for k, v in expected.items() if k not in excluded
    }:
        raise PingstoreError("parameter reference differs beyond input channel count")
    values = snapshot_parameters(
        source, f"{recipe.label(recipe.BASE_K_US)}-execution", "start"
    )
    if any(
        values[f"PING_{name}_input_to_E.weight"].shape != (128, recipe.N_E)
        for name in ("A", "B")
    ):
        raise PingstoreError("parameter reference must have 128 input channels")
    return values


def configuration(source):
    cfg = source.record["execution"].get("configuration")
    if cfg != recipe.configuration():
        raise PingstoreError("run does not match the committed Arnold-tongue recipe")
    return cfg


def recording(source, name, steps):
    with np.load(source.file(f"{name}--spikes.npz"), allow_pickle=False) as raw:
        values = dict(raw)
    expected = {
        f"{name}_{pop}": (steps, len(recipe.INPUT_DIFFERENCES_HZ), count)
        for name in ("A", "B")
        for pop, count in (("E", recipe.N_E), ("I", recipe.N_I))
    }
    if set(values) != set(expected):
        raise PingstoreError("incomplete recorded populations")
    for key, shape in expected.items():
        value = values[key]
        if (
            value.shape != shape
            or value.dtype != np.uint8
            or not ((value == 0) | (value == 1)).all()
        ):
            raise PingstoreError(f"invalid spike array {name}/{key}")
    return values


def compute_contract(source):
    cfg = configuration(source)
    pins = source.record["inputs"]
    if set(pins) not in (set(), {"parameter_reference"}):
        raise PingstoreError("unexpected grid compute inputs")
    preserved = None
    if pins:
        pin = pins["parameter_reference"]
        reference = inputs.source(
            source.directory.parents[2], pin["run_id"], "compute", reference=pin
        )
        preserved = parameter_reference(reference)
    mapping = source.record["scientific_artifacts"]["native_paths"]
    manifest = load_json(source.export / mapping["paired-ping.bundle/manifest.json"])
    if manifest["graph_digest"] != cfg["bundle_digest"]:
        raise PingstoreError("retained network differs from recipe")
    onset = round(recipe.COUPLING_ONSET_MS / recipe.DT_MS)
    state = load_runtime_state(source.unit("prefix-state"))
    if state.completed_steps != onset:
        raise PingstoreError("branch point has incorrect duration")
    recording(source, "prefix", onset)
    input_descriptors = {"prefix": {}, "suffix": {}}
    with np.load(source.export / "inputs.npz", allow_pickle=False) as raw:
        if set(raw.files) != set(recipe.rates()):
            raise PingstoreError("missing input trains")
        shape = (
            round(recipe.T_MS / recipe.DT_MS),
            len(recipe.INPUT_DIFFERENCES_HZ),
            recipe.N_INPUT,
        )
        for key in raw.files:
            value = raw[key]
            if (
                value.shape != shape
                or value.dtype != np.uint8
                or not ((value == 0) | (value == 1)).all()
            ):
                raise PingstoreError("invalid retained input trains")
            for part, selection in (
                ("prefix", slice(None, onset)),
                ("suffix", slice(onset, None)),
            ):
                selected = value[selection]
                digest = hashlib.sha256()
                for offset in range(0, len(selected), 1000):
                    digest.update(
                        selected[offset : offset + 1000].astype(np.float32).tobytes()
                    )
                input_descriptors[part][key] = dict(
                    shape=list(selected.shape),
                    dtype="float32",
                    sha256=digest.hexdigest(),
                )
    expected = {"prefix", *(recipe.label(k) for k in recipe.K_VALUES_US)}
    executions = source.record["execution"].get("simulations", {})
    if set(executions) != expected:
        raise PingstoreError("incomplete coupling grid")
    reference_local = None
    reference_masks = None
    for name in sorted(expected):
        item = executions[name]
        k = (
            0.0
            if name == "prefix"
            else float(
                recipe.K_VALUES_US[
                    [recipe.label(v) for v in recipe.K_VALUES_US].index(name)
                ]
            )
        )
        prefix = name == "prefix"
        if (
            item.get("coupling_us") != k
            or item.get("batches") != len(recipe.INPUT_DIFFERENCES_HZ)
            or item.get("steps") != (onset if prefix else shape[0] - onset)
            or item.get("continued_from_prefix") != (not prefix)
        ):
            raise PingstoreError(
                "execution coverage disagrees with the scientific grid"
            )
        for stage in ("start", "end"):
            native = f"{name}-execution/{stage}"
            metadata = load_json(source.export / mapping[f"{native}/snapshot.json"])
            request = metadata["request"]
            actual_inputs = {
                row["input_id"]: row["value"] for row in request["input_bindings"]
            }
            if actual_inputs != input_descriptors[
                "prefix" if prefix else "suffix"
            ] or request["options"].get("inference_overrides", {}).get(
                "projection_scales"
            ) != recipe.coupling_scales(k):
                raise PingstoreError(
                    "native request did not use the paired inputs and coupling scale"
                )
            values = snapshot_parameters(source, f"{name}-execution", stage)
            cross_names = {f"{cross}.weight" for cross in recipe.cross_ids()}
            if preserved is not None:
                for key, original in preserved.items():
                    if key in cross_names:
                        if k > 0 and not np.array_equal(
                            values[key] != 0, original != 0
                        ):
                            raise PingstoreError(
                                "cross-network mask differs from reference"
                            )
                    elif key in {f"PING_{n}_input_to_E.weight" for n in ("A", "B")}:
                        factor = recipe.N_INPUT // original.shape[0]
                        target = np.tile(original, (factor, 1)) / factor
                        if values[key].shape != target.shape or not np.allclose(
                            values[key], target, atol=1e-9, rtol=1e-6
                        ):
                            raise PingstoreError(
                                "input weights do not preserve reference summed drive"
                            )
                    elif not np.array_equal(values[key], original):
                        raise PingstoreError(
                            "recurrent weights differ from parameter reference"
                        )
            local = {
                key: value for key, value in values.items() if key not in cross_names
            }
            if reference_local is None:
                reference_local = local
            if set(local) != set(reference_local) or any(
                not np.array_equal(value, reference_local[key])
                for key, value in local.items()
            ):
                raise PingstoreError("local parameters changed between paired requests")
            masks = {}
            for key in cross_names:
                value = values[key]
                if not np.allclose(value.sum(axis=0), k, atol=1e-7, rtol=0):
                    raise PingstoreError(
                        "cross-network conductance differs from requested coupling"
                    )
                if k > 0:
                    masks[key] = value != 0
                    if not (masks[key].sum(axis=0) == 8).all():
                        raise PingstoreError(
                            "cross-network fan-in is not exactly eight"
                        )
            if k > 0:
                if reference_masks is None:
                    reference_masks = masks
                if any(
                    not np.array_equal(value, reference_masks[key])
                    for key, value in masks.items()
                ):
                    raise PingstoreError(
                        "cross-network connectivity changed between requests"
                    )
    return cfg
