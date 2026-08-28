"""Validate exp086 evidence without running simulations or phase estimators."""

import numpy as np
from pingstore.contracts import PingstoreError, load_json
from snnlang import load_bundle

from . import recipe


def binary_arrays(path, shapes, dtype):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(shapes):
            raise PingstoreError(f"unexpected array keys: {path.name}")
        arrays = {key: archive[key] for key in shapes}
    for key, array in arrays.items():
        if (
            array.shape != shapes[key]
            or array.dtype != dtype
            or not np.all((array == 0) | (array == 1))
        ):
            raise PingstoreError(f"invalid binary array: {path.name}/{key}")
    return arrays


def recording_shapes(steps):
    return {
        f"population_{index}": (steps, 1, size)
        for index, size in enumerate((recipe.N_E, recipe.N_I, recipe.N_E, recipe.N_I))
    }


def compute_contract(source):
    cfg = recipe.configuration()
    if (
        source.record["inputs"]
        or source.record["execution"].get("configuration") != cfg
        or load_json(source.export / "evidence.json")
        != {
            "schema": "exp086.compute/v1",
            "recipe": cfg,
            "branches": recipe.branches(),
        }
    ):
        raise PingstoreError("exp086 compute recipe or branch grid differs")
    required = [
        "inputs.npz",
        "prefix-spikes.npz",
        "prefix-state/manifest.json",
        "prefix-state/tensors.npz",
    ]
    required.extend(f"branches/{b['label']}/spikes.npz" for b in recipe.branches())
    if any(not (source.export / name).is_file() for name in required):
        raise PingstoreError("exp086 compute evidence is incomplete")
    state = load_json(source.export / "prefix-state/manifest.json")
    if state.get("completed_steps") != round(recipe.COUPLING_ONSET_MS / recipe.DT_MS):
        raise PingstoreError("exp086 retained prefix has wrong duration")
    for branch in recipe.branches():
        bundle = load_bundle(
            source.export / "branches" / branch["label"] / "network.bundle"
        )
        if bundle.manifest["graph_digest"] != cfg["graphs"][branch["label"]]:
            raise PingstoreError("exp086 retained network differs from recipe")
    return cfg


def acquisition(source):
    """Validate the fixed drives and saved branch point without replaying it."""
    from execution import load_runtime_state

    steps = round(recipe.T_MS / recipe.DT_MS)
    onset = round(recipe.COUPLING_ONSET_MS / recipe.DT_MS)
    binary_arrays(
        source.export / "inputs.npz",
        {
            f"drive_A_{recipe.INPUT_RATE_A_HZ:g}_Hz": (steps, 1, recipe.N_INPUT),
            f"drive_B_{recipe.INPUT_RATE_B_HZ:g}_Hz": (steps, 1, recipe.N_INPUT),
        },
        np.float32,
    )
    binary_arrays(
        source.export / "prefix-spikes.npz", recording_shapes(onset), np.uint8
    )
    state = load_runtime_state(source.export / "prefix-state")
    if state.completed_steps != onset:
        raise PingstoreError("exp086 retained prefix has wrong duration")


def analysis(source, cfg):
    result = load_json(source.export / "results.json")
    if (
        source.record["execution"].get("configuration") != recipe.MEASUREMENT
        or result.get("schema") != "exp086.analysis/v1"
        or result.get("recipe") != cfg
        or result.get("measurement") != recipe.MEASUREMENT
        or [r.get("k") for r in result.get("trajectories", [])]
        != recipe.K_VALUES.tolist()
    ):
        raise PingstoreError("exp086 analysis recipe or trajectory grid differs")
    selected = result.get("selected_intermediate")
    if (
        selected not in result["trajectories"]
        or not 0 < selected["k"] < max(recipe.K_VALUES)
        or selected["phase_slips"] < 2
    ):
        raise PingstoreError("invalid exp086 selected intermediate")
    rows = []
    for summary in result["trajectories"]:
        with np.load(
            source.export / f"{recipe.label(summary['k'])}.npz", allow_pickle=False
        ) as archive:
            if set(archive.files) != set(recipe.ARRAY_KEYS):
                raise PingstoreError("exp086 analysis arrays are incomplete")
            arrays = {key: archive[key] for key in recipe.ARRAY_KEYS}
        if any(a.ndim != 1 or a.dtype.kind not in "fi" for a in arrays.values()):
            raise PingstoreError("invalid exp086 analysis array shape or dtype")
        phase_keys = (
            "time_ms",
            "wrapped_phase",
            "unwrapped_phase",
            "relative_velocity_rad_s",
            "relative_velocity_smoothed_rad_s",
        )
        if (
            len(arrays["time_ms"]) < 100
            or any(
                len(arrays[k]) != len(arrays["time_ms"])
                or not np.isfinite(arrays[k]).all()
                for k in phase_keys
            )
            or any(
                len(arrays[k]) != recipe.PHASE_BINS
                for k in (
                    "phase_bin_centres",
                    "phase_density",
                    "mean_velocity_by_phase",
                )
            )
        ):
            raise PingstoreError("exp086 phase coordinates are incomplete")
        steps = round((recipe.T_MS - recipe.COUPLING_ONSET_MS) / recipe.DT_MS)
        if any(
            len(arrays[k]) != steps or not np.isfinite(arrays[k]).all()
            for k in ("rate_e_a", "rate_i_a", "rate_e_b", "rate_i_b")
        ):
            raise PingstoreError("exp086 population-rate coordinates are incomplete")
        rows.append({**summary, **arrays})
    return result, rows
