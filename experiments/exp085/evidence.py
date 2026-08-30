"""Experiment-specific recording contracts and lossless numerical serialization."""

import numpy as np
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.layout import canonical_export_file, canonical_export_unit

from . import recipe


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key]) for key in data.files}


def jobs(schedule):
    steps = round(recipe.T_MS / recipe.DT_MS)
    prc_steps = round(recipe.PRC_T_MS / recipe.DT_MS)
    onset = round(recipe.COUPLING_ONSET_MS / recipe.DT_MS)
    return [
        {"id": "uncoupled", "graph": "none", "steps": steps, "parent_state": None},
        {
            "id": "prc-baseline",
            "graph": "prc",
            "steps": prc_steps,
            "parent_state": None,
        },
        *(
            {"id": row["id"], "graph": "prc", "steps": prc_steps, "parent_state": None}
            for row in schedule
        ),
        {"id": "prefix", "graph": "none", "steps": onset, "parent_state": None},
        *(
            {
                "id": f"pathway-{name}",
                "graph": name,
                "steps": steps - onset,
                "parent_state": "prefix",
            }
            for name, *_ in recipe.PATHWAYS
        ),
    ]


def recording_sizes(job):
    """Channels consumed by the committed analysis for this job."""
    if job["id"] == "prefix":
        return {}
    sizes = {"population_0": recipe.N_E, "population_1": recipe.N_I}
    if job["graph"] != "prc":
        sizes.update(population_2=recipe.N_E, population_3=recipe.N_I)
    elif job["id"] == next(
        f"prc-I-{index:02d}"
        for index, fraction in enumerate(recipe.PRC_PHASE_FRACTIONS)
        if round(float(fraction), 2) == 0.12
    ):
        sizes.update(
            {
                "PING_A_I.voltage": recipe.N_I,
                "PING_A_E_to_I.conductance": recipe.N_I,
                "probe_E_to_PING_A_I_K_EI.conductance": recipe.N_I,
            }
        )
    if job["id"].startswith("pathway-"):
        sizes.pop("population_1")
        if job["id"] not in ("pathway-none", "pathway-e_to_e"):
            sizes.pop("population_3")
        else:
            sizes["PING_B_I_to_E.conductance"] = recipe.N_E
        if job["id"] == "pathway-e_to_e":
            sizes["PING_A_E_to_PING_B_E_K_EE.conductance"] = recipe.N_E
    return sizes


def recording(root, job):
    if job["id"] == "prefix":
        return {}
    data = arrays(canonical_export_file(root, "jobs", job["id"], "recording.npz"))
    for name, size in recording_sizes(job).items():
        value = data.get(name)
        if value is None or value.shape != (job["steps"], 1, size):
            raise PingstoreError(
                f"missing or invalid exp085 recording: {job['id']}/{name}"
            )
        if value.dtype.kind not in "buif" or not np.isfinite(value).all():
            raise PingstoreError(f"invalid exp085 recording values: {name}")
        if name.startswith("population_") and not ((value == 0) | (value == 1)).all():
            raise PingstoreError(f"exp085 spikes are not binary: {name}")
    return data


def compute_export(root, cfg):
    record = load_json(root / "evidence.json")
    if record.get("schema") != "exp085.compute/v1" or record.get("recipe") != cfg:
        raise PingstoreError("inconsistent exp085 compute evidence")
    cycle = record.get("reference_cycle", {})
    left, right = cycle.get("left_step"), cycle.get("next_step")
    if (
        type(left) is not int
        or type(right) is not int
        or not 0 <= left < right < round(recipe.PRC_T_MS / recipe.DT_MS)
    ):
        raise PingstoreError("invalid exp085 reference cycle")
    schedule = recipe.probe_schedule(left, right)
    expected = jobs(schedule)
    if record.get("probes") != schedule or record.get("jobs") != expected:
        raise PingstoreError("incomplete or inconsistent exp085 acquisition grid")
    for name, digest in cfg["graph_hashes"].items():
        if recipe.graph_digest(
            load_json(canonical_export_file(root, "graphs", f"{name}.json"))
        ) != digest:
            raise PingstoreError("exp085 graph differs from the recorded recipe")
    prefix = load_json(canonical_export_file(root, "prefix-state", "manifest.json"))
    if prefix.get("schema") != "tools/snnsim.graph-runtime-state/v1" or prefix.get(
        "completed_steps"
    ) != round(recipe.COUPLING_ONSET_MS / recipe.DT_MS):
        raise PingstoreError("missing or inconsistent exp085 branching state")
    # The state format has its own tensor hash in addition to Pingstore's digest.
    from pingstore.contracts import file_sha256

    if prefix.get("tensors_digest") != "sha256:" + file_sha256(
        canonical_export_file(root, "prefix-state", "tensors.npz")
    ):
        raise PingstoreError("exp085 branching state checksum differs")
    shared_drive = arrays(canonical_export_file(root, "jobs", "uncoupled", "inputs.npz"))
    prc_drive = arrays(canonical_export_file(root, "jobs", "prc-baseline", "inputs.npz"))
    probes = {row["id"]: row for row in schedule}
    onset = round(recipe.COUPLING_ONSET_MS / recipe.DT_MS)
    for job in expected:
        directory = canonical_export_unit(root, "jobs", job["id"])
        request = load_json(directory / "request.json")
        expected_request = {
            **job,
            "seed": recipe.NETWORK_SEED,
            "recording": "full",
            "graph_sha256": cfg["graph_hashes"][job["graph"]],
            "kind": "simulate",
            "executor": "graph",
        }
        if "recording_fields" in request:
            expected_request["recording_fields"] = list(recording_sizes(job))
        if request != expected_request:
            raise PingstoreError("exp085 simulation request differs")
        drive = arrays(directory / "inputs.npz")
        names = {f"drive_A_{recipe.INPUT_RATE_A_HZ:g}_Hz": recipe.N_INPUT}
        if job["graph"] == "prc":
            names.update(
                coupling_matched_pulse_to_E=recipe.N_E,
                coupling_matched_pulse_to_I=recipe.N_E,
            )
        else:
            names[f"drive_B_{recipe.INPUT_RATE_B_HZ:g}_Hz"] = recipe.N_INPUT
        if set(drive) != set(names):
            raise PingstoreError("incomplete exp085 recorded inputs")
        for name, size in names.items():
            value = drive[name]
            if (
                value.shape != (job["steps"], 1, size)
                or not ((value == 0) | (value == 1)).all()
            ):
                raise PingstoreError("invalid exp085 recorded inputs")
            if name.startswith("drive_"):
                reference = (
                    prc_drive[name] if job["graph"] == "prc" else shared_drive[name]
                )
                if job["id"] == "prefix":
                    reference = reference[:onset]
                elif job["parent_state"] == "prefix":
                    reference = reference[onset:]
                if not np.array_equal(value, reference):
                    raise PingstoreError(
                        "exp085 conditions must share recorded input trains"
                    )
            else:
                probe = probes.get(job["id"])
                pulse = np.zeros_like(value)
                if probe is not None and name.endswith("_" + probe["target"]):
                    pulse[
                        probe["arrival_step"]
                        - round(recipe.COUPLING_DELAY_MS / recipe.DT_MS),
                        0,
                    ] = 1
                if not np.array_equal(value, pulse):
                    raise PingstoreError(
                        "exp085 probe input differs from its acquisition schedule"
                    )
        parameters = arrays(directory / "parameters.npz")
        graph = load_json(
            canonical_export_file(root, "graphs", f"{job['graph']}.json")
        )
        # Graph declarations are target-by-source; the executor retains its
        # source-by-target matrices. Preserve those native tensors unchanged.
        shapes = {
            row["id"]: tuple(reversed(row["shape"])) for row in graph["parameters"]
        }
        if set(parameters) != set(shapes) or any(
            parameters[name].shape != shape
            or parameters[name].dtype.kind != "f"
            or not np.isfinite(parameters[name]).all()
            for name, shape in shapes.items()
        ):
            raise PingstoreError("missing or invalid exp085 initialized weights")
        recording(root, job)
    return record


def save_plot_data(root, data):
    saved = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            key = f"array_{len(saved):04d}"
            saved[key] = value
            return {"__array__": key}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        return value

    write_json_atomic(root / "plot-data.json", encode(data))
    np.savez_compressed(root / "plot-arrays.npz", **saved)


def load_plot_data(root):
    saved = arrays(root / "plot-arrays.npz")
    used = set()

    def decode(value):
        if isinstance(value, dict):
            if set(value) == {"__array__"}:
                key = value["__array__"]
                if key not in saved:
                    raise PingstoreError("missing exp085 plot array")
                used.add(key)
                return saved[key]
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        return value

    data = decode(load_json(root / "plot-data.json"))
    if used != set(saved):
        raise PingstoreError("unreferenced exp085 plot arrays")
    return data
