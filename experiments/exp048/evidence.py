"""Validate trained checkpoint roles and retained stream recordings."""

import hashlib

import numpy as np
from experiments.helpers.checkpoints import public_provenance, resolve_checkpoint
from pingstore.contracts import PingstoreError, load_json

from . import recipe


def training_contract(root):
    configs, checkpoints = {}, []
    for seed in recipe.SEEDS:
        name = recipe.cell_name(seed)
        cfg = load_json(root / name / "config.json")
        expected = {
            "model": "ping",
            "dataset": "mnist",
            "seed": seed,
            "dt": recipe.DT,
            "t_ms": recipe.TRAINED_T_MS,
            "n_hidden": recipe.N_E,
            "n_inh": recipe.N_I,
            "n_in": recipe.N_IN,
            "n_out": recipe.N_CLASSES,
            "hidden_sizes": [recipe.N_E],
            "readout_mode": "mem-mean",
            "input_rate": recipe.INPUT_RATE_HZ,
            "ei_strength": 1.0,
            "fr_reg_upper_strength": 0.0,
            "epochs": 50,
            "max_samples": 7000,
        }
        for key, value in expected.items():
            if cfg.get(key) != value:
                raise PingstoreError(f"{name}: training {key} differs from recipe")
        split = cfg.get("dataset_split", {})
        if (
            split.get("checkpoint_selection_partition") != "validation"
            or split.get("official_test_used_during_training") is not False
        ):
            raise PingstoreError("exp048 requires held-out checkpoint selection")
        tau = float(cfg.get("tau_out_ms", 2.0))
        if not np.isfinite(tau) or tau <= 0:
            raise PingstoreError("invalid output time constant")
        try:
            checkpoint = public_provenance(
                resolve_checkpoint(root / name, recipe.CHECKPOINT_ROLE)
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            raise PingstoreError(str(exc)) from exc
        if checkpoint["training_cell"] != name or not 1 <= checkpoint["epoch"] <= 50:
            raise PingstoreError("checkpoint cell or epoch differs")
        configs[name] = cfg
        checkpoints.append(checkpoint)
    return {"configs": configs, "checkpoints": checkpoints}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as raw:
        return {k: raw[k] for k in raw.files}


def recording(path, job):
    data = load_arrays(path)
    steps = sum(round(tau / recipe.DT) for tau, _ in job["segments"])
    for key, value in {
        "T": steps,
        "n_e": recipe.N_E,
        "n_i": recipe.N_I,
        "n_trials": 1,
    }.items():
        a = data[key]
        if a.shape != () or a.dtype.kind not in "iu" or int(a) != value:
            raise PingstoreError(f"invalid stream {key}")
    if data["dt"].shape != () or not np.isclose(float(data["dt"]), recipe.DT):
        raise PingstoreError("stream timestep differs")
    for prefix, population in (
        ("e", recipe.N_E),
        ("i", recipe.N_I),
        ("out", recipe.N_CLASSES),
    ):
        coords = [data[f"{prefix}_{k}"] for k in ("trial", "t", "cell")]
        if any(a.ndim != 1 or a.dtype.kind not in "iu" for a in coords):
            raise PingstoreError("invalid spike coordinates")
        if not len({len(a) for a in coords}) == 1:
            raise PingstoreError("spike coordinate lengths differ")
        for a, limit in zip(coords, (1, steps, population)):
            if np.any(a < 0) or np.any(a >= limit):
                raise PingstoreError("spike coordinate outside recording")
        if len(np.unique(coords[1] * population + coords[2])) != len(coords[1]):
            raise PingstoreError("duplicate spike coordinates")
    return data


def stimulus(path, job):
    data = load_arrays(path)
    n = len(job["segments"])
    labels, pixels = data["labels"], data["pixels"]
    if (
        labels.shape != (n,)
        or labels.dtype.kind not in "iu"
        or np.any(labels < 0)
        or np.any(labels >= recipe.N_CLASSES)
        or pixels.shape != (n, recipe.N_IN)
        or not np.isfinite(pixels).all()
        or np.any(pixels < 0)
        or np.any(pixels > 1)
    ):
        raise PingstoreError("invalid stimulus labels or pixels")
    return data


def stream(directory, job, index):
    meta = load_json(directory / "stream.json")
    if (
        meta.get("job") != job
        or meta.get("stream") != index
        or meta.get("poisson_seed") != job["poisson_seed"] + index
    ):
        raise PingstoreError("stream recipe or random seed differs")
    raw = recording(directory / "rasters.npz", job)
    data = stimulus(directory / "stimulus.npz", job)
    shape = (int(raw["T"]), 1, recipe.N_IN)
    if meta.get("input_shape") != list(shape) or meta.get("input_dtype") != "float32":
        raise PingstoreError("input shape or dtype differs")
    tt, cc = data["input_t"], data["input_cell"]
    if (
        tt.ndim != 1
        or cc.shape != tt.shape
        or tt.dtype.kind not in "iu"
        or cc.dtype.kind not in "iu"
        or np.any(tt < 0)
        or np.any(tt >= shape[0])
        or np.any(cc < 0)
        or np.any(cc >= recipe.N_IN)
        or len(np.unique(tt * recipe.N_IN + cc)) != len(tt)
    ):
        raise PingstoreError("invalid retained input coordinates")
    # Reconstruct exact input bytes, not a new stochastic draw.
    arr = np.zeros(shape, dtype=np.float32)
    arr[tt, 0, cc] = 1
    if hashlib.sha256(arr.tobytes()).hexdigest() != meta.get("input_sha256"):
        raise PingstoreError("retained input hash differs")
    return raw, data


def dense(data, prefix, width):
    result = np.zeros((int(data["T"]), width), dtype=np.int8)
    result[data[f"{prefix}_t"], data[f"{prefix}_cell"]] = 1
    return result


def readout(value):
    if (
        value.shape != (recipe.N_E, recipe.N_CLASSES)
        or value.dtype.kind != "f"
        or not np.isfinite(value).all()
    ):
        raise PingstoreError("invalid trained readout matrix")


def simulation_configuration(config, train, arguments):
    for key in ("model", "dt", "seed", "n_in", "readout_mode", "ei_strength"):
        if config.get(key) != train[key]:
            raise PingstoreError(f"simulator configuration differs: {key}")
    if config.get("n_hidden") not in (train["n_hidden"], [train["n_hidden"]]):
        raise PingstoreError("simulator hidden population differs")
    for flag, key in (
        ("--load-config", "load_config"),
        ("--load-weights", "load_weights"),
        ("--input-file", "input_file"),
    ):
        if config.get(key) != arguments[arguments.index(flag) + 1]:
            raise PingstoreError(f"simulator input identity differs: {key}")


def analysis_rows(result):
    """Validate saved grids without recalculating estimators."""
    for kind, key in (
        ("tau", "tau_sweep_per_seed"),
        ("grid", "grid_sweep_per_seed"),
        ("low", None),
    ):
        jobs = [j for j in recipe.jobs() if j["kind"] == kind]
        if kind == "tau":
            jobs = [j for j in jobs if not j["rate_compensate"]] + [
                j for j in jobs if j["rate_compensate"]
            ]
        rows = (
            result[key]
            if key
            else result["encoding_rate_psychometric"]["per_seed_new_cells"]
        )
        if len(rows) != len(jobs):
            raise PingstoreError("analysis condition grid is incomplete")
        for row, job in zip(rows, jobs):
            expected = {
                "train_seed": job["seed"],
                "tau_ms": job["segments"][0][0],
                "input_rate_hz": job["segments"][0][1],
                "n_total": job["streams"] * len(job["segments"]),
            }
            if kind == "tau":
                expected.update(
                    rate_compensate=job["rate_compensate"],
                    n_streams=job["streams"],
                    n_per_stream=len(job["segments"]),
                )
            if any(row.get(k) != v for k, v in expected.items()):
                raise PingstoreError("analysis condition or count differs")
            correct = row.get("n_correct")
            if type(correct) is not int or not 0 <= correct <= row["n_total"]:
                raise PingstoreError("invalid segment accuracy count")
            scale, field = (1, "accuracy") if kind == "low" else (100, "acc")
            if not np.isclose(row[field], scale * correct / row["n_total"]):
                raise PingstoreError("analysis accuracy and counts differ")
    grids = (
        (
            result["grid_sweep_agg"],
            [
                (t, r)
                for t in sorted(recipe.TAU_GRID_MS)
                for r in sorted(recipe.RATE_GRID_HZ)
            ],
            lambda r: (r["tau_ms"], r["input_rate_hz"]),
        ),
        (
            result["tau_sweep_agg"],
            [(t, flag) for flag in (False, True) for t in sorted(recipe.TAU_SWEEP_MS)],
            lambda r: (r["tau_ms"], r["rate_compensate"]),
        ),
    )
    for rows, keys, key in grids:
        if [key(row) for row in rows] != keys:
            raise PingstoreError("analysis aggregate grid is incomplete")
    psycho = result["encoding_rate_psychometric"]
    expected = {
        "presentation_ms": recipe.TRAINED_T_MS,
        "new_rates_hz": recipe.LOW_RATE_HZ,
        "trained_rate_hz": recipe.INPUT_RATE_HZ,
        "new_streams_per_seed": recipe.LOW_RATE_STREAMS,
        "digits_per_stream": recipe.LOW_RATE_DIGITS_PER_STREAM,
    }
    if any(psycho.get(k) != v for k, v in expected.items()) or [
        r["input_rate_hz"] for r in psycho["curve"]
    ] != sorted(recipe.LOW_RATE_HZ + recipe.RATE_GRID_HZ):
        raise PingstoreError("psychometric curve is incomplete or mismatched")
    for rows, field, error, scale in (
        (result["grid_sweep_agg"], "acc", "acc_sem", 100),
        (result["tau_sweep_agg"], "acc", "acc_sem", 100),
        (psycho["curve"], "accuracy", "accuracy_sem", 1),
    ):
        for row in rows:
            if (
                row["n_seeds"] != len(recipe.SEEDS)
                or not np.isfinite(row[field])
                or not 0 <= row[field] <= scale
                or not np.isfinite(row[error])
                or row[error] < 0
            ):
                raise PingstoreError("invalid aggregate accuracy or uncertainty")


def analysis_figure(data, job, summary):
    steps = [round(tau / recipe.DT) for tau, _ in job["segments"]]
    total, n = sum(steps), len(steps)
    if (
        int(data["T_stream_steps"]) != total
        or not np.array_equal(data["segment_steps"], steps)
        or not np.array_equal(data["segments"], job["segments"])
        or not np.array_equal(data["seg_ends"], np.cumsum(steps) - 1)
    ):
        raise PingstoreError("analysis figure timing differs")
    for key, shape in (
        ("spk_e", (total, recipe.N_E)),
        ("spk_i", (total, recipe.N_I)),
        ("pixels", (n, recipe.N_IN)),
        ("probs", (total, recipe.N_CLASSES)),
        ("labels", (n,)),
        ("seg_preds", (n,)),
        ("seg_correct", (n,)),
        ("pred_per_t", (total,)),
    ):
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise PingstoreError(f"invalid analysis figure {key}")
    for key in ("spk_e", "spk_i", "seg_correct"):
        if not np.all((data[key] == 0) | (data[key] == 1)):
            raise PingstoreError("nonbinary figure spikes or correctness")
    if np.any(data["probs"] < 0) or not np.allclose(data["probs"].sum(axis=1), 1):
        raise PingstoreError("invalid class probabilities")
    for key in ("labels", "seg_correct"):
        if not np.array_equal(data[key], summary[key]):
            raise PingstoreError("figure and numerical headline disagree")
    if job["kind"] == "varying" and not np.array_equal(
        data["seg_preds"], summary["seg_preds"]
    ):
        raise PingstoreError("figure predictions disagree")
