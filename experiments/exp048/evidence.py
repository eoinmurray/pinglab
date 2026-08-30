"""Validate trained checkpoint roles and retained stream recordings."""

import copy
import hashlib
import math
import shutil
import time

import numpy as np
from experiments.helpers.checkpoints import public_provenance, resolve_checkpoint
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)

from . import inputs, measurements, plots, recipe


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


SOURCE = (
    "r2:pinglab/datasets/gamma-gated-sparsity/baseline-20260826/"
    "experiment-runs/exp048/exp048--r003"
)
SOURCE_HASHES = {
    "run.json": "2e1c30d9466d455328e4941471001c9dda316b53dc39559379de68443638ca4d",
    "inventory.json": "0488f5f7e156479ee04f0488650095cdf3bdd95cf88428daaa97bacd87cc11b4",
}
IMPORT = {"schema": "exp048.historical-import/v1", "evidence": "per-seed summaries"}
ANALYSIS = {
    "schema": "exp048.historical-analysis/v1",
    "aggregation": "original seed mean and sample SEM",
}
PRESENTATION = {"schema": "exp048.historical-presentation/v1"}
CARRIED = recipe.FIGURES[:4]
PAYLOAD_NAMES = {
    *recipe.FIGURES,
    "numbers.json",
    "_manifest.json",
    "_dirty.patch",
    "_run.txt",
}
REMOVED_REPLAY_SCRIPT = {
    "path": "run.sh",
    "role": "state",
    "sha256": "8689ec49b3d1a382a76e392db184fbe042535180891984e0df4cb689df7b798f",
    "size_bytes": 247,
}
LEGACY_CONFIG_KEYS = (
    "n_e",
    "n_i",
    "n_in",
    "n_classes",
    "dt",
    "trained_t_ms",
    "tau_headline_ms",
    "n_digits_headline",
    "tau_sweep_ms",
    "tau_grid_ms",
    "rate_grid_hz",
    "input_rate_hz",
    "n_streams",
    "n_grid_streams",
    "n_per_stream",
    "train_seeds",
    "seed",
)


def legacy_configuration():
    config = recipe.configuration()
    return {key: config[key] for key in LEGACY_CONFIG_KEYS}


def validate_numbers(numbers):
    if numbers.get("config") != legacy_configuration():
        raise PingstoreError("historical recipe differs")
    analysis_rows(numbers)
    if (
        numbers["encoding_rate_psychometric"].get("migration_source")
        != "exp065 initial computation"
    ):
        raise PingstoreError("historical low-rate attribution differs")
    for key in ("headline", "varying_headline"):
        row = numbers[key]
        n = (
            recipe.N_DIGITS_HEADLINE
            if key == "headline"
            else len(recipe.VARYING_HEADLINE)
        )
        labels, correct = row["labels"], row["seg_correct"]
        if (
            len(labels) != n
            or len(correct) != n
            or any(type(x) is not int or not 0 <= x < recipe.N_CLASSES for x in labels)
            or any(type(x) is not int or x not in (0, 1) for x in correct)
        ):
            raise PingstoreError("historical headline labels or counts differ")
        if key == "varying_headline":
            preds = row["seg_preds"]
            if (
                row["segments"] != [list(s) for s in recipe.VARYING_HEADLINE]
                or len(preds) != n
                or any(
                    type(x) is not int or not 0 <= x < recipe.N_CLASSES for x in preds
                )
                or [int(a == b) for a, b in zip(labels, preds)] != correct
            ):
                raise PingstoreError("historical varying headline differs")


def archive_files(directory):
    expected = {*SOURCE_HASHES, *(f"payload/{name}" for name in PAYLOAD_NAMES)}
    paths = list(directory.rglob("*"))
    if directory.is_symlink() or any(p.is_symlink() for p in paths):
        raise PingstoreError("historical archive contains symlinks")
    if {p.relative_to(directory).as_posix() for p in paths if p.is_file()} != expected:
        raise PingstoreError("historical archive file set differs")
    for name, digest in SOURCE_HASHES.items():
        if file_sha256(directory / name) != digest:
            raise PingstoreError(f"approved source metadata changed: {name}")
    inventory = load_json(directory / "inventory.json")
    original_rows = inventory["files"]
    if [row for row in original_rows if row.get("path") == "run.sh"] != [
        REMOVED_REPLAY_SCRIPT
    ]:
        raise PingstoreError("historical replay-script inventory differs")
    rows = [row for row in original_rows if row.get("path") != "run.sh"]
    if len(rows) != len(PAYLOAD_NAMES) or {r["path"] for r in rows} != PAYLOAD_NAMES:
        raise PingstoreError("historical payload inventory differs")
    for row in rows:
        path = directory / "payload" / row["path"]
        if (
            path.stat().st_size != row["size_bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise PingstoreError(f"historical payload checksum differs: {row['path']}")
    if inventory["file_count"] != len(original_rows) or inventory[
        "total_size_bytes"
    ] != sum(r["size_bytes"] for r in original_rows):
        raise PingstoreError("historical inventory totals differ")
    original = load_json(directory / "run.json")
    manifest = load_json(directory / "payload/_manifest.json")
    numbers = load_json(directory / "payload/numbers.json")
    if (
        original["experiment"] != recipe.SLUG
        or original["run_id"] != "exp048/r003"
        or original["execution"]["host"] != "local"
        or manifest["run_id"] != "r003"
        or numbers["notebook_run_id"] != "r001"
    ):
        raise PingstoreError("historical producer identity differs")
    validate_numbers(numbers)
    return {
        name: {
            "sha256": file_sha256(directory / name),
            "size_bytes": (directory / name).stat().st_size,
        }
        for name in sorted(expected)
    }


def provenance(directory, files):
    original = load_json(directory / "run.json")
    numbers = load_json(directory / "payload/numbers.json")
    return {
        "schema": "exp048.historical-provenance/v1",
        "gold_2": False,
        "source_uri": SOURCE,
        "source_files": files,
        "source_bytes": sum(row["size_bytes"] for row in files.values()),
        "original_producer": original["execution"],
        "original_code": original["source"],
        "archive_identity": original["run_id"],
        "numbers_identity": numbers["notebook_run_id"],
        "recorded_duration_s": numbers["duration_s"],
        "low_rate_attribution": numbers["encoding_rate_psychometric"][
            "migration_source"
        ],
        "checkpoint_lineage": "unresolved; no operational bank reference asserted",
        "raw_streams_available": False,
        "compute_reproduced": False,
        "identity_discrepancy": "archive r003 versus numerical record r001; unresolved",
    }


def imported(repo, source):
    if (
        source.record["experiment"] != recipe.SLUG
        or source.record["stage"] != "analyse"
        or source.record["inputs"]
        or source.record["execution"]["operation"] != "historical-import"
        or source.record["execution"]["configuration"] != IMPORT
    ):
        raise PingstoreError("not an explicit historical-summary import")
    history = source.record.get("historical")
    if not isinstance(history, dict) or history.get("schema") != "exp048.historical-provenance/v1":
        raise PingstoreError("historical provenance differs")
    if (
        file_sha256(source.export / "numbers.json")
        != history["source_files"]["payload/numbers.json"]["sha256"]
    ):
        raise PingstoreError("imported numerical bytes differ")
    for name in CARRIED:
        if file_sha256(source.export / name) != history["source_files"][f"payload/{name}"]["sha256"]:
            raise PingstoreError(f"imported carried figure differs: {name}")
    return load_json(source.export / "numbers.json"), history


def equivalent(actual, expected):
    """Check replay against saved values, including original floating precision."""
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(equivalent(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(equivalent(a, b) for a, b in zip(actual, expected))
        )
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12
        )
    return type(actual) is type(expected) and actual == expected


def aggregate(numbers):
    result = copy.deepcopy(numbers)
    rows = numbers["tau_sweep_per_seed"]
    result["tau_sweep_agg"] = sum(
        (
            measurements.aggregate_tau_rows(
                [r for r in rows if r["rate_compensate"] == flag]
            )
            for flag in (False, True)
        ),
        [],
    )
    result["grid_sweep_agg"] = measurements.aggregate_grid_rows(
        numbers["grid_sweep_per_seed"]
    )
    low = []
    for rate in recipe.LOW_RATE_HZ:
        rows = [
            r
            for r in numbers["encoding_rate_psychometric"]["per_seed_new_cells"]
            if r["input_rate_hz"] == rate
        ]
        values = np.array([r["accuracy"] for r in rows], dtype=np.float64)
        low.append(
            {
                "tau_ms": recipe.TRAINED_T_MS,
                "input_rate_hz": rate,
                "accuracy": float(values.mean()),
                "accuracy_sem": float(values.std(ddof=1) / math.sqrt(len(values))),
                "n_seeds": len(values),
                "n_total": sum(r["n_total"] for r in rows),
                "source": "exp048 low-rate sweep",
            }
        )
    result["encoding_rate_psychometric"]["curve"] = measurements.rate_curve(
        result["grid_sweep_agg"], low
    )
    if not equivalent(result, numbers):
        raise PingstoreError(
            "historical aggregate replay differs from archived results"
        )
    return result


def analyse_retained(repo, source, *, run_id=None):
    numbers, history = imported(repo, source)
    with inputs.execution(
        repo,
        "analyse",
        sources={"historical": source},
        run_id=run_id,
        configuration=ANALYSIS,
    ) as run:
        result = aggregate(numbers)
        result.pop("notebook_run_id")
        result.pop("duration_s")
        result.update(schema=ANALYSIS["schema"], historical=history)
        run.record["historical"] = history
        write_json_atomic(run.export / "results.json", result)
        write_json_atomic(
            run.scratch / "verification.json",
            {
                "per_seed_rows": sum(
                    len(numbers[k])
                    for k in ("tau_sweep_per_seed", "grid_sweep_per_seed")
                )
                + len(numbers["encoding_rate_psychometric"]["per_seed_new_cells"]),
                "aggregate_replay_matches": True,
                "relative_tolerance": 1e-12,
                "absolute_tolerance": 1e-12,
                "raw_stream_replay": False,
            },
        )
    return run.run_id


def presentation_source(repo, source):
    if source.record["execution"]["configuration"] != ANALYSIS or set(
        source.record["inputs"]
    ) != {"historical"}:
        raise PingstoreError("historical analysis source differs")
    pin = source.record["inputs"]["historical"]
    original = inputs.source(repo, pin["run_id"], "analyse", reference=pin)
    numbers, history = imported(repo, original)
    result = load_json(source.export / "results.json")
    expected = {
        k: v for k, v in numbers.items() if k not in ("notebook_run_id", "duration_s")
    }
    expected.update(schema=ANALYSIS["schema"], historical=history)
    if source.record.get("historical") != history or not equivalent(result, expected):
        raise PingstoreError("historical analysis or lineage differs")
    validate_numbers(result)
    return original, result, history


def present_retained(repo, source, *, run_id=None):
    from experiments.helpers import theme

    original, result, history = presentation_source(repo, source)
    started = time.monotonic()
    with inputs.execution(
        repo,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration=PRESENTATION,
    ) as run:
        run.record["historical"] = history
        run.record["figure_provenance"] = {
            name: {
                "operation": "carried-unchanged",
                "source": original.reference,
                "path": f"export/{name}",
                "sha256": history["source_files"][f"payload/{name}"]["sha256"],
            }
            for name in CARRIED
        }
        for name in set(recipe.FIGURES) - set(CARRIED):
            run.record["figure_provenance"][name] = {
                "operation": "rendered-from-saved-analysis",
                "source": source.reference,
            }
        for name in CARRIED:
            shutil.copyfile(
                original.export / name,
                run.export / name,
            )
        theme.set_paper_mode(True)
        plots.plot_acc_vs_tau(
            result["tau_sweep_agg"], run.export / "acc_vs_tau", run.run_id
        )
        plots.plot_grid_and_rate(
            result["grid_sweep_agg"],
            result["encoding_rate_psychometric"]["curve"],
            run.export / "acc_grid_tau_rate",
            run.run_id,
        )
        write_json_atomic(
            run.export / "numbers.json",
            {
                **result,
                "notebook_run_id": run.run_id,
                "run_id": run.run_id,
                "duration_s": round(time.monotonic() - started, 1),
            },
        )
    return run.run_id
