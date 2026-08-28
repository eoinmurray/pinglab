"""Explicit summary-only history; never substitute summaries for raw streams."""

import copy
import math
import shutil
import time

import numpy as np
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)

from . import evidence, inputs, measurements, plots, recipe

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
    "run.sh",
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
    evidence.analysis_rows(numbers)
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
    rows = inventory["files"]
    if len(rows) != len(PAYLOAD_NAMES) or {r["path"] for r in rows} != PAYLOAD_NAMES:
        raise PingstoreError("historical payload inventory differs")
    for row in rows:
        path = directory / "payload" / row["path"]
        if (
            path.stat().st_size != row["size_bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise PingstoreError(f"historical payload checksum differs: {row['path']}")
    if inventory["file_count"] != len(rows) or inventory["total_size_bytes"] != sum(
        r["size_bytes"] for r in rows
    ):
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
    archive = source.directory / "provenance/archive"
    files = archive_files(archive)
    history = provenance(archive, files)
    if source.record.get("historical") != history:
        raise PingstoreError("historical provenance differs")
    if source.record.get("source_file_mapping") != {
        name: f"provenance/archive/{name}" for name in files
    }:
        raise PingstoreError("historical source file mapping differs")
    if (
        file_sha256(source.export / "numbers.json")
        != files["payload/numbers.json"]["sha256"]
    ):
        raise PingstoreError("imported numerical bytes differ")
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


def analyse(repo, source, *, run_id=None):
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
            run.provenance / "verification.json",
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


def present(repo, source, *, run_id=None):
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
                "path": f"provenance/archive/payload/{name}",
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
                original.directory / "provenance/archive/payload" / name,
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
