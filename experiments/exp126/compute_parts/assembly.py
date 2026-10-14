"""Recipe-specific phase validation and immutable full-result assembly."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from experiments.exp126.recipe import (
    COMPARISON_PHASES,
    PHASE_EXPORT_FILES,
    SLUG,
    assembled_configuration,
    comparison_configuration,
    condition_role,
    implementation_source_digest,
)
from pingstore.contracts import file_sha256, load_json
from pingstore.stages import operation_lock, source_run, stage_run


@dataclass(frozen=True)
class Condition:
    """Scientific bytes from the full run; execution metadata from its phase."""

    directory: Path
    record: dict

    def file(self, name):
        if name not in PHASE_EXPORT_FILES:
            raise ValueError("undeclared scientific role file")
        return self.directory / name


def validate_condition(source):
    cfg = source.record["execution"]["configuration"]
    if cfg != comparison_configuration(cfg.get("seed")):
        raise ValueError("phase differs from the committed matched recipe")
    rows = load_json(source.file("learning.json"))
    if [row["epoch"] for row in rows["epochs"]] != list(range(cfg["epochs"] + 1)):
        raise ValueError("incomplete phase learning history")
    for row in rows["epochs"]:
        for split, count in (("train", 6000), ("validation", 1000)):
            metric = row[split]
            if (
                metric["samples"] != count
                or not np.isfinite(metric["loss"])
                or metric["loss"] < 0
                or not 0 <= metric["accuracy"] <= 1
            ):
                raise ValueError("invalid phase learning metrics")
    if (
        rows["selected_epoch"]
        != min(rows["epochs"][1:], key=lambda row: row["validation"]["loss"])["epoch"]
    ):
        raise ValueError("phase selection is inconsistent")
    if rows["test"]["samples"] != 10000 or not np.isfinite(rows["test"]["loss"]):
        raise ValueError("incomplete phase test metrics")
    shapes = dict(
        input_weight=(cfg["inputs"], cfg["hidden"]),
        input_bias=(1, cfg["hidden"]),
        output_weight=(cfg["hidden"], cfg["outputs"]),
        output_bias=(1, cfg["outputs"]),
    )
    for filename in (
        "initial_weights.npz",
        "selected_weights.npz",
        "final_weights.npz",
        "gradients.npz",
    ):
        with np.load(source.file(filename), allow_pickle=False) as data:
            if set(data.files) != set(shapes) or any(
                data[key].shape != shape or not np.isfinite(data[key]).all()
                for key, shape in shapes.items()
            ):
                raise ValueError("incomplete or invalid model/gradient parameters")
    with np.load(source.file("dataset.npz"), allow_pickle=False) as data:
        train, validation = data["train_indices"], data["validation_indices"]
        labels = data["official_train_labels"]
        if labels.shape != (60000,) or data["official_test_labels"].shape != (10000,):
            raise ValueError("invalid MNIST partitions")
        joined = np.concatenate((train, validation))
        if (
            len(joined) != 7000
            or not np.issubdtype(joined.dtype, np.integer)
            or len(np.unique(joined)) != 7000
            or joined.min() < 0
            or joined.max() >= 60000
        ):
            raise ValueError("invalid or overlapping phase partitions")
        for indices, count in ((train, 600), (validation, 100)):
            if not np.array_equal(
                np.bincount(labels[indices], minlength=10), np.full(10, count)
            ):
                raise ValueError("phase partition is not balanced")
        test_labels = data["official_test_labels"].copy()
    with np.load(source.file("test_predictions.npz"), allow_pickle=False) as data:
        predictions = data["predictions"]
        if (
            not np.array_equal(data["indices"], np.arange(10000))
            or not np.array_equal(data["labels"], test_labels)
            or predictions.shape != (10000,)
            or not np.issubdtype(predictions.dtype, np.integer)
            or (predictions < 0).any()
            or (predictions >= 10).any()
            or abs(
                float(np.mean(predictions == test_labels)) - rows["test"]["accuracy"]
            )
            > 1e-12
        ):
            raise ValueError("incomplete or invalid test predictions")
    trace_shapes = {
        "events": (cfg["steps"], cfg["probe_examples"], cfg["inputs"]),
        "labels": (cfg["probe_examples"],),
        "scores": (cfg["probe_examples"], cfg["outputs"]),
        "loss": (),
    }
    for population, size in (("hidden", cfg["hidden"]), ("output", cfg["outputs"])):
        for state in ("synaptic", "voltage", "spikes"):
            trace_shapes[population + "_" + state] = (
                cfg["steps"],
                cfg["probe_examples"],
                size,
            )
    with np.load(source.file("agreement.npz"), allow_pickle=False) as data:
        if set(data.files) != set(trace_shapes) or any(
            data[key].shape != shape or not np.isfinite(data[key]).all()
            for key, shape in trace_shapes.items()
        ):
            raise ValueError("incomplete phase agreement probe")
    execution = source.record["execution"]
    for field in ("dataset_sha256", "encoding_sha256"):
        if not isinstance(execution.get(field), str) or len(execution[field]) != 64:
            raise ValueError("phase lacks its identity checksum")
    timing = execution["timing"]
    if [row["epoch"] for row in timing["epoch_timings"]] != list(
        range(cfg["epochs"] + 1)
    ):
        raise ValueError("phase lacks epoch timings")
    for row in timing["epoch_timings"]:
        for key in ("optimization_seconds", "evaluation_seconds"):
            if not np.isfinite(row[key]) or row[key] < 0:
                raise ValueError("invalid phase epoch timing")
    for name, values, count in (
        ("inference", timing["inference_seconds"], cfg["inference_repeats"]),
        *[
            (name, values, cfg["timing_repeats"])
            for name, values in execution["microbenchmark"]["seconds"].items()
        ],
    ):
        if (
            len(values) != count
            or not np.isfinite(values).all()
            or (np.asarray(values) <= 0).any()
        ):
            raise ValueError(f"invalid {name} timing coverage")
    if set(execution["microbenchmark"]["seconds"]) != {
        "encoding",
        "forward",
        "forward_backward",
        "optimization_step",
    }:
        raise ValueError("incomplete microbenchmark operations")


def select_phases(repo, identities):
    if not identities or len(set(identities)) != len(identities):
        raise ValueError("provide unique explicit phase identities")
    selected, seeds = {}, set()
    fingerprint = implementation_source_digest(repo)
    provenance = None
    for identity in identities:
        # The phase label is metadata; never infer it from an ID suffix.
        from pingstore.contracts import run_root, validate_operational_run_directory

        record = validate_operational_run_directory(
            run_root(repo / ".pingstore", identity)
        )
        phase = record.get("phase")
        if phase not in COMPARISON_PHASES:
            raise ValueError("unsupported comparison phase")
        source = source_run(
            repo / ".pingstore", identity, stage="compute", experiment=SLUG, phase=phase
        )
        if record.get("implementation") != phase or record["inputs"]:
            raise ValueError("phase identity or independence disagrees with recipe")
        if record.get("implementation_source_sha256") != fingerprint:
            raise ValueError(
                "phase scientific source differs from current assembly recipe"
            )
        identity_provenance = {
            key: record["provenance"].get(key)
            for key in ("git_commit", "lockfile_sha256", "code_dirty")
        }
        if provenance is not None and provenance != identity_provenance:
            raise ValueError("phase source provenance differs")
        provenance = identity_provenance
        validate_condition(source)
        seed = record["execution"]["configuration"]["seed"]
        role = condition_role(phase, seed)
        if role in selected:
            raise ValueError("duplicate phase attempt for a seed; select exactly one")
        selected[role] = source
        seeds.add(seed)
        if set(path.name for path in source.export.iterdir()) != set(
            PHASE_EXPORT_FILES
        ):
            raise ValueError("phase export contains missing or unexpected roles")
    cfg = assembled_configuration(seeds)
    expected = {
        condition_role(phase, seed)
        for phase in COMPARISON_PHASES
        for seed in cfg["seeds"]
    }
    if set(selected) != expected:
        raise ValueError("assembly requires both phases for every selected seed")
    for seed in cfg["seeds"]:
        a, b = (selected[condition_role(phase, seed)] for phase in COMPARISON_PHASES)
        for field in (
            "configuration",
            "device",
            "device_name",
            "cpu_threads",
            "versions",
            "dataset_sha256",
            "encoding_sha256",
        ):
            if a.record["execution"][field] != b.record["execution"][field]:
                raise ValueError(f"incompatible paired phase {field}")
        for filename in ("initial_weights.npz", "dataset.npz"):
            with (
                np.load(a.file(filename), allow_pickle=False) as aa,
                np.load(b.file(filename), allow_pickle=False) as bb,
            ):
                if set(aa.files) != set(bb.files) or any(
                    not np.array_equal(aa[key], bb[key]) for key in aa.files
                ):
                    raise ValueError(f"paired {filename} differs")
    return selected, cfg


def assemble(repo, identities, run_id=None):
    with operation_lock(repo / ".pingstore", exclusive=False):
        selected, cfg = select_phases(repo, identities)
        mapping = {
            role: {name: f"{role}/{name}" for name in PHASE_EXPORT_FILES}
            for role in sorted(selected)
        }
        with stage_run(
            repo,
            SLUG,
            "compute",
            inputs=selected,
            run_id=run_id,
            configuration=cfg,
            operation="assemble",
        ) as run:
            run.record["assembly"] = {
                "output_mapping": mapping,
                "implementation_source_sha256": implementation_source_digest(repo),
            }
            for role, source in sorted(selected.items()):
                unit = run.export / role
                unit.mkdir()
                for name in PHASE_EXPORT_FILES:
                    target = run.export / mapping[role][name]
                    if target.exists():
                        raise FileExistsError(target)
                    shutil.copyfile(source.file(name), target)
                validate_condition(Condition(unit, source.record))
            with (run.directory / "README.md").open("a") as handle:
                handle.write(
                    "\nCompleted phases were assembled without re-running training or inference. Each phase remains an immutable input.\n"
                )
        return run.run_id


def assembled_conditions(repo, full):
    cfg = full.record["execution"]["configuration"]
    expected_configuration = assembled_configuration(cfg.get("seeds", []))
    # Retained v4 results from the original matched protocol remain readable.
    if cfg.get("protocol", {}).get("schema") == "exp126.matched/v2":
        legacy = expected_configuration["protocol"]
        legacy["schema"] = "exp126.matched/v2"
        legacy["snnlab"] = (
            "GraphExecutor with matched custom neuron and surrogate; validation overhead included"
        )
        for key in (
            "snnlab_extension_validation",
            "snnlab_policy_selection",
            "strict_validation_microbenchmark",
        ):
            legacy.pop(key)
    if (
        full.record.get("scope") != "full"
        or cfg.get("schema") != "exp126.assembled/v1"
        or full.record["execution"]["operation"] != "assemble"
        or cfg != expected_configuration
    ):
        raise ValueError("analysis requires an assembled full comparison compute run")
    expected = {
        condition_role(phase, seed)
        for phase in COMPARISON_PHASES
        for seed in cfg["seeds"]
    }
    mapping = {
        role: {name: f"{role}/{name}" for name in PHASE_EXPORT_FILES}
        for role in sorted(expected)
    }
    if (
        set(full.record["inputs"]) != expected
        or full.record["assembly"]["output_mapping"] != mapping
        or set(p.name for p in full.export.iterdir()) != expected
    ):
        raise ValueError(
            "assembled phase coverage or output mapping disagrees with recipe"
        )
    selected = {}
    for seed in cfg["seeds"]:
        for phase in COMPARISON_PHASES:
            role = condition_role(phase, seed)
            pin = full.record["inputs"][role]
            original = source_run(
                repo / ".pingstore",
                pin["run_id"],
                stage="compute",
                experiment=SLUG,
                phase=phase,
                reference=pin,
            )
            if (
                original.record.get("implementation") != phase
                or original.record["execution"]["configuration"]["seed"] != seed
            ):
                raise ValueError("assembled input role disagrees with original phase")
            unit = full.unit(role)
            if set(p.name for p in unit.iterdir()) != set(PHASE_EXPORT_FILES):
                raise ValueError("assembled condition output coverage differs")
            for name in PHASE_EXPORT_FILES:
                if file_sha256(unit / name) != file_sha256(original.file(name)):
                    raise ValueError(
                        "assembled scientific bytes differ from declared phase"
                    )
            selected[role] = Condition(unit, original.record)
    return selected
