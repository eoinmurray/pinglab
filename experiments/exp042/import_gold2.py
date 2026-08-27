"""Explicit, offline Gold-2 subset import; never simulate, fetch or publish."""

import argparse
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp042 import analyse, inputs, recipe
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import stage_run

STATE = "state/experiments/exp042"
DERIVED = "derived/artifacts/data/exp042"
URI = "r2://pinglab/campaigns/gold-2"
SIDECARS = ("config.json", "run.sh", "run.jsonl", "output.log")


def safe_path(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise PingstoreError(f"unsafe archive path: {name}")
    target = root / name
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise PingstoreError(f"symlink in archive path: {name}")
    return target


def make_plan(archive, bank_id):
    archive = Path(archive).absolute()
    original = load_json(safe_path(archive, "run.json"))
    inventory = load_json(safe_path(archive, "inventory.json"))
    if (
        original.get("run_id") != "gold-2"
        or original.get("contract_version") != "runstore/v1"
        or original.get("archive", {}).get("uri") != URI
        or inventory.get("run_id") != "gold-2"
        or inventory.get("contract_version") != "runstore/v1"
    ):
        raise PingstoreError("expected the historical Gold-2 archive")
    rows = inventory.get("files", [])
    files = {}
    for row in rows:
        name = row["path"]
        safe_path(archive, name)
        if name in files or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise PingstoreError("duplicate path or invalid archive checksum")
        if not isinstance(row["size_bytes"], int) or row["size_bytes"] < 0:
            raise PingstoreError("invalid archive size")
        files[name] = row
    if len(files) != inventory.get("file_count") or sum(
        row["size_bytes"] for row in rows
    ) != inventory.get("total_size_bytes"):
        raise PingstoreError("archive inventory totals disagree")
    bank = inputs.source(REPO, bank_id, "compute", experiment="exp022")
    evidence = inputs.bank_evidence(bank)
    checkpoints = {p["training_cell"]: p for p in evidence["checkpoints"]}
    cfg = recipe.configuration()
    selected, jobs, recordings = set(), [], {}

    def retain(name):
        if name not in files:
            raise PingstoreError(f"required archive evidence missing: {name}")
        selected.add(name)

    def family(directory, payload, *, sidecars=True):
        for name in (*SIDECARS, payload) if sidecars else (payload,):
            retain(f"{directory}/{name}")
        return f"{directory}/{payload}"

    for job in recipe.jobs(cfg):
        cell, condition = job["cell"], job["condition"]
        tag = "final_epoch__" + checkpoints[cell]["sha256"][:12]
        directory = (
            f"{STATE}/baseline/{cell}/{tag}"
            if condition == "baseline"
            else f"{STATE}/ovrun/{cell}__{cell}_{condition}_{job['seed_offset']}/{tag}"
        )
        jobs.append(
            {
                "job": job,
                "source": family(
                    directory, "metrics.json", sidecars=condition != "baseline"
                ),
            }
        )
    cell = recipe.cell_name(cfg["raster"]["seed"])
    tag = "final_epoch__" + checkpoints[cell]["sha256"][:12]
    for name, condition in (
        ("cycle", "jitter_sigma_14"),
        ("cell", "cell_jitter_sigma_14"),
    ):
        directory = f"{STATE}/condraster/{cell}_{condition}_s0/{tag}"
        recordings[name] = family(directory, "snapshot.npz")
    for name in (*recipe.FIGURES, "numbers.json", "_manifest.json", "run.sh"):
        retain(f"{DERIVED}/{name}")
    # Retain the original campaign lineage and only this experiment's job logs.
    base = "provenance/source-records/base"
    for name in (
        "run.json",
        "inventory.json",
        "collection-plan.json",
        "collection-status/exp042.json",
        "submissions/collection-submission.json",
    ):
        retain(f"{base}/{name}")
    for name in files:
        if name.startswith(f"{base}/logs/") and "exp042" in name:
            retain(name)
    return {
        "schema": "exp042.gold2-import-plan/v1",
        "archive_uri": URI,
        "metadata": {
            name: file_sha256(archive / name) for name in ("run.json", "inventory.json")
        },
        "bank": bank.reference,
        "bank_evidence": evidence,
        "recipe": cfg,
        "jobs": jobs,
        "recordings": recordings,
        "files": [files[name] for name in sorted(selected)],
        "selected_bytes": sum(files[name]["size_bytes"] for name in selected),
        "excluded": "Overrides, full evaluation rasters, unused snapshots, Pareto and cross-tau outputs; R2 remains unchanged.",
    }


def verify_files(archive, plan):
    for name, digest in plan["metadata"].items():
        if file_sha256(safe_path(archive, name)) != digest:
            raise PingstoreError(f"archive metadata changed: {name}")
    for row in plan["files"]:
        path = safe_path(archive, row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != row["size_bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise PingstoreError(f"archive payload checksum mismatch: {row['path']}")


def validate_configuration(config, training, *, sample=False):
    for key in (
        "dt",
        "t_ms",
        "dataset",
        "seed",
        "ei_strength",
        "ei_ratio",
        "w_in",
    ):
        if key not in training or config.get(key) != training[key]:
            raise PingstoreError(f"historical configuration disagrees with bank: {key}")
    if config.get("tau_gaba") != training.get("tau_gaba_ms"):
        raise PingstoreError("historical inhibitory time constant differs")
    if config.get("n_hidden") not in (training["n_hidden"], [training["n_hidden"]]):
        raise PingstoreError("historical hidden population differs")
    if not config.get("infer") or config.get("model") != "ping":
        raise PingstoreError("historical payload is not PING inference")
    if PurePosixPath(config.get("load_weights", "")).parts[-2:] != (
        training["training_cell_name"],
        "weights_final.pth",
    ):
        raise PingstoreError("historical checkpoint path differs")
    if sample:
        if config.get("sample_index") != 0:
            raise PingstoreError("historical illustrative sample differs")
    elif config.get("max_samples") != recipe.EVAL_MAX_SAMPLES:
        raise PingstoreError("historical evaluation sample count differs")


def validate_science(archive, plan):
    cfg, evidence = plan["recipe"], plan["bank_evidence"]
    numbers = load_json(archive / DERIVED / "numbers.json")
    for checkpoint in evidence["checkpoints"]:
        candidates = [
            p
            for p in numbers.get("checkpoint_provenance", [])
            if p["training_cell"] == checkpoint["training_cell"]
        ]
        if candidates != [checkpoint]:
            raise PingstoreError("historical checkpoint provenance differs from bank")
    for key, value in (
        ("seeds", cfg["seeds"]),
        ("conditions", cfg["conditions"]),
        ("jitter_sigmas_ms", cfg["jitter_sigmas_ms"]),
        ("evaluation_samples_per_condition", cfg["evaluation_samples"]),
        ("raster_sample_idx", cfg["raster"]["sample_index"]),
        ("f_gamma_reference_hz", cfg["f_gamma_reference_hz"]),
    ):
        if numbers["config"].get(key) != value:
            raise PingstoreError(f"historical recipe differs: {key}")
    groups = {key: [] for key in ("results", "jitter_sweep", "cell_jitter_sweep")}
    for item in plan["jobs"]:
        job, path = item["job"], archive / item["source"]
        training = evidence["configurations"][job["cell"]]
        metrics = load_json(path)
        if job["condition"] == "baseline":
            # The historical cache retained only metrics and rasters; full
            # commands are retained in the original shard logs.
            config = metrics.get("config", {})
            for key in (
                "dt",
                "t_ms",
                "dataset",
                "ei_strength",
                "ei_ratio",
                "w_in",
                "n_hidden",
                "n_inh",
            ):
                if config.get(key) != training[key]:
                    raise PingstoreError(
                        f"historical baseline configuration differs: {key}"
                    )
            if PurePosixPath(config.get("load_weights", "")).parts[-2:] != (
                job["cell"],
                "weights_final.pth",
            ):
                raise PingstoreError("historical baseline checkpoint differs")
        else:
            config = load_json(path.parent / "config.json")
            validate_configuration(config, training)
            expected = f"{job['cell']}_{job['condition']}_{job['seed_offset']}.npz"
            if PurePosixPath(config.get("i_override_file", "")).name != expected:
                raise PingstoreError("historical override identity differs")
        metrics = load_json(path)
        if (
            metrics.get("config", {}).get("evaluation_partition")
            != cfg["evaluation_partition"]
        ):
            raise PingstoreError("historical evaluation partition differs")
        groups[job["group"]].append(analyse.measurement(metrics, job, cfg))

    def order(row):
        return row["seed"], row["condition"]

    for group, values in groups.items():
        if sorted(values, key=order) != sorted(numbers[group], key=order):
            raise PingstoreError(
                f"historical numbers disagree with raw metrics: {group}"
            )
    training = evidence["configurations"][recipe.cell_name(cfg["raster"]["seed"])]
    for name in plan["recordings"].values():
        path = archive / name
        config = load_json(path.parent / "config.json")
        validate_configuration(config, training, sample=True)
        if (
            PurePosixPath(config.get("i_override_file", "")).name
            != path.parent.parent.name + "_ov.npz"
        ):
            raise PingstoreError("historical illustrative intervention differs")
        analyse.raster_sample(path, training, cfg)
    return groups


def import_subset(archive, plan):
    archive = Path(archive).absolute()
    if make_plan(archive, plan["bank"]["run_id"]) != plan:
        raise PingstoreError("import plan changed; review a new plan")
    verify_files(archive, plan)
    validate_science(archive, plan)
    ancestors = inputs.lineage(REPO, plan["bank"]["run_id"], plan["bank"])
    bank = ancestors[plan["bank"]["run_id"]]
    with stage_run(
        REPO,
        "exp042",
        "compute",
        inputs={"bank": bank},
        configuration=plan["recipe"],
        operation="historical-import",
    ) as run:
        originals = run.provenance / "gold-2"
        for name in [*plan["metadata"], *(r["path"] for r in plan["files"])]:
            target = originals / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(archive, name), target)
        verify_files(originals, plan)
        write_json_atomic(run.provenance / "import-plan.json", plan)
        for item in plan["jobs"]:
            write_json_atomic(
                run.export / "jobs" / (item["job"]["id"] + ".json"),
                {"job": item["job"], "metrics": load_json(originals / item["source"])},
            )
        for name, source in plan["recordings"].items():
            with np.load(originals / source, allow_pickle=False) as data:
                np.savez_compressed(
                    run.export / f"{name}.npz",
                    **{key: data[key] for key in ("spk_e", "spk_i", "label")},
                )
            with (
                np.load(originals / source, allow_pickle=False) as old,
                np.load(run.export / f"{name}.npz", allow_pickle=False) as new,
            ):
                for key in new.files:
                    if old[key].dtype != new[key].dtype or not np.array_equal(
                        old[key], new[key]
                    ):
                        raise PingstoreError("lossless snapshot extraction failed")
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp042.compute/v1",
                "recipe": plan["recipe"],
                "bank_evidence": plan["bank_evidence"],
                "jobs": recipe.jobs(plan["recipe"]),
                "recordings": ["cycle.npz", "cell.npz"],
            },
        )
        run.record["historical_import"] = {
            "archive_uri": URI,
            "metadata_sha256": plan["metadata"],
            "plan": "provenance/import-plan.json",
            "originals": "provenance/gold-2",
            "selected_files": len(plan["files"]),
            "selected_bytes": plan["selected_bytes"],
            "simulation_executed": False,
            "scientific_scope": "66 historical conditions and two illustrative recordings; subset of the larger historical campaign",
        }
        (run.directory / "README.md").write_text(
            "# Exp042: selective Gold-2 import\n\n"
            "This is a local historical import, not a new simulation. The original\n"
            "Wilkes inference used the three TR-02 final-epoch checkpoints pinned\n"
            "through the existing exp022 bank. Original archive metadata, selected\n"
            "scientific bytes, execution records and comparison figures remain in\n"
            "provenance/gold-2; import-plan.json maps every source and checksum.\n\n"
            "The export retains all 66 current condition metrics and losslessly\n"
            "compressed full-population E/I spikes and labels from the two 14 ms\n"
            "illustrative trials. No upstream training, inference, publication or\n"
            "remote mutation occurred. Excluded historical work remains on R2.\n"
        )
        verify_files(archive, plan)
        for ancestor in ancestors.values():
            ancestor.check_unchanged()
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "import"))
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--source", help="explicit v3 exp022 compute bank, for plan")
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.action == "plan":
            if not args.source or args.plan.exists():
                raise PingstoreError("planning requires --source and a new plan path")
            plan = make_plan(args.archive, args.source)
            write_json_atomic(args.plan, plan)
            args.plan.with_suffix(".files.txt").write_text(
                "\n".join(r["path"] for r in plan["files"]) + "\n"
            )
            print(f"{len(plan['files'])} files, {plan['selected_bytes']} bytes")
        else:
            import_subset(args.archive, load_json(args.plan))
    except (PingstoreError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"exp042 Gold-2 import: {exc}\n")


if __name__ == "__main__":
    main()
