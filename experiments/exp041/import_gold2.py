"""Explicit offline Gold-2 import; retain selected arrays without simulation."""

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp041 import evidence, inputs, recipe
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import stage_run

URI = "r2://pinglab/campaigns/gold-2"
STATE = "state/experiments/exp041"
DERIVED = "derived/artifacts/data/exp041"
BASE = "provenance/source-records/base"
SIDECARS = ("config.json", "run.sh", "run.jsonl", "output.log")
ARRAYS = {
    "infer": ("dt", "pop_e", "pop_i"),
    "snapshot": ("dt", "n_e", "n_i", "label", "spk_e", "spk_i"),
}


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
    files = {}
    for row in inventory["files"]:
        name = row["path"]
        safe_path(archive, name)
        if name in files or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise PingstoreError("duplicate path or invalid archive checksum")
        if type(row["size_bytes"]) is not int or row["size_bytes"] < 0:
            raise PingstoreError("invalid archive size")
        files[name] = row
    if (
        len(files) != inventory["file_count"]
        or sum(row["size_bytes"] for row in files.values())
        != inventory["total_size_bytes"]
    ):
        raise PingstoreError("archive inventory totals disagree")
    bank = inputs.source(REPO, bank_id, "compute", experiment="exp022")
    contract = evidence.training_contract(bank.export)
    checkpoints = evidence.checkpoints(bank.export, contract)
    evidence.histories(bank.export, contract)
    selected, jobs = set(), []

    def retain(name):
        if name not in files:
            raise PingstoreError(f"required archive evidence missing: {name}")
        selected.add(name)

    for cell, checkpoint in zip(contract["cells"], checkpoints, strict=True):
        modes = ("infer", "snapshot") if cell["seed"] == recipe.SEEDS[0] else ("infer",)
        for mode in modes:
            directory = f"{STATE}/{mode}/{cell['cell_name']}__final_epoch__{checkpoint['sha256'][:12]}"
            payload = "pop_traces.npz" if mode == "infer" else "snapshot.npz"
            for name in (*SIDECARS, payload):
                retain(f"{directory}/{name}")
            if mode == "infer":
                retain(f"{directory}/metrics.json")
            jobs.append(
                {"cell": cell, "mode": mode, "directory": directory, "payload": payload}
            )
    for name in (
        "lineage.json",
        f"{DERIVED}/numbers.json",
        f"{STATE}/cache/rows.json",
        f"{BASE}/run.json",
        f"{BASE}/collection-plan.json",
        f"{BASE}/collection-status/exp041.json",
        f"{BASE}/submissions/collection-submission.json",
    ):
        retain(name)
    for name in files:
        if name.startswith(f"{BASE}/logs/") and "exp041" in name:
            retain(name)
    return {
        "schema": "exp041.gold2-import-plan/v1",
        "archive_uri": URI,
        "metadata": {
            n: file_sha256(archive / n) for n in ("run.json", "inventory.json")
        },
        "bank": bank.reference,
        "recipe": recipe.configuration(),
        "training_contract": contract,
        "checkpoints": checkpoints,
        "jobs": jobs,
        "arrays": ARRAYS,
        "files": [files[n] for n in sorted(selected)],
        "selected_source_bytes": sum(files[n]["size_bytes"] for n in selected),
        "excluded": "Snapshot voltages, conductances, input/output recordings and old plots; training weights remain in the pinned bank. Originals stay unchanged in Gold-2.",
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


def validate_config(config, cell, common, mode):
    expected = {
        **{
            k: common[k]
            for k in (
                "dt",
                "t_ms",
                "dataset",
                "ei_strength",
                "ei_ratio",
                "w_in",
                "readout_mode",
                "dales_law",
                "signed_readout",
                "readout_bias",
                "adaptive_threshold",
                "train_leak",
                "state_clamp",
                "trainable_w_ee",
                "trainable_w_ei",
                "trainable_w_ie",
                "trainable_w_ii",
            )
        },
        "seed": cell["seed"],
        "tau_gaba": cell["tau_gaba_ms"],
        "model": "ping",
        "infer": True,
        "scale_w_in": 1.0,
        "scale_w_ei": 1.0,
        "scale_w_ie": 1.0,
        "intervention": [],
        "scale_projection": [],
    }
    for key, value in expected.items():
        if not evidence._same(config.get(key), value):
            raise PingstoreError(f"historical configuration differs: {key}")
    if config.get("n_hidden") not in (common["n_hidden"], [common["n_hidden"]]):
        raise PingstoreError("historical hidden population differs")
    for key, name in (
        ("load_weights", "weights_final.pth"),
        ("load_config", "config.json"),
    ):
        if PurePosixPath(config.get(key, "")).parts[-2:] != (cell["cell_name"], name):
            raise PingstoreError(f"historical checkpoint identity differs: {key}")
    if mode == "infer" and config.get("max_samples") != recipe.EVAL_MAX_SAMPLES:
        raise PingstoreError("historical evaluation sample count differs")
    if mode == "snapshot" and config.get("sample_index") != recipe.RASTER_SAMPLE_IDX:
        raise PingstoreError("historical illustrative sample differs")


def normalized_metrics(metrics, config):
    """Add only two independently verified fields; keep original metrics in provenance."""
    result = copy.deepcopy(metrics)
    for key, value in (("seed", config["seed"]), ("tau_gaba_ms", config["tau_gaba"])):
        if key in result["config"] and not evidence._same(result["config"][key], value):
            raise PingstoreError(f"conflicting historical metric metadata: {key}")
        result["config"][key] = value
    if result["config"].get("load_weights") != config["load_weights"]:
        raise PingstoreError(
            "metrics and command configuration reference different weights"
        )
    return result


def validate_science(archive, plan):
    numbers = load_json(archive / DERIVED / "numbers.json")

    def key(row):
        return row["training_cell"]

    if sorted(numbers["checkpoint_provenance"], key=key) != sorted(
        plan["checkpoints"], key=key
    ):
        raise PingstoreError("historical checkpoint hashes or roles differ from bank")
    common = plan["training_contract"]["common"]
    for name, value in {
        "tau_gaba_sweep_ms": list(recipe.TAU_GABA_SWEEP),
        "seeds": list(recipe.SEEDS),
        "dt": common["dt"],
        "t_ms": common["t_ms"],
        "epochs": common["epochs"],
        "max_samples": common["max_samples"],
        "evaluation_samples": recipe.EVAL_MAX_SAMPLES,
        "dataset": "mnist",
        "f_gamma_band_hz": list(recipe.F_GAMMA_BAND_HZ),
    }.items():
        if not evidence._same(numbers["config"].get(name), value):
            raise PingstoreError(f"historical recipe differs: {name}")
    for job in plan["jobs"]:
        directory = archive / job["directory"]
        config = load_json(directory / "config.json")
        validate_config(config, job["cell"], common, job["mode"])
        if job["mode"] == "infer":
            evidence.population_traces(
                directory / job["payload"], common, recipe.EVAL_MAX_SAMPLES
            )
            normalized_metrics(load_json(directory / "metrics.json"), config)
        else:
            evidence.snapshot(directory / job["payload"], common["dt"], common)


def extract_arrays(source, target, keys):
    """Copy exact NPY bytes into a compressed ZIP, with no dtype/value conversion."""
    target.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    with (
        zipfile.ZipFile(source) as old,
        zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as new,
    ):
        if len(old.namelist()) != len(set(old.namelist())):
            raise PingstoreError("duplicate array names in historical recording")
        for key in keys:
            name = key + ".npy"
            raw = old.read(name)
            new.writestr(name, raw)
            entries[name] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
    with zipfile.ZipFile(target) as new:
        for name, row in entries.items():
            if hashlib.sha256(new.read(name)).hexdigest() != row["sha256"]:
                raise PingstoreError("lossless array extraction failed")
    return entries


def import_subset(archive, plan):
    archive = Path(archive).absolute()
    # JSON normalization keeps tuple/list representation irrelevant to plan equality.
    if json.loads(json.dumps(make_plan(archive, plan["bank"]["run_id"]))) != plan:
        raise PingstoreError("import plan changed; audit a new plan")
    verify_files(archive, plan)
    validate_science(archive, plan)
    ancestors = inputs.lineage(REPO, plan["bank"]["run_id"], plan["bank"])
    bank = ancestors[plan["bank"]["run_id"]]
    with stage_run(
        REPO,
        "exp041",
        "compute",
        inputs={"bank": bank},
        configuration=plan["recipe"],
        operation="historical-import",
    ) as run:
        originals = run.provenance / "gold-2"
        for name in [
            *plan["metadata"],
            *(r["path"] for r in plan["files"] if not r["path"].endswith(".npz")),
        ]:
            target = originals / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(archive, name), target)
            if file_sha256(target) != file_sha256(archive / name):
                raise PingstoreError("original evidence copy mismatch")
        shutil.copyfile(Path(__file__), run.provenance / "import_gold2.py")
        write_json_atomic(run.provenance / "import-plan.json", plan)
        mappings = []
        for job in plan["jobs"]:
            source = archive / job["directory"]
            output = run.export / job["mode"] / job["cell"]["cell_name"]
            target = output / job["payload"]
            arrays = extract_arrays(
                source / job["payload"], target, ARRAYS[job["mode"]]
            )
            mappings.append(
                {
                    "source": job["directory"] + "/" + job["payload"],
                    "target": str(target.relative_to(run.directory)),
                    "operation": "exact NPY bytes, lossless ZIP compression",
                    "arrays": arrays,
                    "target_sha256": file_sha256(target),
                }
            )
            common = plan["training_contract"]["common"]
            if job["mode"] == "infer":
                config = load_json(source / "config.json")
                metrics = normalized_metrics(load_json(source / "metrics.json"), config)
                write_json_atomic(output / "metrics.json", metrics)
                evidence.measurement(
                    output / "metrics.json",
                    job["cell"],
                    common,
                    recipe.EVAL_MAX_SAMPLES,
                )
                evidence.population_traces(target, common, recipe.EVAL_MAX_SAMPLES)
                mappings.append(
                    {
                        "source": job["directory"] + "/metrics.json",
                        "target": str(
                            (output / "metrics.json").relative_to(run.directory)
                        ),
                        "operation": "add seed and tau_gaba_ms from verified sibling config; all other values unchanged",
                        "metadata_source": job["directory"] + "/config.json",
                        "target_sha256": file_sha256(output / "metrics.json"),
                    }
                )
            else:
                evidence.snapshot(target, common["dt"], common)
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp041.compute/v1",
                "config": plan["recipe"],
                "training_contract": plan["training_contract"],
                "checkpoint_provenance": plan["checkpoints"],
            },
        )
        write_json_atomic(run.provenance / "file-mapping.json", mappings)
        producer = load_json(originals / BASE / "run.json")
        status = load_json(originals / BASE / "collection-status/exp041.json")
        submission = load_json(
            originals / BASE / "submissions/collection-submission.json"
        )
        jobs = [
            job for job in submission.get("jobs", []) if job.get("name") == "ggs-exp041"
        ]
        if len(jobs) != 1 or not jobs[0].get("job_id"):
            raise PingstoreError("missing or ambiguous historical Slurm job")
        run.record["historical_import"] = {
            "archive_uri": URI,
            "metadata_sha256": plan["metadata"],
            "plan": "provenance/import-plan.json",
            "mapping": "provenance/file-mapping.json",
            "original_records": "provenance/gold-2",
            "simulation_executed": False,
            "producer": {
                "origin": "slurm",
                "campaign": producer["run_id"],
                "git_commit": producer["source"]["git_commit"],
                "execution_window": status,
                "slurm_job": jobs[0],
            },
            "source_preservation": "Original recordings unchanged in Gold-2; selected NPY entry bytes retained exactly in export. Unselected snapshot arrays are not duplicated locally.",
        }
        (run.directory / "README.md").write_text(
            "# exp041: selective local Gold-2 import\n\n"
            "This is a local import, not a new simulation. The original HPC/Slurm\n"
            "campaign is recorded in run.json and provenance/gold-2. Its 18 final-epoch\n"
            "checkpoint hashes match the pinned exp022 training bank exactly.\n\n"
            "Export contains metrics and full E/I population traces for 18 networks\n"
            "(1,000 official-test images each), plus six seed-42 sample-0 E/I spike\n"
            "snapshots. Selected NPY bytes, including dtypes, are unchanged; only ZIP\n"
            "compression differs. Voltage, conductance and input/output recordings\n"
            "remain in the unchanged Gold-2 originals. No weights or old figures are\n"
            "duplicated. Original metrics are retained; the export adds verified\n"
            "seed and inhibitory-decay metadata without changing measurements.\n\n"
            "import-plan.json and file-mapping.json give every source checksum,\n"
            "selected array checksum and destination. Read the retained script to\n"
            "reproduce the import with a fresh identity. Exp033/046/054 need the\n"
            "derived frequencies; exp046 also needs the same pinned training bank.\n"
            "Their staged-input migrations and publication remain separate work.\n"
        )
        verify_files(archive, plan)
        for ancestor in ancestors.values():
            ancestor.check_unchanged()
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "import"))
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--source", help="explicit v3 exp022 bank for planning")
    parser.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.action == "plan":
            if not args.source or args.plan.exists():
                raise PingstoreError("planning requires --source and a new plan path")
            plan = make_plan(args.archive, args.source)
            verify_files(args.archive, plan)
            validate_science(args.archive, plan)
            write_json_atomic(args.plan, plan)
            print(
                f"{len(plan['jobs'])} jobs; {len(plan['files'])} files; {plan['selected_source_bytes']} source bytes"
            )
        else:
            import_subset(args.archive, load_json(args.plan))
    except (PingstoreError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"exp041 Gold-2 import: {exc}\n")


if __name__ == "__main__":
    main()
