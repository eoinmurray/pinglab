"""Explicit selective Gold-2 import; no simulation, analysis or publication."""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp037 import evidence, inputs, measurements, recipe
from experiments.exp041 import import_gold2 as archive_helpers
from experiments.helpers.frontier import summarize_frontier
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.native import execution_origin
from pingstore.stages import stage_run

URI = archive_helpers.URI
SIDECARS = archive_helpers.SIDECARS
STATE = "state/experiments/exp037"
DERIVED = "derived/artifacts/data/exp037"
REPAIR = "provenance/source-records/repair"
ARRAYS = ("dt", "n_e", "n_i", "label", "spk_e", "spk_i")
safe_path = archive_helpers.safe_path
verify_files = archive_helpers.verify_files


def source_directory(job, checkpoint):
    tag = "best_validation__" + checkpoint["sha256"][:12]
    if job["kind"] == "sweep":
        relative = (
            f"perturb/{job['mode']}_{float(job['level'])}_{job['cell_name']}/{tag}"
        )
    else:
        level = recipe._level_tag(job["level"])
        relative = f"perturb_raster/{job['model']}/seed{job['seed']}/{job['mode']}_{level}_s0/{tag}"
    return f"{STATE}/{relative}"


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
        or sum(r["size_bytes"] for r in files.values()) != inventory["total_size_bytes"]
    ):
        raise PingstoreError("archive inventory totals disagree")
    if recipe.configuration()["evaluation_samples"] != 1000:
        raise PingstoreError("Gold-2 import requires the production recipe")
    bank = inputs.source(REPO, bank_id, "compute", experiment="exp022")
    contract = evidence.training_contract(bank.export)
    evidence.histories(bank.export, contract)
    checkpoints = {c["training_cell"]: c for c in contract["checkpoints"]}
    selected, jobs = set(), []

    def retain(name):
        if name not in files:
            raise PingstoreError(f"required archive evidence missing: {name}")
        selected.add(name)

    for job in recipe.jobs(recipe.configuration()):
        directory = source_directory(job, checkpoints[job["cell_name"]])
        payload = "snapshot.npz" if "sample_index" in job else "metrics.json"
        for name in (*SIDECARS, payload):
            retain(f"{directory}/{name}")
        jobs.append({"job": job, "directory": directory, "payload": payload})
    for name in (
        "lineage.json",
        f"{DERIVED}/numbers.json",
        f"{REPAIR}/run.json",
        f"{REPAIR}/collection-plan.json",
        f"{REPAIR}/logs/exp037/exp037.jsonl",
    ):
        retain(name)
    for name in files:
        if name.startswith(f"{REPAIR}/logs/") and "exp037" in name:
            retain(name)
    return {
        "schema": "exp037.gold2-import-plan/v1",
        "archive_uri": URI,
        "metadata": {
            n: file_sha256(archive / n) for n in ("run.json", "inventory.json")
        },
        "bank": bank.reference,
        "recipe": recipe.configuration(),
        "training_contract": contract,
        "jobs": jobs,
        "arrays": list(ARRAYS),
        "files": [files[n] for n in sorted(selected)],
        "selected_source_bytes": sum(files[n]["size_bytes"] for n in selected),
        "excluded": "Unused snapshot arrays and old plots remain unchanged in Gold-2; weights and histories stay in the pinned exp022 bank.",
    }


def producer_evidence(archive):
    lineage = load_json(archive / "lineage.json")
    producer = load_json(archive / REPAIR / "run.json")
    numbers = load_json(archive / DERIVED / "numbers.json")
    provenance = numbers["collection_provenance"]
    repair = lineage["sources"]["repair"]
    if (
        "exp037" not in lineage["selection"]["repaired_experiment_state"]
        or producer["run_id"] != repair["run_id"]
        or producer["source"]["git_commit"] != repair["source_git_commit"]
        or provenance.get("campaign_id") != repair["run_id"]
        or provenance.get("source_git_commit") != repair["source_git_commit"]
        or provenance.get("experiment") != recipe.SLUG
    ):
        raise PingstoreError("historical repair producer lineage differs")
    paths = sorted(
        (archive / REPAIR / "logs/collection").glob("ggs-repair-exp037_*.out")
    )
    if len(paths) != 1:
        raise PingstoreError("missing or ambiguous historical exp037 Slurm log")
    lines = paths[0].read_text().splitlines()
    match = re.fullmatch(
        r"job=(\d+) host=(\S+) action=run-experiment experiment=exp037", lines[0]
    )
    if not match or paths[0].name != f"ggs-repair-exp037_{match[1]}.out":
        raise PingstoreError("historical Slurm job identity differs")
    events = [
        json.loads(line)
        for line in (archive / REPAIR / "logs/exp037/exp037.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    completed = [
        e
        for e in events
        if e.get("event") == "completed"
        and e.get("experiment") == recipe.SLUG
        and e.get("run_id") == numbers.get("notebook_run_id")
    ]
    if len(completed) != 1 or completed[0].get("quantitative_rows") != 192:
        raise PingstoreError("missing or ambiguous historical exp037 completion")
    plan = load_json(archive / REPAIR / "collection-plan.json")
    experiments = [
        e
        for stage in plan["stages"]
        for e in stage["experiments"]
        if e.get("slug") == recipe.SLUG
    ]
    expected_execution = {
        "mode": "sharded-inference",
        "partition": "ordered-round-robin",
        "shards": 6,
        "workload_contract": {"condition_jobs": 204, "simulator_launches_max": 204},
    }
    if (
        plan.get("campaign_id") != repair["run_id"]
        or plan.get("source", {}).get("git_commit") != repair["source_git_commit"]
        or plan.get("profile") != "production"
        or len(experiments) != 1
        or experiments[0].get("execution") != expected_execution
    ):
        raise PingstoreError("historical exp037 shard plan differs")
    shards = []
    for path in sorted(
        (archive / REPAIR / "logs/collection/exp037-shards").glob("*.out")
    ):
        identity = re.fullmatch(r"(\d+)_(\d+)\.out", path.name)
        content = path.read_text().splitlines()
        header = re.fullmatch(
            r"job=(\d+) host=(\S+) action=run-experiment-shard experiment=exp037",
            content[0],
        )
        if not identity or not header:
            raise PingstoreError("historical exp037 shard identity differs")
        shards.append(
            {
                "array_job_id": identity[1],
                "shard_index": int(identity[2]),
                "job_id": header[1],
                "host": header[2],
                "device_record": content[1],
                "log": str(path.relative_to(archive)),
            }
        )
    if (
        sorted(s["shard_index"] for s in shards) != list(range(6))
        or len({s["array_job_id"] for s in shards}) != 1
    ):
        raise PingstoreError("missing or ambiguous historical exp037 shards")
    return {
        "origin": "slurm",
        "campaign": producer["run_id"],
        "git_commit": producer["source"]["git_commit"],
        "job_id": match[1],
        "job_role": "aggregation",
        "shards": shards,
        "host": match[2],
        "device_record": lines[1],
        "log": str(paths[0].relative_to(archive)),
        "completion": completed[0],
        "campaign_status_as_recorded": producer.get("status"),
        "note": "The campaign manifest remains planned; study completion is established by the retained experiment event and outputs, not a rewritten campaign status.",
    }


def validate_science(archive, plan):
    numbers = load_json(archive / DERIVED / "numbers.json")
    expected = {c["training_cell"]: c for c in plan["training_contract"]["checkpoints"]}
    actual = {c["training_cell"]: c for c in numbers["checkpoint_provenance"]}
    if actual != expected or len(numbers["checkpoint_provenance"]) != len(expected):
        raise PingstoreError("historical checkpoint hashes or roles differ from bank")
    if numbers.get("checkpoint_policy") != recipe.CHECKPOINT_POLICY:
        raise PingstoreError("historical checkpoint selection policy differs")
    points = []
    for entry in plan["jobs"]:
        job = entry["job"]
        directory = archive / entry["directory"]
        train = plan["training_contract"]["configs"][job["cell_name"]]
        evidence.inference_config(load_json(directory / "config.json"), train, job)
        metric = evidence.recordings(directory, train, job)
        if job["kind"] == "sweep":
            points.append(measurements.perturbation_row(metric, job))
    bank = inputs.source(REPO, plan["bank"]["run_id"], "compute", experiment="exp022")
    rows = measurements.baseline_rows(
        evidence.histories(bank.export, plan["training_contract"])
    )
    expected_results = {
        "baseline_results": rows,
        "frontier_summary": summarize_frontier(rows),
        "perturbation": points,
        "perturbation_summary": measurements.summarize_perturbation_rows(points),
    }
    for key, value in expected_results.items():
        if json.loads(json.dumps(value)) != numbers.get(key):
            raise PingstoreError(f"historical numerical replay differs: {key}")
    producer_evidence(archive)


def import_subset(archive, plan):
    archive = Path(archive).absolute()
    if json.loads(json.dumps(make_plan(archive, plan["bank"]["run_id"]))) != plan:
        raise PingstoreError("import plan changed; audit a new plan")
    verify_files(archive, plan)
    validate_science(archive, plan)
    if execution_origin() != "local":
        raise PingstoreError("this historical import must execute locally")
    ancestors = inputs.lineage(REPO, plan["bank"]["run_id"], plan["bank"])
    bank = ancestors[plan["bank"]["run_id"]]
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs={"bank": bank},
        configuration=plan["recipe"],
        operation="historical-import",
    ) as run:
        originals = run.provenance / "gold-2"
        mappings = []

        def copy_file(name, target):
            target.parent.mkdir(parents=True, exist_ok=True)
            source = safe_path(archive, name)
            shutil.copyfile(source, target)
            digest = file_sha256(target)
            if digest != file_sha256(source):
                raise PingstoreError("original evidence copy mismatch")
            mappings.append(
                {
                    "source": name,
                    "target": str(target.relative_to(run.directory)),
                    "operation": "byte-for-byte copy",
                    "target_sha256": digest,
                }
            )

        for name in [
            *plan["metadata"],
            *(r["path"] for r in plan["files"] if not r["path"].endswith(".npz")),
        ]:
            copy_file(name, originals / name)
        shutil.copyfile(Path(__file__), run.provenance / "import_gold2.py")
        shutil.copyfile(
            Path(archive_helpers.__file__), run.provenance / "exp041-import-helpers.py"
        )
        write_json_atomic(run.provenance / "import-plan.json", plan)
        for entry in plan["jobs"]:
            job = entry["job"]
            print(f"[import] {job['path']}", flush=True)
            output = run.export / job["path"]
            target = output / entry["payload"]
            name = entry["directory"] + "/" + entry["payload"]
            if entry["payload"] == "snapshot.npz":
                arrays = archive_helpers.extract_arrays(archive / name, target, ARRAYS)
                mappings.append(
                    {
                        "source": name,
                        "target": str(target.relative_to(run.directory)),
                        "operation": "exact NPY bytes, lossless ZIP compression",
                        "arrays": arrays,
                        "target_sha256": file_sha256(target),
                    }
                )
            else:
                copy_file(name, target)
            for sidecar in SIDECARS:
                copy_file(
                    entry["directory"] + "/" + sidecar,
                    run.provenance / "simulations" / job["path"] / sidecar,
                )
            train = plan["training_contract"]["configs"][job["cell_name"]]
            evidence.recordings(output, train, job)
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp037.compute/v1",
                "recipe": plan["recipe"],
                "training_contract": plan["training_contract"],
                "jobs": recipe.jobs(plan["recipe"]),
            },
        )
        write_json_atomic(
            run.provenance / "file-mapping.json",
            {"schema": "exp037.gold2-file-mapping/v1", "files": mappings},
        )
        run.record["historical_import"] = {
            "archive_uri": URI,
            "metadata_sha256": plan["metadata"],
            "plan": "provenance/import-plan.json",
            "mapping": "provenance/file-mapping.json",
            "original_records": "provenance/gold-2",
            "simulation_executed": False,
            "producer": producer_evidence(originals),
            "source_preservation": "Gold-2 originals unchanged; metrics copied verbatim, all E/I spikes and snapshot metadata retained as exact NPY bytes. Unselected arrays remain in Gold-2.",
        }
        (run.directory / "README.md").write_text(
            "# exp037: selective local Gold-2 import\n\n"
            "This is a local historical import, not a new simulation. The original\n"
            "HPC/Slurm repair campaign, six shards and aggregation job are recorded\n"
            "separately in run.json. Array IDs and per-task job IDs remain distinct.\n"
            "All 36 best-validation checkpoints match the pinned exp022 bank.\n\n"
            "Export retains 192 original perturbation metric records, each evaluating\n"
            "1,000 official-test images. All 12 snapshots retain full E/I spike\n"
            "arrays, dt, population sizes and\n"
            "the actual image label. No trials or neurons were subsampled. Selected\n"
            "NPY bytes and dtypes are unchanged; only ZIP compression differs.\n"
            "Unused snapshot arrays and old figures remain in unchanged Gold-2.\n"
            "Weights and histories are referenced in the bank, not duplicated.\n\n"
            "provenance/import-plan.json and file-mapping.json record the selected\n"
            "source files, checksums, selected arrays and destinations. Original\n"
            "configs, commands, logs, archive metadata and summary are retained.\n"
            "The campaign manifest is preserved as recorded, including its planned\n"
            "status; experiment completion is established by the retained log event.\n"
            "Independent analysis and presentation follow this compute import;\n"
            "publication and article/scientific review remain separate actions.\n"
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
                f"{len(plan['jobs'])} probes; {len(plan['files'])} files; {plan['selected_source_bytes']} source bytes"
            )
        else:
            import_subset(args.archive, load_json(args.plan))
    except (PingstoreError, OSError, ValueError, KeyError, IndexError) as exc:
        parser.exit(1, f"exp037 Gold-2 import: {exc}\n")


if __name__ == "__main__":
    main()
