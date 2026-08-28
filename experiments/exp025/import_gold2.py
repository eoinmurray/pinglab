"""Explicit offline Gold-2 import with a reviewed, checksummed selection plan."""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp025 import evidence, inputs, recipe
from experiments.exp025.compute import payload_arrays
from experiments.exp041 import import_gold2 as archive_helpers
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import stage_run

URI = archive_helpers.URI
STATE = "state/experiments/exp025"
DERIVED = "derived/artifacts/data/exp025"
BANK = "state/checkpoints/current-repair-exp022/cells"
REPAIR = "provenance/source-records/repair"
LOG = f"{REPAIR}/logs/collection/ggs-repair-exp025_34111989.out"
safe_path = archive_helpers.safe_path
verify_files = archive_helpers.verify_files


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
    bank = inputs.source(REPO, bank_id, "compute", experiment="exp022")
    contract = evidence.training_contract(bank.export)
    evidence.histories(bank.export, contract)
    selected = set()

    def retain(name):
        if name not in files:
            raise PingstoreError(f"required archive evidence missing: {name}")
        selected.add(name)

    checkpoints = {c["training_cell"]: c for c in contract["checkpoints"]}
    for name, checkpoint in checkpoints.items():
        weight = files.get(f"{BANK}/{name}/weights_final.pth")
        if not weight or weight["sha256"] != checkpoint["sha256"]:
            raise PingstoreError("repair-bank checkpoint hash differs from pinned bank")
        retain(f"{BANK}/{name}/config.json")
        retain(f"{BANK}/{name}/metrics.json")
    jobs = []
    for job in recipe.jobs(recipe.configuration()):
        name = job["cell_name"]
        directory = (
            f"{BANK}/{name}/infer"
            if job["kind"] == "snapshot"
            else f"{STATE}/{job['path']}__final_epoch__{checkpoints[name]['sha256'][:12]}"
        )
        for filename in archive_helpers.SIDECARS:
            retain(f"{directory}/{filename}")
        if job["kind"] != "snapshot":
            retain(f"{directory}/metrics.json")
        for filename in payload_arrays(job):
            retain(f"{directory}/{filename}")
        jobs.append({"job": job, "directory": directory})
    for name in (
        "lineage.json",
        f"{DERIVED}/numbers.json",
        f"{REPAIR}/run.json",
        f"{REPAIR}/collection-plan.json",
        LOG,
        LOG.removesuffix(".out") + ".err",
        f"{REPAIR}/logs/exp025/exp025.jsonl",
    ):
        retain(name)
    return {
        "schema": "exp025.gold2-import-plan/v1",
        "archive_uri": URI,
        "metadata": {
            n: file_sha256(archive / n) for n in ("run.json", "inventory.json")
        },
        "bank": bank.reference,
        "recipe": recipe.configuration(),
        "training_contract": contract,
        "jobs": jobs,
        "files": [files[n] for n in sorted(selected)],
        "selected_source_bytes": sum(files[n]["size_bytes"] for n in selected),
        "excluded": "Weights are pinned, not copied. COBA PFG recordings, unused array members and old plots remain in Gold-2. No trials or selected E/I spikes are subsampled.",
    }


validate_config = evidence.inference_config


def producer_evidence(archive):
    lineage = load_json(archive / "lineage.json")
    producer = load_json(archive / REPAIR / "run.json")
    first = (archive / LOG).read_text().splitlines()[0]
    match = re.fullmatch(
        r"job=(\d+) host=(\S+) action=run-experiment experiment=exp025", first
    )
    if (
        not match
        or producer["run_id"] != lineage["sources"]["repair"]["run_id"]
        or producer["source"]["git_commit"]
        != lineage["sources"]["repair"]["source_git_commit"]
        or "exp025" not in lineage["selection"]["repaired_experiment_state"]
    ):
        raise PingstoreError("repair producer evidence differs")
    return {
        "origin": "slurm",
        "campaign": producer["run_id"],
        "declared_git_commit": producer["source"]["git_commit"],
        "slurm_job_id": match[1],
        "host": match[2],
        "log": "provenance/gold-2/" + LOG,
        "code_attribution": "Retained campaign and per-job configs declare ac6f4988; command paths name the 73f0883e checkout. Both records are preserved; the directory name alone does not establish executed code.",
    }


def validate_science(archive, plan):
    numbers = load_json(archive / DERIVED / "numbers.json")
    contract = plan["training_contract"]
    historical = [
        c for g in numbers["training_sources"].values() for c in g["checkpoints"]
    ]
    if sorted(historical, key=lambda c: c["training_cell"]) != sorted(
        contract["checkpoints"], key=lambda c: c["training_cell"]
    ):
        raise PingstoreError("historical checkpoint hashes or roles differ from bank")
    for c in contract["cells"]:
        cfg = load_json(archive / BANK / c["cell_name"] / "config.json")
        if cfg != contract["configs"][c["cell_name"]]:
            raise PingstoreError("archived training recipe differs from pinned bank")
    for key, value in {
        "dataset": "mnist",
        "models": recipe.MODELS,
        "epochs": 50,
        "max_samples": 7000,
        "evaluation_samples": 1000,
        "dt": 0.1,
        "t_ms": 200.0,
        "frontier_seeds": recipe.SEEDS,
        "representative_seed": 42,
        "fr_strength_upper": 0.041,
        "rate_target_grid_hz": [t for t in recipe.RATE_TARGET_GRID_HZ if t is not None],
    }.items():
        if not evidence._same(numbers["config"].get(key), value):
            raise PingstoreError(f"historical recipe differs: {key}")
    producer_evidence(archive)
    for row in plan["jobs"]:
        job, directory = row["job"], archive / row["directory"]
        cfg = load_json(directory / "config.json")
        validate_config(cfg, contract["configs"][job["cell_name"]], job)
        if job["kind"] != "snapshot":
            archive_helpers.normalized_metrics(
                load_json(directory / "metrics.json"), cfg
            )


def import_subset(archive, plan):
    archive = Path(archive).absolute()
    if json.loads(json.dumps(make_plan(archive, plan["bank"]["run_id"]))) != plan:
        raise PingstoreError("import plan changed; audit a new plan")
    verify_files(archive, plan)
    validate_science(archive, plan)
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
        shutil.copyfile(
            Path(archive_helpers.__file__), run.provenance / "archive_helpers.py"
        )
        write_json_atomic(run.provenance / "import-plan.json", plan)
        mappings = []
        for row in plan["jobs"]:
            job = row["job"]
            source = archive / row["directory"]
            output = run.export / job["path"]
            output.mkdir(parents=True)
            for filename, keys in payload_arrays(job).items():
                target = output / filename
                arrays = archive_helpers.extract_arrays(source / filename, target, keys)
                mappings.append(
                    {
                        "source": row["directory"] + "/" + filename,
                        "target": str(target.relative_to(run.directory)),
                        "operation": "exact NPY bytes, lossless ZIP compression",
                        "arrays": arrays,
                        "target_sha256": file_sha256(target),
                    }
                )
            if job["kind"] != "snapshot":
                metrics = archive_helpers.normalized_metrics(
                    load_json(source / "metrics.json"),
                    load_json(source / "config.json"),
                )
                target = output / "metrics.json"
                write_json_atomic(target, metrics)
                mappings.append(
                    {
                        "source": row["directory"] + "/metrics.json",
                        "target": str(target.relative_to(run.directory)),
                        "operation": "add seed and tau_gaba_ms from verified sibling config; all other values unchanged",
                        "metadata_source": row["directory"] + "/config.json",
                        "target_sha256": file_sha256(target),
                    }
                )
            evidence.recordings(
                output, plan["training_contract"]["configs"][job["cell_name"]], job
            )
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp025.compute/v1",
                "recipe": plan["recipe"],
                "training_contract": plan["training_contract"],
                "jobs": recipe.jobs(plan["recipe"]),
            },
        )
        write_json_atomic(run.provenance / "file-mapping.json", mappings)
        run.record["historical_import"] = {
            "archive_uri": URI,
            "metadata_sha256": plan["metadata"],
            "plan": "provenance/import-plan.json",
            "mapping": "provenance/file-mapping.json",
            "original_records": "provenance/gold-2",
            "simulation_executed": False,
            "producer": producer_evidence(originals),
            "source_preservation": "Gold-2 originals unchanged; selected NPY bytes and dtypes retained exactly. No trials or selected E/I spikes removed.",
        }
        (run.directory / "README.md").write_text(
            "# exp025: selective local Gold-2 import\n\n"
            "This is a local import, not a new simulation. The original repair-campaign\n"
            "HPC/Slurm producer is recorded separately in run.json and provenance/gold-2.\n"
            "All 48 final-epoch checkpoint hashes match the pinned exp022 bank.\n\n"
            "Retained: 36 frontier metrics, 12 representative-seed PFG metrics with\n"
            "all 1,000 PING trials' E population traces and full E/I spike indices,\n"
            "48 scale-sweep metrics and sample-wise E rates, and two 400 ms digit-0\n"
            "snapshots with full E/I spikes. Snapshots came from the repair bank's\n"
            "legacy infer directories; they were copied, never regenerated upstream.\n"
            "The 48 training histories and weights remain available through the bank pin.\n\n"
            "Only unused NPZ members, COBA PFG recordings and old plots are omitted.\n"
            "COBA PFG uses metrics only in the retained analysis. Selected NPY bytes\n"
            "are unchanged, including dtypes; only ZIP compression differs. Original\n"
            "metrics are preserved; operational copies add verified seed and decay metadata.\n"
            "Every source and selected member checksum is in the plan and mapping.\n\n"
            "The historical COBA/PING gradient-damping difference and code-attribution\n"
            "ambiguity are retained, not corrected. Scientific review and publication\n"
            "are separate work. Gold-2 and existing operational runs remain unchanged.\n"
        )
        verify_files(archive, plan)
        for ancestor in ancestors.values():
            ancestor.check_unchanged()
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "import"))
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--source")
    p.add_argument("--plan", required=True, type=Path)
    a = p.parse_args()
    try:
        if a.action == "plan":
            if not a.source or a.plan.exists():
                raise PingstoreError("planning requires --source and a new plan path")
            plan = make_plan(a.archive, a.source)
            verify_files(a.archive, plan)
            validate_science(a.archive, plan)
            write_json_atomic(a.plan, plan)
            print(
                f"{len(plan['jobs'])} jobs; {len(plan['files'])} files; {plan['selected_source_bytes']} source bytes"
            )
        else:
            print(import_subset(a.archive, load_json(a.plan)))
    except (PingstoreError, OSError, ValueError, KeyError) as exc:
        p.exit(1, f"exp025 Gold-2 import: {exc}\n")


if __name__ == "__main__":
    main()
