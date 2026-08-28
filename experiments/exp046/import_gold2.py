"""Explicit offline Gold-2 import of full E/I spike evidence; never simulate."""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp041 import evidence as bank_evidence
from experiments.exp041 import import_gold2 as archive_helpers
from experiments.exp046 import evidence, inputs, recipe
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import stage_run

URI = archive_helpers.URI
BASE = archive_helpers.BASE
SIDECARS = archive_helpers.SIDECARS
STATE = "state/experiments/exp046"
DERIVED = "derived/artifacts/data/exp046"
ARRAYS = {
    "rasters.npz": (
        "dt",
        "n_trials",
        "T",
        "n_e",
        "n_i",
        "e_trial",
        "e_t",
        "e_cell",
        "i_trial",
        "i_t",
        "i_cell",
    ),
    "per_cell_rates.npz": ("rate_e_per_cell",),
}
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
    checkpoints = evidence.checkpoints(bank.export, contract)
    selected, jobs = set(), []

    def retain(name):
        if name not in files:
            raise PingstoreError(f"required archive evidence missing: {name}")
        selected.add(name)

    for cell, checkpoint in zip(contract["cells"], checkpoints, strict=True):
        directory = f"{STATE}/infer/{cell['cell_name']}/final_epoch__{checkpoint['sha256'][:12]}"
        for name in (*SIDECARS, "metrics.json", *ARRAYS):
            retain(f"{directory}/{name}")
        jobs.append({"cell": cell, "directory": directory})
    for name in (
        "lineage.json",
        f"{DERIVED}/numbers.json",
        f"{BASE}/run.json",
        f"{BASE}/collection-plan.json",
        f"{BASE}/collection-status/exp046.json",
        f"{BASE}/submissions/collection-submission.json",
    ):
        retain(name)
    for name in files:
        if name.startswith(f"{BASE}/logs/") and "exp046" in name:
            retain(name)
    return {
        "schema": "exp046.gold2-import-plan/v1",
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
        "excluded": "Unused output spikes, per-cell I rates, per-sample E rates and old plots; checkpoints remain in the pinned exp022 bank. Gold-2 originals remain unchanged.",
    }


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
        "tau_gabas_ms": list(recipe.TAU_GABA_SWEEP_MS),
        "seeds": list(recipe.SEEDS),
        "evaluation_samples": recipe.EVAL_MAX_SAMPLES,
        "exp041_training_epochs": common["epochs"],
    }.items():
        if not bank_evidence._same(numbers["config"].get(name), value):
            raise PingstoreError(f"historical recipe differs: {name}")
    for job in plan["jobs"]:
        directory = archive / job["directory"]
        config = load_json(directory / "config.json")
        archive_helpers.validate_config(config, job["cell"], common, "infer")
        if config.get("outputs") != ["rasters", "per_cell_rates"]:
            raise PingstoreError("historical recording selection differs")
        archive_helpers.normalized_metrics(
            load_json(directory / "metrics.json"), config
        )
        evidence.recordings(directory, common, recipe.EVAL_MAX_SAMPLES)
    producer_evidence(archive)


def producer_evidence(archive):
    producer = load_json(archive / BASE / "run.json")
    status = load_json(archive / BASE / "collection-status/exp046.json")
    submission = load_json(archive / BASE / "submissions/collection-submission.json")
    jobs = [j for j in submission.get("jobs", []) if j.get("name") == "ggs-exp046"]
    if len(jobs) != 1 or not jobs[0].get("job_id"):
        raise PingstoreError("missing or ambiguous historical Slurm job")
    if status.get("state") != "complete" or status.get("experiment") != "exp046":
        raise PingstoreError("historical exp046 execution is not complete")
    return {
        "origin": "slurm",
        "campaign": producer["run_id"],
        "git_commit": producer["source"]["git_commit"],
        "execution_window": status,
        "slurm_job": jobs[0],
    }


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
            Path(archive_helpers.__file__), run.provenance / "exp041-import-helpers.py"
        )
        write_json_atomic(run.provenance / "import-plan.json", plan)
        mappings = []
        for job in plan["jobs"]:
            print(f"[import] {job['cell']['cell_name']}", flush=True)
            source = archive / job["directory"]
            output = run.export / "infer" / job["cell"]["cell_name"]
            for payload, keys in ARRAYS.items():
                target = output / payload
                arrays = archive_helpers.extract_arrays(source / payload, target, keys)
                mappings.append(
                    {
                        "source": job["directory"] + "/" + payload,
                        "target": str(target.relative_to(run.directory)),
                        "operation": "exact NPY bytes, lossless ZIP compression",
                        "arrays": arrays,
                        "target_sha256": file_sha256(target),
                    }
                )
            config = load_json(source / "config.json")
            metrics = archive_helpers.normalized_metrics(
                load_json(source / "metrics.json"), config
            )
            write_json_atomic(output / "metrics.json", metrics)
            evidence.measurement(
                output / "metrics.json",
                job["cell"],
                plan["training_contract"]["common"],
                recipe.EVAL_MAX_SAMPLES,
            )
            evidence.recordings(
                output, plan["training_contract"]["common"], recipe.EVAL_MAX_SAMPLES
            )
            mappings.append(
                {
                    "source": job["directory"] + "/metrics.json",
                    "target": str((output / "metrics.json").relative_to(run.directory)),
                    "operation": "add seed and tau_gaba_ms from verified sibling config; all other values unchanged",
                    "metadata_source": job["directory"] + "/config.json",
                    "target_sha256": file_sha256(output / "metrics.json"),
                }
            )
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp046.compute/v1",
                "config": plan["recipe"],
                "training_contract": plan["training_contract"],
                "checkpoint_provenance": plan["checkpoints"],
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
            "source_preservation": "Originals unchanged in Gold-2; full E/I sparse spikes and per-cell E rates retained as exact NPY bytes with lossless ZIP compression. Unused arrays are not copied.",
        }
        (run.directory / "README.md").write_text(
            "# exp046: selective local Gold-2 import\n\n"
            "This is a local import, not a new simulation. run.json identifies the\n"
            "original HPC/Slurm producer separately from this import operation.\n"
            "All 18 final-epoch checkpoint hashes match the pinned exp022 bank.\n\n"
            "Export contains all E/I spike indices for 1,000 official-test images\n"
            "per network, timing/population metadata, per-cell E rates and metrics.\n"
            "Selected NPY bytes and dtypes are unchanged; only ZIP compression\n"
            "differs. No trials, E/I spikes or checkpoints were sampled or replaced.\n"
            "Unused output-spike arrays, per-cell I rates and per-sample E rates\n"
            "remain in the unchanged Gold-2 originals. Checkpoints are referenced,\n"
            "not duplicated. Original metrics are retained verbatim; operational\n"
            "metrics add verified seed and inhibitory-decay metadata only.\n\n"
            "provenance/import-plan.json and file-mapping.json record every source\n"
            "checksum, selected array checksum and destination. Analysis requires\n"
            "an explicitly selected exp041 analysis using the same training bank.\n"
            "Presentation and publication are separate actions. The article and\n"
            "its scientific claims still require final review.\n"
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
                f"{len(plan['jobs'])} networks; {len(plan['files'])} files; {plan['selected_source_bytes']} source bytes"
            )
        else:
            import_subset(args.archive, load_json(args.plan))
    except (PingstoreError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"exp046 Gold-2 import: {exc}\n")


if __name__ == "__main__":
    main()
