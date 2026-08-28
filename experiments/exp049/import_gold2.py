"""Explicit selective Gold-2 import; no simulation, analysis or publication."""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp041 import import_gold2 as archive_helpers
from experiments.exp049 import evidence, inputs, measurements, recipe
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import stage_run

URI = archive_helpers.URI
SIDECARS = archive_helpers.SIDECARS
STATE = "state/experiments/exp049"
DERIVED = "derived/artifacts/data/exp049"
BASE = "provenance/source-records/base"
ARRAYS = recipe.ARRAYS
safe_path = archive_helpers.safe_path
verify_files = archive_helpers.verify_files


def source_directory(job, checkpoint):
    return f"{STATE}/{job['kind']}/{job['cell_name']}__final_epoch__{checkpoint['sha256'][:12]}"


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
    checkpoints = {c["training_cell"]: c for c in contract["checkpoints"]}
    selected, jobs = set(), []

    def retain(name):
        if name not in files:
            raise PingstoreError(f"required archive evidence missing: {name}")
        selected.add(name)

    for job in recipe.jobs(recipe.configuration()):
        directory = source_directory(job, checkpoints[job["cell_name"]])
        payloads = recipe.PAYLOADS[job["kind"]]
        for name in (*SIDECARS, *payloads):
            retain(f"{directory}/{name}")
        jobs.append({"job": job, "directory": directory, "payloads": list(payloads)})
    for name in (
        "lineage.json",
        f"{DERIVED}/numbers.json",
        f"{BASE}/run.json",
        f"{BASE}/collection-plan.json",
        f"{BASE}/collection-status/exp049.json",
        f"{BASE}/submissions/collection-submission.json",
        f"{BASE}/logs/exp049/exp049.jsonl",
    ):
        retain(name)
    for name in files:
        if name.startswith(f"{BASE}/logs/") and "exp049" in name:
            retain(name)
    return {
        "schema": "exp049.gold2-import-plan/v1",
        "archive_uri": URI,
        "metadata": {
            n: file_sha256(archive / n) for n in ("run.json", "inventory.json")
        },
        "bank": bank.reference,
        "recipe": recipe.configuration(),
        "training_contract": contract,
        "jobs": jobs,
        "arrays": {k: list(v) for k, v in ARRAYS.items()},
        "files": [files[n] for n in sorted(selected)],
        "selected_source_bytes": sum(files[n]["size_bytes"] for n in selected),
        "excluded": "Unused I-population traces, snapshot voltages/conductances/input/output arrays and unrelated weight matrices remain in Gold-2. Training checkpoints and histories remain in the pinned exp022 bank.",
    }


def producer_evidence(archive):
    lineage = load_json(archive / "lineage.json")
    producer = load_json(archive / BASE / "run.json")
    numbers = load_json(archive / DERIVED / "numbers.json")
    provenance = numbers["collection_provenance"]
    status = load_json(archive / BASE / "collection-status/exp049.json")
    if (
        "exp049" not in lineage["selection"]["base_experiment_state"]
        or "exp049" in lineage["selection"].get("repaired_experiment_state", [])
        or producer["run_id"] != lineage["sources"]["base"]["run_id"]
        or provenance.get("campaign_id") != producer["run_id"]
        or provenance.get("source_git_commit") != producer["source"]["git_commit"]
        or provenance.get("experiment") != recipe.SLUG
        or provenance.get("training_run") != "TR-05"
        or status.get("experiment") != recipe.SLUG
        or status.get("state") != "complete"
    ):
        raise PingstoreError("historical base producer lineage differs")
    paths = sorted((archive / BASE / "logs/collection").glob("ggs-exp049_*.out"))
    if len(paths) != 1:
        raise PingstoreError("missing or ambiguous historical exp049 Slurm log")
    lines = paths[0].read_text().splitlines()
    match = (
        re.fullmatch(
            r"job=(\d+) host=(\S+) action=run-experiment experiment=exp049", lines[0]
        )
        if len(lines) >= 2
        else None
    )
    if not match or paths[0].name != f"ggs-exp049_{match[1]}.out":
        raise PingstoreError("historical Slurm job identity differs")
    events = [
        json.loads(line)
        for line in (archive / BASE / "logs/exp049/exp049.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    completed = [
        e
        for e in events
        if e.get("event") == "completed"
        and e.get("experiment") == recipe.SLUG
        and e.get("run_id") == numbers.get("run_id")
    ]
    if len(completed) != 1 or completed[0].get("quantitative_rows") != 12:
        raise PingstoreError("missing or ambiguous historical exp049 completion")
    return {
        "origin": "slurm",
        "campaign": producer["run_id"],
        "git_commit": producer["source"]["git_commit"],
        "job_id": match[1],
        "host": match[2],
        "device_record": lines[1],
        "log": str(paths[0].relative_to(archive)),
        "completion": completed[0],
        "study_status": status,
        "campaign_status_as_recorded": producer.get("status"),
        "note": "Gold-2 retains this study from the base HPC campaign, not the repair campaign. Original producer records are preserved unchanged.",
    }


def validate_science(archive, plan):
    numbers = load_json(archive / DERIVED / "numbers.json")
    expected = {c["training_cell"]: c for c in plan["training_contract"]["checkpoints"]}
    actual = {c["training_cell"]: c for c in numbers["checkpoint_provenance"]}
    if actual != expected or len(numbers["checkpoint_provenance"]) != len(expected):
        raise PingstoreError("historical checkpoint hashes or roles differ from bank")
    if numbers.get("checkpoint_policy") != recipe.CHECKPOINT_POLICY:
        raise PingstoreError("historical checkpoint selection policy differs")
    for entry in plan["jobs"]:
        job = entry["job"]
        directory = archive / entry["directory"]
        train = plan["training_contract"]["configs"][job["cell_name"]]
        evidence.inference_config(load_json(directory / "config.json"), train, job)
        evidence.recordings(directory, train, job)
    bank = inputs.source(
        REPO,
        plan["bank"]["run_id"],
        "compute",
        experiment="exp022",
        reference=plan["bank"],
    )
    histories = evidence.histories(bank.export, plan["training_contract"])
    curves, summary = {}, []
    checkpoints = {
        c["training_cell"]: c for c in plan["training_contract"]["checkpoints"]
    }
    import torch

    for cell in recipe.bank_cells():
        name = cell["cell_name"]
        paths = {
            kind: archive / source_directory({**cell, "kind": kind}, checkpoints[name])
            for kind in ("infer", "weights_dump")
        }
        with np.load(
            paths["weights_dump"] / "weights_dump.npz", allow_pickle=False
        ) as raw:
            weights = tuple(raw[k] for k in recipe.WEIGHT_ARRAYS)
        state = torch.load(
            bank.export / name / "weights_final.pth",
            map_location="cpu",
            weights_only=True,
        )
        for key, arr in (("W_ei.1", weights[1]), ("W_ie.1", weights[3])):
            actual = state[key].numpy()
            if actual.dtype != arr.dtype or not np.array_equal(arr, actual):
                raise PingstoreError(
                    "retained recurrent weights differ from final checkpoint"
                )
        train = plan["training_contract"]["configs"][name]
        with np.load(paths["infer"] / "pop_traces.npz", allow_pickle=False) as raw:
            result = measurements.endpoint(
                train, load_json(paths["infer"] / "metrics.json"), raw["pop_e"], weights
            )
        summary.append(
            {
                "condition": cell["condition"],
                "seed": cell["seed"],
                **{k: v for k, v in result.items() if k not in ("psd", "freqs_hz")},
            }
        )
        curves[name] = measurements.epoch_curve(
            histories[name]["epochs"], cell["condition"]
        )

    def ordered(rows):
        return sorted(rows, key=lambda r: (r["condition"], r["seed"]))

    if ordered(measurements.clean(summary)) != ordered(numbers["summary"]):
        raise PingstoreError("retained endpoint summary differs from archived results")
    if measurements.rhythmicity(curves) != numbers.get("rhythmicity"):
        raise PingstoreError(
            "retained history rhythmicity differs from archived results"
        )
    if numbers.get("config") != {
        "epochs": recipe.EPOCHS,
        "max_samples": recipe.MAX_SAMPLES,
        "evaluation_samples": recipe.EVAL_MAX_SAMPLES,
        "seeds": recipe.SEEDS,
        "conditions": recipe.CONDITIONS,
        "common_recipe": recipe.COMMON_RECIPE,
    }:
        raise PingstoreError("historical report recipe differs")
    bank.check_unchanged()
    producer_evidence(archive)


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
                    "source_sha256": file_sha256(source),
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
            for payload in entry["payloads"]:
                target = output / payload
                name = entry["directory"] + "/" + payload
                if payload in ARRAYS:
                    arrays = archive_helpers.extract_arrays(
                        safe_path(archive, name), target, ARRAYS[payload]
                    )
                    mappings.append(
                        {
                            "source": name,
                            "source_sha256": file_sha256(archive / name),
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
                "schema": "exp049.compute/v1",
                "recipe": plan["recipe"],
                "training_contract": plan["training_contract"],
                "jobs": recipe.jobs(plan["recipe"]),
            },
        )
        write_json_atomic(run.provenance / "file-mapping.json", {"files": mappings})
        run.record["historical_import"] = {
            "archive_uri": URI,
            "metadata_sha256": plan["metadata"],
            "plan": "provenance/import-plan.json",
            "mapping": "provenance/file-mapping.json",
            "original_records": "provenance/gold-2",
            "simulation_executed": False,
            "producer": producer_evidence(originals),
            "source_preservation": "Gold-2 originals unchanged; selected E-population traces, recurrent weights and E/I snapshots retained as exact NPY bytes. Metric records copied verbatim. No simulation or retraining.",
        }
        write_json_atomic(
            run.provenance / "verification.json",
            {
                "source_checksums_verified": True,
                "checkpoint_records_matched": 12,
                "trained_matrices_matched": 24,
                "archived_summary_rows_reproduced": 12,
                "rhythmicity_reproduced": True,
                "selected_arrays": plan["arrays"],
                "simulation_executed": False,
            },
        )
        (run.directory / "README.md").write_text(
            "# exp049: selective local Gold-2 import\n\n"
            "This is a local import of historical evidence, not a new simulation.\n"
            "run.json records the original base HPC/Slurm campaign, host and job\n"
            "separately from this import operation. The study is not from the repair campaign.\n\n"
            "All 12 TR-05 final-epoch checkpoint records and the 24 trained E/I\n"
            "matrices match the pinned exp022 bank. Retained recordings reproduce\n"
            "all 12 archived summary rows and the history-based rhythmicity summary.\n\n"
            "Export keeps 12 metrics files, 12 full E-population trace arrays,\n"
            "12 sets of initial/trained recurrent E/I matrices, and four full E/I\n"
            "snapshots with metadata. NPY bytes and dtypes are unchanged; only ZIP\n"
            "compression differs. There is no trial, neuron or temporal subsampling.\n"
            "Unused arrays remain in unchanged Gold-2. Training histories and\n"
            "checkpoints stay in the referenced bank.\n\n"
            "provenance/import-plan.json and file-mapping.json record the selected\n"
            "source files, hashes, exact array hashes and destinations. Original\n"
            "configs, commands, logs, archive metadata and summary are retained.\n"
            "Analysis and presentation are separate stages; publication and\n"
            "science/article review require separate approval.\n"
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
            print(import_subset(args.archive, load_json(args.plan)))
    except (PingstoreError, OSError, ValueError, KeyError, IndexError) as exc:
        parser.exit(1, f"exp049 Gold-2 import: {exc}\n")


if __name__ == "__main__":
    main()
