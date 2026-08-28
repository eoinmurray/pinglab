"""Read-only Gold-2 selection planning; never allocate, import or modify runs."""

import argparse
import gzip
import re
import sys
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp080 import evidence, measurements, recipe
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)

URI = "r2://pinglab/campaigns/gold-2"
DERIVED = "derived/artifacts/data/exp080"
BASE = "provenance/source-records/base"
CAMPAIGN = "ggs-production-20260818-4ad223d3"
COMMIT = "4ad223d32620dd9f03698b89f28aedfe944d43ac"
JOB = "33913460"


def safe_path(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
        raise PingstoreError("unsafe Gold-2 path")
    path = Path(root).absolute() / name
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise PingstoreError(f"missing or linked Gold-2 evidence: {name}")
    return path


def producer(archive, numbers):
    lineage = load_json(safe_path(archive, "lineage.json"))
    original = load_json(safe_path(archive, f"{BASE}/run.json"))
    plan = load_json(safe_path(archive, f"{BASE}/collection-plan.json"))
    status = load_json(safe_path(archive, f"{BASE}/collection-status/exp080.json"))
    provenance = numbers["collection_provenance"]
    if (
        lineage["sources"]["base"]["run_id"] != CAMPAIGN
        or original["run_id"] != CAMPAIGN
        or original["source"]["git_commit"] != COMMIT
        or original["source"].get("git_clean") is not True
        or plan["source"]["git_commit"] != COMMIT
        or plan["campaign_id"] != CAMPAIGN
        or provenance.get("campaign_id") != CAMPAIGN
        or provenance.get("source_git_commit") != COMMIT
        or provenance.get("experiment") != recipe.SLUG
        or provenance.get("dependencies") != []
        or provenance.get("training_run") is not None
        or provenance.get("lockfile_sha256") != original["source"]["lockfile"]["sha256"]
        or status.get("state") != "complete"
        or status.get("experiment") != recipe.SLUG
    ):
        raise PingstoreError("exp080 historical producer lineage differs")
    rows = [
        row
        for stage in plan["stages"]
        for row in stage["experiments"]
        if row["slug"] == recipe.SLUG
    ]
    if (
        len(rows) != 1
        or rows[0]["dependencies"] != []
        or rows[0]["training_run"] is not None
        or rows[0]["execution"] != {"mode": "monolithic"}
    ):
        raise PingstoreError("exp080 historical campaign contract differs")
    log = safe_path(archive, f"{BASE}/logs/collection/ggs-exp080_{JOB}.out").read_text()
    match = re.fullmatch(
        r"job=(\d+) host=(\S+) action=run-experiment experiment=exp080",
        log.splitlines()[0],
    )
    if (
        not match
        or match[1] != JOB
        or "exp080 complete: selected 0.5--25 Hz" not in log
    ):
        raise PingstoreError("exp080 historical Slurm completion differs")
    return {
        "campaign_id": CAMPAIGN,
        "source_git_commit": COMMIT,
        "lockfile_sha256": provenance["lockfile_sha256"],
        "slurm_job_id": JOB,
        "host": match[2],
        "status": status,
        "notebook_run_id": numbers["run_id"],
        "command": rows[0]["command"],
        "note": "Original base CUDA decoder calibration; no exp022 bank is consumed.",
    }


def compute_document(numbers):
    cfg = recipe.configuration()
    if numbers["parameters"] != recipe.reported_parameters(cfg):
        raise PingstoreError("historical exp080 scientific recipe differs")
    return {
        "schema": "exp080.compute/v1",
        "recipe": cfg,
        **{
            key: numbers[key]
            for key in (
                "training_dataset",
                "training",
                "evaluation",
                "simulator_validation",
                "runtime_s",
                "environment",
            )
        },
        "illustration": {"kind": "historical-image", "path": "feature_images.png"},
    }


def make_plan(archive, live_metadata):
    archive, live_metadata = Path(archive).absolute(), Path(live_metadata).absolute()
    metadata = {}
    for name in ("run.json", "inventory.json"):
        cached, live = safe_path(archive, name), safe_path(live_metadata, name)
        if cached.read_bytes() != live.read_bytes():
            raise PingstoreError("live R2 metadata differs from cached Gold-2")
        metadata[name] = {
            "size_bytes": cached.stat().st_size,
            "sha256": file_sha256(cached),
        }
    original, inventory = (
        load_json(archive / "run.json"),
        load_json(archive / "inventory.json"),
    )
    if (
        original.get("run_id") != "gold-2"
        or original.get("contract_version") != "runstore/v1"
        or original.get("archive", {}).get("uri") != URI
        or inventory.get("run_id") != "gold-2"
        or inventory.get("contract_version") != "runstore/v1"
    ):
        raise PingstoreError("expected historical Gold-2 archive")
    files = {row["path"]: row for row in inventory["files"]}
    if (
        len(files) != len(inventory["files"])
        or len(files) != inventory["file_count"]
        or sum(row["size_bytes"] for row in files.values())
        != inventory["total_size_bytes"]
    ):
        raise PingstoreError("Gold-2 inventory totals or duplicate paths differ")
    for name, row in files.items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or str(relative) != name
            or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or type(row["size_bytes"]) is not int
            or row["size_bytes"] < 0
        ):
            raise PingstoreError("invalid Gold-2 inventory row")
    selected = {}

    def retain(name, target, encoding="identity"):
        row = {"path": name, **metadata[name]} if name in metadata else files.get(name)
        if row is None:
            raise PingstoreError(f"missing Gold-2 inventory entry: {name}")
        path = safe_path(archive, name)
        if (
            path.stat().st_size != row["size_bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise PingstoreError(f"Gold-2 source checksum differs: {name}")
        selected[name] = {
            **row,
            "target": target,
            "encoding": encoding,
            "retained_bytes": len(gzip.compress(path.read_bytes(), mtime=0))
            if encoding == "gzip"
            else row["size_bytes"],
        }

    for seed in recipe.SEEDS:
        for name in ("decoder.pt", "training.json"):
            relative = f"models/seed-{seed}/{name}"
            retain(f"{DERIVED}/{relative}", f"export/{relative}")
    for name in ("held_out_correctness.npz", "feature_images.png"):
        retain(f"{DERIVED}/{name}", f"export/{name}")
    for name in (
        "lineage.json",
        f"{BASE}/run.json",
        f"{BASE}/collection-plan.json",
        f"{BASE}/collection-status/exp080.json",
        f"{BASE}/logs/collection/ggs-exp080_{JOB}.out",
        f"{BASE}/logs/collection/ggs-exp080_{JOB}.err",
        f"{DERIVED}/numbers.json",
        f"{DERIVED}/decision.json",
        f"{DERIVED}/reproducer.json",
    ):
        retain(name, f"provenance/archive/{name}")
    retain("run.json", "provenance/archive/run.json")
    retain("inventory.json", "provenance/archive/inventory.json.gz", "gzip")
    numbers = load_json(safe_path(archive, f"{DERIVED}/numbers.json"))
    cfg = recipe.configuration()
    _, correctness = evidence.validate(
        archive / DERIVED, cfg, historical=True, document=compute_document(numbers)
    )
    actual = measurements.analyze(correctness, cfg)
    if actual != numbers["decision"] or actual != load_json(
        archive / DERIVED / "decision.json"
    ):
        raise PingstoreError("exp080 historical numerical replay differs")
    producer_record = producer(archive, numbers)
    excluded = [
        row for name, row in files.items() if "exp080" in name and name not in selected
    ]
    return {
        "schema": "exp080.gold2-import-plan/v1",
        "archive_uri": URI,
        "metadata": metadata,
        "producer": producer_record,
        "recipe": cfg,
        "upstream_inputs": {},
        "files": [selected[name] for name in sorted(selected)],
        "source_file_count": len(selected),
        "source_bytes": sum(row["size_bytes"] for row in selected.values()),
        "retained_source_bytes": sum(
            row["retained_bytes"] for row in selected.values()
        ),
        "scientific_export_bytes": sum(
            row["retained_bytes"]
            for row in selected.values()
            if row["target"].startswith("export/")
        ),
        "excluded_experiment_files": excluded,
        "checks": {
            "live_metadata_matches": True,
            "selected_source_checksums": True,
            "checkpoint_roles": True,
            "all_120000_correctness_values_retained": True,
            "exact_numerical_replay": True,
            "producer_lineage": True,
        },
        "notes": [
            "Planning only: no Pingstore allocation, import, training, simulation or publication.",
            "Three validation-selected decoders and full histories remain byte-identical.",
            "All 8 rates, 3 seeds and 5000 test images remain; no subsampling.",
            "The historical illustration has no retained raw features and will be carried unchanged.",
            "Full archive inventory is losslessly gzipped; obsolete SVGs and bookkeeping stay in Gold-2.",
            "Retained-source bytes exclude new evidence, mapping, commands, code provenance and derived runs.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--live-metadata", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    requested_path = args.plan.absolute()
    plan_path = requested_path.resolve()
    if any(
        root == plan_path or root in plan_path.parents
        for root in (
            args.archive.resolve(),
            args.live_metadata.resolve(),
            REPO / ".pingstore",
        )
    ) or any(p.is_symlink() for p in (requested_path, *requested_path.parents)):
        parser.error("plan output must be outside source evidence and Pingstore")
    if args.plan.exists():
        parser.error(
            "use a fresh plan path; existing approval evidence must not be overwritten"
        )
    write_json_atomic(args.plan, make_plan(args.archive, args.live_metadata))


if __name__ == "__main__":
    main()
