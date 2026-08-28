"""Read-only Gold-2 audit and selection planning; never allocate or import runs."""

import gzip
import re
from pathlib import Path, PurePosixPath

from pingstore.contracts import PingstoreError, file_sha256, load_json

from . import evidence, measurements, recipe

URI = "r2://pinglab/campaigns/gold-2"
STATE = "state/experiments/exp047/probe"
DERIVED = "derived/artifacts/data/exp047"
BASE = "provenance/source-records/base"
CAMPAIGN = "ggs-production-20260818-4ad223d3"
COMMIT = "4ad223d32620dd9f03698b89f28aedfe944d43ac"
JOB = "33913459"
SIDECARS = ("config.json", "output.log", "run.jsonl", "run.sh")


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
    status = load_json(safe_path(archive, f"{BASE}/collection-status/exp047.json"))
    provenance = numbers["collection_provenance"]
    if (
        lineage["sources"]["base"]["run_id"] != CAMPAIGN
        or "exp047" not in lineage["selection"]["base_experiment_state"]
        or original["run_id"] != CAMPAIGN
        or original["source"]["git_commit"] != COMMIT
        or original["source"].get("git_clean") is not True
        or plan["source"]["git_commit"] != COMMIT
        or provenance.get("campaign_id") != CAMPAIGN
        or provenance.get("source_git_commit") != COMMIT
        or provenance.get("experiment") != recipe.SLUG
        or provenance.get("dependencies") != []
        or provenance.get("training_run") is not None
        or provenance.get("lockfile_sha256") != original["source"]["lockfile"]["sha256"]
        or status.get("state") != "complete"
        or status.get("experiment") != recipe.SLUG
    ):
        raise PingstoreError("exp047 historical producer lineage differs")
    experiments = [
        row
        for stage in plan["stages"]
        for row in stage["experiments"]
        if row["slug"] == recipe.SLUG
    ]
    if (
        len(experiments) != 1
        or experiments[0]["dependencies"] != []
        or experiments[0]["training_run"] is not None
        or experiments[0]["execution"] != {"mode": "monolithic"}
    ):
        raise PingstoreError("exp047 historical campaign recipe differs")
    log = safe_path(archive, f"{BASE}/logs/collection/ggs-exp047_{JOB}.out").read_text()
    match = re.fullmatch(
        r"job=(\d+) host=(\S+) action=run-experiment experiment=exp047",
        log.splitlines()[0],
    )
    if not match or match[1] != JOB or "[published] " not in log:
        raise PingstoreError("exp047 historical Slurm completion differs")
    return {
        "campaign_id": CAMPAIGN,
        "source_git_commit": COMMIT,
        "lockfile_sha256": provenance["lockfile_sha256"],
        "slurm_job_id": JOB,
        "host": match[2],
        "status": status,
        "notebook_run_id": numbers["run_id"],
        "note": "Original base HPC simulation, not the repaired training-bank branch. Per-probe naive timestamps are retained without inferring a timezone.",
    }


def make_plan(archive, live_metadata):
    """Validate cached scientific evidence against freshly retrieved R2 metadata."""
    archive, live_metadata = Path(archive).absolute(), Path(live_metadata).absolute()
    metadata = {}
    for name in ("run.json", "inventory.json"):
        cached, live = safe_path(archive, name), safe_path(live_metadata, name)
        if cached.read_bytes() != live.read_bytes():
            raise PingstoreError("live R2 metadata differs from the cached archive")
        metadata[name] = {
            "sha256": file_sha256(cached),
            "size_bytes": cached.stat().st_size,
        }
    original = load_json(archive / "run.json")
    inventory = load_json(archive / "inventory.json")
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
    for row in files.values():
        if (
            not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            or type(row["size_bytes"]) is not int
            or row["size_bytes"] < 0
        ):
            raise PingstoreError("invalid Gold-2 inventory row")
    selected = {}

    def retain(name, target, encoding="identity"):
        row = (
            files.get(name)
            if name not in metadata
            else {"path": name, **metadata[name]}
        )
        if row is None:
            raise PingstoreError(f"missing Gold-2 inventory entry: {name}")
        path = safe_path(archive, name)
        if (
            path.stat().st_size != row["size_bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise PingstoreError(f"Gold-2 source checksum differs: {name}")
        retained_bytes = (
            len(gzip.compress(path.read_bytes(), mtime=0))
            if encoding == "gzip"
            else row["size_bytes"]
        )
        selected[name] = {
            **row,
            "target": target,
            "encoding": encoding,
            "retained_bytes": retained_bytes,
        }

    cfg = recipe.configuration()
    rows = {}
    for item in recipe.jobs(cfg):
        prefix = f"{STATE}/{item['id']}"
        retain(f"{prefix}/metrics.json", f"export/probe/{item['id']}/metrics.json")
        for name in SIDECARS:
            retain(f"{prefix}/{name}", f"provenance/simulations/{item['id']}/{name}")
        evidence.simulation_config(
            load_json(safe_path(archive, f"{prefix}/config.json")), cfg, item
        )
        rows[item["id"]] = evidence.metric(
            load_json(safe_path(archive, f"{prefix}/metrics.json")), cfg, item
        )
    for name in (
        "lineage.json",
        f"{BASE}/run.json",
        f"{BASE}/collection-plan.json",
        f"{BASE}/collection-status/exp047.json",
        f"{BASE}/logs/collection/ggs-exp047_{JOB}.out",
        f"{BASE}/logs/collection/ggs-exp047_{JOB}.err",
        f"{DERIVED}/numbers.json",
        f"{DERIVED}/run.sh",
    ):
        retain(name, f"provenance/archive/{name}")
    retain("run.json", "provenance/archive/run.json")
    retain("inventory.json", "provenance/archive/inventory.json.gz", "gzip")
    numbers = load_json(safe_path(archive, f"{DERIVED}/numbers.json"))
    actual = measurements.analyse_rows(rows, cfg)
    if any(
        actual[key] != numbers[key]
        for key in ("config", "definition", "raw", "summary")
    ):
        raise PingstoreError("exp047 historical numerical replay differs")
    producer_record = producer(archive, numbers)
    excluded = [
        row for name, row in files.items() if "exp047" in name and name not in selected
    ]
    return {
        "schema": "exp047.gold2-import-plan/v1",
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
        "scientific_metric_bytes": sum(
            row["size_bytes"]
            for row in selected.values()
            if row["target"].startswith("export/")
        ),
        "excluded_experiment_files": excluded,
        "checks": {
            "metadata_matches_live": True,
            "selected_source_checksums": True,
            "unique_simulations": len(rows),
            "reported_rows": 54,
            "exact_numerical_replay": True,
            "producer_lineage": True,
        },
        "notes": [
            "Planning only: no Pingstore allocation, import, simulation or publication.",
            "All 42 metrics and 168 simulation sidecars retained unchanged; no subsampling.",
            "Full original archive inventory is losslessly gzipped, not replaced by a subset.",
            "Retained-source bytes exclude new recipe, mapping, stage provenance, source patch and derived runs.",
            "Original figures and publication bookkeeping stay in unchanged Gold-2; figures will be redrawn from saved analysis.",
            "Raw spikes and realised weight matrices were not retained historically; no claim of remeasurement from spikes.",
        ],
    }
