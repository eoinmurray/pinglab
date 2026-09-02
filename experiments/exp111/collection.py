"""Explicit campaign orchestration of independent exp111 stages; no publication."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import reserve_stage, stage_reservation

from . import inputs, recipe

STAGES = ("compute", "analyse", "present")


def require_staged(row: dict) -> None:
    if row.get("execution", {}).get("mode") != "exp111-staged":
        raise PingstoreError(
            "legacy exp111 campaign is not conformant; create a new staged plan"
        )


def references(repo: Path, row: dict) -> dict:
    path = Path(row["required_outputs"][0])
    document = load_json(path) if path.is_file() else {}
    if set(document) - {*STAGES, "bank"}:
        raise PingstoreError("unsupported exp111 campaign stage references")
    previous = {}
    if document:
        if "bank" not in document:
            raise PingstoreError("exp111 campaign must pin its bank")
        ref = document["bank"]
        inputs.source(
            repo, ref["run_id"], "compute", experiment="exp022", reference=ref
        )
    for stage in STAGES:
        if stage not in document:
            if any(later in document for later in STAGES[STAGES.index(stage) + 1 :]):
                raise PingstoreError("exp111 campaign has incomplete stage lineage")
            break
        ref = document[stage]
        source = inputs.source(repo, ref["run_id"], stage, reference=ref)
        expected = {"training_bank": document["bank"]}
        if stage == "analyse":
            expected = {"compute": document["compute"]}
        elif stage == "present":
            expected = {"analysis": document["analyse"]}
        if source.record["inputs"] != expected:
            raise PingstoreError("exp111 campaign stage inputs do not match")
        previous[stage] = source
    return previous


def reserve(repo: Path, row: dict, *, origin: str | None = None) -> dict:
    """Reserve every stage before dispatch; never overwrite an incomplete run."""
    require_staged(row)
    path = Path(row["paths"]["state"]) / "stage-reservations.json"
    identities = load_json(path) if path.is_file() else {}
    previous = references(repo, row)
    for stage in STAGES:
        if stage in previous:
            identities[stage] = previous[stage].record["run_id"]
            continue
        identity = identities.get(stage)
        temporary = repo / ".pingstore/runs" / f".{identity}.tmp"
        if identity:
            completed = repo / ".pingstore/runs" / identity
            if completed.exists():
                raise PingstoreError(
                    "completed exp111 reservation lacks campaign reference; explicit recovery required"
                )
        if identity and temporary.is_dir() and not (temporary / "run.json").exists():
            record = stage_reservation(temporary)
            if (
                record["run_id"] == identity
                and record["experiment"] == recipe.SLUG
                and record["stage"] == stage
                and not (temporary / ".writer.lock").exists()
            ):
                continue
        identities[stage] = reserve_stage(
            repo / ".pingstore", recipe.SLUG, stage, origin=origin
        )
    write_json_atomic(path, identities)
    return identities


def campaign_bank(repo: Path, plan: dict):
    identity = load_json(Path(plan["exp022_manifest"])).get("pingstore_run_id")
    if not identity:
        raise PingstoreError(
            "exp111 campaign requires an explicit completed exp022 bank"
        )
    return inputs.source(repo, identity, "compute", experiment="exp022")


def completed(repo: Path, plan: dict, row: dict):
    require_staged(row)
    previous = references(repo, row)
    if set(previous) != set(STAGES):
        raise PingstoreError("exp111 campaign stages are incomplete")
    presentation = previous["present"]
    document = load_json(Path(row["required_outputs"][0]))
    if document["bank"] != campaign_bank(repo, plan).reference:
        raise PingstoreError("exp111 campaign references a different training bank")
    required = ["numbers.json", *recipe.FIGURES]
    if not all((presentation.presentation / name).is_file() for name in required):
        raise PingstoreError("exp111 presentation is incomplete")
    return presentation


def execute(repo: Path, plan: dict, row: dict) -> dict:
    require_staged(row)
    bank = campaign_bank(repo, plan)
    previous = references(repo, row)
    path = Path(row["required_outputs"][0])
    if path.is_file() and load_json(path).get("bank") != bank.reference:
        raise PingstoreError("exp111 campaign already pins a different bank")
    identities = reserve(repo, row)
    refs = {
        "bank": bank.reference,
        **{stage: run.reference for stage, run in previous.items()},
    }
    for stage in STAGES:
        if stage in previous:
            continue
        upstream = {"compute": "bank", "analyse": "compute", "present": "analyse"}[
            stage
        ]
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                f"experiments.exp111.{stage}",
                "--run-id",
                identities[stage],
                "--source",
                refs[upstream]["run_id"],
            ],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="")
        output = inputs.source(repo, identities[stage], stage)
        refs[stage] = output.reference
        write_json_atomic(path, refs)
        references(repo, row)
    completed(repo, plan, row)
    return refs
