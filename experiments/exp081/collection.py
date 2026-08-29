"""Explicit campaign orchestration of independent exp081 stages; no publication."""

import os
import subprocess
import sys
from pathlib import Path

from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import reserve_stage, stage_reservation

from . import inputs, recipe

STAGES = ("compute", "analyse", "present")


def require_staged(row: dict) -> None:
    if row.get("execution", {}).get("mode") != "exp081-staged":
        raise PingstoreError(
            "legacy exp081 campaign is not conformant; create a new staged plan"
        )


def references(repo: Path, row: dict) -> dict:
    path = Path(row["required_outputs"][0])
    document = load_json(path) if path.is_file() else {}
    if set(document) - set(STAGES):
        raise PingstoreError("unsupported exp081 campaign stage references")
    previous = {}
    for stage in STAGES:
        if stage not in document:
            if any(later in document for later in STAGES[STAGES.index(stage) + 1 :]):
                raise PingstoreError("exp081 campaign has incomplete stage lineage")
            break
        ref = document[stage]
        source = inputs.source(repo, ref["run_id"], stage, reference=ref)
        expected = {} if stage == "compute" else {"compute": document["compute"]}
        if stage == "present":
            expected["analysis"] = document["analyse"]
        if source.record["inputs"] != expected:
            raise PingstoreError("exp081 campaign stage inputs do not match")
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
                    "completed exp081 reservation lacks campaign reference; explicit recovery required"
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


def completed(repo: Path, plan: dict, row: dict):
    require_staged(row)
    previous = references(repo, row)
    if set(previous) != set(STAGES):
        raise PingstoreError("exp081 campaign stages are incomplete")
    presentation = previous["present"]
    for name in (
        "numbers.json",
        "empirical_moments.svg",
        "response_distributions.svg",
        "frequency_response.svg",
        "analytical_empirical.svg",
    ):
        if not (presentation.presentation / name).is_file():
            raise PingstoreError("exp081 presentation is incomplete")
    return presentation


def execute(repo: Path, plan: dict, row: dict) -> dict:
    require_staged(row)
    previous = references(repo, row)
    identities = reserve(repo, row)
    refs = {stage: run.reference for stage, run in previous.items()}
    for stage in STAGES:
        if stage in previous:
            continue
        command = [
            sys.executable,
            "-m",
            f"experiments.exp081.{stage}",
            "--run-id",
            identities[stage],
        ]
        if stage != "compute":
            upstream = "compute" if stage == "analyse" else "analyse"
            command += ["--source", refs[upstream]["run_id"]]
        environment = {
            **os.environ,
            "PINGLAB_SMOKE": "1" if plan.get("profile") == "smoke" else "0",
        }
        result = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="")
        output = inputs.source(repo, identities[stage], stage)
        refs[stage] = output.reference
        write_json_atomic(Path(row["required_outputs"][0]), refs)
        references(repo, row)
    completed(repo, plan, row)
    return refs
