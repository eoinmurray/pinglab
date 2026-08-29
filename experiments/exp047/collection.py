"""Explicit campaign orchestration of independent exp047 stages; no publication."""

import os
import subprocess
import sys
from pathlib import Path

from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import reserve_stage, stage_reservation

from . import evidence, inputs, recipe

STAGES = ("compute", "analyse", "present")


def require_staged(row: dict) -> None:
    if row.get("execution", {}).get("mode") != "exp047-staged":
        raise PingstoreError(
            "legacy exp047 campaign is not conformant; create a new staged plan"
        )


def references(repo: Path, row: dict) -> dict:
    path = Path(row["required_outputs"][0])
    document = load_json(path) if path.is_file() else {}
    if set(document) - set(STAGES):
        raise PingstoreError("unsupported exp047 campaign stage references")
    previous = {}
    for stage in STAGES:
        if stage not in document:
            if any(later in document for later in STAGES[STAGES.index(stage) + 1 :]):
                raise PingstoreError("exp047 campaign has incomplete stage lineage")
            break
        ref = document[stage]
        source = inputs.source(repo, ref["run_id"], stage, reference=ref)
        expected = {} if stage == "compute" else {"compute": document["compute"]}
        if stage == "present":
            expected = {"analysis": document["analyse"]}
        if source.record["inputs"] != expected:
            raise PingstoreError("exp047 campaign stage inputs do not match")
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
                    "completed exp047 reservation lacks campaign reference; explicit recovery required"
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
        raise PingstoreError("exp047 campaign stages are incomplete")
    presentation = previous["present"]
    _profile(previous, plan)
    from .present import analysis_source

    _, result = analysis_source(
        repo, previous["analyse"].record["run_id"], previous["analyse"].reference
    )
    numbers = load_json(presentation.presentation / "numbers.json")
    if any(numbers.get(k) != v for k, v in result.items()):
        raise PingstoreError("exp047 presentation numbers differ from analysis")
    required = ["numbers.json", *recipe.FIGURES]
    if not all((presentation.presentation / name).is_file() for name in required):
        raise PingstoreError("exp047 presentation is incomplete")
    return presentation


def _profile(previous, plan):
    if "compute" in previous:
        cfg = evidence.compute_contract(previous["compute"])
        if cfg["profile"] != (
            "smoke" if plan.get("profile") == "smoke" else "production"
        ):
            raise PingstoreError(
                "exp047 campaign profile differs from completed computation"
            )


def execute(repo: Path, plan: dict, row: dict) -> dict:
    require_staged(row)
    previous = references(repo, row)
    _profile(previous, plan)
    identities = reserve(repo, row)
    refs = {stage: run.reference for stage, run in previous.items()}
    for stage in STAGES:
        if stage in previous:
            continue
        command = [
            sys.executable,
            "-m",
            f"experiments.exp047.{stage}",
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
