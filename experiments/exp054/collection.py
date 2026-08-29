"""Explicit campaign orchestration of independent exp054 stages; no publication."""

import os
import subprocess
import sys
from pathlib import Path

from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import reserve_stage, stage_reservation

from . import evidence, inputs, recipe

STAGES = ("compute", "analyse", "present")


def require_staged(row: dict) -> None:
    if row.get("execution", {}).get("mode") != "exp054-staged":
        raise PingstoreError(
            "legacy exp054 campaign is not conformant; create a new staged plan"
        )


def references(repo: Path, row: dict) -> dict:
    path = Path(row["required_outputs"][0])
    document = load_json(path) if path.is_file() else {}
    if set(document) - {*STAGES, "frequencies"}:
        raise PingstoreError("unsupported exp054 campaign stage references")
    previous = {}
    if document:
        if "frequencies" not in document:
            raise PingstoreError("exp054 campaign must pin its exp041 analysis")
        ref = document["frequencies"]
        inputs.source(
            repo, ref["run_id"], "analyse", experiment="exp041", reference=ref
        )
    for stage in STAGES:
        if stage not in document:
            if any(later in document for later in STAGES[STAGES.index(stage) + 1 :]):
                raise PingstoreError("exp054 campaign has incomplete stage lineage")
            break
        ref = document[stage]
        source = inputs.source(repo, ref["run_id"], stage, reference=ref)
        inputs.configuration(source)
        expected = (
            {}
            if stage == "compute"
            else {
                "compute": document["compute"],
                "frequencies": document["frequencies"],
            }
        )
        if stage == "present":
            expected = {"analysis": document["analyse"]}
        if source.record["inputs"] != expected:
            raise PingstoreError("exp054 campaign stage inputs do not match")
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
                    "completed exp054 reservation lacks campaign reference; explicit recovery required"
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
        if identity:
            raise PingstoreError(
                "incomplete exp054 reservation requires explicit recovery or a fresh campaign"
            )
        if os.environ.get("SLURM_JOB_ID"):
            raise PingstoreError(
                "exp054 HPC identities must be reserved before submission"
            )
        identities[stage] = reserve_stage(
            repo / ".pingstore", recipe.SLUG, stage, origin=origin
        )
    write_json_atomic(path, identities)
    return identities


def completed(repo: Path, plan: dict, row: dict):
    require_staged(row)
    previous = references(repo, row)
    if set(previous) != set(STAGES):
        raise PingstoreError("exp054 campaign stages are incomplete")
    presentation = previous["present"]
    document = load_json(Path(row["required_outputs"][0]))
    if document["frequencies"] != campaign_frequencies(repo, plan).reference:
        raise PingstoreError("exp054 campaign pins a different exp041 analysis")
    _profile(previous, plan)
    from .present import analysis_source

    _, _, _, result = analysis_source(repo, previous["analyse"].record["run_id"])
    numbers = load_json(presentation.presentation / "numbers.json")
    if any(numbers.get(k) != v for k, v in result.items()):
        raise PingstoreError("exp054 presentation numbers differ from analysis")
    for name in ("numbers.json", *recipe.FIGURES):
        if not (presentation.presentation / name).is_file():
            raise PingstoreError("exp054 presentation is incomplete")
    return presentation


def execute(repo: Path, plan: dict, row: dict) -> dict:
    require_staged(row)
    frequencies = campaign_frequencies(repo, plan)
    previous = references(repo, row)
    _profile(previous, plan)
    path = Path(row["required_outputs"][0])
    if path.is_file() and load_json(path).get("frequencies") != frequencies.reference:
        raise PingstoreError("exp054 campaign already pins a different exp041 analysis")
    identities = reserve(repo, row)
    refs = {
        "frequencies": frequencies.reference,
        **{stage: run.reference for stage, run in previous.items()},
    }
    for stage in STAGES:
        if stage in previous:
            continue
        command = [
            sys.executable,
            "-m",
            f"experiments.exp054.{stage}",
            "--run-id",
            identities[stage],
        ]
        if stage != "compute":
            upstream = "compute" if stage == "analyse" else "analyse"
            command += ["--source", refs[upstream]["run_id"]]
        if stage == "analyse":
            command += ["--frequency-source", frequencies.record["run_id"]]
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


def campaign_frequencies(repo: Path, plan: dict):
    from experiments.exp041 import collection as upstream

    candidates = [
        row
        for stage in plan["stages"]
        for row in stage["experiments"]
        if row["slug"] == "exp041"
    ]
    if len(candidates) != 1:
        raise PingstoreError("exp054 campaign needs exactly one exp041 dependency")
    upstream.require_staged(candidates[0])
    completed = upstream.references(repo, candidates[0])
    if "analyse" not in completed:
        raise PingstoreError(
            "exp054 requires completed exp041 analysis; upstream execution is separate"
        )
    return completed["analyse"]


def _profile(previous, plan):
    if "compute" in previous:
        cfg = evidence.compute_contract(previous["compute"])
        if cfg["profile"] != (
            "smoke" if plan.get("profile") == "smoke" else "production"
        ):
            raise PingstoreError("exp054 campaign profile differs from computation")
