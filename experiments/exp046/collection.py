"""Explicit campaign orchestration of independent exp046 stages; no publication."""

import os
import subprocess
import sys
from pathlib import Path

from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import reserve_stage, stage_reservation

from . import inputs, recipe

STAGES = ("compute", "analyse", "present")


def require_staged(row: dict) -> None:
    if row.get("execution", {}).get("mode") != "exp046-staged":
        raise PingstoreError(
            "legacy exp046 campaign is not conformant; create a new staged plan"
        )


def references(repo: Path, row: dict) -> dict:
    path = Path(row["required_outputs"][0])
    document = load_json(path) if path.is_file() else {}
    if set(document) - {*STAGES, "bank", "frequencies"}:
        raise PingstoreError("unsupported exp046 campaign stage references")
    previous = {}
    if document:
        if "bank" not in document:
            raise PingstoreError("exp046 campaign must pin its bank")
        ref = document["bank"]
        inputs.source(
            repo, ref["run_id"], "compute", experiment="exp022", reference=ref
        )
        if "frequencies" not in document:
            raise PingstoreError("exp046 campaign must pin its exp041 analysis")
        ref = document["frequencies"]
        inputs.source(
            repo, ref["run_id"], "analyse", experiment="exp041", reference=ref
        )
    for stage in STAGES:
        if stage not in document:
            if any(later in document for later in STAGES[STAGES.index(stage) + 1 :]):
                raise PingstoreError("exp046 campaign has incomplete stage lineage")
            break
        ref = document[stage]
        source = inputs.source(repo, ref["run_id"], stage, reference=ref)
        expected = {"bank": document["bank"]}
        if stage == "analyse":
            expected["compute"] = document["compute"]
            expected["frequencies"] = document["frequencies"]
        elif stage == "present":
            expected = {"analysis": document["analyse"]}
        if source.record["inputs"] != expected:
            raise PingstoreError("exp046 campaign stage inputs do not match")
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
                    "completed exp046 reservation lacks campaign reference; explicit recovery required"
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
        raise PingstoreError("exp046 campaign stages are incomplete")
    presentation = previous["present"]
    document = load_json(Path(row["required_outputs"][0]))
    bank = campaign_bank(repo, plan)
    if document["bank"] != bank.reference:
        raise PingstoreError("exp046 campaign references a different training bank")
    if document["frequencies"] != campaign_frequencies(repo, plan).reference:
        raise PingstoreError("exp046 campaign references a different exp041 analysis")
    cfg = inputs.configuration(previous["compute"])
    if cfg["profile"] != ("smoke" if plan.get("profile") == "smoke" else "production"):
        raise PingstoreError(
            "exp046 campaign profile differs from completed computation"
        )
    required = ["numbers.json", *recipe.FIGURES]
    if not all((presentation.presentation / name).is_file() for name in required):
        raise PingstoreError("exp046 presentation is incomplete")
    return presentation


def campaign_bank(repo: Path, plan: dict):
    identity = load_json(Path(plan["exp022_manifest"])).get("pingstore_run_id")
    if not identity:
        raise PingstoreError(
            "exp046 campaign requires an explicit completed exp022 bank"
        )
    return inputs.source(repo, identity, "compute", experiment="exp022")


def execute(repo: Path, plan: dict, row: dict) -> dict:
    require_staged(row)
    bank = campaign_bank(repo, plan)
    frequencies = campaign_frequencies(repo, plan)
    previous = references(repo, row)
    path = Path(row["required_outputs"][0])
    if path.is_file() and load_json(path).get("bank") != bank.reference:
        raise PingstoreError("exp046 campaign already pins a different bank")
    if path.is_file() and load_json(path).get("frequencies") != frequencies.reference:
        raise PingstoreError("exp046 campaign already pins a different exp041 analysis")
    if "compute" in previous:
        cfg = inputs.configuration(previous["compute"])
        if cfg["profile"] != (
            "smoke" if plan.get("profile") == "smoke" else "production"
        ):
            raise PingstoreError(
                "exp046 campaign profile differs from completed computation"
            )
    identities = reserve(repo, row)
    refs = {
        "bank": bank.reference,
        "frequencies": frequencies.reference,
        **{stage: run.reference for stage, run in previous.items()},
    }
    for stage in STAGES:
        if stage in previous:
            continue
        command = [
            sys.executable,
            "-m",
            f"experiments.exp046.{stage}",
            "--run-id",
            identities[stage],
        ]
        upstream = {"compute": "bank", "analyse": "compute", "present": "analyse"}[
            stage
        ]
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
        raise PingstoreError("exp046 campaign needs exactly one exp041 dependency")
    upstream.require_staged(candidates[0])
    completed = upstream.references(repo, candidates[0])
    if "analyse" not in completed:
        raise PingstoreError(
            "exp046 requires completed exp041 analysis; upstream execution is separate"
        )
    return completed["analyse"]
