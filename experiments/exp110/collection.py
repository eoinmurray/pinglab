"""Campaign orchestration for the exp110 presentation-only synthesis."""

import os
import subprocess
import sys
from pathlib import Path

from experiments.exp037 import collection as exp037_collection
from experiments.exp041 import collection as exp041_collection
from experiments.exp044 import collection as exp044_collection
from experiments.exp046 import collection as exp046_collection
from experiments.exp054 import collection as exp054_collection
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import reserve_stage, source_run, stage_reservation

from . import recipe


def require_staged(row: dict) -> None:
    if row.get("execution", {}).get("mode") != "exp110-present-only":
        raise PingstoreError("exp110 requires a presentation-only staged plan")


def references(repo: Path, row: dict) -> dict:
    path = Path(row["required_outputs"][0])
    document = load_json(path) if path.is_file() else {}
    source_specs = {
        "exp054_analysis": ("analyse", "exp054"),
        "exp041_presentation": ("present", "exp041"),
        "exp046_presentation": ("present", "exp046"),
        "exp037_presentation": ("present", "exp037"),
        "exp044_presentation": ("present", "exp044"),
    }
    if set(document) - {*source_specs, "present"}:
        raise PingstoreError("unsupported exp110 campaign references")
    if not document:
        return {}
    if not set(source_specs) <= set(document):
        raise PingstoreError("exp110 must pin all five source experiments")
    source_refs = {role: document[role] for role in source_specs}
    for role, (stage, experiment) in source_specs.items():
        reference = source_refs[role]
        source_run(
            repo / ".pingstore",
            reference["run_id"],
            stage=stage,
            experiment=experiment,
            reference=reference,
        )
    if "present" not in document:
        return {}
    presentation_ref = document["present"]
    presentation = source_run(
        repo / ".pingstore",
        presentation_ref["run_id"],
        stage="present",
        experiment=recipe.SLUG,
        reference=presentation_ref,
    )
    if presentation.record["inputs"] != source_refs:
        raise PingstoreError("exp110 presentation pins different sources")
    return {"present": presentation}


def reserve(repo: Path, row: dict, *, origin: str | None = None) -> dict:
    require_staged(row)
    path = Path(row["paths"]["state"]) / "stage-reservations.json"
    identities = load_json(path) if path.is_file() else {}
    if references(repo, row):
        return identities
    identity = identities.get("present")
    if identity:
        temporary = repo / ".pingstore/runs" / f".{identity}.tmp"
        if temporary.is_dir() and not (temporary / "run.json").exists():
            record = stage_reservation(temporary)
            if (
                record["run_id"] == identity
                and record["experiment"] == recipe.SLUG
                and record["stage"] == "present"
                and not (temporary / ".writer.lock").exists()
            ):
                return identities
        raise PingstoreError("exp110 reservation requires explicit recovery")
    if os.environ.get("SLURM_JOB_ID"):
        raise PingstoreError("exp110 HPC identity must be reserved before submission")
    identities["present"] = reserve_stage(
        repo / ".pingstore", recipe.SLUG, "present", origin=origin
    )
    write_json_atomic(path, identities)
    return identities


def upstream_sources(repo: Path, plan: dict) -> dict:
    rows = {
        row["slug"]: row
        for stage in plan["stages"]
        for row in stage["experiments"]
        if row["slug"]
        in {"exp037", "exp041", "exp044", "exp046", "exp054"}
    }
    if set(rows) != {
        "exp037",
        "exp041",
        "exp044",
        "exp046",
        "exp054",
    }:
        raise PingstoreError("exp110 requires all five presentation dependencies")
    requirements = {
        "exp054_analysis": (exp054_collection, "exp054", "analyse"),
        "exp041_presentation": (exp041_collection, "exp041", "present"),
        "exp046_presentation": (exp046_collection, "exp046", "present"),
        "exp037_presentation": (exp037_collection, "exp037", "present"),
        "exp044_presentation": (exp044_collection, "exp044", "present"),
    }
    found = {}
    for role, (module, experiment, stage) in requirements.items():
        sources = module.references(repo, rows[experiment])
        if stage not in sources:
            raise PingstoreError(f"exp110 requires completed {experiment} {stage}")
        found[role] = sources[stage]
    return found


def completed(repo: Path, plan: dict, row: dict):
    require_staged(row)
    found = references(repo, row)
    if set(found) != {"present"}:
        raise PingstoreError("exp110 presentation is incomplete")
    presentation = found["present"]
    expected = upstream_sources(repo, plan)
    if presentation.record["inputs"] != {
        role: source.reference for role, source in expected.items()
    }:
        raise PingstoreError("exp110 campaign uses different source presentations")
    if not all((presentation.export / name).is_file() for name in recipe.FIGURES):
        raise PingstoreError("exp110 figure export is incomplete")
    return presentation


def execute(repo: Path, plan: dict, row: dict) -> dict:
    require_staged(row)
    sources = upstream_sources(repo, plan)
    source_refs = {role: source.reference for role, source in sources.items()}
    path = Path(row["required_outputs"][0])
    existing = load_json(path) if path.is_file() else {}
    if any(
        existing.get(role, reference) != reference
        for role, reference in source_refs.items()
    ):
        raise PingstoreError("exp110 campaign already pins different sources")
    found = references(repo, row)
    if found:
        completed(repo, plan, row)
        return {**source_refs, "present": found["present"].reference}
    identity = reserve(repo, row)["present"]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.exp110.present",
            "--source",
            sources["exp054_analysis"].record["run_id"],
            "--exp041-source",
            sources["exp041_presentation"].record["run_id"],
            "--exp046-source",
            sources["exp046_presentation"].record["run_id"],
            "--exp037-source",
            sources["exp037_presentation"].record["run_id"],
            "--exp044-source",
            sources["exp044_presentation"].record["run_id"],
            "--run-id",
            identity,
        ],
        cwd=repo,
        env=os.environ,
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    presentation = source_run(
        repo / ".pingstore", identity, stage="present", experiment=recipe.SLUG
    )
    document = {
        **source_refs,
        "present": presentation.reference,
    }
    write_json_atomic(path, document)
    completed(repo, plan, row)
    return document
