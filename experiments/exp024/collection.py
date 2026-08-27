"""Collection adapter for exp024 stages; no legacy output capture or publication."""

import subprocess
import sys
from pathlib import Path

from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import reserve_stage, source_run, stage_reservation

from . import recipe


def require_staged(row: dict) -> None:
    if row.get("execution", {}).get("mode") != "exp024-staged":
        raise PingstoreError("legacy exp024 campaign needs its original checkout; use explicit stage commands for a new run")


def reserve(repo: Path, row: dict, *, origin: str | None = None) -> dict:
    """Reserve before dispatch; do not alter old or partially executed reservations."""
    require_staged(row)
    path = Path(row["paths"]["state"]) / "stage-reservations.json"
    identities = load_json(path) if path.is_file() else {}
    references_path = Path(row["required_outputs"][0])
    references = load_json(references_path) if references_path.is_file() else {}
    for stage in ("analyse", "present"):
        if stage in references:
            reference = references[stage]
            source_run(repo / ".pingstore", reference["run_id"], stage=stage,
                       experiment=recipe.SLUG, reference=reference)
            identities[stage] = reference["run_id"]
            continue
        identity = identities.get(stage)
        temporary = repo / ".pingstore/runs" / f".{identity}.tmp"
        if identity and temporary.is_dir() and not (temporary / "run.json").exists():
            record = stage_reservation(temporary)
            if record["experiment"] == recipe.SLUG and record["stage"] == stage:
                continue
        identities[stage] = reserve_stage(repo / ".pingstore", recipe.SLUG, stage, origin=origin)
    write_json_atomic(path, identities)
    return identities


def completed(repo: Path, plan: dict, row: dict):
    require_staged(row)
    document = load_json(Path(row["required_outputs"][0]))
    compute_id = load_json(Path(plan["exp022_manifest"]))["pingstore_run_id"]
    compute = source_run(repo / ".pingstore", compute_id, stage="compute", experiment="exp022",
                         reference=document["compute"])
    analysis = source_run(repo / ".pingstore", document["analyse"]["run_id"],
                          stage="analyse", experiment=recipe.SLUG, reference=document["analyse"])
    presentation = source_run(repo / ".pingstore", document["present"]["run_id"],
                              stage="present", experiment=recipe.SLUG, reference=document["present"])
    if (analysis.record["inputs"].get("compute") != compute.reference
            or presentation.record["inputs"].get("analysis") != analysis.reference):
        raise PingstoreError("exp024 collection stage inputs do not match")
    for role, reference in analysis.record["inputs"].items():
        source_run(repo / ".pingstore", reference["run_id"], stage="compute",
                   experiment="exp022", reference=reference)
        if presentation.record["inputs"].get(role) != reference:
            raise PingstoreError("exp024 presentation has different computation inputs")
    if not all((presentation.presentation / filename).is_file()
               for filename in (*recipe.FIGURES, "numbers.json")):
        raise PingstoreError("exp024 presentation is incomplete")
    return presentation


def execute(repo: Path, plan: dict, row: dict) -> dict:
    require_staged(row)
    compute_id = load_json(Path(plan["exp022_manifest"])).get("pingstore_run_id")
    if not compute_id:
        raise PingstoreError("exp024 requires an explicit completed exp022 compute run")
    compute = source_run(repo / ".pingstore", compute_id, stage="compute", experiment="exp022")
    path = Path(row["required_outputs"][0])
    previous = load_json(path) if path.is_file() else {}
    if previous and previous.get("compute") != compute.reference:
        raise PingstoreError("exp024 campaign already references a different computation")
    references = {"compute": compute.reference}
    analysis = None
    if "analyse" in previous:
        analysis = source_run(repo / ".pingstore", previous["analyse"]["run_id"], stage="analyse",
                              experiment=recipe.SLUG, reference=previous["analyse"])
        if analysis.record["inputs"].get("compute") != compute.reference:
            raise PingstoreError("exp024 analysis belongs to a different computation")
        references["analyse"] = analysis.reference
    # Local execution reserves here; scheduler dispatch reserves before sbatch.
    identities = reserve(repo, row)
    source_id = compute_id
    for stage in ("analyse", "present"):
        if stage == "analyse" and analysis is not None:
            source_id = analysis.record["run_id"]
            continue
        result = subprocess.run(
            [sys.executable, "-m", f"experiments.exp024.{stage}",
             "--source", source_id, "--run-id", identities[stage]],
            cwd=repo, check=True, capture_output=True, text=True,
        )
        print(result.stdout, end="")
        output = source_run(repo / ".pingstore", identities[stage], stage=stage, experiment=recipe.SLUG)
        references[stage] = output.reference
        write_json_atomic(path, references)
        source_id = output.record["run_id"]
    completed(repo, plan, row)
    return references
