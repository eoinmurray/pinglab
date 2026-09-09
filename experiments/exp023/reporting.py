"""Audited presentation metadata corrections; never alter retained evidence."""

import copy
import shutil
import time

from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import stage_run

from . import inputs, recipe

AUDITED_COMPUTE = {
    "run_id": "exp023-r011-compute",
    "payload_digest": "sha256:c08f1e26a231130156e54d294da5950f5191411775d79e012d21b5ba8e710671",
}
AUDITED_BASE = "cb00575381d35c01cbca8c1d4d19396a7624cb85"


def corrected_configuration(compute, cfg):
    """Resolve only the specifically audited legacy run, never all v1 recipes."""
    if (
        compute.reference != AUDITED_COMPUTE
        or compute.record["provenance"].get("git_commit") != AUDITED_BASE
        or cfg != recipe.configuration(version=1)
    ):
        raise PingstoreError(
            "refractory metadata correction requires the audited exp023 compute"
        )
    corrected = copy.deepcopy(cfg)
    corrected["schema"] = "exp023.reported-configuration/v1"
    corrected["biophysics"].update(refractory_E_ms=1.2, refractory_I_ms=0.6)
    correction = {
        "schema": "exp023.refractory-metadata-correction/v1",
        "source_recipe_schema": cfg["schema"],
        "declared_ms": {"E": 3.0, "I": 1.5},
        "executed_ms": {"E": 1.2, "I": 0.6},
        "executed_steps": {"E": 12, "I": 6},
        "dt_ms": 0.1,
        "basis": (
            "Audited simulator base used module-level 12/6 step counters derived at "
            "0.25 ms; the production step calls did not use runtime-derived counters. "
            "The retained 0.1-ms scope recordings corroborate 12-step E and 6-step I "
            "post-spike reset holds. This corrects the report, not the computation."
        ),
        "source_was_dirty": compute.record["provenance"].get("code_dirty"),
        "source_audit_limit": "The recorded Git base is not a complete frozen dirty checkout; retained traces corroborate the counter audit.",
        "measurements_changed": False,
        "figures_changed": False,
    }
    return corrected, correction


def correct_presentation(
    repo, identity, analysis, compute, cfg, results, *, run_id=None
):
    source = inputs.source(repo, identity, "present")
    expected = {"analysis": analysis.reference, "compute": compute.reference}
    if source.record["inputs"] != expected:
        raise PingstoreError(
            "metadata source has different exp023 analysis or compute pins"
        )
    original = load_json(source.file("numbers.json"))
    if any(original.get(k) != v for k, v in results.items()):
        raise PingstoreError("metadata source numbers differ from retained analysis")
    if source.record["execution"]["configuration"] != cfg:
        raise PingstoreError("metadata source recipe differs from computation")
    corrected, correction = corrected_configuration(compute, cfg)
    started = time.monotonic()
    with stage_run(
        repo,
        recipe.SLUG,
        "present",
        run_id=run_id,
        inputs={"presentation": source, **{"analysis": analysis, "compute": compute}},
        configuration={
            "schema": "exp023.presentation-metadata/v1",
            "source_recipe": cfg,
            "reported_biophysics": corrected["biophysics"],
        },
        operation="metadata-correction",
    ) as run:
        for path in source.export.iterdir():
            if path.name != "numbers.json":
                shutil.copyfile(path, run.export / path.name)
        write_json_atomic(
            run.export / "numbers.json",
            {
                **original,
                "config": corrected,
                "run_id": run.run_id,
                "duration_s": time.monotonic() - started,
                "git_sha": run.record["provenance"]["git_commit"],
            },
        )
        run.record["metadata_correction"] = correction
        with (run.directory / "README.md").open("a") as history:
            history.write(
                "\nCorrected reported refractory periods from E/I 3/1.5 ms to 1.2/0.6 ms. "
                "The original recipe and all source runs remain unchanged. The source "
                "used 12/6 simulation-step counters at dt=0.1 ms, corroborated by "
                "retained voltage reset holds. All figures and scientific measurements "
                "were reused unchanged; no simulation, remeasurement or drawing ran. "
                "The exported configuration is labelled as a reporting correction.\n"
            )
    return run.run_id
