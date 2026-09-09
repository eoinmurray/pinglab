"""Read-only, checksum-pinned contract for the exp022 refractory replacement bank."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

from experiments.exp022 import campaign, recipe
from pingstore.contracts import PingstoreError, file_sha256
from pingstore.stages import SourceRun, source_run

from helpers.checkpoints import ROLES, public_provenance, resolve_checkpoint

SOURCE_REFERENCE = {
    "run_id": "exp022-r001-compute",
    "payload_digest": "sha256:9e3c93df9541809d1d019fe5290afbf7dff7d07ec14b07160fabe7ad79c9a0a8",
}
INHERITED_COMMIT = "4ad223d32620dd9f03698b89f28aedfe944d43ac"
REPAIRED_COMMIT = "ac6f49884084811e3e05d49e8b45735d514ff245"
CELL_FILES = ("config.json", "metrics.json", "weights.pth", "weights_final.pth")
_REFRACTORY_FLAGS = ("--refractory-e-ms", "--refractory-i-ms", "--refractory-policy")
_LEGACY_DYNAMICS = {"dt_ms", "presentation_duration_ms", "tau_ampa_ms", "tau_gaba_ms"}
_PROVENANCE_FIELDS = (
    "training_cell_name", "training_run_id", "campaign_resolved_parameters",
    "campaign_id", "campaign_manifest_sha256", "campaign_repository_commit",
    "imported_cell_provenance",
)


def _registry(cells: list[dict], label: str) -> dict[str, dict]:
    names = [cell["name"] for cell in cells]
    if len(names) != 102 or len(set(names)) != 102:
        raise PingstoreError(f"{label} registry must contain exactly 102 unique cells")
    return {cell["name"]: cell for cell in cells}


def _partition() -> tuple[dict[str, dict], dict]:
    old = _registry(recipe.LEGACY_CANONICAL_CELLS, "historical")
    current = _registry(recipe.CANONICAL_CELLS, "current")
    reused = sorted(name for name, cell in old.items() if cell["dt_ms"] == 0.1)
    replaced = sorted(set(old) - set(reused))
    new = sorted(set(current) - set(reused))
    diagnostics = sorted(name for name, cell in current.items() if cell["seed"] == 42)
    expected_new = {
        f"ping__dt{dt}__seed{seed}"
        for dt in ("0p05", "0p2", "0p3", "0p6") for seed in (42, 43, 44)
    }
    expected_old = {
        f"ping__dt{dt}__seed{seed}"
        for dt in ("0p05", "0p25", "0p5", "1") for seed in (42, 43, 44)
    }
    if (len(reused) != 90 or set(new) != expected_new or set(replaced) != expected_old
            or len(diagnostics) != 34 or not set(reused) <= set(current)):
        raise PingstoreError("registry drift: expected the approved 90-reused/12-new partition")
    for name in reused:
        if old[name] != current[name]:
            raise PingstoreError(f"reused cell definition changed: {name}")
    for name in new:
        cell = current[name]
        dt_label, seed_label = name.removeprefix("ping__dt").split("__seed")
        if (cell["dt_ms"] != float(dt_label.replace("p", "."))
                or cell["seed"] != int(seed_label) or cell["family"] != "dt"
                or cell["training_run_id"] != "TR-04"):
            raise PingstoreError(f"replacement cell definition changed: {name}")
    if recipe.SCALE.get("refractory_e_ms") != 1.2 or recipe.SCALE.get("refractory_i_ms") != 0.6:
        raise PingstoreError("reuse requires the approved 1.2/0.6-ms model")
    if recipe.SCALE.get("refractory_policy") != "exact":
        raise PingstoreError("reuse requires exact refractory conversion")
    return old, {
        "schema": "exp022.bank-reuse/v1",
        "source": dict(SOURCE_REFERENCE),
        "reused_cells": reused,
        "new_cells": new,
        "replaced_cells": replaced,
        "diagnostics": diagnostics,
        "diagnostic_policy": "regenerate_all_34_seed42_final_epoch_snapshots",
    }


def _historical_parameters(cell: dict) -> dict:
    """Project only the established pre-adoption fields, never mutate saved metadata."""
    samples = cell.get("max_samples", recipe.SUBSET_MAX_SAMPLES)
    epochs = recipe.EPOCHS_STANDARD
    # Coarse historical cells cannot use the new exact-refractory conversion.
    # Their authored dt is restored after deriving the otherwise identical contract.
    scientific = recipe.scientific_contract(dict(cell, dt_ms=0.1), samples, epochs)
    scientific["dynamics"] = {
        key: value for key, value in scientific["dynamics"].items() if key in _LEGACY_DYNAMICS
    }
    scientific["dynamics"]["dt_ms"] = cell["dt_ms"]
    args = recipe.build_train_args(cell, Path("unused-output"), samples, epochs)
    parameters = campaign.resolved_parameters(cell, args, samples, epochs, scientific)
    parameters["arguments"].pop("--out-dir")
    for flag in _REFRACTORY_FLAGS:
        parameters["arguments"].pop(flag)
    return parameters


def _load_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise PingstoreError(f"cannot read retained cell metadata: {path}") from exc
    if not isinstance(value, dict):
        raise PingstoreError(f"retained cell metadata must be an object: {path}")
    return value


def _inspect_cell(source: SourceRun, cell: dict) -> dict:
    name = cell["name"]
    unit = source.unit(name)
    files = list(unit.iterdir())
    if {path.name for path in files} != set(CELL_FILES) or any(
        path.is_symlink() or not path.is_file() for path in files
    ):
        raise PingstoreError(f"{name}: expected exactly the four retained checkpoint-role files")
    config = _load_object(unit / "config.json")
    metrics = _load_object(unit / "metrics.json")
    nested = metrics.get("config")
    if not isinstance(nested, dict):
        raise PingstoreError(f"{name}: missing metrics configuration")
    for key in _PROVENANCE_FIELDS:
        if config.get(key) != metrics.get(key):
            raise PingstoreError(f"{name}: config/metrics disagree on {key}")
    if (config.get("training_cell_name") != name
            or config.get("training_run_id") != cell["training_run_id"]):
        raise PingstoreError(f"{name}: retained cell identity mismatch")
    expected = _historical_parameters(cell)
    observed = copy.deepcopy(config.get("campaign_resolved_parameters"))
    if not isinstance(observed, dict) or not isinstance(observed.get("arguments"), dict):
        raise PingstoreError(f"{name}: missing original resolved parameters")
    observed["arguments"].pop("--out-dir", None)
    if observed != expected:
        raise PingstoreError(f"{name}: retained scientific parameters disagree with the approved recipe")
    saved_expected = campaign._expected_config({"parameters": expected})
    for key, wanted in saved_expected.items():
        if not campaign._same(config.get(key), wanted):
            raise PingstoreError(f"{name}: saved config {key} disagrees with scientific parameters")
        # These three fields were absent from the historical nested metrics config.
        if key in nested or key not in {"model", "hidden_sizes", "surrogate_slope"}:
            if not campaign._same(nested.get(key), wanted):
                raise PingstoreError(f"{name}: metrics config {key} disagrees with scientific parameters")
    if any(key.startswith("refractory_") for payload in (config, nested) for key in payload):
        raise PingstoreError(f"{name}: historical source unexpectedly declares refractory parameters")
    repaired = cell["family"] == "low_w_in" or (
        cell["family"] == "activity_frontier" and cell.get("rate_target_hz") is not None
    )
    original = config.get("imported_cell_provenance")
    if repaired:
        if original is not None:
            raise PingstoreError(f"{name}: repaired training unexpectedly claims inherited provenance")
        original = {
            "repository_commit": config.get("campaign_repository_commit"),
            "campaign_id": config.get("campaign_id"),
            "campaign_manifest_sha256": config.get("campaign_manifest_sha256"),
        }
    if not isinstance(original, dict):
        raise PingstoreError(f"{name}: missing original inherited training provenance")
    commit = REPAIRED_COMMIT if repaired else INHERITED_COMMIT
    if original.get("repository_commit") != commit or config.get("git_sha") != commit[:8]:
        raise PingstoreError(f"{name}: original training revision mismatch")
    if not original.get("campaign_id") or not original.get("campaign_manifest_sha256"):
        raise PingstoreError(f"{name}: missing original campaign identity")
    epochs = metrics.get("epochs")
    if (not isinstance(epochs, list) or len(epochs) != expected["epochs"] or any(
        not isinstance(row, dict) or row.get("ep") != index
        or row.get("samples") != round(expected["max_samples"] * 0.9)
        for index, row in enumerate(epochs, 1)
    )):
        raise PingstoreError(f"{name}: training history does not cover the complete epoch/sample contract")
    try:
        checkpoints = {
            role: public_provenance(resolve_checkpoint(unit, role)) for role in ROLES
        }
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise PingstoreError(f"{name}: invalid retained checkpoint role: {exc}") from exc
    return {
        "source_cell": name,
        "seed": cell["seed"],
        "dt_ms": cell["dt_ms"],
        "training_run_id": cell["training_run_id"],
        "original_training": {"origin": "slurm", **copy.deepcopy(original)},
        "retained_provenance": {key: copy.deepcopy(config[key]) for key in _PROVENANCE_FIELDS
                                if key in config},
        "checkpoint_roles": checkpoints,
        "files": {filename: {"sha256": file_sha256(unit / filename),
                             "size_bytes": (unit / filename).stat().st_size}
                  for filename in CELL_FILES},
        "refractory_interpretation": {
            "source_declaration": "absent; fixed 12-E/6-I timestep counters",
            "e_ms": round(12 * cell["dt_ms"], 12), "i_ms": round(6 * cell["dt_ms"], 12),
            "reuse_validation": "PLAN.md step 4, pinned pre/post full-network preservation",
        },
    }


def inspect_source(repo: Path) -> tuple[SourceRun, dict]:
    """Validate the sole approved source and return its complete inspectable reuse plan."""
    registry, plan = _partition()
    source = source_run(
        Path(repo) / ".pingstore", SOURCE_REFERENCE["run_id"], stage="compute",
        experiment="exp022", reference=SOURCE_REFERENCE,
    )
    units = list(source.export.iterdir())
    if {unit.name for unit in units} != set(registry) or any(
        not unit.is_dir() or unit.is_symlink() for unit in units
    ):
        raise PingstoreError("retained bank must contain exactly the historical 102-cell registry")
    records = {name: _inspect_cell(source, cell) for name, cell in sorted(registry.items())}
    lineage = Counter(row["original_training"]["repository_commit"] for row in records.values())
    scientific = source.record.get("scientific_execution", {})
    if (lineage != {INHERITED_COMMIT: 60, REPAIRED_COMMIT: 42}
            or scientific.get("cells") != 102 or scientific.get("inherited_cells") != 60
            or scientific.get("retrained_cells") != 42 or scientific.get("origin") != "slurm"):
        raise PingstoreError("retained bank original training lineage disagrees with its run record")
    plan["per_cell"] = {name: records[name] for name in plan["reused_cells"]}
    plan["source_operation"] = {
        "origin": source.record["origin"],
        "operation": source.record["execution"].get("operation"),
        "scientific_origin": scientific["origin"],
    }
    source.check_unchanged()
    return source, plan
