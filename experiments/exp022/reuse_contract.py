"""Read-only contract for the exp110 COBA damping replacement bank."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.exp022 import compute, recipe
from experiments.exp022.checkpoints import (
    ROLES,
    public_provenance,
    resolve_checkpoint,
)
from pingstore.contracts import PingstoreError, file_sha256
from pingstore.stages import SourceRun, source_run

SOURCE_REFERENCE = {
    "run_id": "exp022-r007-compute",
    "payload_digest": "sha256:6bfeda8ce5e32bb35748f335338ee6af29bedf76ac606babc9823a466da640c0",
}
CELL_FILES = ("config.json", "metrics.json", "weights.pth", "weights_final.pth")


def replacement_cells() -> list[dict]:
    """The exact exp110 activity-frontier COBA training selection."""
    cells = [
        copy.deepcopy(cell)
        for cell in recipe.CANONICAL_CELLS
        if cell["training_run_id"] == "TR-02" and cell["model"] == "coba"
    ]
    names = {cell["name"] for cell in cells}
    expected = {
        recipe.cell_name("coba", target, seed)
        for target in recipe.RATE_TARGET_GRID_HZ
        for seed in recipe.SEEDS_BASELINE
    }
    if len(cells) != 18 or names != expected:
        raise PingstoreError("registry drift: expected the approved 18 TR-02 COBA cells")
    for cell in cells:
        args = recipe.build_train_args(
            cell,
            Path("unused-output"),
            *recipe.cell_samples_epochs(cell),
        )
        value = args[args.index("--v-grad-dampen") + 1]
        if value != "1000" or cell.get("recipe_overrides") != {
            "--v-grad-dampen": "1000"
        }:
            raise PingstoreError(f"replacement damping contract changed: {cell['name']}")
        if args[args.index("--ei-strength") + 1] != "0":
            raise PingstoreError(f"replacement COBA loop is not disabled: {cell['name']}")
    return cells


def _load_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise PingstoreError(f"cannot read retained cell metadata: {path}") from exc
    if not isinstance(value, dict):
        raise PingstoreError(f"retained cell metadata must be an object: {path}")
    return value


def _current_parameters(cell: dict) -> dict:
    samples, epochs = recipe.cell_samples_epochs(cell)
    args = recipe.build_train_args(cell, Path("unused-output"), samples, epochs)
    parameters = compute.resolved_parameters(
        cell,
        args,
        samples,
        epochs,
        scientific_contract=recipe.scientific_contract(cell, samples, epochs),
    )
    parameters["arguments"].pop("--out-dir")
    return parameters


def _inspect_reused_cell(source: SourceRun, cell: dict) -> dict:
    name = cell["name"]
    unit = source.unit(name)
    files = list(unit.iterdir())
    if {path.name for path in files} != set(CELL_FILES) or any(
        path.is_symlink() or not path.is_file() for path in files
    ):
        raise PingstoreError(
            f"{name}: expected exactly the four operational checkpoint-role files"
        )
    config = _load_object(unit / "config.json")
    metrics = _load_object(unit / "metrics.json")
    nested = metrics.get("config")
    if not isinstance(nested, dict):
        raise PingstoreError(f"{name}: missing metrics configuration")
    for payload, label in ((config, "config"), (metrics, "metrics")):
        if payload.get("training_cell_name") != name:
            raise PingstoreError(f"{name}: {label} cell identity mismatch")
        if payload.get("training_run_id") != cell["training_run_id"]:
            raise PingstoreError(f"{name}: {label} training-run identity mismatch")
    expected = compute._expected_config({"parameters": _current_parameters(cell)})
    for key, wanted in expected.items():
        actual = config.get(key, nested.get(key))
        if not compute._same(actual, wanted):
            raise PingstoreError(
                f"{name}: saved {key}={actual!r} disagrees with {wanted!r}"
            )
    epochs = metrics.get("epochs")
    expected_epochs = recipe.cell_samples_epochs(cell)[1]
    if (
        not isinstance(epochs, list)
        or len(epochs) != expected_epochs
        or any(not isinstance(row, dict) or row.get("ep") != index
               for index, row in enumerate(epochs, 1))
    ):
        raise PingstoreError(f"{name}: incomplete retained training history")
    try:
        checkpoints = {
            role: public_provenance(resolve_checkpoint(unit, role)) for role in ROLES
        }
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise PingstoreError(f"{name}: invalid retained checkpoint roles: {exc}") from exc
    return {
        "source_cell": name,
        "training_run_id": cell["training_run_id"],
        "checkpoint_roles": checkpoints,
        "files": {
            filename: {
                "sha256": file_sha256(unit / filename),
                "size_bytes": (unit / filename).stat().st_size,
            }
            for filename in CELL_FILES
        },
    }


def inspect_source(repo: Path) -> tuple[SourceRun, dict]:
    """Validate the pinned bank and return the exact 84/18 assembly plan."""
    current = {cell["name"]: cell for cell in recipe.CANONICAL_CELLS}
    if len(current) != 102:
        raise PingstoreError("current exp022 registry must contain 102 unique cells")
    replacements = {cell["name"] for cell in replacement_cells()}
    reused = sorted(set(current) - replacements)
    new = sorted(replacements)
    diagnostics = sorted(
        name for name, cell in current.items() if cell["seed"] == 42
    )
    if len(reused) != 84 or len(new) != 18 or len(diagnostics) != 34:
        raise PingstoreError("registry drift: expected 84 reused, 18 new and 34 probes")
    source = source_run(
        Path(repo) / ".pingstore",
        SOURCE_REFERENCE["run_id"],
        stage="compute",
        experiment="exp022",
        reference=SOURCE_REFERENCE,
    )
    units = list(source.export.iterdir())
    if {unit.name for unit in units} != set(current) or any(
        not unit.is_dir() or unit.is_symlink() for unit in units
    ):
        raise PingstoreError("retained bank must contain exactly the current 102-cell registry")
    records = {
        name: _inspect_reused_cell(source, current[name]) for name in reused
    }
    plan = {
        "schema": "exp022.bank-reuse/v2",
        "purpose": "exp110_coba_gradient_damping_alignment",
        "source": dict(SOURCE_REFERENCE),
        "reused_cells": reused,
        "new_cells": new,
        "replaced_cells": new,
        "diagnostics": diagnostics,
        "diagnostic_policy": "regenerate_all_34_seed42_final_epoch_snapshots",
        "per_cell": records,
        "source_operation": {
            "origin": source.record["origin"],
            "operation": source.record["execution"].get("operation"),
        },
    }
    source.check_unchanged()
    return source, plan
