"""Resolve exp022 checkpoints by scientific role, with verified provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

ROLES = {
    "best_validation": "weights.pth",
    "final_epoch": "weights_final.pth",
}

PURPOSE_ROLES = {
    "deployment_performance": "best_validation",
    "endpoint_dynamics": "final_epoch",
}


def checkpoint_policy(purpose: str) -> dict[str, str]:
    """Resolve a scientific analysis purpose to its checkpoint role."""
    try:
        role = PURPOSE_ROLES[purpose]
    except KeyError as exc:
        raise ValueError(
            f"unknown checkpoint purpose {purpose!r}; "
            f"expected one of {sorted(PURPOSE_ROLES)}"
        ) from exc
    return {"purpose": purpose, "role": role}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(train_dir: Path, role: str) -> dict:
    """Return a verified checkpoint record for ``role`` or fail closed."""
    train_dir = Path(train_dir).resolve()
    if role not in ROLES:
        raise ValueError(f"unknown checkpoint role {role!r}; expected one of {sorted(ROLES)}")
    metrics_path = train_dir / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read checkpoint metadata from {metrics_path}") from exc
    recorded = metrics.get("checkpoints", {}).get(role)
    if not isinstance(recorded, dict):
        raise RuntimeError(f"{metrics_path} does not register checkpoint role {role!r}")
    filename = recorded.get("filename")
    if filename != ROLES[role]:
        raise RuntimeError(
            f"{metrics_path} maps {role!r} to {filename!r}, expected {ROLES[role]!r}"
        )
    path = train_dir / filename
    if not path.is_file():
        raise RuntimeError(f"missing {role} checkpoint: {path}")
    digest = sha256_file(path)
    if recorded.get("sha256") != digest:
        raise RuntimeError(f"checkpoint hash mismatch for {path}")
    expected_epoch = (
        metrics.get("best_epoch")
        if role == "best_validation"
        else metrics.get("config", {}).get("epochs")
    )
    if int(recorded.get("epoch", -1)) != int(expected_epoch):
        raise RuntimeError(
            f"checkpoint epoch mismatch for {path}: {recorded.get('epoch')} != {expected_epoch}"
        )
    return {
        "training_cell": metrics.get("training_cell_name", train_dir.name),
        "role": role,
        "filename": filename,
        "epoch": int(recorded["epoch"]),
        "sha256": digest,
        "path": path,
    }


def public_provenance(record: dict) -> dict:
    """Drop the host-specific path before publishing checkpoint provenance."""
    return {key: record[key] for key in ("training_cell", "role", "filename", "epoch", "sha256")}


def checkpoint_provenance(train_dirs: Iterable[Path], role: str) -> list[dict]:
    records = [public_provenance(resolve_checkpoint(path, role)) for path in train_dirs]
    return sorted(records, key=lambda row: row["training_cell"])


def training_horizon(train_dirs: Iterable[Path]) -> int:
    """Return the common configured epoch count, failing on mixed inputs."""
    horizons = set()
    for train_dir in train_dirs:
        metrics_path = Path(train_dir).resolve() / "metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text())
            horizons.add(int(metrics["config"]["epochs"]))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"cannot resolve training horizon from {metrics_path}") from exc
    if not horizons:
        raise ValueError("training horizon requires at least one training directory")
    if len(horizons) != 1:
        raise RuntimeError(f"mixed upstream training horizons: {sorted(horizons)}")
    return horizons.pop()


def cache_tag(record: dict) -> str:
    """Stable suffix preventing cache reuse across checkpoint roles or contents."""
    return f"{record['role']}__{record['sha256'][:12]}"
