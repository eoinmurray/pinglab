"""Resolve exp022 checkpoints by scientific role, with verified provenance.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import torch
from pingstore.library import PingstoreError, file_sha256

ROLES = {
    "best_validation": "weights.pth",
    "final_epoch": "weights_final.pth",
}

PURPOSE_ROLES = {
    "deployment_performance": "best_validation",
    "endpoint_dynamics": "final_epoch",
}


def checkpoint_tensors(path, training):
    n_in, n_e, n_i, n_out = (
        training[k] for k in ("n_in", "n_hidden", "n_inh", "n_out")
    )
    for key in (
        "adaptive_threshold",
        "train_leak",
        "signed_readout",
        "readout_bias",
        "state_clamp",
        "trainable_w_ee",
        "trainable_w_ii",
    ):
        if training.get(key, False) is not False:
            raise PingstoreError(f"unsupported checkpoint dynamics: {key}")
    if training.get("dales_law") is not True:
        raise PingstoreError("requires Dale-constrained bank weights")
    if training.get("hidden_sizes", [n_e]) != [n_e] or training["readout_mode"] not in (
        "mem-mean",
        "spike-count",
    ):
        raise PingstoreError(
            "requires one E/I layer with mean pre-reset voltage readout"
        )
    shapes = {
        "W_ff.0": (n_in, n_e),
        "W_ff.1": (n_e, n_out),
        "W_ee.1": (n_e, n_e),
        "W_ei.1": (n_e, n_i),
        "W_ie.1": (n_i, n_e),
        "W_ii.1": (n_i, n_i),
    }
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise PingstoreError("checkpoint must contain exactly six matrices")
    for key, shape in shapes.items():
        value = state[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(value.shape) != shape
            or not torch.isfinite(value).all()
        ):
            raise PingstoreError(f"invalid checkpoint tensor: {key}")
        if key in ("W_ee.1", "W_ii.1") and torch.count_nonzero(value):
            raise PingstoreError(f"unsupported same-population recurrence: {key}")
        if key in ("W_ei.1", "W_ie.1") and (value < 0).any():
            raise PingstoreError(f"negative recurrent weight: {key}")
    return state


def epoch_metrics(train_dir: Path) -> list[dict]:
    """Read the complete epoch record, with a legacy/in-progress JSONL fallback."""
    metrics_path = Path(train_dir) / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text())
        if "epochs" in metrics:
            rows = metrics["epochs"]
            if not isinstance(rows, list) or not all(
                isinstance(row, dict) for row in rows
            ):
                raise RuntimeError(f"invalid epoch records in {metrics_path}")
            return rows
    jsonl_path = Path(train_dir) / "metrics.jsonl"
    if not jsonl_path.is_file():
        return []
    return [
        json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()
    ]


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


def resolve_checkpoint(train_dir: Path, role: str) -> dict:
    """Return a verified checkpoint record for ``role`` or fail closed."""
    train_dir = Path(train_dir).resolve()
    if role not in ROLES:
        raise ValueError(
            f"unknown checkpoint role {role!r}; expected one of {sorted(ROLES)}"
        )
    metrics_path = train_dir / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read checkpoint metadata from {metrics_path}"
        ) from exc
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
    digest = file_sha256(path)
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
    return {
        key: record[key]
        for key in ("training_cell", "role", "filename", "epoch", "sha256")
    }


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
            raise RuntimeError(
                f"cannot resolve training horizon from {metrics_path}"
            ) from exc
    if not horizons:
        raise ValueError("training horizon requires at least one training directory")
    if len(horizons) != 1:
        raise RuntimeError(f"mixed upstream training horizons: {sorted(horizons)}")
    return horizons.pop()


def cache_tag(record: dict) -> str:
    """Stable suffix preventing cache reuse across checkpoint roles or contents."""
    return f"{record['role']}__{record['sha256'][:12]}"
