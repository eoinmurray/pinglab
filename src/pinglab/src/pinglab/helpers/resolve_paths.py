from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentPaths:
    config_path: Path
    artifact_dir: Path


def resolve_paths(*, experiment_dir: Path, artifacts_root: Path) -> ExperimentPaths:
    artifact_dir = artifacts_root / experiment_dir.name
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        config_path=experiment_dir / "config.json",
        artifact_dir=artifact_dir,
    )
