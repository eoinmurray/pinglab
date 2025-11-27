
import yaml
from pathlib import Path

from pinglab.types import ExperimentConfig


def load_config(path: Path) -> ExperimentConfig:
    with path.open() as f:
        data = yaml.safe_load(f)
    return ExperimentConfig.model_validate(data)
