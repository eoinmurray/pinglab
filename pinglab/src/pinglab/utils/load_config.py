"""Configuration file loading utilities."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from pinglab.types import ExperimentConfig


def load_config(path: Path | str) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.

    Parameters:
        path: Path to the YAML configuration file

    Returns:
        Validated ExperimentConfig object

    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the file contains invalid YAML
        ValidationError: If the config doesn't match the expected schema
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in {path}: {e}") from e

    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as e:
        raise ValidationError.from_exception_data(
            title=f"Invalid config in {path}",
            line_errors=e.errors(),
        ) from e
