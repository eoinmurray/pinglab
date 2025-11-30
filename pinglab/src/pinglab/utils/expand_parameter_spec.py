"""Parameter sweep expansion utilities."""

import itertools
from typing import Any

import numpy as np


def expand_parameter_spec(
    spec: dict[str, dict[str, Any]] | None,
    mode: str,
) -> list[dict[str, Any]]:
    """
    Expand a parameter specification into a list of parameter dictionaries.

    Parameters:
        spec: Dict mapping parameter names to spec dicts with 'type', 'start', 'stop', 'num', etc.
              Supported types: 'linspace', 'range', 'values'
        mode: Expansion mode ('grid' for Cartesian product, 'random' not yet implemented)

    Returns:
        List of dicts, each containing parameter overrides for one run
    """
    if not spec:
        return [{}]  # No parameters to sweep

    # Generate values for each parameter
    param_values = {}
    for param_name, param_spec in spec.items():
        param_type = param_spec.get("type")

        if param_type == "linspace":
            values = np.linspace(
                param_spec["start"], param_spec["stop"], param_spec["num"]
            )
        elif param_type == "range":
            values = np.arange(
                param_spec["start"], param_spec["stop"], param_spec.get("step", 1)
            )
        elif param_type == "values":
            values = param_spec["values"]
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

        param_values[param_name] = values

    # Generate combinations based on mode
    if mode == "grid":
        combinations = list(itertools.product(*param_values.values()))
        param_names = list(param_values.keys())
        param_dicts = [dict(zip(param_names, combo)) for combo in combinations]
    elif mode == "random":
        raise NotImplementedError("Random mode not yet implemented")
    else:
        raise ValueError(f"Unknown rungroup mode: {mode}")

    return param_dicts
