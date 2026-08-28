"""Pure aggregation of retained correctness; no simulation or file writes."""

from typing import Any

import numpy as np

from . import recipe


def analyze(correctness: np.ndarray, cfg: dict | None = None) -> dict[str, Any]:
    cfg = recipe.configuration() if cfg is None else cfg
    if (
        correctness.dtype != np.bool_
        or correctness.ndim != 3
        or correctness.shape[:2] != (len(cfg["rates_hz"]), len(cfg["seeds"]))
        or correctness.shape[2] == 0
    ):
        raise ValueError("expected nonempty boolean rate/seed/image correctness array")
    rows = []
    for rate_index, rate in enumerate(cfg["rates_hz"]):
        per_seed = correctness[rate_index].mean(axis=1)
        rows.append(
            {
                "rate_hz": rate,
                "accuracy": float(per_seed.mean()),
                "minimum_seed_accuracy": float(per_seed.min()),
                "maximum_seed_accuracy": float(per_seed.max()),
                "per_seed_accuracy": per_seed.tolist(),
            }
        )
    floor = next(
        (
            row["rate_hz"]
            for row in rows
            if row["minimum_seed_accuracy"] >= cfg["useful_accuracy"]
        ),
        None,
    )
    decision = {
        "criterion_crossed": floor is not None,
        "r_train_hz": floor,
        "recommendation": {"floor_hz": floor, "ceiling_hz": max(cfg["rates_hz"])},
        "rows": rows,
    }
    return decision
