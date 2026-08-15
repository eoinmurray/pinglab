"""Small helpers for replicated exp022 activity-frontier results."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt

import numpy as np


def summarize_frontier(rows: list[dict]) -> list[dict]:
    """Aggregate independent seeds at each model × rate-target point."""
    grouped: dict[tuple[str, float | None], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["rate_target_hz"])].append(row)

    summary = []
    for (model, rate_target_hz), points in grouped.items():
        points = sorted(points, key=lambda point: point["seed"])
        item = {
            "model": model,
            "rate_target_hz": rate_target_hz,
            "rate_target_display": points[0]["rate_target_display"],
            "seeds": [point["seed"] for point in points],
            "n_seeds": len(points),
            "cell_names": [point["cell_name"] for point in points],
            "statistic": "mean_across_independent_seeds",
            "uncertainty": "sem_across_independent_seeds",
        }
        for field in ("best_acc", "final_acc", "rate_e"):
            values = np.asarray([point[field] for point in points], dtype=float)
            item[field] = float(values.mean())
            item[f"{field}_sem"] = (
                float(values.std(ddof=1) / sqrt(len(values)))
                if len(values) > 1
                else 0.0
            )
        summary.append(item)
    return summary
