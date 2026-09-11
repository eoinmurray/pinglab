"""Numerical measurements of retained exp038 evidence; no simulation."""

from collections import defaultdict
from math import sqrt

import numpy as np

from . import recipe


def summarize_frontier(rows: list[dict]) -> list[dict]:
    """Aggregate retained training histories at each model x rate target."""
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


def summarize_ei_points(points: list[dict]) -> list[dict]:
    """Aggregate the E→I sweep across independently trained seeds."""
    summary = []
    for ei in sorted({float(point["ei_strength"]) for point in points}):
        rows = [point for point in points if float(point["ei_strength"]) == ei]
        row = {"ei_strength": ei}
        for field in ("acc", "hid_rate_hz", "inh_rate_hz"):
            values = np.asarray(
                [float(point.get(field) or 0.0) for point in rows], dtype=float
            )
            row[field] = float(values.mean())
            row[f"{field}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(row)
    return summary


def raster(directory, train, job):
    path = directory if directory.is_file() else directory / "recording.npz"
    with np.load(path, allow_pickle=False) as d:
        e, i = d["spk_e"], d["spk_i"]
        if e.ndim == 3:
            e = e[:, 0, :]
        if i.ndim == 3:
            i = i[:, 0, :]
        label = int(d["label"])
    rng = np.random.default_rng(0)
    ei = np.sort(rng.choice(e.shape[1], recipe.EI_RASTER_N_E_PLOT, replace=False))
    ii = np.sort(rng.choice(i.shape[1], recipe.EI_RASTER_N_I_PLOT, replace=False))
    result = {
        "label": label,
        "dt": float(train["dt"]),
        "t_ms": float(train["t_ms"]),
        "e": e[:, ei].astype(bool),
        "i": i[:, ii].astype(bool),
    }
    if job["kind"] == "rate_raster":
        seconds = float(train["t_ms"]) / 1000.0
        result.update(
            spike_rate=job["input_rate"],
            e_rate_hz=float(e.sum() / (e.shape[1] * seconds)),
            i_rate_hz=float(i.sum() / (i.shape[1] * seconds)) if i.shape[1] else 0.0,
        )
    else:
        result.update(seed=job["seed"], ei_strength=job["ei_strength"])
    return result
