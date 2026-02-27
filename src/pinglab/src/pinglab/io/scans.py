from __future__ import annotations

from typing import Any

import numpy as np


def collect_scans(meta: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    scans: list[tuple[str, dict[str, Any]]] = []
    scan_default = meta.get("scan")
    if isinstance(scan_default, dict):
        scans.append(("scan", scan_default))
    for key in sorted(meta.keys()):
        if key.startswith("scan_") and isinstance(meta.get(key), dict):
            scans.append((key, meta[key]))
    return scans


def linspace_from_scan(scan: dict[str, Any]) -> np.ndarray:
    start = float(scan.get("start", 0.0))
    stop = float(scan.get("stop", start))
    steps = int(scan.get("steps", 1))
    if steps <= 0:
        raise ValueError("scan.steps must be >= 1")
    return np.linspace(start, stop, steps)


def scan_variant(parameter: str, scan_key: str) -> str:
    if parameter.endswith(".std"):
        return "std"
    if parameter.endswith(".mean"):
        return "mean"
    return scan_key
