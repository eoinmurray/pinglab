from __future__ import annotations

import copy
from typing import Any


def set_edges_enabled(spec: dict[str, Any], disabled_ids: set[str]) -> dict[str, Any]:
    updated = copy.deepcopy(spec)
    for edge in updated.get("edges", []):
        edge_id = str(edge.get("id", ""))
        edge["enabled"] = edge_id not in disabled_ids
    return updated


def layer_bounds_from_spec(spec: dict[str, Any]) -> list[tuple[int, int, str]]:
    bounds: list[tuple[int, int, str]] = []
    cursor = 0
    for node in spec.get("nodes", []):
        if node.get("kind") != "population":
            continue
        size = int(node.get("size", 0))
        if size <= 0:
            continue
        label = str(node.get("id", f"pop_{len(bounds) + 1}"))
        bounds.append((cursor, cursor + size, label))
        cursor += size
    return bounds
