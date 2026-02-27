from __future__ import annotations

import copy
import re
from typing import Any


_INDEXED_SEGMENT_RE = re.compile(r"^([^\[\]]+)\[([^\[\]]+)\]$")


def _find_by_id(items: list[Any], item_id: str, *, path: str) -> dict[str, Any]:
    matches = [x for x in items if isinstance(x, dict) and str(x.get("id")) == item_id]
    if not matches:
        raise ValueError(f"path '{path}': id '{item_id}' not found")
    if len(matches) > 1:
        raise ValueError(f"path '{path}': id '{item_id}' is ambiguous")
    return matches[0]


def _resolve_root_alias(spec: dict[str, Any], segment: str, *, path: str) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for value in spec.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict) and str(item.get("id")) == segment:
                matches.append(item)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"path '{path}': root alias '{segment}' is ambiguous")
    return matches[0]


def _resolve_segment(current: Any, segment: str, *, path: str, is_root: bool) -> Any:
    indexed = _INDEXED_SEGMENT_RE.match(segment)
    if indexed:
        key = indexed.group(1)
        item_id = indexed.group(2)
        if not isinstance(current, dict):
            raise ValueError(f"path '{path}': segment '{segment}' requires an object")
        if key not in current:
            raise ValueError(f"path '{path}': key '{key}' not found")
        items = current[key]
        if not isinstance(items, list):
            raise ValueError(f"path '{path}': key '{key}' is not a list")
        return _find_by_id(items, item_id, path=path)

    if isinstance(current, dict):
        if segment in current:
            return current[segment]
        if is_root:
            alias = _resolve_root_alias(current, segment, path=path)
            if alias is not None:
                return alias
        raise ValueError(f"path '{path}': key '{segment}' not found")

    if isinstance(current, list):
        try:
            idx = int(segment)
        except ValueError as exc:
            raise ValueError(f"path '{path}': list segment '{segment}' is not an integer") from exc
        if idx < 0 or idx >= len(current):
            raise ValueError(f"path '{path}': list index {idx} out of range")
        return current[idx]

    raise ValueError(f"path '{path}': cannot traverse segment '{segment}' on scalar value")


def overwrite_spec_value_inplace(spec: dict[str, Any], path: str, value: Any) -> None:
    """
    Overwrite a value inside a graph spec using a path, mutating in place.

    Supported path forms:
    - dot notation: ``sim.seed``
    - list index: ``nodes.0.size``
    - id lookup in list field: ``edges[e_to_e].w.std``
    - root alias by id: ``e_to_e.w.std`` (resolves across top-level list fields)
    """
    if not isinstance(spec, dict):
        raise ValueError("spec must be an object")
    segments = [s for s in path.split(".") if s]
    if not segments:
        raise ValueError("path must not be empty")

    current: Any = spec
    for i, segment in enumerate(segments[:-1]):
        current = _resolve_segment(current, segment, path=path, is_root=(i == 0))

    last = segments[-1]
    indexed = _INDEXED_SEGMENT_RE.match(last)
    if indexed:
        key = indexed.group(1)
        item_id = indexed.group(2)
        if not isinstance(current, dict):
            raise ValueError(f"path '{path}': segment '{last}' requires an object")
        if key not in current:
            raise ValueError(f"path '{path}': key '{key}' not found")
        items = current[key]
        if not isinstance(items, list):
            raise ValueError(f"path '{path}': key '{key}' is not a list")
        target = _find_by_id(items, item_id, path=path)
        target.clear()
        if isinstance(value, dict):
            target.update(value)
            return
        raise ValueError(f"path '{path}': replacing an id-target requires an object value")

    if isinstance(current, dict):
        if last not in current:
            raise ValueError(f"path '{path}': key '{last}' not found")
        current[last] = value
        return

    if isinstance(current, list):
        try:
            idx = int(last)
        except ValueError as exc:
            raise ValueError(f"path '{path}': list segment '{last}' is not an integer") from exc
        if idx < 0 or idx >= len(current):
            raise ValueError(f"path '{path}': list index {idx} out of range")
        current[idx] = value
        return

    raise ValueError(f"path '{path}': cannot assign on scalar value")


def overwrite_spec_value(spec: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """
    Return a copied spec with one value overwritten by path.
    """
    out = copy.deepcopy(spec)
    overwrite_spec_value_inplace(out, path, value)
    return out


def spec_with_overwrite(spec: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """
    Alias for overwrite_spec_value for readability at call sites.
    """
    return overwrite_spec_value(spec, path, value)
