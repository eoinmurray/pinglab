import pytest

from pinglab.io import (
    overwrite_spec_value,
    overwrite_spec_value_inplace,
    spec_with_overwrite,
)


def _base_spec() -> dict:
    return {
        "schema_version": "pinglab-graph.v1",
        "sim": {"seed": 0},
        "nodes": [
            {"id": "E", "kind": "population", "type": "E", "size": 400},
        ],
        "edges": [
            {"id": "e_to_e", "kind": "EE", "from": "E", "to": "E", "w": {"mean": 0.0, "std": 0.01}},
        ],
        "inputs": {},
    }


def test_overwrite_spec_value_updates_nested_path_and_returns_copy() -> None:
    spec = _base_spec()
    out = overwrite_spec_value(spec, "sim.seed", 7)
    assert out["sim"]["seed"] == 7
    assert spec["sim"]["seed"] == 0


def test_overwrite_spec_value_updates_edge_value_by_root_alias() -> None:
    spec = _base_spec()
    out = overwrite_spec_value(spec, "e_to_e.w.std", 0.007)
    assert out["edges"][0]["w"]["std"] == 0.007
    assert spec["edges"][0]["w"]["std"] == 0.01


def test_overwrite_spec_value_updates_edge_value_by_indexed_segment() -> None:
    spec = _base_spec()
    overwrite_spec_value_inplace(spec, "edges[e_to_e].w.mean", 0.11)
    assert spec["edges"][0]["w"]["mean"] == 0.11


def test_spec_with_overwrite_returns_copy() -> None:
    spec = _base_spec()
    out = spec_with_overwrite(spec, "e_to_e.w.std", 0.005)
    assert out["edges"][0]["w"]["std"] == 0.005
    assert spec["edges"][0]["w"]["std"] == 0.01


def test_overwrite_spec_value_raises_on_missing_path() -> None:
    spec = _base_spec()
    with pytest.raises(ValueError, match="not found"):
        overwrite_spec_value_inplace(spec, "sim.unknown", 1)
