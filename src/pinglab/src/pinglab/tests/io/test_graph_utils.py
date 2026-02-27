from pinglab.io import layer_bounds_from_spec, set_edges_enabled


def test_set_edges_enabled_sets_enabled_flag_from_disabled_ids() -> None:
    spec = {
        "edges": [
            {"id": "a", "enabled": True},
            {"id": "b", "enabled": True},
        ]
    }
    out = set_edges_enabled(spec, {"b"})
    assert out["edges"][0]["enabled"] is True
    assert out["edges"][1]["enabled"] is False
    assert spec["edges"][1]["enabled"] is True


def test_layer_bounds_from_spec_accumulates_population_ranges() -> None:
    spec = {
        "nodes": [
            {"id": "input", "kind": "input", "size": 0},
            {"id": "E1", "kind": "population", "size": 3},
            {"id": "E2", "kind": "population", "size": 2},
            {"id": "I", "kind": "population", "size": 1},
        ]
    }
    assert layer_bounds_from_spec(spec) == [
        (0, 3, "E1"),
        (3, 5, "E2"),
        (5, 6, "I"),
    ]
