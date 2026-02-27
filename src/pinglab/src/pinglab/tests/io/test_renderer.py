from pathlib import Path

import pinglab.io.renderer as renderer
from pinglab.io.renderer import build_graphviz_from_spec, render_graphviz_config


def test_build_graphviz_from_spec_includes_nodes_and_edges() -> None:
    spec = {
        "nodes": [
            {"id": "input_1", "kind": "input", "type": "tonic", "size": 0},
            {"id": "E", "kind": "population", "type": "E", "size": 4},
            {"id": "I", "kind": "population", "type": "I", "size": 2},
        ],
        "edges": [
            {"id": "in_to_e", "kind": "input", "from": "input_1", "to": "E", "w": {"mean": 1, "std": 0}},
            {"id": "i_to_e", "kind": "IE", "from": "I", "to": "E", "enabled": False},
        ],
    }

    dot = build_graphviz_from_spec(spec, title="Test Graph")
    source = dot.source

    assert "label=\"Test Graph\"" in source
    assert "input_1" in source
    assert "E" in source
    assert "I" in source
    assert "input_1 -> E" in source
    assert "style=dashed" in source


def test_build_graphviz_from_spec_groups_e_and_i_on_rows() -> None:
    spec = {
        "nodes": [
            {"id": "E1", "kind": "population", "type": "E", "size": 4},
            {"id": "E2", "kind": "population", "type": "E", "size": 4},
            {"id": "I1", "kind": "population", "type": "I", "size": 2},
            {"id": "I2", "kind": "population", "type": "I", "size": 2},
        ],
        "edges": [],
    }

    source = build_graphviz_from_spec(spec).source
    assert "{ rank=same; I1; I2; }" in source
    assert "{ rank=same; E1; E2; }" in source
    assert "I1 -> E1 [style=invis weight=10]" in source
    assert "E1 -> E2 [style=invis weight=20]" in source


def test_render_graphviz_config_writes_file(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"nodes":[{"id":"A","kind":"population","type":"E","size":1},{"id":"B","kind":"population","type":"I","size":1}],"edges":[{"id":"a_to_b","kind":"EI","from":"A","to":"B"}]}', encoding="utf-8")

    def fake_render(self, filename: str, directory: str, format: str, cleanup: bool) -> str:
        output_file = Path(directory) / f"{filename}.{format}"
        output_file.write_text("fake-image", encoding="utf-8")
        return str(output_file)

    monkeypatch.setattr(renderer.Digraph, "render", fake_render)

    output_path = tmp_path / "graph.png"
    rendered_path = render_graphviz_config(config_path, output_path)

    assert rendered_path.exists()
    assert rendered_path.suffix == ".png"


def test_build_graphviz_from_spec_groups_input_rows_by_population() -> None:
    spec = {
        "nodes": [
            {"id": "in_e", "kind": "input", "type": "tonic", "size": 0},
            {"id": "in_i", "kind": "input", "type": "tonic", "size": 0},
            {"id": "E1", "kind": "population", "type": "E", "size": 4},
            {"id": "I1", "kind": "population", "type": "I", "size": 2},
        ],
        "edges": [
            {"id": "in_e_to_e1", "kind": "input", "from": "in_e", "to": "E1"},
            {"id": "in_i_to_i1", "kind": "input", "from": "in_i", "to": "I1"},
        ],
        "inputs": {
            "in_e": {"input_population": "e"},
            "in_i": {"input_population": "i"},
        },
    }

    source = build_graphviz_from_spec(spec).source
    assert "{ rank=source; in_e; }" in source
    assert "{ rank=sink; in_i; }" in source
