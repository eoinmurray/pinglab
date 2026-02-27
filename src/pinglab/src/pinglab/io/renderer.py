from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphviz import Digraph
from graphviz.backend import ExecutableNotFound


def build_graphviz_from_spec(
    spec: dict[str, Any],
    *,
    title: str | None = None,
    include_disabled_edges: bool = True,
) -> Digraph:
    """Build a Graphviz directed graph from a pinglab graph spec."""
    graph = Digraph(comment=title or "")
    graph_attrs = {"rankdir": "TB", "fontsize": "18", "fontname": "Menlo"}
    if title:
        graph_attrs["labelloc"] = "t"
        graph_attrs["label"] = title
    graph.attr(**graph_attrs)
    graph.attr(
        "graph",
        size="8,8!",
        ratio="fill",
        ranksep="0.35",
        nodesep="0.2",
        dpi="180",
        fontname="Menlo",
        forcelabels="true",
        margin="0.05",
        pad="0.05",
    )
    graph.attr("node", style="filled", fontsize="12", fontname="Menlo")
    graph.attr(
        "edge",
        fontsize="10",
        fontname="Menlo",
        fontcolor="#6b7280",
    )

    node_by_id = {
        str(node.get("id", "")): node
        for node in spec.get("nodes", [])
        if isinstance(node, dict) and node.get("id") is not None
    }
    input_specs = spec.get("inputs", {})

    e_row_nodes: list[str] = []
    i_row_nodes: list[str] = []
    e_input_row_nodes: list[str] = []
    i_input_row_nodes: list[str] = []

    for node_id, node in node_by_id.items():
        kind = str(node.get("kind", ""))
        node_type = str(node.get("type", ""))
        size = int(node.get("size", 0))

        if kind == "population":
            fillcolor = "#fca5a5" if node_type == "E" else "#93c5fd" if node_type == "I" else "#d1d5db"
            shape = "square"
            if node_type == "E":
                e_row_nodes.append(node_id)
            elif node_type == "I":
                i_row_nodes.append(node_id)
        else:
            fillcolor = "#d1d5db"
            shape = "square"
            population_hint = ""
            if isinstance(input_specs, dict):
                raw_input_spec = input_specs.get(node_id, {})
                if isinstance(raw_input_spec, dict):
                    population_hint = str(raw_input_spec.get("input_population", "")).strip().lower()
            if not population_hint:
                # Fallback: infer from first outgoing edge target population type.
                for raw_edge in spec.get("edges", []):
                    if not isinstance(raw_edge, dict):
                        continue
                    if str(raw_edge.get("from", "")) != node_id:
                        continue
                    dst_id = str(raw_edge.get("to", ""))
                    dst_node = node_by_id.get(dst_id, {})
                    if isinstance(dst_node, dict):
                        dst_type = str(dst_node.get("type", "")).strip().upper()
                        if dst_type in {"E", "I"}:
                            population_hint = dst_type.lower()
                            break
            if population_hint.startswith("e"):
                e_input_row_nodes.append(node_id)
            elif population_hint.startswith("i"):
                i_input_row_nodes.append(node_id)

        lines = [node_id]
        if node_type:
            lines.append(f"type={node_type}")
        lines.append(f"n={size}")
        label = "\\n".join(lines)

        graph.node(node_id, label=label, shape=shape, fillcolor=fillcolor)

    if i_row_nodes:
        graph.body.append("{ rank=same; " + "; ".join(i_row_nodes) + "; }")
    if e_row_nodes:
        graph.body.append("{ rank=same; " + "; ".join(e_row_nodes) + "; }")
    if i_row_nodes and e_row_nodes:
        # Force I row above E row while allowing compact rank spacing.
        graph.edge(i_row_nodes[0], e_row_nodes[0], style="invis", weight="10")
    if len(e_row_nodes) > 1:
        # Preserve config order for E populations within their row.
        for left, right in zip(e_row_nodes[:-1], e_row_nodes[1:]):
            graph.edge(left, right, style="invis", weight="20")
    if e_input_row_nodes:
        graph.body.append("{ rank=source; " + "; ".join(e_input_row_nodes) + "; }")
    if i_input_row_nodes:
        graph.body.append("{ rank=sink; " + "; ".join(i_input_row_nodes) + "; }")

    for edge in spec.get("edges", []):
        if not isinstance(edge, dict):
            continue

        enabled = bool(edge.get("enabled", True))
        if not enabled and not include_disabled_edges:
            continue

        src = str(edge.get("from", ""))
        dst = str(edge.get("to", ""))
        edge_id = str(edge.get("id", ""))
        kind = str(edge.get("kind", ""))
        if not src or not dst:
            continue

        attrs: dict[str, str] = {}

        if not enabled:
            attrs["style"] = "dashed"
            attrs["color"] = "#9ca3af"

        graph.edge(src, dst, **attrs)

    return graph


def render_graphviz_spec(
    spec: dict[str, Any],
    output_path: str | Path,
    *,
    title: str | None = None,
    image_format: str = "png",
    include_disabled_edges: bool = True,
) -> Path:
    """Render a pinglab graph spec to an image via Graphviz."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    graph = build_graphviz_from_spec(
        spec,
        title=title,
        include_disabled_edges=include_disabled_edges,
    )

    # graphviz.render expects a filename without extension.
    try:
        rendered = graph.render(
            filename=output.stem,
            directory=str(output.parent),
            format=image_format,
            cleanup=True,
        )
    except ExecutableNotFound as exc:
        raise RuntimeError(
            "Graphviz 'dot' executable was not found on PATH. Install Graphviz "
            "(e.g. 'brew install graphviz') to enable image rendering."
        ) from exc
    return Path(rendered)


def render_graphviz_config(
    config_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
    image_format: str = "png",
    include_disabled_edges: bool = True,
) -> Path:
    """Load a graph config JSON file and render it via Graphviz."""
    config_file = Path(config_path)
    spec = json.loads(config_file.read_text(encoding="utf-8"))
    return render_graphviz_spec(
        spec,
        output_path,
        title=title,
        image_format=image_format,
        include_disabled_edges=include_disabled_edges,
    )
