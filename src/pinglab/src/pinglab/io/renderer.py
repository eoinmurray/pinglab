"""Graphviz-based network topology diagram renderer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import graphviz

# ── colour palettes ──────────────────────────────────────────────────────────

_LIGHT: dict[str, str] = {
    "bg": "white",
    "fg": "#333333",
    "node_E": "#ffd6d6",
    "node_I": "#d6e8ff",
    "node_input": "#f0f0f0",
    "node_border": "#333333",
    "edge_EE": "#cc0000",
    "edge_EI": "#dd6600",
    "edge_IE": "#0055cc",
    "edge_II": "#003399",
    "edge_input": "#888888",
}

_DARK: dict[str, str] = {
    "bg": "#18181b",
    "fg": "#cccccc",
    "node_E": "#6b1a1a",
    "node_I": "#1a3a6b",
    "node_input": "#2a2a2e",
    "node_border": "#cccccc",
    "edge_EE": "#ff6b6b",
    "edge_EI": "#ffaa44",
    "edge_IE": "#4dabf7",
    "edge_II": "#748ffc",
    "edge_input": "#aaaaaa",
}


def _edge_color(kind: str, theme: dict[str, str]) -> str:
    return {
        "EE": theme["edge_EE"],
        "EI": theme["edge_EI"],
        "IE": theme["edge_IE"],
        "II": theme["edge_II"],
        "input": theme["edge_input"],
    }.get(kind.upper(), theme["fg"])


def _edge_label(edge: dict[str, Any]) -> str:
    w = edge.get("w", {})
    mean = float(w.get("mean", 0.0))
    std = float(w.get("std", 0.0))
    delay = edge.get("delay_ms")
    label = f"μ={mean:.2f} σ={std:.2f}"
    if delay is not None:
        label += f"  d={float(delay):.1f}ms"
    return label


def _build(spec: dict[str, Any], theme: dict[str, str]) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        graph_attr={
            "bgcolor": theme["bg"],
            "rankdir": "LR",
            "size": "6,6!",
            "ratio": "fill",
            "margin": "0.5",
            "fontcolor": theme["fg"],
            "fontname": "Helvetica",
            "pad": "0.5",
        },
        node_attr={
            "fontname": "Helvetica",
            "fontcolor": theme["fg"],
            "color": theme["node_border"],
            "penwidth": "1.5",
            "style": "filled",
        },
        edge_attr={
            "fontname": "Helvetica",
            "fontsize": "9",
        },
    )

    for node in spec.get("nodes", []):
        node_id = str(node.get("id", ""))
        kind = str(node.get("kind", ""))
        ntype = str(node.get("type", ""))
        size = node.get("size", 0)

        if kind == "input":
            inputs = spec.get("inputs", {})
            mode = inputs.get(node_id, {}).get("mode", ntype)
            label = f"{node_id}\\n[{mode}]"
            dot.node(node_id, label=label, shape="box", fillcolor=theme["node_input"])
        elif kind == "population":
            label = f"{node_id}\\nn={size}"
            fill = theme["node_E"] if ntype == "E" else theme["node_I"]
            dot.node(node_id, label=label, shape="ellipse", fillcolor=fill)

    for edge in spec.get("edges", []):
        src = str(edge.get("from", ""))
        dst = str(edge.get("to", ""))
        kind = str(edge.get("kind", ""))
        color = _edge_color(kind, theme)
        label = _edge_label(edge)
        dot.edge(src, dst, label=label, color=color, fontcolor=color, penwidth="1.5")

    return dot


def save_graph_diagram(spec: dict[str, Any], path: Path) -> None:
    """Render network topology diagram. Saves <path>_light.png and <path>_dark.png."""
    base = str(Path(path)).removesuffix(".png")
    for suffix, theme in (("_light", _LIGHT), ("_dark", _DARK)):
        dot = _build(spec, theme)
        dot.render(filename=base + suffix, format="png", cleanup=True)
