"""Graphviz-based network topology diagram renderer."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import graphviz
from PIL import Image

# ── palettes ─────────────────────────────────────────────────────────────────

_LIGHT: dict[str, str] = {
    "bg":           "#FFFFFF",
    "fg":           "#1A1A1A",
    "node_E_fill":  "#FDE8E8",
    "node_E_pen":   "#C0392B",
    "node_I_fill":  "#E8F0FE",
    "node_I_pen":   "#2563EB",
    "node_in_fill": "#F5F5F5",
    "node_in_pen":  "#9CA3AF",
    "edge_EE":      "#C0392B",
    "edge_EI":      "#D97706",
    "edge_IE":      "#2563EB",
    "edge_II":      "#4F46E5",
    "edge_in":      "#9CA3AF",
}

_DARK: dict[str, str] = {
    "bg":           "#18181B",
    "fg":           "#E4E4E7",
    "node_E_fill":  "#3B1010",
    "node_E_pen":   "#F87171",
    "node_I_fill":  "#0D1F3C",
    "node_I_pen":   "#60A5FA",
    "node_in_fill": "#27272A",
    "node_in_pen":  "#71717A",
    "edge_EE":      "#F87171",
    "edge_EI":      "#FBB040",
    "edge_IE":      "#60A5FA",
    "edge_II":      "#818CF8",
    "edge_in":      "#71717A",
}

_FONT = "Courier New"


def _edge_color(kind: str, theme: dict[str, str]) -> str:
    return {
        "EE": theme["edge_EE"],
        "EI": theme["edge_EI"],
        "IE": theme["edge_IE"],
        "II": theme["edge_II"],
    }.get(kind.upper(), theme["edge_in"])


def _edge_label(edge: dict[str, Any]) -> str:
    w = edge.get("w", {})
    mean = float(w.get("mean", 0.0))
    std  = float(w.get("std",  0.0))
    delay = edge.get("delay_ms")
    label = f"μ={mean:.2f} σ={std:.2f}"
    if delay is not None:
        label += f"  d={float(delay):.1f}ms"
    return label


def _build(spec: dict[str, Any], theme: dict[str, str]) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        graph_attr={
            "bgcolor":   theme["bg"],
            "rankdir":   "LR",
            "nodesep":   "0.6",
            "ranksep":   "0.8",
            "margin":    "0.2",
            "pad":       "0.2",
            "fontname":  _FONT,
            "fontcolor": theme["fg"],
        },
        node_attr={
            "fontname":  _FONT,
            "fontsize":  "11",
            "fontcolor": theme["fg"],
            "shape":     "box",
            "style":     "filled",
            "width":     "1.1",
            "height":    "1.1",
            "fixedsize": "true",
        },
        edge_attr={
            "fontname":  _FONT,
            "fontsize":  "8",
        },
    )

    for node in spec.get("nodes", []):
        node_id = str(node.get("id", ""))
        kind    = str(node.get("kind", ""))
        ntype   = str(node.get("type", ""))
        size    = node.get("size", 0)

        if kind == "input":
            mode  = spec.get("inputs", {}).get(node_id, {}).get("mode", ntype)
            label = f"{node_id}\n[{mode}]"
            dot.node(node_id, label=label,
                     fillcolor=theme["node_in_fill"],
                     color=theme["node_in_pen"],
                     penwidth="1.5")
        elif kind == "population":
            label = f"{node_id}\nn={size}"
            fill = theme["node_E_fill"] if ntype == "E" else theme["node_I_fill"]
            pen  = theme["node_E_pen"]  if ntype == "E" else theme["node_I_pen"]
            dot.node(node_id, label=label,
                     fillcolor=fill, color=pen, penwidth="2.0")

    for edge in spec.get("edges", []):
        src   = str(edge.get("from", ""))
        dst   = str(edge.get("to",   ""))
        kind  = str(edge.get("kind", ""))
        color = _edge_color(kind, theme)
        label = _edge_label(edge)
        dot.edge(src, dst, label=label, color=color,
                 fontcolor=color, penwidth="1.5")

    return dot


def _render_square(dot: graphviz.Digraph, out_path: Path, bg_hex: str) -> None:
    """Render dot graph to a square PNG, padding with bg colour as needed."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    dot.render(filename=str(tmp_path.with_suffix("")), format="png", cleanup=True)

    img = Image.open(tmp_path).convert("RGBA")
    w, h = img.size
    size  = max(w, h)
    r, g, b = int(bg_hex[1:3], 16), int(bg_hex[3:5], 16), int(bg_hex[5:7], 16)
    canvas = Image.new("RGBA", (size, size), (r, g, b, 255))
    canvas.paste(img, ((size - w) // 2, (size - h) // 2), img)
    canvas.convert("RGB").save(out_path, "PNG")
    tmp_path.unlink(missing_ok=True)


def save_graph_diagram(spec: dict[str, Any], path: Path) -> None:
    """Render network topology to <path>_light.png and <path>_dark.png (square)."""
    base = str(Path(path)).removesuffix(".png")
    for suffix, theme in (("_light", _LIGHT), ("_dark", _DARK)):
        _render_square(_build(spec, theme), Path(base + suffix + ".png"), theme["bg"])
