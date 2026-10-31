"""Compile and draw an illustrative Bosman-style A/B-to-C circuit.

Separate sensory and attention spike inputs drive the two V1-like circuits.
Their excitatory populations converge symmetrically on a V4-like E/I circuit.
An independent background spike-input port also drives the receiver's E/I cells.
Sizes and weights are illustrative; this script does not simulate activity.

Run from the repository root with `.venv/bin/python
notes/ctc-attention/figures/draw.py`. The native bundle and SVG/PNG diagram
are written beside this script, using SNNLab's circuit renderer.
"""

import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import matplotlib
from snnlab import lang, viz

matplotlib.use("Agg")
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from experiments import theme  # noqa: E402 - resolve the shared repository palette.

SIZES = dict(E=128, I=32, input=64)
DIRECTORY = Path(__file__).resolve().parent


def author_network():
    """Build three recurrent E/I circuits with equal afferent pathways into C."""
    net = lang.Network("CTC attention: A and B converge on C", dt=0.1 * lang.ms)
    cells, drives = {}, {}
    for circuit in ("A", "B", "C"):
        with net.group(circuit):
            for kind in ("E", "I"):
                inhibitory = kind == "I"
                name = f"{circuit}_{kind}"
                cells[name] = net.population(
                    name,
                    size=SIZES[kind],
                    neuron=lang.COBA_LIF(
                        tau_mem=(5.0 if inhibitory else 20.0) * lang.ms,
                        capacitance_nf=0.5 if inhibitory else 1.0,
                        leak_us=0.10 if inhibitory else 0.05,
                        resting_mv=-65.0,
                        threshold_mv=-50.0,
                        reset_mv=-65.0,
                        initial_voltage_mv=-65.0,
                        refractory=(0.6 if inhibitory else 1.2) * lang.ms,
                    ),
                )
    for circuit in ("A", "B"):
        for kind in ("sensory", "attention"):
            name = f"{kind}_{circuit}"
            drives[name] = net.input(
                name,
                shape=("time", "batch", SIZES["input"]),
                signal_type="spikes",
                unit="spike",
            )
    drives["background_C"] = net.input(
        "background_C",
        shape=("time", "batch", SIZES["input"]),
        signal_type="spikes",
        unit="spike",
    )

    def connect(source, target, incoming, *, inhibitory=False, recurrent=False):
        external = source in drives
        count = SIZES["input"] if external else SIZES[source[-1]]
        net.connect(
            drives[source] if external else cells[source].spikes,
            cells[target].inhibitory if inhibitory else cells[target].excitatory,
            name=f"{source}_to_{target}",
            synapse=(
                lang.GABA(tau=6 * lang.ms) if inhibitory else lang.AMPA(tau=2 * lang.ms)
            ),
            weight=lang.Constant(incoming / count),
            initialization_scaling="direct",
            constraint=lang.NonNegative(),
            connection="recurrent" if recurrent else "feedforward",
            delay=0.1 * lang.ms if recurrent else None,
        )

    for circuit in ("A", "B", "C"):
        connect(f"{circuit}_E", f"{circuit}_E", 0.5, recurrent=True)
        connect(f"{circuit}_E", f"{circuit}_I", 1.0, recurrent=True)
        connect(f"{circuit}_I", f"{circuit}_E", 2.0, inhibitory=True, recurrent=True)
        connect(f"{circuit}_I", f"{circuit}_I", 1.0, inhibitory=True, recurrent=True)
    for circuit in ("A", "B"):
        connect(f"sensory_{circuit}", f"{circuit}_E", 0.9)
        connect(f"attention_{circuit}", f"{circuit}_E", 0.1)
        connect(f"{circuit}_E", "C_E", 0.5)
        connect(f"{circuit}_E", "C_I", 0.5)
        for kind in ("E", "I"):
            connect("C_E", f"{circuit}_{kind}", 0.1, recurrent=True)
    for kind in ("E", "I"):
        connect("background_C", f"C_{kind}", 0.1)
    return lang.compile(net)


def optional_connection(source, target):
    """Mark same-type local recurrence and descending excitation as optional."""
    return source == target or (source == "C_E" and target.split("_")[0] in ("A", "B"))


def drawing_graph(bundle):
    """Repeat feedback endpoints in an inset without changing the native graph."""
    graph = lang.diagram(bundle, view="circuit")
    feedback = tuple(
        edge
        for edge in graph.edges
        if edge.source == "C_E" and edge.target.startswith(("A_", "B_"))
    )
    endpoints = {name for edge in feedback for name in (edge.source, edge.target)}
    duplicates = tuple(
        replace(node, id=f"feedback_{node.id}")
        for node in graph.nodes
        if node.id in endpoints
    )
    return replace(
        graph,
        name="Converging inputs and local E/I circuits",
        title="A / B → C · Illustrative circuit",
        nodes=graph.nodes + duplicates,
        edges=tuple(edge for edge in graph.edges if edge not in feedback)
        + tuple(
            replace(
                edge, source=f"feedback_{edge.source}", target=f"feedback_{edge.target}"
            )
            for edge in feedback
        ),
        groups=graph.groups
        + (
            type(graph.groups[0])(
                id="feedback",
                label="Optional feedback",
                members=tuple(n.id for n in duplicates),
            ),
        ),
    )


def presentation(graph):
    """Keep feedforward routes central and optional feedback in a separate inset."""
    positions = dict(
        sensory_A=(20, 240),
        attention_A=(20, 405),
        A_E=(400, 240),
        A_I=(680, 240),
        sensory_B=(20, 780),
        attention_B=(20, 945),
        B_E=(400, 780),
        B_I=(680, 780),
        C_E=(1260, 510),
        C_I=(1540, 510),
        background_C=(1260, 930),
        feedback_C_E=(1930, 510),
        feedback_A_E=(2330, 220),
        feedback_A_I=(2330, 440),
        feedback_B_E=(2330, 720),
        feedback_B_I=(2330, 940),
    )
    positions = {
        name: (x + (100 if name.startswith("feedback_") else 0), y)
        for name, (x, y) in positions.items()
    }

    def symbol(identifier):
        identifier = identifier.removeprefix("feedback_")
        first, second = identifier.split("_")
        if first in ("sensory", "attention", "background"):
            return rf"{'U' if first == 'attention' else 'X'}_{{{second}}}"
        return rf"{second}_{{{first}}}"

    labels = {name: rf"${symbol(name)}$" for name in positions}
    ports = dict(
        ul=(0.25, 0),
        ur=(0.75, 0),
        bl=(0.25, 1),
        br=(0.75, 1),
        in_upper=(0, 0.25),
        in_lower=(0, 0.75),
        out_upper=(1, 0.25),
        out_lower=(1, 0.75),
        bottom_input=(0.95, 1),
    )
    node_layouts = {
        name: viz.NodeLayout(
            position=point,
            anchor="top_left",
            label=labels[name],
            min_size=(180, 130),
            ports=ports,
        )
        for name, point in positions.items()
    }
    node_layouts["background_C"] = replace(
        node_layouts["background_C"],
        annotations=(
            viz.DiagramLabel("Background C", anchor="bottom", offset=(0, 16)),
        ),
    )
    edge_layouts = {}
    for edge in graph.edges:
        source, target = edge.source, edge.target
        label = viz.DiagramLabel(rf"$W_{{{symbol(target)},{symbol(source)}}}$")
        if source.startswith("feedback_"):
            source_y = dict(
                feedback_A_E=0.15,
                feedback_A_I=0.38,
                feedback_B_E=0.62,
                feedback_B_I=0.85,
            )[target]
            port = f"to_{target}"
            node_layouts[source] = replace(
                node_layouts[source],
                ports={
                    **node_layouts[source].ports,
                    port: (1, source_y),
                },
            )
            route = dict(source_port=port, target_port="left", routing="polyline")
            label = replace(label, position=0.75, offset=(0, -25))
        elif source == target:
            x, y = positions[source]
            excitatory = source.endswith("_E")
            loop_y = y - 65 if excitatory else y + 195
            route = dict(
                source_port="ur" if excitatory else "br",
                target_port="ul" if excitatory else "bl",
                waypoints=((x + 135, loop_y), (x + 45, loop_y)),
            )
            label = replace(label, position=0.5, offset=(0, 25))
        elif source in ("A_E", "B_E") and target == "C_I":
            upper = source == "A_E"
            route = dict(
                source_port="br" if upper else "ur",
                target_port="top" if upper else "bottom",
                waypoints=((535, 440 if upper else 710), (1630, 440 if upper else 710)),
            )
            label = replace(
                label,
                position=0.82 if upper else 0.62,
                offset=(0, -25 if upper else 25),
            )
        elif source in ("A_E", "B_E") and target == "C_E":
            upper = source == "A_E"
            route = dict(
                source_port="br" if upper else "ur",
                target_port="in_upper" if upper else "in_lower",
                waypoints=(
                    (535, 485 if upper else 735),
                    (1080, 485 if upper else 735),
                    (1080, 542.5 if upper else 607.5),
                ),
            )
            label = replace(label, position=0.5, offset=(0, 25))
        elif source == "background_C":
            excitatory_target = target == "C_E"
            route = dict(
                source_port="top" if excitatory_target else "right",
                target_port="bottom" if excitatory_target else "bottom_input",
                waypoints=() if excitatory_target else ((1711, 995),),
            )
            label = replace(label, position=0.5, offset=(55, 0))
        elif source.startswith("sensory_"):
            route = dict(source_port="right", target_port="in_upper")
            label = replace(label, position=0.5, offset=(0, -25))
        elif source.startswith("attention_"):
            route = dict(
                source_port="right",
                target_port="in_lower",
                waypoints=(
                    (300, positions[source][1] + 65),
                    (300, positions[target][1] + 97.5),
                ),
            )
            label = replace(label, position=0.6, offset=(15, 0))
        elif source.endswith("_E"):
            route = dict(source_port="out_upper", target_port="in_upper")
            label = replace(label, position=0.5, offset=(0, -25))
        else:
            route = dict(source_port="in_lower", target_port="out_lower")
            label = replace(label, position=0.5, offset=(0, 25))
        edge_layouts[edge.id] = viz.EdgeLayout(
            label=label, **{"routing": "curved", **route}
        )
    titles = dict(
        A="A · V1 stimulus 1",
        B="B · V1 stimulus 2",
        C="C · V4 receiver",
        feedback="Optional feedback",
    )
    layout = viz.DiagramLayout(
        canvas=viz.CanvasLayout(size=(1920, 1080), fit="contain", padding=70),
        nodes=node_layouts,
        edges=edge_layouts,
        groups={
            name: viz.GroupLayout(
                label=viz.DiagramLabel(title, offset=(-76, -76)), padding=100
            )
            for name, title in titles.items()
        },
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=12), gap=24
        ),
    )
    palette = viz.DiagramTheme(
        background=theme.PAPER,
        ink=theme.INK_BLACK,
        muted=theme.MUTED,
        inhibitory=theme.DEEP_RED,
        line=theme.RULE_WARM,
        population_fill=theme.PAPER_TINT,
        population_border=theme.INK_BLACK,
        input_fill=theme.PAPER_TINT,
        input_border=theme.INK_BLACK,
        font_name="DejaVu Sans",
        label_size=18,
        secondary_size=14,
    )
    return layout, palette


def embed_font_outlines(path):
    """Keep DejaVu Sans identical in PNG and SVG, including browser previews."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path as VectorPath
    from matplotlib.textpath import TextPath

    namespace = "{http://www.w3.org/2000/svg}"
    tree = ET.parse(path)
    commands = {
        VectorPath.MOVETO: "M",
        VectorPath.LINETO: "L",
        VectorPath.CURVE3: "Q",
        VectorPath.CURVE4: "C",
        VectorPath.CLOSEPOLY: "Z",
    }
    for parent in list(tree.getroot().iter()):
        for index, element in enumerate(list(parent)):
            if element.tag != namespace + "text":
                continue
            text = element.text or ""
            glyphs = TextPath(
                (0, 0),
                text,
                size=float(element.get("font-size")),
                prop=FontProperties(family="DejaVu Sans"),
            )
            outline = " ".join(
                commands[code]
                + (
                    " " + " ".join(f"{v:.5f}" for v in vertices)
                    if code != VectorPath.CLOSEPOLY
                    else ""
                )
                for vertices, code in glyphs.iter_segments(curves=True, simplify=False)
            )
            group = ET.Element(
                namespace + "g",
                {
                    "id": element.get("id"),
                    "data-snnlab-id": element.get("data-snnlab-id", ""),
                    "aria-label": text,
                    "transform": f"translate({element.get('x')} {element.get('y')}) scale(1 -1)",
                },
            )
            ET.SubElement(group, namespace + "title").text = text
            ET.SubElement(
                group, namespace + "path", {"d": outline, "fill": element.get("fill")}
            )
            parent.remove(element)
            parent.insert(index, group)
    tree.write(path, encoding="unicode", xml_declaration=True)


def main():
    bundle = author_network()
    bundle.write(DIRECTORY / "network.bundle", visualise=False)
    graph = drawing_graph(bundle)
    # Native metadata stays in the bundle; compact cards need only their symbols.
    graph = replace(graph, nodes=tuple(replace(n, detail="") for n in graph.nodes))
    layout, palette = presentation(graph)
    with matplotlib.rc_context({"savefig.bbox": None}):
        diagram = viz.resolve_diagram(graph, layout=layout, theme=palette)
        elements = []
        for element in diagram.scene["elements"]:
            metadata = element.get("data", {})
            if element["type"] == "arrow":
                source = metadata["source"].removeprefix("feedback_")
                target = metadata["target"].removeprefix("feedback_")
                local = source.split("_")[0] == target.split("_")[0]
                if local:
                    colour = (
                        theme.DEEP_RED
                        if metadata["role"] == "inhibitory"
                        else theme.INK_BLACK
                    )
                    colour = matplotlib.colors.to_rgb(colour)
                    # Composite on white for matching SVG and PNG faintness.
                    colour = matplotlib.colors.to_hex(
                        tuple(0.45 * c + 0.55 for c in colour)
                    )
                    element = {**element, "strokeColor": colour, "strokeWidth": 2.0}
                elif source.startswith(("A_", "B_", "C_")):
                    element = {**element, "strokeWidth": 3.2}
                if optional_connection(source, target):
                    element = {**element, "strokeStyle": "dashed"}
            if metadata.get("node_kind") in ("population", "input"):
                is_title = element["type"] == "image"
                if element["type"] == "text" and "annotation" not in metadata:
                    continue
                if is_title:
                    box = diagram.nodes[metadata["id"]]
                    element = {
                        **element,
                        "y": box.y + (box.height - element["height"]) / 2,
                    }
            elements.append(element)
        diagram = replace(diagram, scene={**diagram.scene, "elements": elements})
        for extension in ("svg", "png"):
            output = DIRECTORY / f"network.{extension}"
            viz.render_diagram(diagram, output)
            if extension == "svg":
                embed_font_outlines(output)
    print(DIRECTORY / "network.svg")
    print(DIRECTORY / "network.png")


if __name__ == "__main__":
    main()
