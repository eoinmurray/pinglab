"""Deterministic, deliberately styled Graphviz reports."""

from __future__ import annotations

import html
import subprocess
import tempfile
from collections.abc import Collection
from pathlib import Path

from .compiler import Bundle

PALETTE = {
    "ink": "#243044",
    "muted": "#667085",
    "line": "#CBD5E1",
    "input": "#F5F7FA",
    "exc": "#F5F7FA",
    "inh": "#F5F7FA",
    "neutral": "#F5F7FA",
    "output": "#FFF4D6",
    "mod": "#F1EAFE",
    "train": "#EAF2FF",
}


def _q(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _node_id(value: str) -> str:
    return "n_" + "".join(c if c.isalnum() else "_" for c in value)


def _card(title: str, subtitle: str, badge: str, fill: str) -> str:
    return (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="15"><B>{html.escape(title)}</B></FONT></TD></TR>'
        f'<TR><TD HEIGHT="5"></TD></TR>'
        f'<TR><TD ALIGN="LEFT"><FONT COLOR="{PALETTE["muted"]}" POINT-SIZE="10">{html.escape(subtitle)}</FONT></TD></TR>'
        f'<TR><TD HEIGHT="7"></TD></TR>'
        f'<TR><TD ALIGN="LEFT" BGCOLOR="{fill}" CELLPADDING="4"><FONT COLOR="{PALETTE["ink"]}" POINT-SIZE="9">{html.escape(badge.upper())}</FONT></TD></TR>'
        "</TABLE>>"
    )


def _dot(
    bundle: Bundle,
    view: str,
    *,
    expand_groups: Collection[str] = (),
) -> str:
    if view not in {"circuit", "training", "expanded"}:
        raise ValueError("view must be circuit, training, or expanded")
    graph = bundle.graph
    if len(graph["projections"]) + len(graph["operations"]) > 120 and view != "circuit":
        raise ValueError(
            "graph is too dense for a legible expanded static view; filter or collapse it"
        )
    groups = {g["id"]: g for g in graph["groups"]}
    collapsed = view in {"circuit", "training"}
    expanded_groups = set(expand_groups)
    if expanded_groups and not collapsed:
        raise ValueError("expand_groups is only available for circuit and training views")
    unknown_groups = expanded_groups - groups.keys()
    if unknown_groups:
        unknown = ", ".join(sorted(unknown_groups))
        raise ValueError(f"cannot expand unknown groups: {unknown}")
    member_group = {member: g["id"] for g in graph["groups"] for member in g["members"]}
    lines = [
        "digraph snnlang {",
        'graph [rankdir=LR, bgcolor="#FFFFFF", pad="0.35", nodesep="0.55", ranksep="1.05",',
        '  splines=spline, outputorder=edgesfirst, fontname="Helvetica", compound=true, newrank=true];',
        f'node [shape=plain, fontname="Helvetica", fontcolor="{PALETTE["ink"]}"];',
        f'edge [fontname="Helvetica", fontsize=9, fontcolor="{PALETTE["muted"]}", color="{PALETTE["line"]}", penwidth=1.7, arrowsize=0.72];',
    ]

    rendered: set[str] = set()
    for row in graph["inputs"]:
        node = row["id"]
        lines.append(
            f'{_q(node)} [id={_q(_node_id(node))}, class="node input", label={_card(node, " × ".join(map(str, row["shape"])), row["signal_type"], PALETTE["input"])}, '
            f'shape=box, style="rounded,filled", fillcolor="{PALETTE["neutral"]}", color="{PALETTE["line"]}", margin="0.16,0.12"];'
        )
        rendered.add(node)

    if collapsed:
        for group_id, group in sorted(groups.items()):
            if group_id in expanded_groups:
                continue
            pops = [p for p in graph["populations"] if p["id"] in group["members"]]
            ops = [o for o in graph["operations"] if o["id"] in group["members"]]
            if not pops and not ops:
                continue
            detail = " · ".join(
                [f"{p['id']} {p['size']}" for p in pops]
                + ([f"{len(ops)} ops"] if ops else [])
            )
            lines.append(
                f'{_q(group_id)} [id={_q(_node_id(group_id))}, class="node component", label={_card(group_id, detail, "component", PALETTE["neutral"])}, '
                f'shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#AEBBCA", penwidth=1.5, margin="0.18,0.14"];'
            )
            rendered.add(group_id)

    for pop in graph["populations"]:
        if (
            collapsed
            and pop.get("group") in groups
            and pop.get("group") not in expanded_groups
        ):
            continue
        fill = PALETTE["exc"]
        title_lower = pop["id"].lower()
        if title_lower == "i" or title_lower.endswith("_i") or "inhib" in title_lower:
            fill = PALETTE["inh"]
        badge = "spiking population" if pop["spiking"] else "analogue population"
        pop_detail = f"{pop['size']:,} units · {pop['neuron']['kind']}"
        pop_title = (
            pop["id"].replace("_", " ")
            if pop.get("group") in expanded_groups
            else pop["id"]
        )
        lines.append(
            f'{_q(pop["id"])} [id={_q(_node_id(pop["id"]))}, class="node population {"spiking" if pop["spiking"] else "analogue"}", label={_card(pop_title, pop_detail, badge, fill)}, '
            f'shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#B8C4D1", penwidth=1.4, margin="0.18,0.14"];'
        )
        rendered.add(pop["id"])

    for group_id in sorted(expanded_groups):
        members = [
            pop["id"]
            for pop in graph["populations"]
            if pop.get("group") == group_id
        ]
        title = group_id.replace("_", " ")
        lines.append(
            f'subgraph {_q("cluster_" + _node_id(group_id))} {{ label={_q(title)}; '
            f'color="#8FA3B8"; fontcolor="{PALETTE["ink"]}"; fontname="Helvetica-Bold"; '
            'fontsize=15; penwidth=1.8; style="rounded"; margin=18; labeljust="l";'
        )
        for member in members:
            lines.append(f"{_q(member)};")
        lines.append("}")

    if view == "expanded":
        for op in graph["operations"]:
            label = _card(
                op["id"], op["kind"].replace("_", " "), "operation", PALETTE["neutral"]
            )
            lines.append(
                f'{_q(op["id"])} [id={_q(_node_id(op["id"]))}, class="node operation", label={label}, shape=box, style="rounded,filled", '
                f'fillcolor="#FFFFFF", color="{PALETTE["line"]}", margin="0.14,0.1"];'
            )
            rendered.add(op["id"])

    for output in graph["outputs"]:
        lines.append(
            f'{_q("out:" + output["id"])} [id={_q(_node_id("out_" + output["id"]))}, class="node output", '
            f"label={_card(output['id'], 'named graph interface', 'output', PALETTE['output'])}, "
            f'shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#D8B85B", penwidth=1.5, margin="0.16,0.12"];'
        )
        rendered.add("out:" + output["id"])

    def mapped(owner: str) -> str:
        group = member_group.get(owner)
        return (
            group
            if collapsed and group in rendered and group not in expanded_groups
            else owner
        )

    emitted_edges: set[tuple[str, str, str]] = set()
    for projection in graph["projections"]:
        source = mapped(projection["source"].partition(".")[0])
        target = mapped(projection["target"].partition(".")[0])
        if source == target or source not in rendered or target not in rendered:
            continue
        polarity = projection["polarity"]
        # Circuit diagrams follow the lab's ink-plus-one-accent figure palette:
        # ordinary signal flow is ink and inhibition earns the red accent.
        color = "#C44A55" if polarity == "inhibitory" else PALETTE["ink"]
        arrow = (
            "normal"
            if polarity == "excitatory"
            else "tee"
            if polarity == "inhibitory"
            else "diamond"
        )
        style = "dashed" if polarity == "modulatory" else "solid"
        connection = projection["connection"]
        if connection == "feedback":
            style = "dashed"
        # Direction, arrowhead and line style carry connection semantics in the
        # collapsed view. Repeating "feedback" makes reciprocal labels collide.
        label = projection["id"] if view != "circuit" else ""
        key = (source, target, polarity)
        if key in emitted_edges and view == "circuit":
            continue
        emitted_edges.add(key)
        lines.append(
            f'{_q(source)} -> {_q(target)} [id={_q("e_" + _node_id(projection["id"]))}, class="edge {polarity} {connection}", color="{color}", '
            f"arrowhead={arrow}, style={style}, label={_q(label)}, constraint={'false' if connection == 'feedback' else 'true'}, "
            f"penwidth={2.2 if connection in {'recurrent', 'feedback'} else 1.7}];"
        )

    if collapsed:
        # Preserve computation flow across collapsed component boundaries.
        for op in graph["operations"]:
            target = mapped(op["id"])
            for source_signal in op["sources"]:
                source = mapped(source_signal.partition(".")[0])
                key = (source, target, "operation")
                if (
                    source != target
                    and source in rendered
                    and target in rendered
                    and key not in emitted_edges
                ):
                    emitted_edges.add(key)
                    lines.append(
                        f'{_q(source)} -> {_q(target)} [color="#8794A6", arrowhead=vee, '
                        'label="signal", penwidth=1.5];'
                    )

    if view == "expanded":
        for op in graph["operations"]:
            for source_signal in op["sources"]:
                source = source_signal.partition(".")[0]
                if source in rendered:
                    lines.append(
                        f'{_q(source)} -> {_q(op["id"])} [color="#8794A6", arrowhead=vee];'
                    )
        signal_owner = {f"{o['id']}.value": o["id"] for o in graph["operations"]}
        signal_owner |= {f"{p['id']}.voltage": p["id"] for p in graph["populations"]}
        signal_owner |= {f"{p['id']}.spikes": p["id"] for p in graph["populations"]}
        for output in graph["outputs"]:
            owner = signal_owner.get(
                output["signal"], output["signal"].partition(".")[0]
            )
            if owner in rendered:
                lines.append(
                    f'{_q(owner)} -> {_q("out:" + output["id"])} [color="#C49A28", penwidth=2.0];'
                )
    else:
        operation_group = {o["id"]: mapped(o["id"]) for o in graph["operations"]}
        for output in graph["outputs"]:
            owner = output["signal"].partition(".")[0]
            owner = operation_group.get(owner, mapped(owner))
            if owner in rendered:
                lines.append(
                    f'{_q(owner)} -> {_q("out:" + output["id"])} [color="#C49A28", penwidth=2.0];'
                )

    if view == "training" and bundle.training:
        trainable = {
            pid
            for g in bundle.training["parameter_groups"]
            if not g["frozen"]
            for pid in g["parameters"]
        }
        frozen = {
            pid
            for g in bundle.training["parameter_groups"]
            if g["frozen"]
            for pid in g["parameters"]
        }
        for group in bundle.training["parameter_groups"]:
            gid = "train:" + group["id"]
            state = "frozen" if group["frozen"] else "trainable"
            fill = PALETTE["neutral"] if group["frozen"] else PALETTE["train"]
            group_detail = f"{len(group['parameters'])} tensors · lr {group['lr']:g}"
            lines.append(
                f"{_q(gid)} [label={_card(group['id'], group_detail, state, fill)}, "
                f'shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#7B9ACA", margin="0.15,0.1"];'
            )
        for index, objective in enumerate(bundle.training["objectives"]):
            oid = f"objective:{index}"
            lines.append(
                f"{_q(oid)} [label={_card(objective['kind'].replace('_', ' '), objective['target'], 'objective', PALETTE['output'])}, "
                f'shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#D8B85B", margin="0.15,0.1"];'
            )
            matching = next(
                (
                    o["id"]
                    for o in graph["outputs"]
                    if o["signal"] == objective["prediction"]
                ),
                None,
            )
            if matching:
                lines.append(
                    f'{_q("out:" + matching)} -> {_q(oid)} [color="#C49A28", penwidth=2.0];'
                )
        for group in bundle.training["parameter_groups"]:
            targets = {
                member_group.get(p["id"].partition(".")[0], p["id"].partition(".")[0])
                for p in graph["parameters"]
                if p["id"] in group["parameters"]
            }
            for target in sorted(targets):
                if target in rendered:
                    lines.append(
                        f'{_q("train:" + group["id"])} -> {_q(target)} [style=dotted, color="#5279B7", '
                        f"arrowhead={'none' if group['frozen'] else 'vee'}, constraint=false];"
                    )
        del trainable, frozen

    lines.append("}")
    return "\n".join(lines) + "\n"


def visualise_bundle(
    bundle: Bundle,
    path: Path,
    *,
    view: str,
    scale: int = 1,
    expand_groups: Collection[str] = (),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dot = _dot(bundle, view, expand_groups=expand_groups)
    suffix = path.suffix.lower()
    if suffix not in {".svg", ".png", ".pdf", ".dot"}:
        raise ValueError("visual output must be .svg, .png, .pdf, or .dot")
    if suffix == ".dot":
        path.write_text(dot)
        return path
    with tempfile.TemporaryDirectory(prefix="snnlang-") as temp:
        dot_path = Path(temp) / "network.dot"
        dot_path.write_text(dot)
        args = ["dot", f"-T{suffix[1:]}", str(dot_path), "-o", str(path)]
        if suffix == ".png":
            args.insert(1, f"-Gdpi={144 * scale}")
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Graphviz failed: {result.stderr.strip()}")
    return path
