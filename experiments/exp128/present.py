"""Draw structural overviews of the browser-only teaching models.

These schematics document the authored inputs and membrane, rather than a saved
simulation bundle. Run present.py to regenerate report/images/*-network.svg;
no model is executed and no scientific run evidence is created.
"""

from pathlib import Path

from snnlab import viz


def diagrams():
    """Return display schematics for the browser models, with no simulation claims."""
    diagrams = {}
    for name, sources in (("single", ("P",)), ("dual", ("E", "I"))):
        visual = viz.Diagram(
            name="exp128_" + name,
            title="Single synaptic input"
            if name == "single"
            else "Excitatory and inhibitory inputs",
            nodes=tuple(
                viz.DiagramNode(s, s, "Prescribed spikes", "", kind="input")
                for s in sources
            )
            + (viz.DiagramNode("N", "N", "LIF membrane", "", kind="population"),),
            edges=tuple(
                viz.DiagramEdge(
                    s,
                    "N",
                    id=s + "_to_N",
                    role="signal"
                    if name == "single"
                    else "inhibitory"
                    if s == "I"
                    else "excitatory",
                )
                for s in sources
            ),
            metadata=dict(view="illustrative browser-model schematic"),
        )
        positions = {sources[0]: (0, -150 if name == "dual" else 0), "N": (500, 0)}
        if name == "dual":
            positions["I"] = (0, 150)
        layout = viz.DiagramLayout(
            nodes={
                k: viz.NodeLayout(position=p, label=f"${k}$", min_size=(240, 160))
                for k, p in positions.items()
            },
            edges={
                s + "_to_N": viz.EdgeLayout(
                    label=viz.DiagramLabel(rf"$W_{{N,{s}}}$"),
                    source_port="right",
                    target_port="left"
                    if name == "single"
                    else "top"
                    if s == "E"
                    else "bottom",
                    waypoints=None if name == "single" else ((500, positions[s][1]),),
                )
                for s in sources
            },
            defaults=viz.LayoutDefaults(
                node=viz.NodeLayout(content_align="center", padding=16), gap=30
            ),
        )
        diagrams[name] = viz.resolve_diagram(
            visual, layout=layout, theme=viz.DiagramTheme(secondary_size=13)
        )
    return diagrams


def present():
    """Regenerate the authored SVG figures from SNNLab's layout specifications."""
    destination = Path(__file__).parent / "report" / "images"
    for name, diagram in diagrams().items():
        viz.render_diagram(diagram, destination / (name + "-network.svg"))


if __name__ == "__main__":
    present()
