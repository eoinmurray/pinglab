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
    visual = viz.Diagram(
        name="exp129_synaptic_membrane",
        title="Illustrative synaptic input",
        nodes=(
            viz.DiagramNode("P", "P", "Prescribed spikes", "", kind="input"),
            viz.DiagramNode("N", "N", "HH membrane", "", kind="population"),
        ),
        edges=(viz.DiagramEdge("P", "N", id="P_to_N", role="excitatory"),),
        metadata=dict(view="illustrative browser-model schematic"),
    )
    layout = viz.DiagramLayout(
        nodes={
            k: viz.NodeLayout(position=p, label=f"${k}$", min_size=(240, 160))
            for k, p in dict(P=(0, 0), N=(500, 0)).items()
        },
        edges=dict(
            P_to_N=viz.EdgeLayout(
                label=viz.DiagramLabel(r"$W_{N,P}$"),
                source_port="right",
                target_port="left",
            )
        ),
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=16), gap=30
        ),
    )
    diagrams["membrane"] = viz.resolve_diagram(
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
