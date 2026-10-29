"""Draw explicit retained analysis and illustrations; never simulate or aggregate.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import argparse
import shutil
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp080 import recipe
from experiments.exp080.custom import evidence, inputs
from pingstore.library import PingstoreError, load_json, write_json_atomic
from snnlab import viz


def schematic_diagrams():
    """Lay out the documented model; these are illustrations, not execution evidence."""
    specifications = [
        {
            "name": "decoder",
            "title": "Membrane features and learned decoder",
            "nodes": [
                ("X", "Pixel spike events", "input", (-360, 0)),
                ("M", "Nonspiking membranes", "population", (0, 0)),
                ("F", "Mean voltage features", "operation", (600, 0)),
                ("H", "ReLU decoder", "population", (600, 300)),
                ("R", "Class logits", "population", (0, 300)),
            ],
            "edges": [
                ("X", "M", "excitatory"),
                ("M", "F", "average"),
                ("F", "H", "signal"),
                ("H", "R", "signal"),
            ],
        }
    ]
    diagrams = {}
    ports = dict(
        top_left=(0.25, 0),
        top_right=(0.75, 0),
        bottom_left=(0.25, 1),
        bottom_right=(0.75, 1),
    )
    for spec in specifications:
        visual = viz.Diagram(
            name="exp080_" + spec["name"],
            title=spec["title"],
            nodes=tuple(
                viz.DiagramNode(identifier, identifier, detail, "", kind=kind)
                for identifier, detail, kind, _ in spec["nodes"]
            ),
            edges=tuple(
                viz.DiagramEdge(
                    source,
                    target,
                    id=source + "_to_" + target,
                    role="signal" if role == "average" else role,
                )
                for source, target, role in spec["edges"]
            ),
            metadata=dict(
                view="illustrative documented model; no historical bundle reconstructed"
            ),
        )
        positions = {
            identifier: position for identifier, _, _, position in spec["nodes"]
        }
        routes = {}
        for source, target, role in spec["edges"]:
            label = (
                r"$\langle V-E_L\rangle_t$"
                if role == "average"
                else rf"$W_{{{target},{source}}}$"
            )
            if (source, target) == ("E", "I"):
                route = dict(source_port="bottom_left", target_port="top_left")
            elif (source, target) == ("I", "E"):
                route = dict(source_port="top_right", target_port="bottom_right")
            elif source == "J":
                route = dict(source_port="top", target_port="bottom")
            elif positions[source][0] == positions[target][0]:
                route = dict(source_port="bottom", target_port="top")
            elif positions[source][0] > positions[target][0]:
                route = dict(source_port="left", target_port="right")
            else:
                route = dict(source_port="right", target_port="left")
            routes[source + "_to_" + target] = viz.EdgeLayout(
                label=viz.DiagramLabel(
                    label, offset=(0, -65) if role == "average" else (0, 0)
                ),
                **route,
            )
        layout = viz.DiagramLayout(
            nodes={
                identifier: viz.NodeLayout(
                    position=position,
                    label=f"${identifier}$",
                    min_size=(240, 160),
                    ports=ports,
                )
                for identifier, position in positions.items()
            },
            edges=routes,
            defaults=viz.LayoutDefaults(
                node=viz.NodeLayout(content_align="center", padding=16), gap=30
            ),
        )
        with matplotlib.rc_context({"savefig.bbox": None}):
            diagrams[spec["name"]] = viz.resolve_diagram(
                visual, layout=layout, theme=viz.DiagramTheme(secondary_size=13)
            )
    return diagrams


def draw_schematics():
    """Regenerate report-local SVG illustrations without executing scientific stages."""
    destination = Path(__file__).parent / "report" / "images"
    destination.mkdir(exist_ok=True)
    with matplotlib.rc_context({"savefig.bbox": None}):
        for name, diagram in schematic_diagrams().items():
            viz.render_diagram(diagram, destination / (name + "-network.svg"))


def present(identity: str, *, run_id: str | None = None) -> str:
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    pin = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, pin["run_id"], "compute", reference=pin)
    result = load_json(analysis.export / "results.json")
    if (
        result.get("schema") != "exp080.analysis/v1"
        or result.get("recipe") != cfg
        or result.get("parameters") != recipe.reported_parameters(cfg)
        or result.get("decision") != load_json(analysis.export / "decision.json")
    ):
        raise PingstoreError("exp080 analysis recipe or decision disagrees")
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": analysis, "compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        historical = compute.record["execution"].get("operation") == "historical-import"
        if historical:
            evidence.require(
                bool(compute.record.get("historical_import", {}).get("producer")),
                "historical import lacks original producer identity",
            )
        illustration = result["illustration"]
        if illustration != load_json(compute.export / "evidence.json")["illustration"]:
            raise PingstoreError("exp080 illustration disagrees with compute source")
        evidence.illustration(compute.export, illustration, cfg, historical=historical)
        plot_training(result["training"], run.export)
        plot_psychometric(result["decision"], run.export, cfg)
        if illustration["kind"] == "historical-image":
            shutil.copyfile(
                compute.file(illustration["path"]), run.export / "feature_images.png"
            )
            run.record["retained_figures"] = {
                "feature_images.png": {
                    "source": compute.reference,
                    "path": illustration["path"],
                    "regenerated": False,
                },
            }
        else:
            with np.load(
                compute.file(illustration["path"]), allow_pickle=False
            ) as samples:
                plot_feature_images(
                    samples["image"],
                    samples["features_mV"],
                    samples["rates_hz"],
                    run.export,
                )
        write_json_atomic(run.export / "numbers.json", result)
        write_json_atomic(run.export / "decision.json", result["decision"])
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp080 v4 analyse run ID"
    )
    parser.add_argument(
        "--run-id", help="source-neutral identity reserved before dispatch"
    )
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


def plot_training(records: list[dict[str, Any]], output: Path) -> None:
    theme.apply()
    fig, axis = plt.subplots(figsize=(5.1, 3.1), constrained_layout=True)
    for record in records:
        axis.plot(
            [row["epoch"] for row in record["history"]],
            [row["validation_accuracy"] for row in record["history"]],
            label=f"Seed {record['seed']}",
        )
    axis.set(xlabel="Epoch", ylabel="Mixed-rate validation accuracy", ylim=(0, 1))
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.savefig(output / "training_history.svg", metadata={"Date": None})
    plt.close(fig)


def plot_psychometric(decision: dict[str, Any], output: Path, cfg: dict) -> None:
    theme.apply()
    rows = decision["rows"]
    rates = np.asarray([row["rate_hz"] for row in rows])
    accuracy = np.asarray([row["accuracy"] for row in rows])
    lower = np.asarray([row["minimum_seed_accuracy"] for row in rows])
    upper = np.asarray([row["maximum_seed_accuracy"] for row in rows])
    fig, axis = plt.subplots(figsize=(5.1, 3.2), constrained_layout=True)
    axis.plot(rates, accuracy, color=theme.INK_BLACK, marker="o")
    axis.fill_between(
        rates, lower, upper, color=theme.INK_BLACK, alpha=0.14, linewidth=0
    )
    axis.axhline(cfg["chance_accuracy"], color="#777777", linestyle=":", label="Chance")
    axis.axhline(
        cfg["useful_accuracy"],
        color="#777777",
        linestyle="--",
        label="Practical criterion",
    )
    if decision["criterion_crossed"]:
        axis.axvline(
            decision["r_train_hz"],
            color=theme.DEEP_RED,
            linewidth=1.0,
            label="Selected floor",
        )
    axis.set_xscale("log")
    axis.set_xticks(cfg["rates_hz"])
    axis.set_xticklabels([f"{rate:g}" for rate in cfg["rates_hz"]])
    axis.set(
        xlabel="Maximum-pixel encoding rate (Hz)",
        ylabel="Held-out accuracy",
        ylim=(0, 1),
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=7)
    fig.savefig(output / "psychometric.svg", metadata={"Date": None})
    plt.close(fig)


def plot_feature_images(
    image: np.ndarray, features: np.ndarray, rates: np.ndarray, output: Path
) -> None:
    theme.apply()
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.0), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Input image")
    for axis, feature, rate in zip(axes[1:], features, rates, strict=True):
        axis.imshow(feature.reshape(28, 28), cmap="magma", vmin=0, vmax=65)
        axis.set_title(f"{rate:g} Hz")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    theme.label_panels(axes)
    fig.savefig(output / "feature_images.svg", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    if "--draw-schematics" in sys.argv[1:]:
        parser = argparse.ArgumentParser(
            description="Render illustrative model diagrams only."
        )
        parser.add_argument("--draw-schematics", action="store_true", required=True)
        parser.parse_args()
        draw_schematics()
    else:
        main()
