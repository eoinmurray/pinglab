"""Exp024 presentation: draw completed analysis without reading training cells.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
from snnlab import viz

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "libraries")]


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp024 import recipe
from pingstore.library import (
    PingstoreError,
    load_json,
    source_run,
    stage_run,
    write_json_atomic,
)

SEEDS = recipe.SEEDS
MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}


def schematic_diagrams():
    """Lay out the documented model; these are illustrations, not execution evidence."""
    specifications = [
        {
            "name": "coba",
            "title": "COBA classifier",
            "nodes": [
                ("X", "Poisson spikes", "input", (-360, 0)),
                ("E", "Excitatory neurons", "population", (0, 0)),
                ("R", "Class readout", "population", (360, 0)),
            ],
            "edges": [("X", "E", "excitatory"), ("E", "R", "signal")],
        },
        {
            "name": "ping",
            "title": "PING classifier",
            "nodes": [
                ("X", "Poisson spikes", "input", (-360, 0)),
                ("E", "Excitatory neurons", "population", (0, 0)),
                ("I", "Inhibitory neurons", "population", (0, 300)),
                ("R", "Class readout", "population", (360, 0)),
            ],
            "edges": [
                ("X", "E", "excitatory"),
                ("E", "I", "excitatory"),
                ("I", "E", "inhibitory"),
                ("E", "R", "signal"),
            ],
        },
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
            name="exp024_" + spec["name"],
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
                label=viz.DiagramLabel(label), **route
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


def plot_model_curves(cells: dict, model: str, out_path: Path) -> None:
    """Three panels for one model — loss, validation accuracy, and firing rate vs
    epoch — three seeds overlaid. COBA shows E only (I is silent); PING shows
    E (solid) and I (dashed)."""
    from matplotlib.lines import Line2D

    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    cells = {k: v for k, v in cells.items() if k[0] == model}
    color = MODEL_COLORS[model]
    # Column-width three-panel row (H11): built at the ~6.9 in column so the fonts
    # and line weights match the rest of the collection, not 2x-shrunk on display.
    fig, (axL, axA, axR) = plt.subplots(1, 3, figsize=(6.9, 2.5), dpi=150)
    for (_, _seed), met in sorted(cells.items()):
        eps = np.array([e["ep"] for e in met["epochs"]])
        axL.plot(
            eps,
            [e["loss"] for e in met["epochs"]],
            color=color,
            lw=1.2,
            alpha=0.8,
        )
        axL.plot(
            eps,
            [e["test_loss"] for e in met["epochs"]],
            color=color,
            lw=1.0,
            ls="--",
            alpha=0.6,
        )
        axA.plot(
            eps,
            [e["acc"] for e in met["epochs"]],
            color=color,
            lw=1.2,
            alpha=0.85,
        )
        axR.plot(
            eps,
            [e["test_rate_e"] for e in met["epochs"]],
            color=color,
            lw=1.2,
            alpha=0.85,
        )
        if model == "ping":
            axR.plot(
                eps,
                [e["test_rate_i"] for e in met["epochs"]],
                color=color,
                lw=1.0,
                ls="--",
                alpha=0.85,
            )
    axL.set_title("loss", loc="left", fontweight="semibold")
    axL.set_ylabel("loss")
    axL.legend(
        handles=[
            Line2D([0], [0], color=color, lw=2, label="train"),
            Line2D([0], [0], color=color, lw=2, ls="--", label="validation"),
        ],
        frameon=False,
        fontsize=theme.SIZE_LEGEND,
    )
    axA.set_title("validation accuracy", loc="left", fontweight="semibold")
    axA.set_ylabel("accuracy (%)")
    axA.set_ylim(0, 100)
    axR.set_title("firing rate", loc="left", fontweight="semibold")
    axR.set_ylabel("rate (Hz)")
    if model == "ping":
        axR.legend(
            handles=[
                Line2D([0], [0], color=color, lw=2, label="E"),
                Line2D([0], [0], color=color, lw=2, ls="--", label="I"),
            ],
            frameon=False,
            fontsize=theme.SIZE_LEGEND,
        )
    for ax in (axL, axA, axR):
        ax.set_xlabel("epoch")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    theme.label_panels((axL, axA, axR))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confidence_inflation(cells: dict, conv_ep: dict, out_path: Path) -> None:
    """Draw retained accuracy, cross-entropy and rate; markers come from analysis."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    # Column-width three-panel row (H11), matching plot_model_curves and siblings.
    fig, (axA, axL, axR) = plt.subplots(1, 3, figsize=(6.9, 2.5), dpi=150)

    for (model, seed), m in sorted(cells.items()):
        color = MODEL_COLORS[model]
        eps = np.array([e["ep"] for e in m["epochs"]])
        label = model.upper() if seed == SEEDS[0] else None
        axA.plot(
            eps,
            [e["acc"] for e in m["epochs"]],
            color=color,
            lw=1.2,
            alpha=0.85,
            label=label,
        )
        axL.plot(
            eps,
            [e["test_loss"] for e in m["epochs"]],
            color=color,
            lw=1.2,
            alpha=0.85,
            label=label,
        )
        axR.plot(
            eps,
            [e["test_rate_e"] for e in m["epochs"]],
            color=color,
            lw=1.2,
            alpha=0.85,
            label=label,
        )

    for model, ce in conv_ep.items():
        if ce is None:
            continue
        for ax in (axA, axL, axR):
            ax.axvline(ce, color=MODEL_COLORS[model], lw=0.8, ls=":", alpha=0.6)

    axA.set_title("validation accuracy", loc="left", fontweight="semibold")
    axA.set_ylabel("accuracy (%)")
    axA.set_ylim(0, 100)
    axA.legend(frameon=False, fontsize=theme.SIZE_LEGEND, loc="lower right")
    axL.set_title("validation CE", loc="left", fontweight="semibold")
    axL.set_ylabel("CE loss")
    axL.set_yscale("log")
    axR.set_title("E firing rate", loc="left", fontweight="semibold")
    axR.set_ylabel("rate (Hz)")
    for ax in (axA, axL, axR):
        ax.set_xlabel("epoch")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    theme.label_panels((axA, axL, axR))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def present(identity: str, *, run_id: str | None = None) -> str:
    analysis = source_run(
        REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    results = load_json(analysis.export / "results.json")
    curves = load_json(analysis.export / "curves.json")
    if (
        results.get("schema") != "exp024.analysis/v1"
        or curves.get("schema") != "exp024.curves/v1"
    ):
        raise PingstoreError("unsupported exp024 analysis payload")
    inputs = {"analysis": analysis}
    for role, ref in analysis.record["inputs"].items():
        inputs[role] = source_run(
            REPO / ".pingstore",
            ref["run_id"],
            stage="compute",
            experiment="exp022",
            reference=ref,
        )
    if "compute" not in inputs:
        raise PingstoreError("exp024 analysis must pin its exp022 computation")
    cells = {(cell["model"], cell["seed"]): cell for cell in curves["cells"]}
    expected = {(model, seed) for model in recipe.MODELS for seed in recipe.SEEDS}
    if set(cells) != expected or len(curves["cells"]) != len(expected):
        raise PingstoreError("presentation requires all six analysed baselines")
    markers = {
        model: results["models"][model]["accuracy_marker_epoch_mean"]
        for model in recipe.MODELS
    }
    started = time.monotonic()
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs=inputs,
        run_id=run_id,
        configuration=results["config"],
    ) as run:
        for model in recipe.MODELS:
            plot_model_curves(cells, model, run.export / f"{model}_curves.svg")
        plot_confidence_inflation(
            cells, markers, run.export / "confidence_inflation.svg"
        )
        lineage = [
            {
                "file": name,
                "operation": "render",
                "source_run": identity,
                "source_paths": ["curves.json", "results.json"],
            }
            for name in recipe.FIGURES
        ]
        run.record["presentation_lineage"] = lineage
        write_json_atomic(
            run.export / "numbers.json",
            {
                **results,
                "run_id": run.run_id,
                "git_sha": run.record["provenance"]["git_commit"],
                "duration_s": time.monotonic() - started,
                "final": {
                    row["name"]: {
                        "acc": row["final_acc"],
                        "rate_e": row["final_e_rate_hz"],
                        "rate_i": row["final_i_rate_hz"],
                    }
                    for row in results["cells"]
                },
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp024 analyse run ID"
    )
    parser.add_argument("--run-id", help="unused identity reserved before dispatch")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


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
