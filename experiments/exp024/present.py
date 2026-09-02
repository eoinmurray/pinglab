"""Exp024 presentation: draw completed analysis without reading training cells."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp024 import recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import source_run, stage_run

from helpers import theme

SEEDS = recipe.SEEDS
MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}


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
    for (_, seed), met in sorted(cells.items()):
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
    analysis = source_run(REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG)
    results = load_json(analysis.export / "results.json")
    curves = load_json(analysis.export / "curves.json")
    if results.get("schema") != "exp024.analysis/v1" or curves.get("schema") != "exp024.curves/v1":
        raise PingstoreError("unsupported exp024 analysis payload")
    inputs = {"analysis": analysis}
    for role, ref in analysis.record["inputs"].items():
        inputs[role] = source_run(REPO / ".pingstore", ref["run_id"], stage="compute",
                                  experiment="exp022", reference=ref)
    if "compute" not in inputs:
        raise PingstoreError("exp024 analysis must pin its exp022 computation")
    cells = {(cell["model"], cell["seed"]): cell for cell in curves["cells"]}
    expected = {(model, seed) for model in recipe.MODELS for seed in recipe.SEEDS}
    if set(cells) != expected or len(curves["cells"]) != len(expected):
        raise PingstoreError("presentation requires all six analysed baselines")
    markers = {model: results["models"][model]["accuracy_marker_epoch_mean"]
               for model in recipe.MODELS}
    started = time.monotonic()
    with stage_run(REPO, recipe.SLUG, "present", inputs=inputs, run_id=run_id,
                   configuration=results["config"]) as run:
        for model in recipe.MODELS:
            plot_model_curves(cells, model, run.export / f"{model}_curves.svg")
        plot_confidence_inflation(cells, markers, run.export / "confidence_inflation.svg")
        lineage = [{"file": name, "operation": "render", "source_run": identity,
                    "source_paths": ["curves.json", "results.json"]} for name in recipe.FIGURES]
        run.record["presentation_lineage"] = lineage
        write_json_atomic(run.export / "numbers.json", {
            **results, "run_id": run.run_id, "git_sha": run.record["provenance"]["git_commit"],
            "duration_s": time.monotonic() - started,
            "final": {row["name"]: {"acc": row["final_acc"], "rate_e": row["final_e_rate_hz"],
                                      "rate_i": row["final_i_rate_hz"]} for row in results["cells"]},
        })
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp024 analyse run ID")
    parser.add_argument("--run-id", help="unused identity reserved before dispatch")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
