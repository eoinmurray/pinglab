"""Render paired fine-tuning evidence and native architecture diagrams."""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp171 import recipe
from experiments.exp171.compute import write_json
from pingstore.library import source_run, stage_run
from snnlab import lang, viz


def circuit(bundle, path, *, teacher=False):
    native = lang.diagram(bundle, view="circuit")
    positions = {"drive": (-440, 0), "E": (0, 0), "I": (0, 360), "readout": (440, 0)}
    if not teacher:
        positions["artificial"] = (-440, 360)
    native = replace(
        native,
        title="Frozen PING calibration teacher"
        if teacher
        else "COBA fine-tuning with private GABA",
        groups=(),
        nodes=tuple(node for node in native.nodes if node.id in positions),
        edges=tuple(
            edge
            for edge in native.edges
            if edge.source in positions and edge.target in positions
        ),
    )
    labels = {
        "input_to_E": r"$W_{EX}^{P}$" if teacher else r"$W_{EX}^{C}$ (trained)",
        "readout_projection": r"$W_{RE}^{P}$" if teacher else r"$W_{RE}^{C}$ (trained)",
        "E_to_I": r"$W_{IE}^{P}$" if teacher else "$0$ (frozen)",
        "I_to_E": r"$W_{EI}^{P}$" if teacher else "$0$ (frozen)",
        "artificial_to_E": r"$1\,\mu S\;\delta_{jk}$ (frozen)",
    }
    routes = {
        "input_to_E": dict(source_port="right", target_port="left"),
        "readout_projection": dict(source_port="right", target_port="left"),
        "E_to_I": dict(source_port="bottom_right", target_port="top_right"),
        "I_to_E": dict(source_port="top_left", target_port="bottom_left"),
        "artificial_to_E": dict(
            source_port="top",
            target_port="lower_left",
            routing="polyline",
            waypoints=((-300, 180), (-190, 180)),
        ),
    }
    layout = viz.DiagramLayout(
        nodes={
            key: viz.NodeLayout(
                position=position,
                min_size=(220, 140),
                ports={
                    "top_left": (0.25, 0),
                    "top_right": (0.75, 0),
                    "bottom_left": (0.25, 1),
                    "bottom_right": (0.75, 1),
                    "lower_left": (0, 0.8),
                },
            )
            for key, position in positions.items()
        },
        edges={
            edge.id: viz.EdgeLayout(
                **routes[edge.id], label=viz.DiagramLabel(labels[edge.id])
            )
            for edge in native.edges
        },
    )
    viz.render_diagram(native, path, layout=layout)


def raster_figure(recording, summary, cfg):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1, 1.4]},
    )
    for ax, key, colour, title, label in [
        (
            axes[0],
            "artificial_events",
            theme.DEEP_RED,
            "Artificial inhibitory input events",
            "Private GABA channel",
        ),
        (axes[1], "E_spikes", theme.INK_BLACK, "Excitatory spikes", "E neuron index"),
    ]:
        t, cell = np.nonzero(recording[key])
        ax.scatter(
            t * cfg["dt_ms"],
            cell,
            s=3 if key == "E_spikes" else 1,
            marker=".",
            linewidths=0,
            color=colour,
            rasterized=True,
        )
        ax.set(
            xlim=(0, cfg["duration_ms"]),
            ylim=(-0.5, cfg["n_e"] - 0.5),
            ylabel=label,
            title=title,
            yticks=[0, 256, 512, 768, 1023],
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (ms)")
    theme.label_panels(axes)
    selection = (
        "First correct image in fixed validation batch"
        if summary.get("selection_policy")
        == "first_correct_in_retained_validation_batch"
        else "First validation image"
    )
    fig.suptitle(
        f"Best-validation checkpoint: epoch {summary['epoch']} | digit {summary['label']} | predicted {summary['predicted']}\n"
        f"{selection}; draw {summary['validation_draw'] + 1} of {cfg['validation_draws']}; E {summary['e_rate_hz']:.1f} Hz"
        + (
            f"; {summary['event_count']} artificial events/channel"
            if "event_count" in summary
            else ""
        )
    )
    return fig


def ceiling_label(target):
    return "Off" if target is None else f"{target:g} Hz"


def frontier_figure(cases, *, partition="validation"):
    theme.apply()
    fig, ax = plt.subplots(figsize=(6.5, 3.4), layout="constrained")
    offsets = {
        "coba": {
            1.0: ((-2, -8), "right", "top"),
            2.5: ((5, -7), "left", "top"),
            5.0: ((5, -5), "left", "top"),
            10.0: ((5, -5), "left", "top"),
            25.0: ((4, -8), "left", "top"),
        },
        "ping": {
            1.0: ((-8, -2), "right", "top"),
            2.5: ((-8, 3), "right", "bottom"),
            5.0: ((-5, 8), "right", "bottom"),
            10.0: ((0, 9), "center", "bottom"),
            25.0: ((0, 9), "center", "bottom"),
        },
        "artificial": {
            1.0: ((-7, 5), "right", "bottom"),
            2.5: ((-7, 6), "right", "bottom"),
            5.0: ((18, 0), "left", "center"),
            10.0: ((5, 6), "left", "bottom"),
            25.0: ((5, -8), "left", "top"),
        },
    }
    xmax = 1.0
    for name, label, colour, marker in (
        ("coba", "COBA", theme.DEEP_RED, "s"),
        ("ping", "PING", theme.INK_BLACK, "D"),
        ("artificial", "Artificial I", theme.ELECTRIC_CYAN, "^"),
    ):
        points = sorted(
            [
                (
                    case["target_hz"],
                    (case["test"] if "test" in case else case["history"][-1])
                    if name == "artificial"
                    else case["controls"][name],
                )
                for case in cases
            ],
            key=lambda item: item[1]["e_rate_hz"],
        )
        xmax = max(xmax, max(point["e_rate_hz"] for _, point in points))
        ax.plot(
            [p["e_rate_hz"] for _, p in points],
            [p["accuracy_pct"] for _, p in points],
            marker=marker,
            ms=4,
            lw=1.4,
            color=colour,
            label=label,
        )
        for target, point in points:
            xy = (point["e_rate_hz"], point["accuracy_pct"])
            if target is None:
                ax.scatter(
                    [xy[0]],
                    [xy[1]],
                    s=60,
                    marker="*",
                    color=colour,
                    edgecolor=theme.INK_BLACK,
                    linewidths=0.7,
                    zorder=6,
                )
                offset, horizontal, vertical = (
                    ((14, 5), "left", "bottom")
                    if name == "ping"
                    else ((-6, -8), "right", "top")
                    if name == "coba"
                    else ((8, -8), "left", "top")
                )
                text = f"{label}\n{xy[1]:.0f}% @ {xy[0]:.0f} Hz"
            else:
                offset, horizontal, vertical = offsets[name][target]
                text = f"{target:g}"
            ax.annotate(
                text,
                xy,
                xytext=offset,
                textcoords="offset points",
                ha=horizontal,
                va=vertical,
                fontsize=theme.SIZE_ANNOTATION,
                color=colour,
            )
    ax.set(
        xlabel="test-set E firing rate (Hz)"
        if partition == "test"
        else "hidden E firing rate (Hz)",
        ylabel=f"{partition} accuracy (%)",
        xlim=(-xmax * 0.03, xmax * 1.12),
        ylim=(40, 100),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False,
        fontsize=theme.SIZE_LEGEND,
        loc="lower right",
        title="★ unpenalised",
        title_fontsize=theme.SIZE_ANNOTATION,
    )
    return fig


def learning_figure(results):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    history = results["history"]
    epochs = [row["epoch"] for row in history]
    for ax, key, label in zip(
        axes,
        ["cross_entropy", "accuracy_pct", "e_rate_hz"],
        [
            "Cross-entropy",
            "Validation accuracy (%)",
            "E firing rate (Hz)",
        ],
        strict=True,
    ):
        for index, case in enumerate(results.get("sweep", [results])):
            rows = case["history"]
            ax.plot(
                [r["epoch"] for r in rows],
                [r[key] for r in rows],
                "o-",
                color=theme.CYCLE[index],
                label=ceiling_label(case["target_hz"])
                if "sweep" in results
                else "Validation",
            )
        ax.set(
            xlabel="Fine-tuning epoch",
            ylabel=label,
            xticks=epochs
            if len(epochs) <= 6
            else epochs[:: max(1, (len(epochs) - 1) // 4)],
        )
        ax.spines[["top", "right"]].set_visible(False)
    training_rows = [row for row in history if row.get("training_loss") is not None]
    if training_rows:
        axes[0].plot(
            [row["epoch"] for row in training_rows],
            [row["training_loss"] for row in training_rows],
            "s--",
            color=theme.DEEP_RED,
            label="Training",
        )
    for index, case in enumerate(results.get("sweep", [results])):
        measured = case.get("final_training")
        if not measured:
            continue
        axes[0].plot(
            [measured["epoch"]],
            [measured["cross_entropy"]],
            "s",
            color=theme.CYCLE[index] if "sweep" in results else theme.DEEP_RED,
            label=None if "sweep" in results else "Training (final)",
        )
    if "sweep" in results:
        axes[1].legend(frameon=False, fontsize=8, title="E ceiling")
    axes[0].legend(frameon=False)
    theme.label_panels(axes)
    return fig


def present(identity, run_id=None):
    started = time.perf_counter()
    analysis = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="analyse"
    )
    pin = analysis.record["inputs"]["compute"]
    computed = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        experiment=recipe.SLUG,
        stage="compute",
        reference=pin,
    )
    cfg = computed.record["execution"]["configuration"]
    results = json.loads(analysis.file("results.json").read_text())
    inputs = {"analyse": analysis, "compute": computed}
    for role, pin in analysis.record["inputs"].items():
        if role.startswith(("cell_", "test_evaluation")):
            inputs[role] = source_run(
                ROOT / ".pingstore",
                pin["run_id"],
                experiment=recipe.SLUG,
                stage="compute",
                reference=pin,
            )
    recorded = None
    if "raster" in analysis.record["inputs"]:
        pin = analysis.record["inputs"]["raster"]
        recorded = source_run(
            ROOT / ".pingstore",
            pin["run_id"],
            experiment=recipe.SLUG,
            stage="compute",
            phase="raster",
            reference=pin,
        )
        inputs["raster"] = recorded
    with stage_run(
        ROOT,
        recipe.SLUG,
        "present",
        inputs=inputs,
        configuration=cfg,
        run_id=run_id,
    ) as run:
        with matplotlib.rc_context():
            theme.set_paper_mode(True)
            theme.apply()
            fig = learning_figure(results)
            for ext in ("svg", "png"):
                fig.savefig(run.export / ("learning-curves." + ext), dpi=180)
            plt.close(fig)
            if "sweep" in results:
                fig = frontier_figure(
                    results.get("test_frontier", results["sweep"]),
                    partition="test" if "test_frontier" in results else "validation",
                )
                for ext in ("svg", "png"):
                    fig.savefig(run.export / ("activity-frontier." + ext), dpi=180)
                plt.close(fig)
            raster_source = recorded or (
                computed if cfg.get("execution_backend") == "snnlab.sim.train" else None
            )
            if raster_source:
                with np.load(raster_source.file("raster.npz")) as z:
                    fig = raster_figure(z, results["raster"], cfg)
                    for ext in ("svg", "png"):
                        fig.savefig(run.export / ("raster." + ext), dpi=190)
                    plt.close(fig)
        for name, is_teacher in [("network", False), ("teacher", True)]:
            bundle = lang.load_bundle(
                computed.restore_bundle(
                    name + ".bundle", run.scratch / (name + ".bundle")
                )
            )
            for ext in ("svg", "png"):
                circuit(
                    bundle, run.export / (name + "-circuit." + ext), teacher=is_teacher
                )
        results["timings"]["presentation_seconds"] = time.perf_counter() - started
        results["timings"]["stages_total_seconds"] = sum(
            results["timings"][key]
            for key in ("compute_seconds", "analysis_seconds", "presentation_seconds")
        )
        if "sweep" in results:
            results["timings"]["sweep_compute_seconds"] = sum(
                cell["timings"]["compute_seconds"] for cell in results["sweep"]
            )
            results["timings"]["stages_total_seconds"] += (
                results["timings"]["sweep_compute_seconds"]
                - results["timings"]["compute_seconds"]
            )
        write_json(run.export / "numbers.json", results)
        run.record["execution"].update(
            no_simulation=True, elapsed_seconds=time.perf_counter() - started
        )
        analysis.check_unchanged()
        computed.check_unchanged()
        if recorded:
            recorded.check_unchanged()
        for role, cell in inputs.items():
            if role.startswith("cell_"):
                cell.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    present(args.source, args.run_id)
