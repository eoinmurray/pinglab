"""Render training curves, sample rasters, executed network circuits and numbers."""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pinglab-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp124 import recipe
from experiments.exp124.analyse import network_definition, parameter_table
from matplotlib.ticker import MaxNLocator
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab import lang, viz


def draw_networks(recording, export):
    """Project the executed definitions, including historical pre-bundle evidence."""
    for model in recipe.MODELS:
        bundle = lang.Bundle(
            graph=network_definition(recording, model, "graph.json"),
            training=network_definition(recording, model, "training.json"),
            manifest={},
            diagnostics=[],
        )
        diagram = lang.diagram(bundle, view="circuit")
        # Circuit views omit reductions; omit their disconnected output boxes too.
        connected = {
            endpoint
            for edge in diagram.edges
            for endpoint in (edge.source, edge.target)
        }
        diagram = replace(
            diagram, nodes=tuple(node for node in diagram.nodes if node.id in connected)
        )
        viz.render_diagram(diagram, export / f"{model}_network.svg")


def accuracy_percent(row, split):
    value = row[split]["accuracy"]
    return float("nan") if value is None else 100 * value


def draw_training_curves(conditions, cfg, export):
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    ):
        for condition in recipe.MODELS:
            rows = conditions[condition]["epochs"]
            figure, axes = plt.subplots(
                1, 2, figsize=(7.1, 3.8), constrained_layout=True
            )
            for split, color in (("train", "#202020"), ("validation", "#a34d45")):
                epochs = [row["epoch"] for row in rows]
                axes[0].plot(
                    epochs,
                    [row[split]["loss"] for row in rows],
                    color=color,
                    marker="o",
                    markersize=2.5,
                    label="Online training"
                    if split == "train"
                    and cfg.get("training_measurement")
                    == "online_pre_update_minibatches"
                    else split.capitalize(),
                )
                axes[1].plot(
                    epochs,
                    [accuracy_percent(row, split) for row in rows],
                    color=color,
                    marker="o",
                    markersize=2.5,
                )
            axes[0].set(title="A  Classification loss", ylabel="Cross-entropy")
            maximum = max(
                accuracy_percent(row, split)
                for row in rows
                for split in ("train", "validation")
                if row[split]["accuracy"] is not None
            )
            axes[1].set(
                title="B  Classification accuracy",
                ylabel="Accuracy (%)",
                ylim=(0, min(100, max(20, maximum * 1.15))),
            )
            axes[1].axhline(5, linestyle=":", color="#777777", linewidth=1)
            for axis in axes:
                axis.set(xlabel="Completed epoch", xlim=(0, cfg["epochs"]))
                axis.set_box_aspect(3 / 4)
                axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                axis.spines[["top", "right"]].set_visible(False)
            axes[0].legend(frameon=False)
            figure.suptitle(
                f"{condition.upper()} — {'full SHD' if cfg['full_dataset'] else 'reduced-data'} training, two hidden layers\n"
                f"{cfg['train_examples']:,} training / {cfg['validation_examples']:,} validation",
                fontsize=10,
            )
            for extension in ("svg",):
                filename = f"full_{condition}_training_curves.{extension}"
                if (export / filename).exists():
                    raise ValueError(f"retained export collides with {filename}")
                figure.savefig(export / filename, dpi=180)
            plt.close(figure)


def render(source, recording, export):
    cfg = source.record["execution"]["configuration"]
    results = load_json(source.file("results.json"))
    conditions = results["conditions"]
    sample = load_json(source.file("raster_measurements.json"))
    numbers = {
        "full_training": {
            "configuration": cfg,
            "conditions": conditions,
            **source.record["execution"]["source_measurements"],
        },
        "sample_raster": sample,
        "parameter_table": results.get("parameter_table") or parameter_table(recording),
    }
    draw_training_curves(conditions, cfg, export)
    draw_networks(recording, export)
    filename = (
        "recording.npz" if recording.file("recording.npz").exists() else "rasters.npz"
    )
    with (
        np.load(recording.file(filename), allow_pickle=False) as spikes,
        plt.rc_context({"font.size": 8, "axes.titlesize": 9}),
    ):
        for model in recipe.MODELS:
            figure, axes = plt.subplots(
                4,
                1,
                figsize=(7.1, 6.5),
                constrained_layout=True,
                gridspec_kw={"height_ratios": [2.4, 1, 2.4, 1]},
            )
            for layer in (1, 2):
                axis, correlation_axis = axes[2 * (layer - 1) : 2 * layer]
                excitatory = spikes[f"{model}_e{layer}"]
                times, neurons = np.nonzero(excitatory)
                axis.scatter(
                    times * sample["dt_ms"],
                    neurons,
                    color="#202020",
                    marker="|",
                    s=7,
                    linewidths=0.5,
                    rasterized=True,
                    label="E",
                )
                total = excitatory.shape[1]
                if model == "ping":
                    inhibitory = spikes[f"{model}_i{layer}"]
                    times, neurons = np.nonzero(inhibitory)
                    axis.scatter(
                        times * sample["dt_ms"],
                        neurons + total,
                        color="#a34d45",
                        marker="|",
                        s=7,
                        linewidths=0.5,
                        rasterized=True,
                        label="I",
                    )
                    axis.axhline(total - 0.5, color="#888888", linewidth=0.5)
                    total += inhibitory.shape[1]
                axis.set(
                    title=f"{'AC'[layer - 1]}  Layer {layer}",
                    ylabel="Neuron index",
                    xlabel="Time from utterance onset (ms)",
                    ylim=(-0.5, total - 0.5),
                    xlim=(0, sample["duration_ms"]),
                )
                axis.spines[["top", "right"]].set_visible(False)
                axis.legend(frameon=False, loc="upper right", markerscale=2)
                correlation_max = 1.0
                for population, color in (("e", "#202020"), ("i", "#a34d45")):
                    if population == "i" and model != "ping":
                        continue
                    acf = sample["models"][model]["autocorrelations"][
                        f"{population}{layer}"
                    ]
                    correlation_max = max(
                        correlation_max,
                        max(
                            (v for v in acf["correlation"] if v is not None), default=1
                        ),
                    )
                    correlation_axis.plot(
                        acf["lag_ms"],
                        [
                            np.nan if value is None else value
                            for value in acf["correlation"]
                        ],
                        color=color,
                        linewidth=0.9,
                        label=population.upper(),
                    )
                    if all(value is None for value in acf["correlation"]):
                        correlation_axis.text(
                            0.98,
                            0.8 if population == "e" else 0.5,
                            f"{population.upper()}: undefined (silent or too short)",
                            color=color,
                            ha="right",
                            transform=correlation_axis.transAxes,
                        )
                correlation_axis.axhline(
                    1, color="#888888", linewidth=0.7, linestyle=":"
                )
                correlation_axis.set(
                    title=f"{'BD'[layer - 1]}  Layer {layer} spike autocorrelogram",
                    xlabel="Lag (ms)",
                    ylabel="ACF",
                    xlim=(0, acf["lag_ms"][-1]),
                    ylim=(-0.05, correlation_max * 1.1),
                )
                correlation_axis.spines[["top", "right"]].set_visible(False)
                correlation_axis.legend(frameon=False, loc="upper right")
            figure.suptitle(
                f"{model.upper()} — trained two-layer response, class {sample['class']}",
                fontsize=10,
            )
            for extension in ("svg",):
                figure.savefig(export / f"{model}_sample_raster.{extension}", dpi=180)
            plt.close(figure)
    write_json_atomic(export / "numbers.json", numbers)


def present(identity):
    source = source_run(
        REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    ref = source.record["inputs"]["compute"]
    recording = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        stage="compute",
        experiment=recipe.SLUG,
        reference=ref,
    )
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": source},
        configuration=source.record["execution"]["configuration"],
    ) as run:
        render(source, recording, run.export)
        recording.check_unchanged()
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
