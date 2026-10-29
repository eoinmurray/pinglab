"""Render validated SHD training curves, diagnostic rasters and report numbers.

Run `uv run python experiments/exp124/present.py --source <analyse-run-id>`.
The selected analysis pins its compute ancestor and payload digest. Curves and
numeric summaries come from that analysis, while rasters and architecture
figures come from the same executed compute evidence.

Circuit layouts use saved native definitions, including historical compiled
graphs and training definitions for older inputs. Presentation writes a separate
immutable Pingstore run with six SVGs and numbers.json. It never changes an
article pin or launches an upstream stage.
"""

import argparse
import os
import shutil
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pinglab-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp124 import recipe
from experiments.exp124.analyse import (
    network_definition,
    recorded_populations,
)
from matplotlib.ticker import MaxNLocator, PercentFormatter
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from snnlab import lang, viz
from snnlab.sim import RecordingData, SignalInfo, TrainingHistory
from snnlab.viz import plot_learning_curves, plot_raster


def circuit_diagram(bundle):
    """Lay out native weighted projections; omit recording and scoring aliases."""
    import matplotlib

    native = lang.diagram(bundle, view="circuit")
    known = {node.id for node in native.nodes if node.kind in ("input", "population")}
    symbols = dict(drive="X", E1="E_1", E2="E_2", I1="I_1", I2="I_2", readout="R")
    positions = dict(
        drive=(-360, 0),
        E1=(0, 0),
        E2=(360, 0),
        readout=(720, 0),
        I1=(0, 300),
        I2=(360, 300),
    )
    routes = {
        f"E{k}_to_I{k}": viz.EdgeLayout(
            source_port="bottom_left", target_port="top_left"
        )
        for k in (1, 2)
    }
    routes.update(
        {
            f"I{k}_to_E{k}": viz.EdgeLayout(
                source_port="top_right", target_port="bottom_right"
            )
            for k in (1, 2)
        }
    )
    title = (
        "Two-layer PING classifier" if "I1" in known else "Two-layer COBA classifier"
    )
    diagram = replace(
        native,
        title=title,
        groups=(),
        nodes=tuple(
            replace(node, detail=node.detail.splitlines()[0])
            for node in native.nodes
            if node.id in known
        ),
        edges=tuple(edge for edge in native.edges if edge.id is not None),
    )
    ports = dict(
        upper_left=(0, 0.3),
        lower_left=(0, 0.7),
        upper_right=(1, 0.3),
        lower_right=(1, 0.7),
        top_left=(0.25, 0),
        top_right=(0.75, 0),
        bottom_left=(0.25, 1),
        bottom_right=(0.75, 1),
    )
    layout = viz.DiagramLayout(
        nodes={
            identifier: viz.NodeLayout(
                position=position,
                label=f"${symbols[identifier]}$",
                min_size=(240, 160),
                ports=ports,
            )
            for identifier, position in positions.items()
            if identifier in known
        },
        edges={
            edge.id: replace(
                routes.get(edge.id, viz.EdgeLayout()),
                label=viz.DiagramLabel(
                    rf"$W_{{{symbols[edge.target]},{symbols[edge.source]}}}$"
                ),
            )
            for edge in diagram.edges
        },
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=16), gap=30
        ),
    )
    with matplotlib.rc_context({"savefig.bbox": None}):
        return viz.resolve_diagram(
            diagram, layout=layout, theme=viz.DiagramTheme(secondary_size=13)
        )


def save_circuit(bundle, path):
    """Export the resolved native architecture without inheriting plot cropping."""
    import matplotlib

    circuit = circuit_diagram(bundle)
    with matplotlib.rc_context({"savefig.bbox": None}):
        for suffix in ("svg", "png"):
            viz.render_diagram(circuit, Path(path).with_suffix("." + suffix))


def redraw_networks(presentation, *, run_id=None):
    """Preserve an explicit presentation's evidence and replace only its diagrams."""
    previous = source_run(
        REPO / ".pingstore", presentation, experiment="exp124", stage="present"
    )
    sources = {
        role: source_run(REPO / ".pingstore", pin["run_id"], reference=pin)
        for role, pin in previous.record["inputs"].items()
    }
    pin = sources["analysis"].record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore", pin["run_id"], reference=pin, stage="compute"
    )
    specs = [
        (compute, f"{mode}/train/network.bundle", f"{mode}_network")
        for mode in ("coba", "ping")
    ]
    with stage_run(
        REPO,
        previous.record["experiment"],
        "present",
        inputs={**sources, "figure_source": previous},
        configuration=previous.record["execution"]["configuration"],
        run_id=run_id,
    ) as run:
        shutil.copytree(previous.export, run.export, dirs_exist_ok=True)
        for source, prefix, stem in specs:
            bundle_path = (
                source.restore_bundle(prefix, run.scratch / (stem + ".bundle"))
                if source.record.get("scientific_artifacts", {}).get("native_paths")
                else source.unit(*prefix.split("/"))
            )
            save_circuit(lang.load_bundle(bundle_path), run.export / (stem + ".svg"))
        run.record["execution"]["snnlab_version"] = version("snnlab")
        previous.check_unchanged()
    return run.run_id


def draw_networks(recording, export):
    """Project the executed definitions, including historical pre-bundle evidence."""
    for model in recording.record["execution"]["configuration"]["models"]:
        bundle = lang.Bundle(
            graph=network_definition(recording, model, "graph.json"),
            training=network_definition(recording, model, "training.json"),
            manifest={},
            diagnostics=[],
        )
        save_circuit(bundle, export / f"{model}_network.svg")


def draw_training_curves(conditions, native_metrics, cfg, export):
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    ):
        for condition in cfg["models"]:
            rows = conditions[condition]["epochs"]
            figure, axes = plt.subplots(
                1, 2, figsize=(7.1, 3.8), constrained_layout=True
            )
            artists = plot_learning_curves(
                {"cross_entropy": axes[0], "accuracy": axes[1]},
                TrainingHistory.from_metrics(native_metrics[condition]),
                metrics=("cross_entropy", "accuracy"),
                colors={"train": "#202020", "validation": "#a34d45"},
                labels={"train": "Online training", "validation": "Validation"},
                marker="o",
                markersize=2.5,
            )
            axes[0].set(title="A  Classification loss", ylabel="Cross-entropy")
            maximum = max(
                row[split]["accuracy"]
                for row in rows
                for split in ("train", "validation")
                if row[split]["accuracy"] is not None
            )
            axes[1].set(
                title="B  Classification accuracy",
                ylabel="Accuracy (%)",
                ylim=(0, min(1, max(0.2, maximum * 1.15))),
            )
            axes[1].yaxis.set_major_formatter(PercentFormatter(1, symbol=""))
            axes[1].axhline(0.05, linestyle=":", color="#777777", linewidth=1)
            for axis in axes:
                axis.set(xlabel="Completed epoch", xlim=(0, cfg["epochs"]))
                axis.set_box_aspect(3 / 4)
                axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                axis.spines[["top", "right"]].set_visible(False)
            axes[0].legend(
                handles=[
                    artists[split, "cross_entropy"][0]
                    for split in ("train", "validation")
                ],
                labels=("Online training", "Validation"),
                frameon=False,
            )
            figure.suptitle(
                f"{condition.upper()} — {'full SHD' if cfg['full_dataset'] else 'reduced-data'} training, two hidden layers\n"
                f"{cfg['train_examples']:,} training / {cfg['validation_examples']:,} validation",
                fontsize=10,
            )
            figure.savefig(export / f"full_{condition}_training_curves.svg", dpi=180)
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
    }
    draw_training_curves(conditions, results["training_metrics"], cfg, export)
    draw_networks(recording, export)
    filename = (
        "recording.npz" if recording.file("recording.npz").exists() else "rasters.npz"
    )
    with (
        np.load(recording.file(filename), allow_pickle=False) as spikes,
        plt.rc_context({"font.size": 8, "axes.titlesize": 9}),
    ):
        for model, populations in recorded_populations(recording).items():
            signals = {pop: spikes[f"{model}_{pop}"] for pop in populations}
            saved_info = (
                recording.record["execution"].get("recording_info", {}).get(model)
            )
            # Historical dense rasters predate native coordinate exports; their
            # contiguous cells and full-window steps are explicit in the source.
            info = {
                pop: SignalInfo(**saved_info[f"{pop.upper()}.spikes"])
                if saved_info
                else SignalInfo(
                    axes=("time", "cell"),
                    unit="spike",
                    dt_ms=sample["dt_ms"],
                    cell_ids=tuple(range(values.shape[1])),
                    step_indices=tuple(range(values.shape[0])),
                    start_step=0,
                    end_step=values.shape[0],
                    array_dtype=str(values.dtype),
                )
                for pop, values in signals.items()
            }
            native_recording = RecordingData(signals, info)
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
                plot_raster(
                    axis,
                    native_recording,
                    signal=f"e{layer}",
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
                    plot_raster(
                        axis,
                        native_recording,
                        signal=f"i{layer}",
                        offset=total,
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
            figure.savefig(export / f"{model}_sample_raster.svg", dpi=180)
            plt.close(figure)
    write_json_atomic(export / "numbers.json", numbers)


def present(identity):
    source = source_run(
        REPO / ".pingstore",
        identity,
        stage="analyse",
        experiment=recipe.configuration()["experiment"],
    )
    ref = source.record["inputs"]["compute"]
    recording = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        stage="compute",
        experiment=source.record["experiment"],
        reference=ref,
    )
    with stage_run(
        REPO,
        source.record["experiment"],
        "present",
        inputs={"analysis": source},
        configuration=source.record["execution"]["configuration"],
    ) as run:
        render(source, recording, run.export)
        recording.check_unchanged()
    return run.run_id


if __name__ == "__main__":
    if any(arg.startswith("--redraw-from") for arg in sys.argv[1:]):
        parser = argparse.ArgumentParser(
            description="Refresh only the circuit diagrams of an explicit presentation."
        )
        parser.add_argument("--redraw-from", dest="presentation", required=True)
        parser.add_argument("--run-id")
        print(redraw_networks(**vars(parser.parse_args())), flush=True)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--source", required=True)
        present(parser.parse_args().source)
