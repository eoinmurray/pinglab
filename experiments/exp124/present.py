"""Render validated SHD training curves, diagnostic rasters and report numbers.

Run `uv run python experiments/exp124/present.py --source <analyse-run-id>`.
The selected analysis pins its compute ancestor and payload digest. Curves and
numeric summaries come from that analysis, while rasters and architecture
figures come from the same executed compute evidence.

New runs reuse their saved native circuit diagrams; historical compiled graph
and training definitions support older inputs. Presentation writes a separate
immutable Pingstore run with six SVGs and numbers.json. It never changes an
article pin or launches an upstream stage.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
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
from snnlab import lang
from snnlab.sim import RecordingData, SignalInfo, TrainingHistory
from snnlab.viz import plot_learning_curves, plot_raster


def draw_networks(recording, export):
    """Project the executed definitions, including historical pre-bundle evidence."""
    for model in recording.record["execution"]["configuration"]["models"]:
        saved = recording.file(model, "circuit.svg")
        if saved.exists():
            shutil.copyfile(saved, export / f"{model}_network.svg")
            continue
        bundle = lang.Bundle(
            graph=network_definition(recording, model, "graph.json"),
            training=network_definition(recording, model, "training.json"),
            manifest={},
            diagnostics=[],
        )
        bundle.visualise(export / f"{model}_network.svg", view="circuit")


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
