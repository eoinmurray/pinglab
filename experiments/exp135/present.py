"""Matched training curves, held-out sample rasters and native network views."""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp135 import recipe
from matplotlib.ticker import MaxNLocator
from pingstore.contracts import write_json_atomic
from pingstore.stages import source_run, stage_run


def save(figure, destination, stem):
    for extension in ("svg", "png", "pdf"):
        figure.savefig(destination / f"{stem}.{extension}", dpi=180)
    plt.close(figure)


def training_curves(report, model, destination):
    rows = report["models"][model]["epochs"]
    cfg = report["configuration"]
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 3.8), layout="constrained")
    for axis, metric in zip(axes, ("loss", "accuracy"), strict=True):
        for split, colour, label in (
            ("train", "#202020", "Online training"),
            ("validation", "#a34d45", "Validation"),
        ):
            selected = [row for row in rows if f"{split}_{metric}" in row]
            scale = 100 if metric == "accuracy" else 1
            axis.plot(
                [r["epoch"] for r in selected],
                [scale * r[f"{split}_{metric}"] for r in selected],
                "o-",
                color=colour,
                label=label,
                markersize=2.5,
            )
        axis.set(xlabel="Completed epoch", xlim=(0, cfg["epochs"]))
        axis.set_box_aspect(3 / 4)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        axis.spines[["top", "right"]].set_visible(False)
    largest_loss = max(
        row[key]
        for case in report["models"].values()
        for row in case["epochs"]
        for key in ("train_loss", "validation_loss")
        if key in row
    )
    largest_accuracy = max(
        100 * row[key]
        for case in report["models"].values()
        for row in case["epochs"]
        for key in ("train_accuracy", "validation_accuracy")
        if key in row
    )
    axes[0].set(
        title="A  Classification loss",
        ylabel="Cross-entropy",
        ylim=(0, max(0.1, largest_loss * 1.1)),
    )
    axes[1].set(
        title="B  Classification accuracy",
        ylabel="Accuracy (%)",
        ylim=(0, min(100, max(20, 115 / cfg["n_classes"], largest_accuracy * 1.15))),
    )
    axes[1].axhline(100 / cfg["n_classes"], linestyle=":", color="#777777", linewidth=1)
    axes[0].legend(frameon=False)
    case = report["models"][model]
    figure.suptitle(
        f"{model.upper()} — reduced-data DVS Gesture training, two hidden layers\n"
        f"{case['n_train']:,} training / {case['n_validation']:,} validation",
        fontsize=10,
    )
    save(figure, destination, f"{model}-training-curves")


def raster(report, source, model, destination):
    sample = report["models"][model]["sample"]
    figure, axes = plt.subplots(
        4,
        1,
        figsize=(7.1, 6.5),
        layout="constrained",
        gridspec_kw={"height_ratios": [2.4, 1, 2.4, 1]},
    )
    with np.load(source.file(model, "spikes.npz"), allow_pickle=False) as spikes:
        for layer in (1, 2):
            axis, ac_axis = axes[2 * (layer - 1) : 2 * layer]
            offset, maximum = 0, 1.0
            for kind, colour, limit in (
                ("E", "#202020", report["configuration"]["raster_cells"]["excitatory"]),
                ("I", "#a34d45", report["configuration"]["raster_cells"]["inhibitory"]),
            ):
                pop = f"block{layer}_{kind}"
                if pop not in sample["populations"]:
                    continue
                size = sample["populations"][pop]
                cells = np.linspace(0, size - 1, min(size, limit), dtype=np.int64)
                events = spikes[pop]
                shown = events[np.isin(events[:, 2], cells)]
                axis.scatter(
                    shown[:, 0] * sample["dt_ms"],
                    np.searchsorted(cells, shown[:, 2]) + offset,
                    marker="|",
                    s=7,
                    linewidths=0.5,
                    color=colour,
                    rasterized=True,
                    label=kind,
                )
                if offset:
                    axis.axhline(offset - 0.5, color="#888888", lw=0.5)
                offset += len(cells)
                acf = sample["activity"][pop]["autocorrelation"]
                values = np.asarray(
                    [np.nan if value is None else value for value in acf["correlation"]]
                )
                ac_axis.plot(acf["lag_ms"], values, color=colour, label=kind, lw=0.9)
                if np.isfinite(values).any():
                    maximum = max(maximum, float(np.nanmax(values)))
                else:
                    ac_axis.text(
                        0.02,
                        0.8 if kind == "E" else 0.5,
                        f"{kind}: undefined (silent or too short)",
                        transform=ac_axis.transAxes,
                        ha="left",
                        color=colour,
                        fontsize=6,
                    )
            axis.set(
                title=f"{'AC'[layer - 1]}  Layer {layer}",
                xlabel="Time from crop start (ms)",
                ylabel="Displayed neuron",
                xlim=(0, sample["presentation_ms"]),
                ylim=(-0.5, offset - 0.5),
            )
            ac_axis.axhline(1, color="#888888", linestyle=":", lw=0.7)
            ac_axis.set(
                title=f"{'BD'[layer - 1]}  Layer {layer} spike autocorrelogram",
                xlabel="Lag (ms)",
                ylabel="ACF",
                xlim=(0, acf["lag_ms"][-1]),
                ylim=(-0.05, maximum * 1.1),
            )
            axis.spines[["top", "right"]].set_visible(False)
            ac_axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False, loc="upper right", markerscale=2)
            ac_axis.legend(frameon=False, loc="upper right")
    figure.suptitle(
        f"{model.upper()} — final weights; true class {sample['label']}, prediction {sample['prediction']}",
        fontsize=10,
    )
    save(figure, destination, f"{model}-sample-raster")


def present(source, *, run_id=None):
    upstream = source_run(
        REPO / ".pingstore", source, stage="analyse", experiment=recipe.SLUG
    )
    pin = upstream.record["inputs"]["training"]
    training = source_run(
        REPO / ".pingstore",
        pin["run_id"],
        stage="compute",
        experiment=recipe.SLUG,
        reference=pin,
    )
    report = json.loads(upstream.file("analysis.json").read_text())
    with stage_run(
        REPO, recipe.SLUG, "present", inputs={"analysis": upstream}, run_id=run_id
    ) as run:
        with plt.rc_context(
            {
                "font.family": "sans-serif",
                "font.size": 8,
                "axes.titlesize": 9,
                "axes.labelsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
            }
        ):
            for model in recipe.MODELS:
                training_curves(report, model, run.export)
                raster(report, training, model, run.export)
                for view in ("circuit", "training", "expanded"):
                    for extension in ("svg", "png"):
                        shutil.copyfile(
                            training.file(
                                model,
                                "training",
                                "network.bundle",
                                "reports",
                                f"{view}.{extension}",
                            ),
                            run.export / f"{model}-network-{view}.{extension}",
                        )
        training.check_unchanged()
        write_json_atomic(run.export / "numbers.json", report)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
