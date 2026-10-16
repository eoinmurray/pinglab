"""Render training curves from an explicit Scikit Digits analysis."""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pinglab-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp125.recipe import RASTER, SLUG
from experiments.helpers.figsave import save_figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def render_network(cfg, export):
    """Draw the executed external-backend architecture from its recorded recipe."""
    fig, axis = plt.subplots(figsize=(180 / 25.4, 55 / 25.4))
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    boxes = [
        (0.02, f"{cfg['inputs']} pixel inputs\nBernoulli events"),
        (0.39, f"{cfg['hidden']} hidden LIF\nLearned bias"),
        (0.76, f"{cfg['outputs']} output LIF\nLearned bias"),
    ]
    for x, label in boxes:
        axis.add_patch(FancyBboxPatch((x, 0.44), 0.22, 0.32,
            boxstyle="round,pad=0.01", facecolor="#f3f3f3", edgecolor="#202020"))
        axis.text(x + 0.11, 0.60, label, ha="center", va="center", fontsize=9)
    for start, end in ((0.25, 0.38), (0.62, 0.75)):
        axis.add_patch(FancyArrowPatch((start, 0.60), (end, 0.60),
            arrowstyle="-|>", mutation_scale=12, color="#202020"))
        axis.text((start + end) / 2, 0.84, "Dense signed\nweights", ha="center", fontsize=8)
    axis.text(0.50, 0.29, f"LIF decay {cfg['beta']:g} · threshold {cfg['threshold']:g} · delayed subtractive reset",
              ha="center", fontsize=8)
    axis.text(0.50, 0.12, f"{cfg['steps'] * cfg['dt_ms']:g} ms presentation → output spike counts → largest-count digit",
              ha="center", fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    save_figure(fig, export / "network")
    fig.savefig("/tmp/exp125-network.png", dpi=180)
    plt.close(fig)


def render_raster(numbers, recording, export):
    sample, cfg = numbers["sample_raster"], numbers["configuration"]
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 110 / 25.4),
                             gridspec_kw={"width_ratios": [1.6, 1]}, constrained_layout=True)
    with np.load(recording.file("spikes.npz"), allow_pickle=False) as spikes:
        for row, name in enumerate(("hidden", "output")):
            raster_axis, correlation_axis = axes[row]
            step, neuron = np.nonzero(spikes[name])
            raster_axis.scatter((step + 1) * cfg["dt_ms"], neuron, marker="|",
                                s=12 if name == "hidden" else 25, linewidths=0.6, color="#202020")
            raster_axis.set(title=f"{'AC'[row]}  {name.capitalize()} LIF spikes",
                            xlabel="Time from image onset (ms)", ylabel="Neuron index",
                            xlim=(0, cfg["steps"] * cfg["dt_ms"]), ylim=(-0.5, spikes[name].shape[1] - 0.5))
            if name == "output":
                raster_axis.set_yticks(range(cfg["outputs"]))
            values = sample["populations"][name]
            lags = np.asarray(values["lag_ms"])
            correlation = np.asarray([np.nan if value is None else value for value in values["correlation"]])
            positive = lags > 0
            correlation_axis.plot(lags[positive], correlation[positive], color="#202020")
            correlation_axis.axhline(1, color="#777777", linestyle=":", linewidth=0.8)
            finite = correlation[positive & np.isfinite(correlation)]
            correlation_axis.set(title=f"{'BD'[row]}  {name.capitalize()} count autocorrelation",
                xlabel="Lag (ms)", ylabel="Normalized autocorrelation",
                xlim=(0, sample["max_lag_ms"]), ylim=(0, 1.1 * max(1, finite.max() if finite.size else 1)))
            if not finite.size:
                correlation_axis.text(0.5, 0.5, "Undefined: silent population", transform=correlation_axis.transAxes,
                                      ha="center", fontsize=7)
            for axis in (raster_axis, correlation_axis):
                axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Validation-selected LIF response — digit {sample['class']}, predicted {sample['prediction']}", fontsize=10)
    save_figure(fig, export / "sample_raster")
    fig.savefig("/tmp/exp125-sample-raster.png", dpi=180)
    plt.close(fig)


def present(identity):
    source = source_run(REPO / ".pingstore", identity, stage="analyse", experiment=SLUG)
    numbers = load_json(source.file("results.json"))
    cfg = source.record["execution"]["configuration"]
    numbers["execution"] = source.record["execution"]["source_measurements"]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "svg.fonttype": "none"})
    recording = None
    if "sample_raster" in numbers:
        ref = source.record["inputs"]["raster"]
        recording = source_run(REPO / ".pingstore", ref["run_id"], stage="compute",
                               experiment=SLUG, phase=RASTER["phase"], reference=ref)
    with stage_run(REPO, SLUG, "present", inputs={"analyse": source}, configuration=cfg) as run:
        render_network(cfg, run.export)
        fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 76 / 25.4), constrained_layout=True)
        rows = numbers["epochs"]
        epochs = [row["epoch"] for row in rows]
        for split, color in (("train", "#202020"), ("validation", "#a34d45")):
            axes[0].plot(epochs, [row[split]["loss"] for row in rows], color=color, label=split.capitalize())
            axes[1].plot(epochs, [100 * row[split]["accuracy"] for row in rows], color=color)
        axes[0].set(title="A  Classification loss", ylabel="Cross-entropy")
        axes[1].set(title="B  Classification accuracy", ylabel="Accuracy (%)", ylim=(0, 102))
        axes[1].axhline(10, color="#777777", linestyle=":", linewidth=0.8)
        for axis in axes:
            axis.axvline(numbers["selected_epoch"], color="#777777", linestyle="--", linewidth=0.8)
            axis.set(xlabel="Completed epoch", xlim=(0, cfg["epochs"]))
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False)
        save_figure(fig, run.export / "training_curves")
        fig.savefig(run.scratch / "training_curves.png", dpi=160)
        # A preview is disposable; the scientific exports remain vector files.
        import shutil
        shutil.copyfile(run.scratch / "training_curves.png", "/tmp/exp125-training-curves.png")
        plt.close(fig)
        if recording is not None:
            render_raster(numbers, recording, run.export)
            recording.check_unchanged()
        write_json_atomic(run.export / "numbers.json", numbers)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
