"""Draw learning curves and timing from one explicitly named analysis."""

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
from experiments.exp126.recipe import SLUG
from experiments.helpers.figsave import save_figure
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def present(identity):
    source = source_run(REPO / ".pingstore", identity, stage="analyse", experiment=SLUG)
    numbers = load_json(source.file("results.json"))
    cfg = source.record["execution"]["configuration"]
    benchmark = source.record["execution"]["benchmark"]
    numbers["benchmark"] = benchmark
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "svg.fonttype": "none"})
    with stage_run(
        REPO, SLUG, "present", inputs={"analyse": source}, configuration=cfg
    ) as run:
        fig, axes = plt.subplots(
            1, 2, figsize=(180 / 25.4, 76 / 25.4), constrained_layout=True
        )
        epochs = [row["epoch"] for row in numbers["epochs"]]
        for split, color in (("train", "#202020"), ("validation", "#a34d45")):
            axes[0].plot(
                epochs,
                [row[split]["loss"] for row in numbers["epochs"]],
                color=color,
                label=split.capitalize(),
            )
            axes[1].plot(
                epochs,
                [100 * row[split]["accuracy"] for row in numbers["epochs"]],
                color=color,
            )
        axes[0].set(title="A  Classification loss", ylabel="Cross-entropy")
        axes[1].set(
            title="B  Classification accuracy", ylabel="Accuracy (%)", ylim=(0, 102)
        )
        axes[1].axhline(10, color="#777777", linestyle=":", linewidth=0.8)
        for axis in axes:
            axis.axvline(
                numbers["selected_epoch"],
                color="#777777",
                linestyle="--",
                linewidth=0.8,
            )
            axis.set(xlabel="Completed epoch", xlim=(0, cfg["epochs"]))
            axis.spines[["top", "right"]].set_visible(False)
        axes[0].legend(frameon=False)
        save_figure(fig, run.export / "training_curves")
        plt.close(fig)
        fig, axes = plt.subplots(
            1, 2, figsize=(180 / 25.4, 76 / 25.4), constrained_layout=True
        )
        rows = benchmark["timing"]["epoch_timings"]
        for key, label, color in (
            ("optimization_seconds", "Optimization", "#202020"),
            ("evaluation_seconds", "Train + validation evaluation", "#a34d45"),
        ):
            axes[0].plot(epochs, [row[key] for row in rows], label=label, color=color)
        seconds = benchmark["timing"]["inference_seconds"]
        axes[1].plot(range(1, len(seconds) + 1), seconds, "o-", color="#202020")
        axes[1].axhline(
            benchmark["test_inference_median_seconds"],
            color="#a34d45",
            linestyle="--",
            label="Median",
        )
        axes[0].set(
            title="A  Epoch wall time",
            xlabel="Completed epoch",
            ylabel="Time (s)",
            ylim=(0, None),
        )
        axes[1].set(
            title="B  Full test-set inference",
            xlabel="Timing repeat",
            ylabel="Time for 10,000 images (s)",
            ylim=(0, max(seconds) * 1.15),
            xticks=range(1, len(seconds) + 1),
        )
        for axis in axes:
            axis.legend(frameon=False, fontsize=7)
            axis.spines[["top", "right"]].set_visible(False)
        save_figure(fig, run.export / "timing")
        plt.close(fig)
        write_json_atomic(run.export / "numbers.json", numbers)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed analyse run ID")
    parser.add_argument(
        "--baseline", help="explicit original baseline present run to reuse"
    )
    arguments = parser.parse_args()
    identity = arguments.source
    source = source_run(REPO / ".pingstore", identity, stage="analyse", experiment=SLUG)
    if (
        source.record["execution"]["configuration"].get("schema")
        == "exp126.current-profile-analysis/v1"
    ):
        if arguments.baseline:
            parser.error("--baseline is unsupported for diagnostic presentation")
        from experiments.exp126.compute_parts.current_profiling import (
            present as current_present,
        )

        current_present(identity)
    elif (
        source.record["execution"]["configuration"].get("schema")
        == "exp126.profile-analysis/v1"
    ):
        if arguments.baseline:
            parser.error("--baseline is not supported for profiling presentation")
        from experiments.exp126.compute_parts.profiling import (
            present as profile_present,
        )

        profile_present(identity)
    elif (
        source.record["execution"]["configuration"].get("schema")
        == "exp126.comparison/v1"
    ):
        from experiments.exp126.compute_parts.comparison_present import (
            present as matched_present,
        )

        matched_present(identity, arguments.baseline)
    else:
        if arguments.baseline:
            parser.error("--baseline is only supported for comparison presentation")
        present(identity)
