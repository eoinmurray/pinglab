"""Render retained agreement, learning and timing comparisons."""

import csv
import os
import shutil

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pinglab-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp126.compute import REPO
from experiments.exp126.recipe import CONFIG, SLUG
from experiments.helpers.figsave import save_figure
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run

COLORS = {"snntorch": "#202020", "snnlab": "#a34d45"}


def finish(fig, axes, stem):
    for axis in np.asarray(axes).flat:
        axis.spines[["top", "right"]].set_visible(False)
    save_figure(fig, stem)
    plt.close(fig)


def validation_plot(run, result):
    pairs = [pair for pair in result["pairs"] if pair.get("validation_policy")]
    if not pairs:
        return
    pair = pairs[0]
    execution = pair["execution"]["snnlab"]
    operations = ("forward", "forward_backward", "optimization_step")
    fig, axes = plt.subplots(
        1, 2, figsize=(180 / 25.4, 85 / 25.4), constrained_layout=True
    )
    for policy, color, offset in (
        ("structural", "#a34d45", -0.14),
        ("strict", "#202020", 0.14),
    ):
        benchmark = (
            execution["microbenchmark"]
            if policy == "structural"
            else execution["strict_microbenchmark"]
        )
        for x, operation in enumerate(operations):
            values = np.asarray(benchmark["seconds"][operation]) * 1000
            axes[0].scatter(
                x + offset + np.linspace(-0.03, 0.03, len(values)),
                values,
                color=color,
                s=12,
                alpha=0.7,
                label=policy if x == 0 else None,
            )
            axes[0].plot(
                [x + offset - 0.06, x + offset + 0.06],
                [np.median(values)] * 2,
                color=color,
            )
        counts = [
            pair["validation_policy"]["profiles"][policy][operation].get(
                "cudaStreamSynchronize", 0
            )
            for operation in ("forward", "forward_backward")
        ]
        axes[1].bar(
            np.arange(2) + offset, counts, width=0.26, color=color, label=policy
        )
    axes[0].set(
        title="A  Default and strict timing",
        xticks=range(3),
        xticklabels=["Forward", "Fwd + bwd", "Full step"],
        ylabel="Wall time (ms)",
        ylim=(0, None),
    )
    axes[1].set(
        title="B  Profiled synchronization counts",
        xticks=range(2),
        xticklabels=["Forward", "Fwd + bwd"],
        ylabel="CUDA stream synchronization calls",
    )
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    finish(fig, axes, run.export / "validation_policy")
    result["validation_policy_figure_seed"] = pair["seed"]
    with (run.export / "validation_policy_table.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "policy",
                "operation",
                "median_seconds",
                "stream_synchronizations",
            ),
        )
        writer.writeheader()
        for pair in pairs:
            execution = pair["execution"]["snnlab"]
            for policy, benchmark in (
                ("structural", execution["microbenchmark"]),
                ("strict", execution["strict_microbenchmark"]),
            ):
                for operation in operations:
                    writer.writerow(
                        dict(
                            seed=pair["seed"],
                            policy=policy,
                            operation=operation,
                            median_seconds=float(
                                np.median(benchmark["seconds"][operation])
                            ),
                            stream_synchronizations=pair["validation_policy"][
                                "profiles"
                            ][policy]
                            .get(operation, {})
                            .get("cudaStreamSynchronize")
                            if operation != "optimization_step"
                            else None,
                        )
                    )


def present(identity, baseline_identity=None):
    source = source_run(REPO / ".pingstore", identity, stage="analyse", experiment=SLUG)
    result = load_json(source.file("results.json"))
    if result.get("schema") != "exp126.comparison/v1":
        raise ValueError("not a comparison analysis")
    inputs = {"analyse": source}
    if baseline_identity is not None:
        baseline = source_run(
            REPO / ".pingstore", baseline_identity, stage="present", experiment=SLUG
        )
        numbers = load_json(baseline.file("numbers.json"))
        if (
            baseline.record["execution"]["configuration"] != CONFIG
            or "benchmark" not in numbers
        ):
            raise ValueError("baseline must be an original-protocol presentation")
        inputs["baseline"] = baseline
        result["baseline"] = numbers
    for pair in result["pairs"]:
        pair["execution"] = {
            name: source.record["execution"]["source_benchmarks"][identity]
            for name, identity in pair["run_ids"].items()
        }
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "svg.fonttype": "none"})
    with stage_run(
        REPO,
        SLUG,
        "present",
        inputs=inputs,
        configuration=source.record["execution"]["configuration"],
    ) as run:
        if baseline_identity is not None:
            for stem in ("training_curves", "timing"):
                for extension in ("svg", "pdf"):
                    filename = f"{stem}.{extension}"
                    shutil.copyfile(
                        baseline.file(filename), run.export / f"baseline_{filename}"
                    )
        with np.load(source.file("agreement.npz"), allow_pickle=False) as data:
            fig, axes = plt.subplots(
                2, 2, figsize=(180 / 25.4, 130 / 25.4), constrained_layout=True
            )
            time = np.arange(result["protocol"]["steps"]) * result["protocol"]["dt_ms"]
            cell = int(data["snntorch_hidden_spikes"][:, 0].sum(0).argmax())
            for name, color in COLORS.items():
                axes[0, 0].plot(
                    time,
                    data[name + "_hidden_synaptic"][:, 0, cell],
                    color=color,
                    label=name,
                    linestyle="-" if name == "snntorch" else "--",
                )
                axes[0, 1].plot(
                    time,
                    data[name + "_hidden_voltage"][:, 0, cell],
                    color=color,
                    linestyle="-" if name == "snntorch" else "--",
                )
                spikes = data[name + "_output_spikes"][:, 0]
                k, c = np.nonzero(spikes)
                axes[1, 0].scatter(
                    time[k],
                    c + (-0.12 if name == "snntorch" else 0.12),
                    s=8,
                    color=color,
                    label=name,
                )
            a = np.concatenate(
                [
                    data["snntorch_gradient_" + key].ravel()
                    for key in (
                        "input_weight",
                        "input_bias",
                        "output_weight",
                        "output_bias",
                    )
                ]
            )
            b = np.concatenate(
                [
                    data["snnlab_gradient_" + key].ravel()
                    for key in (
                        "input_weight",
                        "input_bias",
                        "output_weight",
                        "output_bias",
                    )
                ]
            )
            selected = np.linspace(0, len(a) - 1, min(2000, len(a))).astype(int)
            axes[1, 1].scatter(
                a[selected],
                b[selected],
                s=2,
                color="#202020",
                alpha=0.4,
                rasterized=True,
            )
            limits = [
                min(a[selected].min(), b[selected].min()),
                max(a[selected].max(), b[selected].max()),
            ]
            axes[1, 1].plot(limits, limits, color="#a34d45", linestyle="--")
            axes[0, 0].set(
                title="A  Hidden synaptic current",
                ylabel="Dimensionless current",
                xlabel="Time (ms)",
            )
            axes[0, 1].set(
                title="B  Hidden membrane state",
                ylabel="Dimensionless membrane",
                xlabel="Time (ms)",
            )
            axes[1, 0].set(
                title="C  Output spike agreement",
                ylabel="Output neuron",
                xlabel="Time (ms)",
                yticks=range(10),
            )
            axes[1, 1].set(
                title="D  Parameter gradient agreement",
                xlabel="snnTorch gradient",
                ylabel="snnlab gradient",
            )
            axes[0, 0].legend(frameon=False)
            finish(fig, axes, run.export / "equation_agreement")
            result["probe_hidden_cell"] = cell
        fig, axes = plt.subplots(
            2, 2, figsize=(180 / 25.4, 130 / 25.4), constrained_layout=True
        )
        for pair in result["pairs"]:
            for name, color in COLORS.items():
                rows = pair["learning"][name]["epochs"]
                epochs = [row["epoch"] for row in rows]
                for split, style in (("train", "-"), ("validation", "--")):
                    label = f"{name} {split}" if pair is result["pairs"][0] else None
                    axes[0, 0].plot(
                        epochs,
                        [row[split]["loss"] for row in rows],
                        color=color,
                        linestyle=style,
                        alpha=0.7,
                        label=label,
                    )
                    axes[0, 1].plot(
                        epochs,
                        [100 * row[split]["accuracy"] for row in rows],
                        color=color,
                        linestyle=style,
                        alpha=0.7,
                    )
                seconds = np.cumsum(
                    [
                        row["optimization_seconds"]
                        for row in pair["execution"][name]["timing"]["epoch_timings"]
                    ]
                )
                axes[1, 0].plot(
                    seconds,
                    [100 * row["validation"]["accuracy"] for row in rows],
                    color=color,
                    alpha=0.7,
                    label=name if pair is result["pairs"][0] else None,
                )
            axes[1, 1].plot(
                [0, 1],
                [100 * pair["metrics"][name]["test_accuracy"] for name in COLORS],
                color="#777777",
                alpha=0.5,
            )
            for x, (name, color) in enumerate(COLORS.items()):
                axes[1, 1].scatter(
                    x, 100 * pair["metrics"][name]["test_accuracy"], color=color, s=18
                )
        axes[0, 0].set(
            title="A  Train and validation loss",
            xlabel="Completed epoch",
            ylabel="Cross-entropy",
        )
        axes[0, 1].set(
            title="B  Classification accuracy",
            xlabel="Completed epoch",
            ylabel="Accuracy (%)",
            ylim=(0, 102),
        )
        axes[1, 0].set(
            title="C  Time to validation performance",
            xlabel="Cumulative optimization time (s)",
            ylabel="Validation accuracy (%)",
            ylim=(0, 102),
        )
        axes[1, 1].set(
            title="D  Held-out accuracy by replicate",
            xticks=[0, 1],
            xticklabels=list(COLORS),
            ylabel="Test accuracy (%)",
            ylim=(0, 102),
        )
        axes[0, 0].legend(frameon=False, fontsize=6)
        axes[1, 0].legend(frameon=False)
        finish(fig, axes, run.export / "learning_comparison")
        fig, axes = plt.subplots(
            1, 2, figsize=(180 / 25.4, 80 / 25.4), constrained_layout=True
        )
        operations = ("encoding", "forward", "forward_backward", "optimization_step")
        for pair in result["pairs"]:
            for name, color in COLORS.items():
                offset = -0.13 if name == "snntorch" else 0.13
                execution = pair["execution"][name]
                for x, operation in enumerate(operations):
                    values = (
                        np.asarray(execution["microbenchmark"]["seconds"][operation])
                        * 1000
                    )
                    jitter = np.linspace(-0.04, 0.04, len(values))
                    axes[0].scatter(
                        x + offset + jitter,
                        values,
                        color=color,
                        s=12,
                        alpha=0.6,
                        label=name if x == 0 and pair is result["pairs"][0] else None,
                    )
                    axes[0].plot(
                        [x + offset - 0.07, x + offset + 0.07],
                        [np.median(values)] * 2,
                        color=color,
                    )
                values = execution["timing"]["inference_seconds"]
                x = list(result["pairs"]).index(pair) + offset
                axes[1].scatter(
                    x + np.linspace(-0.04, 0.04, len(values)),
                    values,
                    color=color,
                    s=12,
                    alpha=0.6,
                )
                axes[1].plot([x - 0.07, x + 0.07], [np.median(values)] * 2, color=color)
        axes[0].set(
            title="A  Fixed-batch timing repeats",
            xticks=range(4),
            xticklabels=["Encode", "Forward", "Fwd + bwd", "Full step"],
            ylabel="Wall time (ms)",
            ylim=(0, None),
        )
        axes[1].set(
            title="B  Full-test inference repeats",
            xticks=range(len(result["pairs"])),
            xticklabels=[p["seed"] for p in result["pairs"]],
            xlabel="Paired training seed",
            ylabel="Time for 10,000 images (s)",
            ylim=(0, None),
        )
        axes[0].legend(frameon=False)
        finish(fig, axes, run.export / "timing_comparison")
        with (run.export / "comparison_table.csv").open("w") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("seed", "measurement", "snntorch", "snnlab", "speedup"),
            )
            writer.writeheader()
            writer.writerows(result["table"])
        validation_plot(run, result)
        write_json_atomic(run.export / "numbers.json", result)
    return run.run_id
