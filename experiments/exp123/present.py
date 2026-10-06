"""Render saved true-class score trajectories without upstream execution."""

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[2] / "tools"),
]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp123 import recipe
from experiments.helpers import theme
from pingstore.stages import source_run, stage_run


def save(fig, destination):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(destination.with_suffix("." + ext), dpi=300)
    plt.close(fig)


def plot_grid(report, source, destination, coordinate):
    fig, axes = plt.subplots(
        5, 2, figsize=(180 / 25.4, 170 / 25.4), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06)
    colours = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    for j, (ax, image) in enumerate(zip(axes.flat, report["image_reports"])):
        with np.load(
            source.unit(f"image{image['image_index']:05d}") / "trajectories.npz",
            allow_pickle=False,
        ) as data:
            x = data["time_ms"] if coordinate == "time" else data["cycle_position"]
            score = (
                data["true_score"]
                if coordinate == "time"
                else data["true_score_by_cycle"]
            )
            if len(x):
                for ti, (tau, colour) in enumerate(
                    zip(report["design"]["tau_gaba_ms"], colours)
                ):
                    ax.plot(
                        x,
                        score[ti].mean(axis=0),
                        color=colour,
                        lw=1.3,
                        label=f"GABA {tau:g} ms",
                    )
                ax.set_xlim(x[0], x[-1])
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No common cycles",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=theme.SIZE_ANNOTATION,
                )
            ax.set_ylim(0, 1.04)
            ax.set_title(
                f"Image {image['image_index']:02d} · digit {image['label']}",
                fontsize=theme.SIZE_TITLE,
            )
            if coordinate == "time":
                ax.set_xticks([0, 100, 200])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=theme.SIZE_TICK)
            if j % 2 == 0:
                ax.set_ylabel("True-class\nsoftmax score", fontsize=theme.SIZE_LABEL)
            else:
                ax.tick_params(labelleft=False)
            if j >= 8:
                ax.set_xlabel(
                    "Time (ms)" if coordinate == "time" else "Cycle position",
                    fontsize=theme.SIZE_LABEL,
                )
    theme.label_panels(axes.flat)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncols=3,
            fontsize=theme.SIZE_LEGEND,
        )
    save(fig, destination)


def plot_decisions(report, destination):
    fig, axes = plt.subplots(
        2, 10, figsize=(180 / 25.4, 90 / 25.4), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.06)
    taus = np.array(report["design"]["tau_gaba_ms"])
    for j, image in enumerate(report["image_reports"]):
        entries = [
            next(
                d
                for d in report["decisions"]
                if d["image_index"] == image["image_index"] and d["tau_gaba_ms"] == tau
            )
            for tau in taus
        ]
        for k, coordinate in enumerate(("ms", "cycles")):
            ax = axes[k, j]
            q = np.array(
                [
                    d[f"quartiles_{coordinate}"]
                    if d[f"quartiles_{coordinate}"] is not None
                    else [np.nan] * 3
                    for d in entries
                ]
            )
            ax.fill_between(taus, q[:, 0], q[:, 2], color=theme.DEEP_RED, alpha=0.15)
            ax.plot(
                taus, q[:, 1], color=theme.INK_BLACK, marker="o", markersize=3, lw=1
            )
            ax.vlines(taus, q[:, 0], q[:, 2], color=theme.DEEP_RED, lw=0.8)
            upper = (
                (210 if np.nanmax(q[:, 2], initial=0) > 100 else 110) if k == 0 else 5.2
            )
            ax.set_ylim(0, upper)
            ax.set_yticks(
                np.arange(0, 201, 25)
                if upper == 210
                else np.arange(0, 101, 10)
                if k == 0
                else np.arange(0, 5.1, 0.5)
            )
            if not np.isfinite(q).any():
                ax.text(
                    0.5,
                    0.45,
                    "None",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=theme.SIZE_ANNOTATION,
                )
            if j == 0:
                theme.label_panel(ax, "A" if k == 0 else "B", x=-0.08, y=1.12)
            ax.set_title(
                f"{image['image_index']:02d} · {image['label']}",
                fontsize=theme.SIZE_ANNOTATION,
            )
            ax.set_xticks(taus, [f"{tau:g}" for tau in taus])
            ax.set_xlim(2, 13)
            if j == 0:
                ax.set_ylabel(
                    "Time (ms)" if k == 0 else "Cycle position",
                    fontsize=theme.SIZE_LABEL,
                )
            elif upper != 210:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=theme.SIZE_TICK)
    fig.supxlabel("GABA decay (ms)", fontsize=theme.SIZE_LABEL)
    fig.suptitle(
        "Sustained correct decisions · median and interquartile range",
        fontsize=theme.SIZE_TITLE,
        fontweight="bold",
    )
    save(fig, destination)


def plot_accuracy(report, destination):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(180 / 25.4, 110 / 25.4),
        layout="constrained",
        gridspec_kw={"height_ratios": [2, 1]},
        sharex="col",
    )
    colours = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    for j, (key, xlabel) in enumerate(
        (
            ("time", "Elapsed time (ms)"),
            ("cycles", "Cycle position"),
            ("input", "Cumulative input spikes"),
        )
    ):
        curve = report["curves"][key]
        x = np.array(curve["grid"])
        coverage = np.array(curve["trial_coverage"])
        for ti, (tau, colour) in enumerate(
            zip(report["design"]["tau_gaba_ms"], colours)
        ):
            y, low, high = [
                np.array(curve[name][ti], dtype=float) * 100
                for name in ("accuracy", "ci_low", "ci_high")
            ]
            y[coverage < 0.8] = np.nan
            low[coverage < 0.8] = np.nan
            high[coverage < 0.8] = np.nan
            axes[0, j].plot(x, y, color=colour, lw=1.3, label=f"GABA {tau:g} ms")
            axes[0, j].fill_between(x, low, high, color=colour, alpha=0.12)
        axes[0, j].set_ylim(0, 100)
        axes[0, j].set_yticks([0, 25, 50, 75, 100])
        axes[0, j].set_title(("Physical time", "Gamma cycles", "Input budget")[j])
        axes[1, j].plot(x, coverage * 100, color=theme.INK_BLACK, lw=1.3)
        axes[1, j].axhline(80, color=theme.GREY_MID, lw=0.7, ls="--")
        axes[1, j].set_ylim(0, 105)
        axes[1, j].set_yticks([0, 50, 100])
        axes[1, j].set_xlabel(xlabel)
        retained = np.flatnonzero(coverage >= 0.5)
        end = min(len(x) - 1, retained[-1] + 1) if len(retained) else len(x) - 1
        axes[1, j].set_xlim(x[0], x[end])
        if j == 0:
            axes[0, j].set_ylabel("Accuracy (%)")
            axes[1, j].set_ylabel("Paired trials\nretained (%)")
        else:
            for row in range(2):
                axes[row, j].tick_params(labelleft=False)
    theme.label_panels(axes.flat)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncols=3, fontsize=theme.SIZE_LEGEND
    )
    save(fig, destination)


def present(identity, accuracy_identity=None, pareto_identity=None):
    source = source_run(
        recipe.REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    report = json.loads(source.file("results.json").read_text())
    if (
        report.get("schema") != "exp123.analysis/v1"
        or len(report["image_reports"]) != 10
        or "decisions" not in report
    ):
        raise ValueError("unsupported analysis payload")
    theme.set_paper_mode(True)
    theme.apply()
    plt.rcParams["svg.fonttype"] = "none"
    inputs = {"analyse": source}
    accuracy_report = None
    pareto_report = None
    if accuracy_identity is not None:
        accuracy_source = source_run(
            recipe.REPO / ".pingstore",
            accuracy_identity,
            stage="analyse",
            experiment=recipe.SLUG,
        )
        accuracy_report = json.loads(accuracy_source.file("accuracy.json").read_text())
        if accuracy_report.get("schema") != "exp123.accuracy.analysis/v1":
            raise ValueError("unsupported accuracy analysis")
        inputs["accuracy_analyse"] = accuracy_source
    if pareto_identity is not None:
        pareto_source = source_run(
            recipe.REPO / ".pingstore",
            pareto_identity,
            stage="analyse",
            experiment=recipe.SLUG,
        )
        pareto_report = json.loads(pareto_source.file("pareto.json").read_text())
        if pareto_report.get("schema") != "exp123.pareto.analysis/v1":
            raise ValueError("unsupported pareto analysis")
        inputs["pareto_analyse"] = pareto_source
    with stage_run(
        recipe.REPO,
        recipe.SLUG,
        "present",
        inputs=inputs,
        configuration=dict(
            schema="exp123.presentation/v1",
            paper_mode=True,
            style="shared paper theme; trajectory panel labels and decision row labels A/B",
            draw_aggregation="mean within each image and tau",
            show_encoding_traces=False,
            decision_layout="time across top row, cycles across bottom row; median and interquartile range",
            accuracy_minimum_display_coverage=0.8,
            accuracy_axis_extent="last coordinate with at least 50% paired-trial coverage plus one grid step",
        ),
    ) as run:
        plot_grid(report, source, run.export / "confidence_time", "time")
        plot_grid(report, source, run.export / "confidence_cycles", "cycles")
        plot_decisions(report, run.export / "sustained_decisions")
        if accuracy_report is not None:
            plot_accuracy(accuracy_report, run.export / "accuracy_comparison")
            (run.export / "accuracy_numbers.json").write_text(
                json.dumps(accuracy_report, indent=2, allow_nan=False) + "\n"
            )
        if pareto_report is not None:
            from experiments.exp123.pareto_plots import plot_tradeoffs, plot_frontiers

            plot_tradeoffs(pareto_report, run.export / "deadline_tradeoffs", save)
            plot_frontiers(pareto_report, run.export / "pareto_frontiers", save)
            (run.export / "pareto_numbers.json").write_text(
                json.dumps(pareto_report, indent=2, allow_nan=False) + "\n"
            )
        (run.export / "numbers.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        result = run.run_id
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--accuracy-source")
    parser.add_argument("--pareto-source")
    args = parser.parse_args()
    present(args.source, args.accuracy_source, args.pareto_source)
