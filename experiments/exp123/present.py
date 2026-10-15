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
    with plt.rc_context({"savefig.bbox": None}):
        for ext in ("svg", "pdf"):
            fig.savefig(destination.with_suffix("." + ext), dpi=300)
    plt.close(fig)


def plot_grid(report, source, destination, coordinate):
    rows, columns = 3, 3
    images = report["image_reports"][:9]
    fig, axes = plt.subplots(
        rows, columns,
        figsize=tuple(mm / 25.4 for mm in recipe.FIGURE_SIZE_MM),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06)
    colours = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    for j, (ax, image) in enumerate(zip(axes.flat, images)):
        with np.load(
            source.unit(f"image{image['image_index']:05d}") / "trajectories.npz",
            allow_pickle=False,
        ) as data:
            x = data["time_ms"] if coordinate == "time" else data["completed_volley_count"]
            score = (
                data["true_score"]
                if coordinate == "time"
                else data["true_score_by_completed_volley"]
            )
            if coordinate == "cycles" and len(x):
                x = np.concatenate(([0], x))
                score = np.concatenate((data["true_score"][:, :, :1], score), axis=2)
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
                        marker=None if coordinate == "time" else "o",
                        markersize=2.5,
                    )
                ax.set_xlim(x[0], x[-1])
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No common completed volleys",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=theme.SIZE_ANNOTATION,
                )
            ax.set_ylim(0, 1.04)
            ax.set_title(
                f"Image {image['image_index']:02d} · digit {image['label']}"
                if coordinate == "time"
                else f"{image['image_index']:02d} · digit {image['label']}",
                fontsize=theme.SIZE_TITLE,
            )
            if coordinate == "time":
                ax.set_xticks([0, 100, 200])
            elif len(x):
                ax.set_xticks(np.arange(0, int(x[-1]) + 1))
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=theme.SIZE_TICK)
            if j % columns == 0:
                ax.set_ylabel("True-class\nsoftmax score", fontsize=theme.SIZE_LABEL)
            else:
                ax.tick_params(labelleft=False)
            if coordinate == "time" and j >= (rows - 1) * columns:
                ax.set_xlabel(
                    "Time (ms)",
                    fontsize=theme.SIZE_LABEL,
                )
    if coordinate == "cycles":
        fig.supxlabel("Completed volley count", fontsize=theme.SIZE_LABEL)
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
    images = report["image_reports"][:9]
    image_indices = {image["image_index"] for image in images}
    volley_medians = [
        d["quartiles_completed_volley_count"][1]
        for d in report["decisions"]
        if d["image_index"] in image_indices
        and d["quartiles_completed_volley_count"] is not None
    ]
    volley_lower = int(np.floor(min(volley_medians))) if volley_medians else 1
    volley_upper = int(np.ceil(max(volley_medians))) if volley_medians else 2
    fig, axes = plt.subplots(
        2, len(images),
        figsize=tuple(mm / 25.4 for mm in recipe.FIGURE_SIZE_MM),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.015, h_pad=0.06)
    taus = np.array(report["design"]["tau_gaba_ms"])
    for j, image in enumerate(images):
        entries = [
            next(
                d
                for d in report["decisions"]
                if d["image_index"] == image["image_index"] and d["tau_gaba_ms"] == tau
            )
            for tau in taus
        ]
        for k, coordinate in enumerate(("ms", "completed_volley_count")):
            ax = axes[k, j]
            q = np.array(
                [
                    d[f"quartiles_{coordinate}"]
                    if d[f"quartiles_{coordinate}"] is not None
                    else [np.nan] * 3
                    for d in entries
                ]
            )
            ax.plot(
                q[:, 1], taus, color=theme.INK_BLACK,
                marker="o", markersize=3.7, lw=1, zorder=3,
            )
            if k == 0:
                ax.set_xlim(0, 200)
                ax.set_xticks(np.arange(0, 201, 50))
                ax.set_xticks(np.arange(0, 201, 10), minor=True)
            else:
                ax.set_xlim(volley_lower - 0.2, volley_upper + 0.2)
                ax.set_xticks(np.arange(volley_lower, volley_upper + 1))
            if k == 0:
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment("right")
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
            ax.set_yticks(taus, [f"{tau:g}" for tau in taus])
            ax.set_ylim(2, 13)
            if j == 0:
                ax.set_ylabel(
                    "GABA decay (ms)",
                    fontsize=theme.SIZE_LABEL,
                )
            else:
                ax.tick_params(labelleft=False)
            if j == len(images) // 2:
                ax.set_xlabel(
                    "Time (ms)" if k == 0 else "Completed volley count",
                    fontsize=theme.SIZE_LABEL,
                )
            ax.tick_params(labelsize=theme.SIZE_TICK)
    fig.suptitle(
        "Stable correct decisions · median",
        fontsize=theme.SIZE_TITLE,
        fontweight="bold",
    )
    save(fig, destination)


def plot_accuracy(report, destination):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(recipe.FIGURE_SIZE_MM[0] / 25.4, 70 / 25.4),
        layout="constrained",
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
            y = np.array(curve["accuracy"][ti], dtype=float) * 100
            y[coverage < 0.8] = np.nan
            axes[j].plot(x, y, color=colour, lw=1.3, label=f"GABA {tau:g} ms")
        axes[j].set_ylim(0, 100)
        axes[j].set_yticks([0, 25, 50, 75, 100])
        axes[j].set_title(("Physical time", "Gamma cycles", "Input budget")[j])
        axes[j].set_xlabel(xlabel)
        retained = np.flatnonzero(coverage >= 0.5)
        end = min(len(x) - 1, retained[-1] + 1) if len(retained) else len(x) - 1
        axes[j].set_xlim(x[0], x[end])
        if j == 0:
            axes[j].set_ylabel("Accuracy (%)")
        else:
            axes[j].tick_params(labelleft=False)
    theme.label_panels(axes.flat)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside upper center", ncols=3, fontsize=theme.SIZE_LEGEND
    )
    save(fig, destination)


def present(identity, overview_identity=None):
    source = source_run(
        recipe.REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    if source.record["execution"]["configuration"].get("schema") == "exp123.shared.measurement/v1":
        if overview_identity is not None:
            raise ValueError("shared-sweep presentations take one analysis input; omit --overview-source")
        from experiments.exp123.shared_plots import present_shared

        theme.set_paper_mode(True)
        theme.apply()
        plt.rcParams["svg.fonttype"] = "none"
        with stage_run(recipe.REPO, recipe.SLUG, "present", inputs={"analyse": source},
                       configuration=dict(schema="exp123.shared.presentation/v1", paper_mode=True,
                                          figures="six figures from one uniform sweep; evidence traces coloured by mean I-volley interval frequency, higher darker, shared scale across nine panels and both coordinates; pooled sustained decisions in 5 ms / 1 volley by 5 Hz rectangular bins with shared log counts and frequency-bin medians; coverage-limited volleys; 80% population-accuracy first-crossing cost/latency frontier in house colours with matched mean I-frequency and nonlinear GABA-decay colour scales",
                                          show_uncertainty_estimates=False)) as run:
            present_shared(source, run, save)
            result = run.run_id
        return result
    if overview_identity is None:
        raise ValueError("legacy analyses require --overview-source; shared-sweep analyses do not")
    report = json.loads(source.file("results.json").read_text())
    if (
        report.get("schema") != "exp123.analysis/v1"
        or len(report["image_reports"]) != 9
        or "decisions" not in report
    ):
        raise ValueError("unsupported analysis payload")
    theme.set_paper_mode(True)
    theme.apply()
    plt.rcParams["svg.fonttype"] = "none"
    overview = source_run(
        recipe.REPO / ".pingstore", overview_identity, stage="analyse", experiment=recipe.SLUG
    )
    overview_report = json.loads(overview.file("metrics.json").read_text())
    if overview_report.get("schema") != "exp123.overview.analysis/v1":
        raise ValueError("unsupported inhibitory-overview analysis")
    raster_ref = source.record["inputs"]["compute"]
    raster_compute = source_run(
        recipe.REPO / ".pingstore", raster_ref["run_id"], stage="compute",
        experiment=recipe.SLUG, reference=raster_ref,
    )
    inputs = {"analyse": source, "overview_analyse": overview, "raster_compute": raster_compute}
    accuracy_report = json.loads(source.file("accuracy.json").read_text())
    pareto_report = json.loads(source.file("pareto.json").read_text())
    if accuracy_report.get("schema") != "exp123.accuracy.analysis/v1":
        raise ValueError("unsupported accuracy analysis")
    if pareto_report.get("schema") != "exp123.pareto.analysis/v1":
        raise ValueError("unsupported pareto analysis")
    with stage_run(
        recipe.REPO,
        recipe.SLUG,
        "present",
        inputs=inputs,
        configuration=dict(
            schema="exp123.presentation/v1",
            paper_mode=True,
            figure_size_mm=recipe.FIGURE_SIZE_MM,
            style="shared paper theme; trajectory panel labels and decision row labels A/B",
            draw_aggregation="mean within each image and tau",
            show_encoding_traces=False,
            time_trajectory_layout="3 rows by 3 columns; first nine images, indices 0–8",
            cycle_trajectory_layout="3 rows by 3 columns; same nine images, indices 0–8",
            volley_sampling="cumulative softmax at the inhibitory trough after each accepted peak; final unbounded peak excluded",
            volley_zero="initial softmax score at physical time zero; illustrative origin, not a detected volley",
            decision_layout="nine images, indices 0–8; shared time x limits 0–200 ms with ticks every 10 ms and labels every 50 ms; shared completed volley range spanning displayed medians rounded out to integers with 0.2 padding; GABA decay on y; median points connected across decay settings",
            accuracy_minimum_display_coverage=0.8,
            show_uncertainty_estimates=False,
            overview_layout="four overview sweep metrics in a 2×2 grid on the left; three main-protocol E/I rasters stacked on the right",
            overview_raster_tau_ms=recipe.TAUS_MS,
            overview_raster_encoding_seed=42,
            accuracy_layout="one row of three accuracy panels A–C; coverage panels omitted",
            accuracy_figure_size_mm=[recipe.FIGURE_SIZE_MM[0], 70],
            tradeoff_layout="accuracy versus mean cumulative network spikes; three decay curves with dashed equal-elapsed-time connectors; 25 Hz encoding only",
            tradeoff_figure_size_mm=[recipe.FIGURE_SIZE_MM[0], 120],
            accuracy_axis_extent="last coordinate with at least 50% paired-trial coverage plus one grid step",
        ),
    ) as run:
        from experiments.exp123.overview_plot import plot_inhibitory_overview

        plot_inhibitory_overview(
            overview_report, raster_compute, run.export / "inhibitory_overview", save
        )
        overview_numbers = dict(overview_report)
        overview_numbers["design"] = dict(overview_report["design"])
        overview_numbers["raster_design"] = dict(
            image_index=0, encoding_seed=42, tau_gaba_ms=recipe.TAUS_MS,
            input_rate_hz=recipe.RATE_HZ,
        )
        overview_numbers["design"]["conditions"] = [
            c for c in overview_report["design"]["conditions"] if c["sweep"] == "tau"
        ]
        for key in ("rows", "summaries", "psd_summaries"):
            overview_numbers[key] = [r for r in overview_report[key] if r["sweep"] == "tau"]
        (run.export / "overview_numbers.json").write_text(
            json.dumps(overview_numbers, indent=2, allow_nan=False) + "\n"
        )
        plot_grid(report, source, run.export / "confidence_time", "time")
        plot_grid(report, source, run.export / "confidence_cycles", "cycles")
        plot_decisions(report, run.export / "sustained_decisions")
        plot_accuracy(accuracy_report, run.export / "accuracy_comparison")
        (run.export / "accuracy_numbers.json").write_text(
            json.dumps(accuracy_report, indent=2, allow_nan=False) + "\n"
        )
        from experiments.exp123.pareto_plots import plot_frontiers, plot_tradeoffs

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
    parser.add_argument("--overview-source", help="Required only for legacy analyses with a separate overview")
    args = parser.parse_args()
    present(args.source, args.overview_source)
