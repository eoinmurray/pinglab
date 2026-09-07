"""Render manuscript-owned synthesis figures from explicit validated sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp037 import plots as exp037_plots
from experiments.exp041 import plots as exp041_plots
from experiments.exp046 import plots as exp046_plots
from experiments.exp054 import plots as exp054_plots
from experiments.exp054.present import analysis_source
from experiments.exp110 import plots, recipe
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure
from pingstore.contracts import PingstoreError, load_json
from pingstore.stages import source_run, stage_run


def _source_figure(identity: str, experiment: str, name: str):
    source = source_run(
        REPO / ".pingstore", identity, stage="present", experiment=experiment
    )
    path = source.export / name
    if not path.is_file():
        raise PingstoreError(f"{experiment} presentation lacks {name}")
    return source, path


def _presentation_analysis(presentation, experiment: str):
    reference = presentation.record["inputs"]["analysis"]
    return source_run(
        REPO / ".pingstore", reference["run_id"], stage="analyse",
        experiment=experiment, reference=reference,
    )


def build_cycle_participation_compound(
    exp041_path: Path, exp046_path: Path, output_stem: Path
) -> None:
    """Redraw saved measurements as two equal-width panels above six distributions."""
    rates = load_json(exp041_path)
    cycles = load_json(exp046_path)
    if (rates.get("schema") != "exp041.analysis/v1"
            or cycles.get("schema") != "exp046.analysis/v1"
            or len(cycles.get("per_tau", {})) != 6):
        raise PingstoreError("unsupported cycle-participation analysis summaries")
    previous_paper_mode = theme.PAPER_MODE
    theme.set_paper_mode(True)
    theme.apply()
    fig = plt.figure(figsize=(180 / 25.4, 3.8))
    try:
        rows = fig.add_gridspec(2, 1, height_ratios=(1.5, 1), hspace=0.68)
        top = rows[0].subgridspec(1, 2, wspace=0.36)
        bottom = rows[1].subgridspec(1, 6, wspace=0.22)
        top_axes = [fig.add_subplot(top[0, i]) for i in range(2)]
        bottom_axes = [fig.add_subplot(bottom[0, 0])]
        bottom_axes.extend(
            fig.add_subplot(bottom[0, i], sharey=bottom_axes[0]) for i in range(1, 6)
        )
        exp041_plots.plot_quantitative_law(
            rates["aggregate"], rates["fit"], output_stem, axes=top_axes
        )
        exp046_plots.plot_distribution(
            cycles["per_tau"], output_stem, axes=bottom_axes,
            percentages=False, panel_labels="CDEFGH",
        )
        for ax, tau in zip(bottom_axes, sorted(float(k.removeprefix("tau_")) for k in cycles["per_tau"])):
            for bar in ax.patches:
                bar.set_facecolor(theme.INK_BLACK)
                bar.set_edgecolor(theme.INK_BLACK)
            ax.set_title(f"{tau:g} ms", fontsize=theme.SIZE_LABEL)
            ax.tick_params(axis="y", labelleft=ax is bottom_axes[0])
        bottom_axes[0].set_ylabel("Neuron–cycle fraction", fontsize=theme.SIZE_LABEL)
        fig.text(0.54, 0.045, "Spikes per neuron per cycle", ha="center", fontsize=theme.SIZE_LABEL)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.94)
        save_figure(fig, output_stem, formats=("png", "pdf"))
    finally:
        plt.close(fig)
        theme.set_paper_mode(previous_paper_mode)
        theme.apply()


def build_robustness_compound(
    exp037_path: Path, exp044_path: Path, output_stem: Path
) -> None:
    """Redraw retained perturbation and timestep summaries as equal-width panels."""
    perturbation_document = load_json(exp037_path)
    timestep_document = load_json(exp044_path)
    perturbation = perturbation_document.get("plot_data")
    timestep = timestep_document.get("aggregate")
    if (
        perturbation_document.get("schema") not in ("exp037.analysis/v1", "exp037.analysis/v2")
        or not isinstance(perturbation, dict)
        or perturbation.get("use_pct") is not True
        or timestep_document.get("schema") != "exp044.analysis/v1"
        or not isinstance(timestep, list)
        or not timestep
    ):
        raise PingstoreError("unsupported robustness presentation summaries")

    previous_paper_mode = theme.PAPER_MODE
    theme.set_paper_mode(True)
    theme.apply()
    fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 3.4))
    exp037_plots.plot_perturbation_curves(
        perturbation, output_stem, "", axes=axes[:2]
    )
    axes[0].set_title("Spike deletion", loc="left")
    axes[1].set_title("Spike insertion", loc="left")
    axes[0].set_xlabel("Deletion probability (%)")
    for ax in axes[:2]:
        for line in ax.lines:
            line.set_markersize(3)
    legend = axes[1].get_legend()
    fig.legend(legend.legend_handles, [text.get_text() for text in legend.get_texts()],
               loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, frameon=False)
    legend.remove()
    axes[1].tick_params(axis="y", labelleft=False)
    # Wrap the source label to fit one panel of the three-panel compound.
    axes[1].set_xlabel("Added rate / baseline E rate (%)" if perturbation.get("relative_test_baseline") else "Added rate / reference E rate (%)")
    axes[1].set_xlabel(axes[1].get_xlabel().replace(" / ", " /\n"))

    rate_axis = axes[2]
    dts = [row["dt_ms"] for row in timestep]
    rate_axis.errorbar(
        dts,
        [row["e_rate_hz"]["mean"] for row in timestep],
        yerr=[row["e_rate_hz"]["sem"] for row in timestep],
        marker="D",
        markersize=4,
        linewidth=1.2,
        capsize=2,
        color=theme.INK_BLACK,
    )
    rate_axis.set_xscale("log")
    rate_axis.set_xticks(dts)
    rate_axis.set_xticklabels([f"{value:g}" for value in dts])
    rate_axis.set(xlabel="timestep (ms)", ylim=(0, 50))
    rate_axis.set_ylabel("hidden E rate (Hz)")
    rate_axis.set_title("Timestep", loc="left", fontsize=theme.SIZE_LABEL)
    rate_axis.spines["top"].set_visible(False)
    accuracy_axis = rate_axis.twinx()
    accuracy_axis.errorbar(
        dts,
        [row["acc"]["mean"] for row in timestep],
        yerr=[row["acc"]["sem"] for row in timestep],
        marker="s",
        markersize=3,
        linewidth=1.2,
        capsize=2,
        color=theme.GREY_MID,
    )
    accuracy_axis.set_ylim(0, 100)
    accuracy_axis.set_ylabel("test accuracy (%)", color=theme.GREY_MID)
    accuracy_axis.tick_params(axis="y", labelcolor=theme.GREY_MID)
    accuracy_axis.spines["top"].set_visible(False)
    theme.label_panels(axes)
    fig.subplots_adjust(left=0.075, right=0.92, bottom=0.29, top=0.87, wspace=0.60)
    save_figure(fig, output_stem, formats=("png", "pdf"))
    plt.close(fig)
    theme.set_paper_mode(previous_paper_mode)
    theme.apply()


def present(
    identity: str,
    exp041_identity: str,
    exp046_identity: str,
    exp037_identity: str,
    exp044_identity: str,
    *,
    run_id: str | None = None,
) -> str:
    analysis, source_recipe, coordinates, _ = analysis_source(REPO, identity)
    exp041, _ = _source_figure(
        exp041_identity, "exp041", recipe.RATE_FREQUENCY_SOURCE
    )
    exp046, _ = _source_figure(
        exp046_identity, "exp046", recipe.CYCLE_COUNT_SOURCE
    )
    exp037, exp037_figure = _source_figure(
        exp037_identity, "exp037", recipe.PERTURBATION_SOURCE
    )
    exp044, exp044_figure = _source_figure(
        exp044_identity, "exp044", recipe.TIMESTEP_SOURCE
    )
    rate_analysis = _presentation_analysis(exp041, "exp041")
    cycle_analysis = _presentation_analysis(exp046, "exp046")
    with (
        stage_run(
            REPO,
            recipe.SLUG,
            "present",
            inputs={
                "exp054_analysis": analysis,
                "exp041_presentation": exp041,
                "exp046_presentation": exp046,
                "exp041_analysis": rate_analysis,
                "exp046_analysis": cycle_analysis,
                "exp037_presentation": exp037,
                "exp044_presentation": exp044,
            },
            run_id=run_id,
            configuration=recipe.configuration(source_recipe),
        ) as run,
        exp054_plots.configured(source_recipe),
    ):
        mean_field = coordinates["mean_field"]
        plots.build_onset_super_compound(
            coordinates["grid"],
            mean_field["sweep"],
            mean_field["hopf"],
            mean_field["criticality"],
            mean_field["frequency_vs_tau_gaba"],
            {float(key): value for key, value in mean_field["spiking_exp041"].items()},
            run.export / "onset_super_compound",
        )
        build_cycle_participation_compound(
            rate_analysis.export / "results.json",
            cycle_analysis.export / "results.json",
            run.export / "cycle_participation_compound",
        )
        build_robustness_compound(
            exp037_figure,
            exp044_figure,
            run.export / "robustness_compound",
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp054 analysis run")
    parser.add_argument(
        "--exp041-source", required=True, help="completed exp041 presentation run"
    )
    parser.add_argument(
        "--exp046-source", required=True, help="completed exp046 presentation run"
    )
    parser.add_argument(
        "--exp037-source", required=True, help="completed exp037 presentation run"
    )
    parser.add_argument(
        "--exp044-source", required=True, help="completed exp044 presentation run"
    )
    parser.add_argument("--run-id", help="fresh v4 identity reserved before dispatch")
    arguments = parser.parse_args()
    present(
        arguments.source,
        arguments.exp041_source,
        arguments.exp046_source,
        arguments.exp037_source,
        arguments.exp044_source,
        run_id=arguments.run_id,
    )


if __name__ == "__main__":
    main()
