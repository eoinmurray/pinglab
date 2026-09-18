"""Render manuscript-owned synthesis figures from explicit validated sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.exp033 import evidence as exp033_evidence
from experiments.exp033 import recipe as exp033_recipe
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp037 import plots as exp037_plots
from experiments.exp041 import plots as exp041_plots
from experiments.exp046 import plots as exp046_plots
from experiments.exp054 import evidence as exp054_evidence
from experiments.exp054 import plots as exp054_plots
from experiments.exp054 import recipe as exp054_recipe
from experiments.exp110 import plots, recipe
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure
from pingstore.contracts import PingstoreError, load_json
from pingstore.stages import source_run, stage_run

CANONICAL_PRESENTATION_SOURCES = {
    "exp041": "exp041-r005-present",
    "exp046": "exp046-r010-present",
    "exp037": "exp037-r020-present",
    "exp044": "exp044-r009-present",
}


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


def _exp054_analysis(identity: str):
    analysis = source_run(
        REPO / ".pingstore", identity, stage="analyse", experiment="exp054"
    )
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("exp110 requires an independent exp054 analysis")
    reference = analysis.record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore", reference["run_id"], stage="compute",
        experiment="exp054", reference=reference,
    )
    cfg = exp054_recipe.validate(analysis.record["execution"]["configuration"])
    if compute.record["execution"]["configuration"] != cfg:
        raise PingstoreError("exp054 analysis and compute recipes differ")
    coordinates = exp054_evidence.read(analysis.export)
    if (
        coordinates.get("schema") != "exp054.analysis/v2"
        or coordinates.get("recipe") != cfg
    ):
        raise PingstoreError("unsupported exp054 analysis coordinates")
    return analysis, cfg, coordinates


def _exp033_analysis(identity: str):
    analysis = source_run(
        REPO / ".pingstore", identity, stage="analyse", experiment="exp033"
    )
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("exp110 requires an independent exp033 analysis")
    reference = analysis.record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore", reference["run_id"], stage="compute",
        experiment="exp033", reference=reference,
    )
    cfg = exp033_recipe.validate(analysis.record["execution"]["configuration"])
    if compute.record["execution"]["configuration"] != cfg:
        raise PingstoreError("exp033 analysis and compute recipes differ")
    coordinates = exp033_evidence.read(analysis.export)
    numbers = load_json(analysis.export / "results.json")
    frequency = numbers.get("results", {}).get("frequency_vs_tau_gaba", {})
    if numbers.get("slug") != "exp033" or set(frequency) != {"mean_field"}:
        raise PingstoreError("unsupported independent exp033 analysis")
    return analysis, cfg, coordinates, numbers


def _spiking_frequency_medians(
    document: dict, tau_grid: list[float]
) -> dict[float, float]:
    if document.get("schema") != "exp041.analysis/v1":
        raise PingstoreError("unsupported exp041 frequency analysis")
    rows = document.get("results", [])
    expected = {(tau, seed) for tau in tau_grid for seed in (42, 43, 44)}
    observed = [(row.get("tau_gaba_ms"), row.get("seed")) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise PingstoreError("exp110 requires all 18 exp041 frequency rows")
    if any(
        not np.isfinite(row.get("f_gamma_hz", np.nan))
        or row["f_gamma_hz"] <= 0
        for row in rows
    ):
        raise PingstoreError("invalid exp041 frequencies")
    return {
        tau: float(
            np.median(
                [row["f_gamma_hz"] for row in rows if row["tau_gaba_ms"] == tau]
            )
        )
        for tau in tau_grid
    }


def build_cycle_participation_compound(
    exp041_path: Path, exp046_path: Path, output_stem: Path
) -> None:
    """Redraw saved measurements as two equal-width panels above six distributions."""
    rates = load_json(exp041_path)
    cycles = load_json(exp046_path)
    if (rates.get("schema") != "exp041.analysis/v1"
            or cycles.get("schema") != "exp046.analysis/v2"
            or len(cycles.get("per_tau_equal_network", {})) != 6):
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
        for axis in top_axes:
            axis.set_xlabel("Spectral $f_\\mathrm{peak}$ (Hz)", fontsize=theme.SIZE_LABEL)
        rate_labels = sorted(
            (text for text in top_axes[0].texts if text.get_text().endswith(" ms")),
            key=lambda text: text.xy[0],
        )
        for index, text in enumerate(rate_labels):
            text.set_ha("right" if index % 2 else "left")
            text.set_position((-5, 3) if index % 2 else (5, -3))
        exp046_plots.plot_equal_network_distribution(
            cycles["per_tau_equal_network"], output_stem, axes=bottom_axes,
            panel_labels="CDEFGH",
        )
        for ax in bottom_axes:
            ax.tick_params(axis="y", labelleft=ax is bottom_axes[0])
        bottom_axes[0].set_ylabel("Equal-network\nmean fraction", fontsize=theme.SIZE_LABEL, color="black")
        fig.text(0.54, 0.045, "Spikes per neuron–cycle", ha="center", fontsize=theme.SIZE_LABEL, color="black")
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
    exp054_identity: str,
    exp033_identity: str,
    exp041_identity: str,
    exp046_identity: str,
    exp037_identity: str,
    exp044_identity: str,
    *,
    run_id: str | None = None,
) -> str:
    exp054_analysis, exp054_cfg, exp054_coordinates = _exp054_analysis(exp054_identity)
    exp033_analysis, exp033_cfg, exp033_coordinates, exp033_numbers = _exp033_analysis(
        exp033_identity
    )
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
    rate_numbers = load_json(rate_analysis.export / "results.json")
    mean_field = exp033_numbers["results"]
    tau_grid = [
        row["tau_gaba_ms"]
        for row in mean_field["frequency_vs_tau_gaba"]["mean_field"]
    ]
    measured_frequency = _spiking_frequency_medians(rate_numbers, tau_grid)
    with (
        stage_run(
            REPO,
            recipe.SLUG,
            "present",
            inputs={
                "exp054_analysis": exp054_analysis,
                "exp033_analysis": exp033_analysis,
                "exp041_presentation": exp041,
                "exp046_presentation": exp046,
                "exp041_analysis": rate_analysis,
                "exp046_analysis": cycle_analysis,
                "exp037_presentation": exp037,
                "exp044_presentation": exp044,
            },
            run_id=run_id,
            configuration=recipe.configuration(exp054_cfg, exp033_cfg),
        ) as run,
        exp054_plots.configured(exp054_cfg),
    ):
        plots.build_onset_super_compound(
            exp054_coordinates["grid"],
            exp033_coordinates["sweep"],
            mean_field["hopf"],
            mean_field["criticality"],
            mean_field["frequency_vs_tau_gaba"]["mean_field"],
            measured_frequency,
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
        "--theory-source", required=True, help="completed exp033 analysis run"
    )
    parser.add_argument("--run-id", help="fresh v4 identity reserved before dispatch")
    arguments = parser.parse_args()
    present(
        arguments.source,
        arguments.theory_source,
        CANONICAL_PRESENTATION_SOURCES["exp041"],
        CANONICAL_PRESENTATION_SOURCES["exp046"],
        CANONICAL_PRESENTATION_SOURCES["exp037"],
        CANONICAL_PRESENTATION_SOURCES["exp044"],
        run_id=arguments.run_id,
    )


if __name__ == "__main__":
    main()
