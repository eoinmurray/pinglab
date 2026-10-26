"""Run the independent present stage for exp110.

Invoke this CLI from the repository root. The experiment-local implementation
owns argument validation, explicit source pins and Pingstore stage completion.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
from contextlib import contextmanager

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from experiments import theme
from experiments.exp110 import recipe
from experiments.exp110 import recipe as exp054_recipe
from experiments.exp110 import recipe as exp117_recipe
from experiments.exp110 import recipe as source_exp054_custom_plots_recipe
from experiments.exp110.custom import source_exp054_custom_evidence as exp054_evidence
from experiments.exp110.recipe import MODELS, TAU_GABA_REFERENCE_MS
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from pingstore.contracts import PingstoreError, load_json
from pingstore.stages import source_run, stage_run

DEFAULT_FORMATS = ("svg", "pdf")


def save_figure(fig, stem, formats=DEFAULT_FORMATS) -> list[Path]:
    """Write *fig* to <stem>.<ext> for each ext in *formats*.

    *stem* is a path without an extension (a trailing .png/.pdf/.svg is
    stripped, so existing call sites that pass "foo.png" keep working).
    Format is inferred from the extension; rcParams (dpi, bbox, fonttype)
    come from the active theme profile.
    """
    stem = Path(stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    out: list[Path] = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    return out


def _label_panel(axis, label, *, y=1.04):
    return theme.label_panel(axis, label, x=-0.08, y=y)


def build_onset_super_compound(grid, results, hopf, sweep, mf, meas, out_path):
    """Combine the exp054 coupling map with its retained onset comparison."""
    previous_paper_mode = theme.PAPER_MODE
    theme.set_paper_mode(True)
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    fig = plt.figure(figsize=(180 / 25.4, 6.4), dpi=150)
    gs = GridSpec(
        3,
        3,
        figure=fig,
        height_ratios=[1.25, 0.92, 1.05],
        hspace=0.65,
        wspace=0.60,
        top=0.95,
        bottom=0.06,
        left=0.07,
        right=0.90,
    )

    raster_letters = ("D", "E", "F")
    for index, (values, title, vmax, fmt, marked) in enumerate(turnon_map_panels(grid)):
        axis = fig.add_subplot(gs[0, index])
        draw_turnon_map(
            axis,
            values,
            title=title,
            vmax_color=vmax,
            fmt=fmt,
            mark=marked,
            show_y=index == 0,
            mark_labels=raster_letters if marked else None,
            cell_fontsize=3.0,
        )
        for text in list(axis.texts):
            if text.get_text() not in raster_letters:
                text.remove()
        axis.set_title("", loc="center")
        axis.set_title(
            ("E rate (Hz)", "I rate (Hz; clipped)", "Lobe–trough contrast")[index],
            fontsize=theme.SIZE_LABEL,
        )
        axis.set_xlabel(r"$W_{EI}$ (µS)")
        if index == 0:
            axis.set_ylabel(r"$W_{IE}$ (µS)")
        color_axis = axis.inset_axes((1.04, 0, 0.045, 1))
        colorbar = fig.colorbar(axis.images[0], cax=color_axis)
        colorbar.ax.tick_params(labelsize=theme.SIZE_TICK)
        colorbar.locator = plt.MaxNLocator(3)
        colorbar.update_ticks()
        _label_panel(axis, "ABC"[index])

    for index, (label, wei_index, wie_index) in enumerate(TURNON_POINTS):
        axis = fig.add_subplot(gs[1, index])
        draw_turnon_raster(
            axis,
            grid[wie_index][wei_index],
            label=label,
            wei_i=wei_index,
            wie_i=wie_index,
            show_label=False,
        )
        axis.set_title(
            ("Loop off", "Intermediate coupling", "Strong coupling")[index],
            fontsize=theme.SIZE_LABEL,
        )
        cell = grid[wie_index][wei_index]
        n_e = cell["e"].shape[1]
        n_i = 0 if cell["i"] is None else cell["i"].shape[1]
        axis.set_yticks((n_e / 2, n_e + n_i / 2), ("E", "I"))
        _label_panel(axis, raster_letters[index])

    eigen_axis = fig.add_subplot(gs[2, 0])
    drives = np.array([row["I_ext"] for row in results])
    eigen_real = np.array([[value[0] for value in row["eigs"]] for row in results])
    eigen_imag = np.array([[value[1] for value in row["eigs"]] for row in results])
    scatter = None
    for index in range(eigen_real.shape[1]):
        scatter = eigen_axis.scatter(
            eigen_real[:, index],
            eigen_imag[:, index],
            c=drives,
            cmap="magma",
            s=4,
            linewidths=0,
        )
    eigen_axis.axvline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    if hopf:
        omega = hopf["omega_star"]
        eigen_axis.scatter(
            [0, 0],
            [omega, -omega],
            facecolors="none",
            edgecolors=theme.ELECTRIC_CYAN,
            s=60,
            lw=1.4,
            zorder=5,
        )
    assert scatter is not None
    color_axis = eigen_axis.inset_axes((0.27, 0.82, 0.43, 0.04))
    colorbar = fig.colorbar(
        scatter, cax=color_axis, orientation="horizontal", ticks=(0, 2, 4)
    )
    colorbar.set_label("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_TICK, labelpad=2)
    colorbar.ax.tick_params(labelsize=theme.SIZE_TICK)
    colorbar.ax.xaxis.set_label_position("top")
    eigen_axis.set_xlabel("Re$(\\lambda)$ (ms$^{-1}$)", fontsize=theme.SIZE_LABEL)
    eigen_axis.set_ylabel("Im$(\\lambda)$ (ms$^{-1}$)", fontsize=theme.SIZE_LABEL)
    eigen_axis.set_title(
        "Hopf crossing",
        loc="left",
        fontsize=theme.SIZE_LABEL,
        fontweight="semibold",
    )
    _label_panel(eigen_axis, "G")
    _despine(eigen_axis)

    amplitude_axis = fig.add_subplot(gs[2, 1])
    amplitude_axis.plot(
        [row["I_ext"] for row in sweep["up"]],
        [row["amp"] for row in sweep["up"]],
        "o-",
        color=theme.INK_BLACK,
        lw=1.2,
        ms=4,
        label="drive ↑",
    )
    amplitude_axis.plot(
        [row["I_ext"] for row in sweep["down"]],
        [row["amp"] for row in sweep["down"]],
        "s--",
        color=theme.DEEP_RED,
        lw=1.0,
        ms=4,
        markerfacecolor="none",
        label="drive ↓",
    )
    amplitude_axis.axvline(hopf["I_ext_star"], color=theme.AMBER, lw=0.6, ls=":")
    amplitude_axis.set_xlabel("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_LABEL)
    amplitude_axis.set_ylabel("E amplitude (ms$^{-1}$)", fontsize=theme.SIZE_LABEL)
    amplitude_axis.set_title(
        "Mean-field amplitude",
        loc="left",
        fontsize=theme.SIZE_LABEL,
        fontweight="semibold",
    )
    amplitude_axis.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    _label_panel(amplitude_axis, "H")
    _despine(amplitude_axis)

    frequency_axis = fig.add_subplot(gs[2, 2])
    decay = [row["tau_gaba_ms"] for row in mf if row["f_star_Hz"] is not None]
    frequency = [row["f_star_Hz"] for row in mf if row["f_star_Hz"] is not None]
    frequency_axis.plot(
        decay,
        frequency,
        "o-",
        color=theme.INK_BLACK,
        lw=1.4,
        label="mean-field",
    )
    if meas:
        measured_decay = sorted(meas)
        frequency_axis.plot(
            measured_decay,
            [meas[value] for value in measured_decay],
            "s--",
            color=theme.DEEP_RED,
            lw=1.3,
            label="spiking median",
        )
    frequency_axis.set_xlabel("$\\tau_\\text{GABA}$ (ms)", fontsize=theme.SIZE_LABEL)
    frequency_axis.set_ylabel("frequency (Hz)", fontsize=theme.SIZE_LABEL)
    frequency_axis.set_title(
        "Frequency comparison",
        loc="left",
        fontsize=theme.SIZE_LABEL,
        fontweight="semibold",
    )
    frequency_axis.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    _label_panel(frequency_axis, "I")
    _despine(frequency_axis)

    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)
    theme.set_paper_mode(previous_paper_mode)
    theme.apply()


CANONICAL_PRESENTATION_SOURCES = {
    "exp041": "exp041-r005-present",
    "exp046": "exp046-r010-present",
    "exp037": "exp037-r020-present",
    "exp044": "exp044-r009-present",
}

SHARED_TAU_GRID_MS = [4.5, 6.0, 9.0, 12.0, 18.0, 27.0]


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
        REPO / ".pingstore",
        reference["run_id"],
        stage="analyse",
        experiment=experiment,
        reference=reference,
    )


def _exp054_analysis(identity: str):
    analysis = source_run(
        REPO / ".pingstore", identity, stage="analyse", experiment="exp054"
    )
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("exp110 requires an independent exp054 analysis")
    reference = analysis.record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore",
        reference["run_id"],
        stage="compute",
        experiment="exp054",
        reference=reference,
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


def _exp117_analysis(identity: str):
    analysis = source_run(
        REPO / ".pingstore", identity, stage="analyse", experiment="exp117"
    )
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("exp110 requires an independent exp117 analysis")
    reference = analysis.record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore",
        reference["run_id"],
        stage="compute",
        experiment="exp117",
        reference=reference,
    )
    cfg = exp117_recipe.source_exp117_recipe_validate(
        analysis.record["execution"]["configuration"]
    )
    compute_cfg = exp117_recipe.source_exp117_recipe_validate(
        compute.record["execution"]["configuration"]
    )
    if cfg.get("schema") != "exp117.recipe/v3" or compute_cfg != cfg:
        raise PingstoreError("exp117 analysis and compute recipes differ")
    numbers = load_json(analysis.export / "results.json")
    coordinates = load_json(analysis.export / "plot_coordinates.json")
    if (
        numbers.get("schema") != "exp117.analysis/v3"
        or numbers.get("configuration") != cfg
        or coordinates.get("schema") != "exp117.plot-coordinates/v4"
    ):
        raise PingstoreError("unsupported independent exp117 analysis")
    return analysis, cfg, coordinates, numbers


def _adapt_exp117_figure2(
    coordinates: dict,
) -> tuple[list[dict], dict, dict, list[dict]]:
    """Map exp117 analysis fields to exp110's established Figure 2 contract."""
    rows = coordinates.get("rows")
    hopf_source = coordinates.get("hopf")
    criticality = coordinates.get("criticality")
    frequency_rows = coordinates.get("frequency_vs_tau_GABA")
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(hopf_source, dict)
        or not isinstance(criticality, dict)
        or not isinstance(frequency_rows, list)
    ):
        raise PingstoreError("incomplete exp117 Figure 2 coordinates")

    results = []
    for row in rows:
        drive = row.get("I_ext_nA")
        eigenvalues = row.get("eigenvalues_per_ms")
        if (
            not np.isfinite(drive)
            or not isinstance(eigenvalues, list)
            or len(eigenvalues) != 4
            or any(
                not isinstance(value, list)
                or len(value) != 2
                or not np.isfinite(value).all()
                for value in eigenvalues
            )
        ):
            raise PingstoreError("invalid exp117 eigenvalue coordinates")
        results.append({"I_ext": float(drive), "eigs": eigenvalues})

    hopf = {
        "I_ext_star": hopf_source.get("I_ext_star_nA"),
        "omega_star": hopf_source.get("omega_Hopf_rad_per_ms"),
    }
    if not all(np.isfinite(value) for value in hopf.values()):
        raise PingstoreError("invalid exp117 Hopf coordinates")

    relative_drive = criticality.get("relative_drive_nA")
    amplitude_up = criticality.get("amplitude_up_Hz")
    amplitude_down = criticality.get("amplitude_down_Hz")
    if (
        not all(
            isinstance(values, list)
            for values in (relative_drive, amplitude_up, amplitude_down)
        )
        or len(relative_drive) == 0
        or len({len(relative_drive), len(amplitude_up), len(amplitude_down)}) != 1
        or not all(
            np.isfinite(value)
            for values in (relative_drive, amplitude_up, amplitude_down)
            for value in values
        )
    ):
        raise PingstoreError("invalid exp117 criticality coordinates")
    absolute_drive = [hopf["I_ext_star"] + value for value in relative_drive]
    sweep = {
        "up": [
            {"I_ext": drive, "amp": amplitude / 1000}
            for drive, amplitude in zip(absolute_drive, amplitude_up, strict=True)
        ],
        "down": [
            {"I_ext": drive, "amp": amplitude / 1000}
            for drive, amplitude in zip(absolute_drive, amplitude_down, strict=True)
        ],
    }

    frequency_by_tau = {}
    for row in frequency_rows:
        tau = row.get("tau_GABA_ms")
        frequency = row.get("f_Hopf_Hz")
        if (
            not np.isfinite(tau)
            or not np.isfinite(frequency)
            or tau in frequency_by_tau
        ):
            raise PingstoreError("invalid exp117 frequency coordinates")
        frequency_by_tau[float(tau)] = float(frequency)
    if set(frequency_by_tau) != set(SHARED_TAU_GRID_MS):
        raise PingstoreError("exp110 requires the exact shared inhibitory-decay grid")
    mean_field = [
        {"tau_gaba_ms": tau, "f_star_Hz": frequency_by_tau[tau]}
        for tau in SHARED_TAU_GRID_MS
    ]
    return results, hopf, sweep, mean_field


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
        not np.isfinite(row.get("f_gamma_hz", np.nan)) or row["f_gamma_hz"] <= 0
        for row in rows
    ):
        raise PingstoreError("invalid exp041 frequencies")
    return {
        tau: float(
            np.median([row["f_gamma_hz"] for row in rows if row["tau_gaba_ms"] == tau])
        )
        for tau in tau_grid
    }


def build_cycle_participation_compound(
    exp041_path: Path, exp046_path: Path, output_stem: Path
) -> None:
    """Redraw saved measurements as two equal-width panels above six distributions."""
    rates = load_json(exp041_path)
    cycles = load_json(exp046_path)
    if (
        rates.get("schema") != "exp041.analysis/v1"
        or cycles.get("schema") != "exp046.analysis/v2"
        or len(cycles.get("per_tau_equal_network", {})) != 6
    ):
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
        plot_quantitative_law(
            rates["aggregate"], rates["fit"], output_stem, axes=top_axes
        )
        for axis in top_axes:
            axis.set_xlabel(
                "Spectral $f_\\mathrm{peak}$ (Hz)", fontsize=theme.SIZE_LABEL
            )
        rate_labels = sorted(
            (text for text in top_axes[0].texts if text.get_text().endswith(" ms")),
            key=lambda text: text.xy[0],
        )
        for index, text in enumerate(rate_labels):
            text.set_ha("right" if index % 2 else "left")
            text.set_position((-5, 3) if index % 2 else (5, -3))
        plot_equal_network_distribution(
            cycles["per_tau_equal_network"],
            output_stem,
            axes=bottom_axes,
            panel_labels="CDEFGH",
        )
        for ax in bottom_axes:
            ax.tick_params(axis="y", labelleft=ax is bottom_axes[0])
        bottom_axes[0].set_ylabel(
            "Equal-network\nmean fraction", fontsize=theme.SIZE_LABEL, color="black"
        )
        fig.text(
            0.54,
            0.045,
            "Spikes per neuron–cycle",
            ha="center",
            fontsize=theme.SIZE_LABEL,
            color="black",
        )
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.94)
        save_figure(fig, output_stem, formats=("svg", "pdf"))
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
        perturbation_document.get("schema")
        not in ("exp037.analysis/v1", "exp037.analysis/v2")
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
    plot_perturbation_curves(perturbation, output_stem, "", axes=axes[:2])
    axes[0].set_title("Spike deletion", loc="left")
    axes[1].set_title("Spike insertion", loc="left")
    axes[0].set_xlabel("Deletion probability (%)")
    for ax in axes[:2]:
        for line in ax.lines:
            line.set_markersize(3)
    legend = axes[1].get_legend()
    fig.legend(
        legend.legend_handles,
        [text.get_text() for text in legend.get_texts()],
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=3,
        frameon=False,
    )
    legend.remove()
    axes[1].tick_params(axis="y", labelleft=False)
    # Wrap the source label to fit one panel of the three-panel compound.
    axes[1].set_xlabel(
        "Added rate / baseline E rate (%)"
        if perturbation.get("relative_test_baseline")
        else "Added rate / reference E rate (%)"
    )
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
    save_figure(fig, output_stem, formats=("svg", "pdf"))
    plt.close(fig)
    theme.set_paper_mode(previous_paper_mode)
    theme.apply()


def present(
    exp054_identity: str,
    exp117_identity: str,
    exp041_identity: str,
    exp046_identity: str,
    exp037_identity: str,
    exp044_identity: str,
    *,
    run_id: str | None = None,
) -> str:
    exp054_analysis, exp054_cfg, exp054_coordinates = _exp054_analysis(exp054_identity)
    exp117_analysis, exp117_cfg, exp117_coordinates, _ = _exp117_analysis(
        exp117_identity
    )
    exp041, _ = _source_figure(exp041_identity, "exp041", recipe.RATE_FREQUENCY_SOURCE)
    exp046, _ = _source_figure(exp046_identity, "exp046", recipe.CYCLE_COUNT_SOURCE)
    exp037, exp037_figure = _source_figure(
        exp037_identity, "exp037", recipe.PERTURBATION_SOURCE
    )
    exp044, exp044_figure = _source_figure(
        exp044_identity, "exp044", recipe.TIMESTEP_SOURCE
    )
    rate_analysis = _presentation_analysis(exp041, "exp041")
    cycle_analysis = _presentation_analysis(exp046, "exp046")
    rate_numbers = load_json(rate_analysis.export / "results.json")
    theory_results, theory_hopf, theory_sweep, mean_field = _adapt_exp117_figure2(
        exp117_coordinates
    )
    measured_frequency = _spiking_frequency_medians(rate_numbers, SHARED_TAU_GRID_MS)
    with (
        stage_run(
            REPO,
            recipe.SLUG,
            "present",
            inputs={
                "exp054_analysis": exp054_analysis,
                "exp117_analysis": exp117_analysis,
                "exp041_presentation": exp041,
                "exp046_presentation": exp046,
                "exp041_analysis": rate_analysis,
                "exp046_analysis": cycle_analysis,
                "exp037_presentation": exp037,
                "exp044_presentation": exp044,
            },
            run_id=run_id,
            configuration=recipe.configuration(exp054_cfg, exp117_cfg),
        ) as run,
        configured(exp054_cfg),
    ):
        build_onset_super_compound(
            exp054_coordinates["grid"],
            theory_results,
            theory_hopf,
            theory_sweep,
            mean_field,
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


def plot_quantitative_law(
    rows: list[dict], fit: dict, out_path: Path, *, axes=None
) -> None:
    """Seed-mean E rate and accuracy versus gamma frequency, with the saved affine fit."""
    theme.apply()

    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(6.5, 4.2),
            sharex=True,
            gridspec_kw={"hspace": 0.12, "height_ratios": [1.6, 1.0]},
        )
    ax_rate, ax_acc = axes
    fig = ax_rate.figure

    f_gammas: list[float] = []
    for row in rows:
        tau = row["tau_gaba_ms"]
        fg_mu, fg_se = row["f_gamma_hz"]["mean"], row["f_gamma_hz"]["sem"]
        er_mu, er_se = row["e_rate_hz"]["mean"], row["e_rate_hz"]["sem"]
        ac_mu, ac_se = row["acc"]["mean"], row["acc"]["sem"]
        f_gammas.append(fg_mu)
        ax_rate.errorbar(
            fg_mu,
            er_mu,
            xerr=fg_se,
            yerr=er_se,
            fmt="o",
            markersize=6 if standalone else 3,
            color=theme.INK_BLACK,
            capsize=3 if standalone else 2,
            label=f"τ_GABA = {tau:g} ms"
            if standalone and tau == TAU_GABA_REFERENCE_MS
            else None,
        )
        ax_rate.annotate(
            f" {tau:g} ms",
            (fg_mu, er_mu),
            fontsize=theme.SIZE_ANNOTATION,
            color=theme.MUTED,
            xytext=(9 if standalone else 4, -3),
            textcoords="offset points",
            va="center",
        )
        ax_acc.errorbar(
            fg_mu,
            ac_mu,
            xerr=fg_se,
            yerr=ac_se,
            fmt="o",
            markersize=6 if standalone else 3,
            color=theme.INK_BLACK,
            capsize=3 if standalone else 2,
        )

    fg_arr = np.array(f_gammas)
    p_fit, a_fit, r2 = fit["p_affine"], fit["a_affine"], fit["r2_affine"]
    if p_fit is not None:
        xs = np.linspace(0, fg_arr.max() * 1.1, 200)
        ax_rate.plot(
            xs,
            p_fit * xs + a_fit,
            color=theme.DEEP_RED,
            lw=1.2,
            ls="--",
            label=(
                "$r_E = a + p · f_γ$"
                + ("  " if standalone else "\n")
                + f"(a = {a_fit:.2f} Hz, p = {p_fit:.3f},"
                + (" " if standalone else "\n")
                + f"R² = {r2:.3f})"
            ),
        )
    ax_rate.set_ylabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    if standalone:
        ax_acc.set_xlabel(
            "Measured $f_γ$ (Hz) — peak of trained-network population PSD",
            fontsize=theme.SIZE_LABEL,
        )
    else:
        for ax in axes:
            ax.set_xlabel("Measured $f_γ$ (Hz)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylim(0, 100)
    ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.6)
    for ax in (ax_rate, ax_acc):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=theme.SIZE_TICK)
    if not standalone:
        ax_rate.margins(x=0.15, y=0.35)
        ax_acc.set_xlim(ax_rate.get_xlim())
    if not standalone:
        ax_rate.set_ylim(bottom=0)
        for line in ax_rate.lines:
            if line.get_label().startswith("$r_E"):
                line.set_label("Affine fit")
    ax_rate.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    theme.label_panels((ax_rate, ax_acc))
    if standalone:
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        plt.close(fig)


def plot_equal_network_distribution(
    per_tau: dict, out_path: Path, *, axes=None, panel_labels=None
) -> None:
    """Plot equal-network means in one monochrome row."""
    theme.apply()
    taus_sorted = sorted(float(k.removeprefix("tau_")) for k in per_tau)
    standalone = axes is None
    if standalone:
        fig, grid = plt.subplots(
            1,
            len(taus_sorted),
            figsize=(6.9, 2.15),
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        axes = grid.ravel()
    axes = list(axes)
    labels_for_panels = (
        list(panel_labels)
        if panel_labels is not None
        else [chr(ord("A") + i) for i in range(len(taus_sorted))]
    )
    if len(axes) != len(taus_sorted) or len(labels_for_panels) != len(taus_sorted):
        raise ValueError("one axis and panel label are required per decay condition")
    fig = axes[0].figure
    labels = ["0", "1", "2", "≥3"]
    keys = ("frac_zero", "frac_one", "frac_two", "frac_three_plus")
    for i, tau in enumerate(taus_sorted):
        ax = axes[i]
        means = [per_tau[f"tau_{tau:g}"][key] for key in keys]
        ax.bar(labels, means, color="black", edgecolor="black", lw=0.5)
        ax.set_title(
            "$\\tau_{\\mathrm{GABA}}$\n" + f"{tau:g} ms",
            fontsize=theme.SIZE_TITLE,
            color="black",
            loc="center",
            pad=8,
        )
        ax.text(
            0.03,
            0.97,
            labels_for_panels[i],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=theme.SIZE_PANEL,
            fontweight="bold",
            color="black",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color("black")
        ax.tick_params(colors="black", labelsize=theme.SIZE_LABEL)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        ax.grid(False)
    if standalone:
        fig.supxlabel(
            "Spikes per neuron–cycle", fontsize=theme.SIZE_LABEL, color="black"
        )
        fig.supylabel(
            "Equal-network\nmean fraction", fontsize=theme.SIZE_LABEL, color="black"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        plt.close(fig)


MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}

MODEL_MARKERS = {"coba": "s", "ping": "D"}


def plot_perturbation_curves(
    data: dict,
    out_path: Path,
    run_id: str,
    *,
    axes=None,
) -> None:
    """Plot saved across-seed means and sample-SD envelopes."""
    theme.apply()
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3), sharey=True)
    fig = axes[0].figure
    use_pct = data["use_pct"]

    # Left panel: drop (as % of spikes dropped)
    ax_drop = axes[0]
    for model in MODELS:
        row = data["panels"]["drop"][model]
        xs, means, lo, hi = (row[k] for k in ("x", "mean", "lo", "hi"))
        ax_drop.plot(
            xs,
            means,
            marker=MODEL_MARKERS[model],
            markersize=5,
            linewidth=1.4,
            color=MODEL_COLORS[model],
            label=model.upper(),
            zorder=3 if model == "coba" else 2,
        )
        ax_drop.fill_between(
            xs,
            lo,
            hi,
            color=MODEL_COLORS[model],
            alpha=0.15,
            linewidth=0,
        )
    ax_drop.set_xlabel("Spike-deletion probability (%)", fontsize=theme.SIZE_LABEL)
    ax_drop.set_title(
        "Drop — Bernoulli deletion", fontsize=theme.SIZE_LABEL, loc="left", pad=4
    )
    ax_drop.set_xlim(-2, 102)
    ax_drop.axhline(10.0, ls="--", color=theme.MUTED, lw=0.7, alpha=0.6)
    ax_drop.text(
        0.02,
        12,
        "chance",
        transform=ax_drop.get_yaxis_transform(),
        fontsize=theme.SIZE_ANNOTATION,
        color=theme.MUTED,
        va="bottom",
    )

    # Right panel: add
    ax_add = axes[1]
    if use_pct:
        for model in MODELS:
            row = data["panels"]["add"][model]
            xs, means, lo, hi = (row[k] for k in ("x", "mean", "lo", "hi"))
            ax_add.plot(
                xs,
                means,
                marker=MODEL_MARKERS[model],
                markersize=5,
                linewidth=1.4,
                color=MODEL_COLORS[model],
                label=model.upper(),
                zorder=3 if model == "coba" else 2,
            )
            ax_add.fill_between(
                xs,
                lo,
                hi,
                color=MODEL_COLORS[model],
                alpha=0.15,
                linewidth=0,
            )
        ax_add.set_xlabel(
            "Nominal added rate / baseline E rate (%)"
            if data.get("relative_test_baseline")
            else "Nominal added rate / reference E rate (%)",
            fontsize=theme.SIZE_LABEL,
        )
        ax_add.set_title(
            "Add — Bernoulli insertion",
            fontsize=theme.SIZE_LABEL,
            loc="left",
            pad=4,
        )
        maximum = max(max(row["x"]) for row in data["panels"]["add"].values())
        margin = max(1.0, 0.03 * maximum)
        ax_add.set_xlim(-margin, maximum + margin)
    else:
        for model in MODELS:
            row = data["panels"]["add"][model]
            xs, means, lo, hi = (row[k] for k in ("x", "mean", "lo", "hi"))
            ax_add.plot(
                xs,
                means,
                marker=MODEL_MARKERS[model],
                markersize=5,
                linewidth=1.4,
                color=MODEL_COLORS[model],
                label=model.upper(),
                zorder=3 if model == "coba" else 2,
            )
            ax_add.fill_between(
                xs,
                lo,
                hi,
                color=MODEL_COLORS[model],
                alpha=0.15,
                linewidth=0,
            )
        ax_add.set_xlabel("Nominal added rate (Hz / neuron)", fontsize=theme.SIZE_LABEL)
        ax_add.set_title(
            "Add — Bernoulli insertion",
            fontsize=theme.SIZE_LABEL,
            loc="left",
            pad=4,
        )
    ax_add.axhline(10.0, ls="--", color=theme.MUTED, lw=0.7, alpha=0.6)

    for ax in axes:
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=theme.SIZE_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    ax_drop.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    legend_handles, legend_labels = ax_add.get_legend_handles_labels()
    legend_handles.append(
        Patch(
            facecolor=theme.INK_BLACK,
            edgecolor="none",
            alpha=0.15,
            label="±1 SD band",
        )
    )
    legend_labels.append("±1 SD band")
    ax_add.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        fontsize=theme.SIZE_LEGEND,
        frameon=False,
    )
    # The writeup caption carries the takeaway rather than a figure title.
    if standalone:
        theme.label_panels(axes)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)  # line/curve plot: SVG + PDF
        plt.close(fig)


_DEFAULT = source_exp054_custom_plots_recipe.source_exp054_recipe_configuration()

DT_MS = _DEFAULT["dt_ms"]

WEI_MEAN_GRID = _DEFAULT["wei_mean"]

WIE_MEAN_GRID = _DEFAULT["wie_mean"]

GRID_WIN_MS = _DEFAULT["display_window_ms"]

TURNON_POINTS = source_exp054_custom_plots_recipe.turnon_points(_DEFAULT)


@contextmanager
def configured(cfg):
    names = {
        "DT_MS": cfg["dt_ms"],
        "BIN_MS": cfg["bin_ms"],
        "WEI_MEAN_GRID": cfg["wei_mean"],
        "WIE_MEAN_GRID": cfg["wie_mean"],
        "GRID_DISPLAY_STRIDE": cfg["display_stride"],
        "GRID_WIN_MS": cfg["display_window_ms"],
        "TURNON_POINTS": source_exp054_custom_plots_recipe.turnon_points(cfg),
    }
    old = {key: globals().get(key) for key in names}
    paper = theme.PAPER_MODE
    try:
        globals().update(names)
        with plt.rc_context():
            theme.apply()
            plt.rcParams["savefig.bbox"] = "standard"
            yield
    finally:
        for key, value in old.items():
            if value is None:
                globals().pop(key, None)
            else:
                globals()[key] = value
        theme.set_paper_mode(paper)


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def turnon_map_panels(grid):
    """The three scalar-map specs (E rate, I rate edge-clipped, contrast) for the
    turn-on grid: (values 2-D array, title, colour vmax, cell-label fmt, mark A/B/C).
    Single source of truth so exp054 Fig 1 and the onset super-compound agree."""
    e = np.array([[c["rate_hz"] for c in row] for row in grid])
    i = np.array([[c["rate_i_hz"] for c in row] for row in grid])
    ct = np.array([[c["contrast"] for c in row] for row in grid])
    i_fin = i[np.isfinite(i)]
    i_cap = float(np.nanpercentile(i_fin, 92)) if i_fin.size else None
    e_max = float(np.nanmax(e)) if np.isfinite(e).any() else 1.0
    return [
        (e, "E firing rate (Hz)", e_max, "{:.0f}", False),
        (i, "I firing rate (Hz; edge clipped)", i_cap, "{:.0f}", False),
        (ct, "lobe–trough contrast", 1.0, "{:.2f}", True),
    ]


def draw_turnon_map(
    ax,
    vals,
    *,
    title,
    vmax_color,
    fmt,
    mark=False,
    show_y=True,
    mark_labels=None,
    cell_fontsize=4.5,
):
    """Draw one W_EI × W_IE scalar heatmap into ax (per-cell value labels + optional
    sample-point markers). Shared by exp054 Figure 1 and the onset super-compound —
    restyle here and both figures update. mark_labels overrides the circle letters
    (the super-compound relabels them D/E/F to match its raster panel letters).
    cell_fontsize scales the per-cell value labels down for the small super-compound
    panels, where the default (tuned for Figure 1's wider maps) would overlap."""
    ax.imshow(
        vals, origin="lower", aspect="equal", cmap="Greys", vmin=0.0, vmax=vmax_color
    )
    xt = range(0, len(WEI_MEAN_GRID), 2)
    ax.set_xticks(list(xt))
    ax.set_xticklabels(
        [f"{WEI_MEAN_GRID[t]:.1f}" for t in xt], fontsize=theme.SIZE_TICK - 1
    )
    if show_y:
        yt = range(0, len(WIE_MEAN_GRID), 2)
        ax.set_yticks(list(yt))
        ax.set_yticklabels(
            [f"{WIE_MEAN_GRID[t]:.1f}" for t in yt], fontsize=theme.SIZE_TICK - 1
        )
        ax.set_ylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
    else:
        ax.set_yticks([])
    ax.set_xlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
    ax.set_title(
        title, loc="center", fontsize=theme.SIZE_LABEL - 1, fontweight="semibold"
    )
    for iy in range(vals.shape[0]):
        for ix in range(vals.shape[1]):
            v = vals[iy, ix]
            if np.isfinite(v):
                frac = (v / vmax_color) if vmax_color else 0.0
                ax.text(
                    ix,
                    iy,
                    fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    color="white" if frac > 0.55 else theme.INK_BLACK,
                )
    if mark:
        labels = (
            mark_labels if mark_labels is not None else [p[0] for p in TURNON_POINTS]
        )
        for (_, wei_i, wie_i), disp in zip(TURNON_POINTS, labels, strict=True):
            ax.text(
                wei_i,
                wie_i,
                disp,
                ha="center",
                va="center",
                zorder=6,
                fontsize=theme.SIZE_TICK,
                fontweight="bold",
                color=theme.INK_BLACK,
                clip_on=False,
                bbox={
                    "boxstyle": "circle,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": theme.INK_BLACK,
                    "linewidth": 1.2,
                },
            )


def draw_turnon_raster(ax, cell, *, label, wei_i, wie_i, show_label=True):
    """Draw one representative sample-point raster into ax (E black below, I red
    above). Shared by exp054 Figure 1 and the onset super-compound. show_label=False
    drops the leading letter from the title (the super-compound identifies each
    raster by its prominent panel letter instead)."""
    e_ = cell["e"]
    n_e = e_.shape[1]
    T = e_.shape[0]
    t_ms = np.arange(T) * DT_MS
    e_idx, e_t = np.where(e_.T)
    ax.scatter(t_ms[e_t], e_idx, s=0.5, c=theme.INK_BLACK, marker="|", linewidths=0.35)
    total = n_e
    if cell["i"] is not None:
        ii = cell["i"]
        n_i = ii.shape[1]
        i_idx, i_t = np.where(ii.T)
        ax.scatter(
            t_ms[i_t], n_e + i_idx, s=0.5, c=theme.DEEP_RED, marker="|", linewidths=0.35
        )
        ax.axhline(n_e, color=theme.GREY_MID, lw=0.5, alpha=0.6)
        total = n_e + n_i
    ax.set_ylim(0, total)
    ax.set_xlim(0, GRID_WIN_MS)
    ax.set_yticks([])
    prefix = f"{label}   " if show_label else ""
    ax.set_title(
        f"{prefix}W_EI={WEI_MEAN_GRID[wei_i]:.1f}, W_IE={WIE_MEAN_GRID[wie_i]:.1f}"
        f"   contrast={cell['contrast']:.2f}",
        loc="left",
        fontsize=theme.SIZE_TICK - 1,
    )
    ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    _despine(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp054 analysis run")
    parser.add_argument(
        "--theory-source", required=True, help="completed exp117 analysis run"
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
