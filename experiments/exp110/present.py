"""Render manuscript-owned synthesis figures from explicit validated sources."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

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


def _panel_font(height: int, fraction: float = 0.035) -> ImageFont.FreeTypeFont:
    import matplotlib

    path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    return ImageFont.truetype(str(path), max(12, round(height * fraction)))


def _relabel_four_panel(image: Image.Image, labels: str) -> Image.Image:
    """Replace the source figure's A-D labels while preserving its plot pixels."""
    image = image.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = _panel_font(height)
    positions = (
        (0.026, 0.016),
        (0.508, 0.016),
        (0.026, 0.535),
        (0.508, 0.535),
    )
    for label, (x_fraction, y_fraction) in zip(labels, positions):
        x, y = round(width * x_fraction), round(height * y_fraction)
        draw.rectangle(
            (x - 8, y - 7, x + round(width * 0.035), y + round(height * 0.052)),
            fill="white",
        )
        draw.text((x, y), label, font=font, fill="black")
    return image


def build_performance_transfer_compound(
    exp025_path: Path, exp038_path: Path, output_stem: Path
) -> None:
    """Stack the two retained four-panel figures as manuscript panels A-H."""
    with (
        Image.open(exp025_path) as first_source,
        Image.open(exp038_path) as second_source,
    ):
        first = _relabel_four_panel(first_source.copy(), "ABCD")
        second = _relabel_four_panel(second_source.copy(), "EFGH")
    if first.width != second.width:
        target_width = max(first.width, second.width)
        first = first.resize(
            (target_width, round(first.height * target_width / first.width)),
            Image.Resampling.LANCZOS,
        )
        second = second.resize(
            (target_width, round(second.height * target_width / second.width)),
            Image.Resampling.LANCZOS,
        )
    gap = round(first.width * 0.012)
    composite = Image.new(
        "RGB", (first.width, first.height + gap + second.height), "white"
    )
    composite.paste(first, (0, 0))
    composite.paste(second, (0, first.height + gap))
    composite.save(output_stem.with_suffix(".png"), dpi=(300, 300))
    composite.save(output_stem.with_suffix(".pdf"), "PDF", resolution=300)


def _rasterize_svg(source: Path, destination: Path, *, width: int = 2070) -> None:
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        raise PingstoreError("rsvg-convert is required for manuscript SVG composition")
    subprocess.run(
        [renderer, "-w", str(width), "-o", str(destination), str(source)],
        check=True,
        capture_output=True,
        text=True,
    )


def _relabel_cycle_panels(image: Image.Image) -> Image.Image:
    """Replace the six source labels A-F with manuscript labels C-H."""
    image = image.convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = _panel_font(height, 0.056)
    draw.rectangle((0, 0, width, round(height * 0.09)), fill="white")
    for label, condition, x_fraction in zip(
        "CDEFGH",
        ("4.5 ms", "6 ms", "9 ms", "12 ms", "18 ms", "27 ms"),
        (0.039, 0.199, 0.359, 0.519, 0.679, 0.839),
    ):
        x = round(width * x_fraction)
        draw.text((x, 2), f"{label}  {condition}", font=font, fill="black")
    return image


def build_cycle_participation_compound(
    exp041_path: Path, exp046_path: Path, output_stem: Path, scratch: Path
) -> None:
    """Combine rate-frequency and cycle-count evidence as panels A-H."""
    rate_png = scratch / "rate-vs-frequency.png"
    cycles_png = scratch / "spikes-per-cycle.png"
    _rasterize_svg(exp041_path, rate_png)
    _rasterize_svg(exp046_path, cycles_png)
    with Image.open(rate_png) as rate_source, Image.open(cycles_png) as cycle_source:
        rate = rate_source.convert("RGB")
        cycles = _relabel_cycle_panels(cycle_source.copy())
    gap = round(rate.width * 0.012)
    composite = Image.new(
        "RGB", (rate.width, rate.height + gap + cycles.height), "white"
    )
    composite.paste(rate, (0, 0))
    composite.paste(cycles, (0, rate.height + gap))
    composite.save(output_stem.with_suffix(".png"), dpi=(300, 300))
    composite.save(output_stem.with_suffix(".pdf"), "PDF", resolution=300)


def build_robustness_compound(
    exp037_path: Path, exp044_path: Path, output_stem: Path
) -> None:
    """Redraw retained perturbation and timestep summaries as equal-width panels."""
    perturbation_document = load_json(exp037_path)
    timestep_document = load_json(exp044_path)
    perturbation = perturbation_document.get("plot_data")
    timestep = timestep_document.get("aggregate")
    if (
        perturbation_document.get("schema") != "exp037.analysis/v1"
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
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
    model_styles = {
        "coba": (theme.DEEP_RED, "s"),
        "ping": (theme.INK_BLACK, "D"),
    }
    for axis, mode, title in zip(
        axes[:2],
        ("drop", "add"),
        ("Spike deletion", "Spike addition"),
        strict=True,
    ):
        for model, (color, marker) in model_styles.items():
            row = perturbation["panels"][mode][model]
            axis.plot(
                row["x"],
                row["mean"],
                marker=marker,
                markersize=4,
                linewidth=1.2,
                color=color,
                label=model.upper(),
            )
            axis.fill_between(
                row["x"], row["lo"], row["hi"], color=color, alpha=0.15, linewidth=0
            )
        axis.axhline(10.0, ls="--", color=theme.MUTED, lw=0.7, alpha=0.6)
        axis.set_ylim(0, 100)
        axis.yaxis.set_major_locator(mticker.MultipleLocator(20))
        axis.grid(True, axis="y", alpha=0.15, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(title, loc="left", fontsize=theme.SIZE_LABEL)
    axes[0].set(xlim=(-2, 102), xlabel="spike-deletion probability (%)")
    axes[0].set_ylabel("test accuracy (%)")
    maximum = max(max(row["x"]) for row in perturbation["panels"]["add"].values())
    axes[1].set(
        xlim=(-0.03 * maximum, 1.03 * maximum),
        xlabel="added rate / reference E rate (%)",
    )
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].legend(frameon=False, loc="upper right")

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
    rate_axis.set(xlabel="integration timestep (ms)", ylim=(0, 50))
    rate_axis.set_ylabel("hidden E rate (Hz)")
    rate_axis.set_title("Integration timestep", loc="left", fontsize=theme.SIZE_LABEL)
    rate_axis.spines["top"].set_visible(False)
    accuracy_axis = rate_axis.twinx()
    accuracy_axis.errorbar(
        dts,
        [row["acc"]["mean"] for row in timestep],
        yerr=[row["acc"]["sem"] for row in timestep],
        marker="s",
        markersize=4,
        linewidth=1.2,
        capsize=2,
        color=theme.DEEP_RED,
    )
    accuracy_axis.set_ylim(0, 100)
    accuracy_axis.set_ylabel("test accuracy (%)", color=theme.DEEP_RED)
    accuracy_axis.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    accuracy_axis.spines["top"].set_visible(False)
    theme.label_panels(axes)
    fig.subplots_adjust(left=0.065, right=0.94, bottom=0.2, top=0.88, wspace=0.38)
    save_figure(fig, output_stem, formats=("png", "pdf"))
    plt.close(fig)
    theme.set_paper_mode(previous_paper_mode)
    theme.apply()


def present(
    identity: str,
    exp025_identity: str,
    exp038_identity: str,
    exp041_identity: str,
    exp046_identity: str,
    exp037_identity: str,
    exp044_identity: str,
    *,
    run_id: str | None = None,
) -> str:
    analysis, source_recipe, coordinates, _ = analysis_source(REPO, identity)
    exp025, exp025_figure = _source_figure(
        exp025_identity, "exp025", recipe.PERFORMANCE_SOURCE
    )
    exp038, exp038_figure = _source_figure(
        exp038_identity, "exp038", recipe.TRANSFER_SOURCE
    )
    exp041, exp041_figure = _source_figure(
        exp041_identity, "exp041", recipe.RATE_FREQUENCY_SOURCE
    )
    exp046, exp046_figure = _source_figure(
        exp046_identity, "exp046", recipe.CYCLE_COUNT_SOURCE
    )
    exp037, exp037_figure = _source_figure(
        exp037_identity, "exp037", recipe.PERTURBATION_SOURCE
    )
    exp044, exp044_figure = _source_figure(
        exp044_identity, "exp044", recipe.TIMESTEP_SOURCE
    )
    with (
        stage_run(
            REPO,
            recipe.SLUG,
            "present",
            inputs={
                "exp054_analysis": analysis,
                "exp025_presentation": exp025,
                "exp038_presentation": exp038,
                "exp041_presentation": exp041,
                "exp046_presentation": exp046,
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
        build_performance_transfer_compound(
            exp025_figure,
            exp038_figure,
            run.export / "performance_transfer_compound",
        )
        build_cycle_participation_compound(
            exp041_figure,
            exp046_figure,
            run.export / "cycle_participation_compound",
            run.scratch,
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
        "--exp025-source", required=True, help="completed exp025 presentation run"
    )
    parser.add_argument(
        "--exp038-source", required=True, help="completed exp038 presentation run"
    )
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
        arguments.exp025_source,
        arguments.exp038_source,
        arguments.exp041_source,
        arguments.exp046_source,
        arguments.exp037_source,
        arguments.exp044_source,
        run_id=arguments.run_id,
    )


if __name__ == "__main__":
    main()
