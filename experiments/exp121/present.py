"""Compose retained decision and rhythm measurements using snnviz."""

# ruff: noqa: E402
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp121 import recipe as R
from experiments.helpers import theme
from pingstore.stages import source_run, stage_run
from snnlab import viz as snnviz  # noqa: TID251


def render(source, export, study="input", figure_number=1):
    prefix = "" if study == "input" else f"{study}--"
    settings = {
        "input": ("INPUT RATE", "Maximum input rate (Hz)", "Hz", 1.0),
        "gaba": ("GABA DECAY", "GABA decay time (ms)", "ms", 6.0),
        "capacitance": ("E/I CAPACITANCE", "E/I capacitance multiplier", "×", 1.0),
        "leak": ("E/I LEAK CONDUCTANCE", "E/I leak multiplier", "×", 1.0),
        "ampa": ("AMPA DECAY", "AMPA decay time (ms)", "ms", 2.0),
        "inhibition": ("I→E INHIBITORY STRENGTH", "I→E strength multiplier", "×", 1.0),
        "threshold": (
            "EXCITATORY SPIKE THRESHOLD",
            "E spike threshold (mV)",
            "mV",
            1.0,
        ),
    }
    heading, xlabel, unit, factor = settings[study]
    with np.load(source.file(prefix + "measurements.npz")) as saved:
        stable = saved["stable_correct_ms"]
        bins = saved["histogram_edges_ms"]
        histograms = saved["histogram_counts"]
        rates = (
            saved["control_values"]
            if "control_values" in saved
            else saved["input_rates_hz"]
        )
    with np.load(source.file(prefix + "rhythm-measurements.npz")) as saved:
        if not np.array_equal(
            rates,
            saved["control_values"]
            if "control_values" in saved
            else saved["input_rates_hz"],
        ):
            raise ValueError("Condition mismatch")
        frequency, cv = saved["frequency_quartiles_25"], saved["cv_quartiles_25"]
    rates = rates * factor
    grid = snnviz.FigureGrid(
        rows=2,
        columns=(1, 1, 1.15),
        bounds=(0.075, 0.13, 0.90, 0.73),
        row_gap=0.17,
        column_gap=(0.065, 0.11),
    )
    for j in range(4):
        grid.place(f"histogram-{j}", row=j // 2, column=j % 2)
    grid.place("frequency", row=0, column=2)
    grid.place("cv", row=1, column=2)
    fig = grid.figure(figsize=(7.08, 4.8), dpi=240)
    theme.set_paper_mode()
    theme.apply()
    top = grid.add_axes(fig, "frequency")
    bottom = grid.add_axes(fig, "cv")
    palette = snnviz.Theme()
    colors = [palette.ink, palette.mid_grey, palette.cyan, palette.accent]
    order = np.argsort(rates)
    limit = max(5, np.ceil(histograms.max() / 5) * 5) * 1.16
    for j, k in enumerate(order):
        ax = grid.add_axes(fig, f"histogram-{j}")
        ax.bar(
            bins[:-1],
            histograms[k],
            width=np.diff(bins),
            align="edge",
            color=colors[k],
            edgecolor="white",
            linewidth=0.35,
        )
        ax.set(xlim=(0, 400), ylim=(0, limit), xticks=(0, 200, 400))
        ax.set_title(f"{'ABCD'[j]}  {rates[k]:g} {unit}", loc="left", fontsize=7)
        ax.text(
            0.97,
            0.95,
            f"{np.isfinite(stable[k]).sum()}/100 correct",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6,
        )
        if j % 2 == 0:
            ax.set_ylabel("Trials / 20 ms bin", fontsize=7)
        else:
            ax.tick_params(labelleft=False)
        if j >= 2 or len(rates) == 3 and j == 1:
            ax.set_xlabel("Stable-correct time (ms)", fontsize=6.5)
        else:
            ax.tick_params(labelbottom=False)
    for ax, values, title, ylabel in [
        (
            top,
            frequency,
            f"{chr(65 + len(rates))}  NETWORK FREQUENCY",
            "Frequency (Hz)",
        ),
        (bottom, cv, f"{chr(66 + len(rates))}  BURST VARIABILITY", "Interval CV"),
    ]:
        q = values[:, order]
        ax.plot(rates[order], q[1], color=palette.ink)
        for j, k in enumerate(order):
            ax.errorbar(
                rates[k],
                q[1, j],
                yerr=[[q[1, j] - q[0, j]], [q[2, j] - q[1, j]]],
                color=colors[k],
                fmt="o",
                markersize=4,
                capsize=3,
            )
        ax.set(
            xlabel=xlabel.replace("capacitance multiplier", "capacitance\nmultiplier"),
            ylabel=ylabel,
            xticks=rates[order],
            xlim=(
                rates.min() - 0.08 * np.ptp(rates),
                rates.max() + 0.08 * np.ptp(rates),
            ),
        )
        ax.set_title(title, loc="left", fontsize=6.5)
    fig.text(
        0.09, 0.96, f"{heading} · 100 PAIRED IMAGES · 400 ms", weight="bold", fontsize=8
    )
    for ext in ("png", "svg", "pdf"):
        fig.savefig(
            export / f"figure-{figure_number}.{ext}", dpi=300, facecolor="white"
        )
    plt.close(fig)


def present(identity):
    source = source_run(
        ROOT / ".pingstore", identity, stage="analyse", experiment="exp121"
    )
    with stage_run(
        ROOT, "exp121", "present", inputs={"analysis": source}, configuration=R.CONFIG
    ) as run:
        studies = source.record["execution"]["configuration"].get("studies", ["input"])
        for number, study in enumerate(studies, start=1):
            render(source, run.export, study, number)
            prefix = "" if study == "input" else f"{study}--"
            for name in ("decisions.json", "rhythm-results.json"):
                shutil.copyfile(
                    source.file(prefix + name), run.export / (prefix + name)
                )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
