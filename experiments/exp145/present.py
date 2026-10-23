"""Present full-window and 200 ms rasters and four AI diagnostics independently.

Run present.py --source ANALYSE_ID. The analysis run identifies its validated
compute input, whose retained spikes supply the raster after burn-in removal.
Four diagnostic panels summarize the same measurement window.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp145.custom.plotting import diagnostic_figure
from experiments.exp145.custom.recording import read_recording
from pingstore.stages import source_run, stage_run


def style_circuit(source, target):
    """Restyle the executed bundle's SVG without changing its circuit or layout."""
    svg = source.read_text().replace(
        'font-family="Helvetica Neue"',
        'font-family="JetBrains Mono, IBM Plex Mono, Menlo, Consolas, DejaVu Sans Mono, monospace"',
    )
    # Native boxes were sized for a narrower font; keep monospace labels inside them.
    for original, size in (
        ("11.00", theme.SIZE_LABEL),
        ("14.00", theme.SIZE_PANEL),
        ("18.00", theme.SIZE_TITLE),
    ):
        svg = svg.replace(f'font-size="{original}"', f'font-size="{size}"')
    for original, color in {
        "#24292f": theme.INK_BLACK,
        "#aa5b63": theme.DEEP_RED,
        "#536d82": theme.ELECTRIC_CYAN,
        "#59636e": theme.MUTED,
        "#756b5b": theme.GREY_MID,
        "#f3f6f8": theme.PAPER,
        "#f5f4f1": theme.PAPER,
    }.items():
        svg = svg.replace(original, color)
    target.write_text(svg)


def present(args):
    theme.apply()
    plt.rcParams["svg.fonttype"] = "none"
    source = source_run(
        REPO / ".pingstore", args.source, experiment="exp145", stage="analyse"
    )
    # Resolve the recorded ancestry explicitly; no latest-run selection.
    inputs = source.record["inputs"]
    compute_id = inputs["compute"]["run_id"]
    compute = source_run(
        REPO / ".pingstore", compute_id, experiment="exp145", stage="compute"
    )
    cfg = source.record["execution"]["configuration"]
    summary = json.loads(source.file("results.json").read_text())
    start = round(cfg["burn_in_ms"] / cfg["dt_ms"])
    window_ms = cfg["duration_ms"] - cfg["burn_in_ms"]
    with (
        np.load(source.export / "measurements.npz") as data,
        stage_run(
            REPO,
            "exp145",
            "present",
            inputs={"analyse": source, "compute": compute},
            configuration=cfg,
            run_id=args.run_id,
        ) as run,
    ):
        recording = read_recording(compute.unit("recording"), run.scratch / "reader")
        style_circuit(compute.file("circuit.svg"), run.export / "circuit.svg")
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
        for population, color, offset in (
            ("E", theme.INK_BLACK, 0),
            ("I", theme.DEEP_RED, cfg["n_e"]),
        ):
            times, cells = np.nonzero(recording.signals[f"{population}.spikes"][start:])
            times_ms = times * cfg["dt_ms"]
            for axis, selected, marker_size in (
                (axes[0], slice(None), 1),
                (axes[1], times_ms < min(200.0, window_ms), 3),
            ):
                axis.scatter(
                    times_ms[selected],
                    cells[selected] + offset,
                    s=marker_size,
                    c=color,
                    linewidths=0,
                    rasterized=True,
                )
        axes[0].set(
            xlim=(0, window_ms),
            xlabel="Time after burn-in (ms)",
            ylabel="Neuron index",
            title="Shared-input conductance network",
        )
        axes[1].set(
            xlim=(0, min(200.0, window_ms)),
            xlabel="Time after burn-in (ms)",
            ylabel="Neuron index",
            title="First 200 ms after burn-in",
        )
        for axis in axes:
            axis.set_ylim(0, cfg["n_e"] + cfg["n_i"])
        theme.label_panels(axes)
        fig.savefig(run.export / "onset.png")
        fig.savefig(run.export / "onset.svg")
        plt.close(fig)
        diagnostics = diagnostic_figure(data, summary, cfg)
        diagnostics.savefig(run.export / "ai-diagnostics.png")
        diagnostics.savefig(run.export / "ai-diagnostics.svg")
        plt.close(diagnostics)
        shutil.copyfile(source.file("results.json"), run.export / "numbers.json")
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    print(present(parser.parse_args()))


if __name__ == "__main__":
    main()
