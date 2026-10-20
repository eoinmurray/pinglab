"""Render paired membrane-sweep dynamics for the article.

Run present.py --source ANALYSIS_RUN. Read validated analysis metrics and raster
events, then save PNG and SVG figures plus numerical summaries in a new
presentation run. The compute ancestor supplies the single circuit diagram.

Presentation does not simulate or measure activity. Parameter tables belong
in the report and read original native snapshots through its pinned ancestry."""

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
from experiments.exp134 import recipe
from pingstore.stages import source_run, stage_run


def plot_overview(report, events, sweep, destination):
    rows = sorted(
        (r for r in report["summaries"] if r["sweep"] in (sweep, "baseline")),
        key=lambda r: r["multiplier"],
    )
    values = [r["multiplier"] for r in rows]
    fig = plt.figure(figsize=(180 / 25.4, 105 / 25.4), layout="constrained")
    groups = fig.add_gridspec(1, 2)
    metric_grid = groups[0].subgridspec(2, 2)
    raster_grid = groups[1].subgridspec(3, 1)
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.06)
    axes = [fig.add_subplot(metric_grid[i, j]) for i in range(2) for j in range(2)]
    xlabel = "Leak multiplier" if sweep == "leak" else "Capacitance multiplier"
    for ax, (key, title, ylabel) in zip(
        axes,
        [
            ("psd_frequency_hz", "PSD frequency", "Spectral peak (Hz)"),
            ("participation_e_pct", "E participation", "Participation (%)"),
            ("e_spikes_per_second", "E spike rate", "E spikes/s"),
            ("rhythmicity_contrast", "Rhythmicity", "Lobe–trough contrast"),
        ],
        strict=True,
    ):
        ys = [r[key] if r[key] is not None else np.nan for r in rows]
        ax.plot(values, ys, "o-", color=theme.INK_BLACK, linewidth=1.3, markersize=3)
        ax.axvline(1.0, color=theme.FAINT, ls="--", lw=0.7)
        ax.set(xscale="log", xlabel=xlabel, ylabel=ylabel, title=title)
        ax.set_xticks([0.5, 1, 2], labels=["0.5", "1", "2"])
        ax.minorticks_off()
        if key == "participation_e_pct":
            ax.set_ylim(
                0, max(30, 1.08 * max((v for v in ys if np.isfinite(v)), default=0))
            )
        if key == "rhythmicity_contrast":
            ax.set_ylim(0, 1.05)
        if key == "e_spikes_per_second":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    theme.label_panels(axes)
    axes = [fig.add_subplot(raster_grid[i, 0]) for i in range(3)]
    anchors = [
        next(r for r in rows if r["multiplier"] == value) for value in (0.5, 1.0, 2.0)
    ]
    for ax, row in zip(axes, anchors, strict=True):
        name = row["condition_id"]
        for key, colour, offset in (
            ("spk_e", theme.INK_BLACK, 0),
            ("spk_i", theme.DEEP_RED, 1024),
        ):
            ax.scatter(
                events[f"{name}--{key}--step"] * report["design"]["dt_ms"],
                events[f"{name}--{key}--neuron"] + offset,
                s=0.22,
                color=colour,
                linewidths=0,
                rasterized=True,
            )
        ax.axhline(1023.5, color=theme.FAINT, lw=0.5)
        symbol = r"$g_L$" if sweep == "leak" else r"$C_m$"
        ax.set(
            xlim=(0, 200),
            ylim=(-1, 1280),
            xlabel="Time (ms)",
            ylabel="Neuron",
            title=f"{symbol} × {row['multiplier']:g}",
        )
        ax.set_yticks([0, 1024, 1280])
        ax.set_xticks([0, 100, 200])
    theme.label_panels(axes, labels=["E", "F", "G"])
    with plt.rc_context({"savefig.bbox": None}):
        for extension in ("png", "svg"):
            fig.savefig(destination.with_suffix("." + extension), dpi=300)
    plt.close(fig)


def present(identity):
    source = source_run(
        REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    report = json.loads(source.file("metrics.json").read_text())
    if report.get("schema") != "exp134.analysis/v1":
        raise ValueError("unsupported membrane-sweep analysis")
    pin = source.record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore",
        pin["run_id"],
        stage="compute",
        experiment=recipe.SLUG,
        reference=pin,
    )
    theme.set_paper_mode(True)
    theme.apply()
    plt.rcParams["svg.fonttype"] = "none"
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analyse": source, "network_diagrams": compute},
        configuration=dict(
            schema="exp134.presentation/v1",
            figure_size_mm=[180, 105],
            show_uncertainty_estimates=False,
        ),
    ) as run:
        with np.load(source.file("rasters.npz"), allow_pickle=False) as data:
            events = {key: data[key].copy() for key in data.files}
        for sweep in ("leak", "capacitance"):
            plot_overview(report, events, sweep, run.export / f"{sweep}_overview")
        diagram = compute.export / "network.svg"
        if not diagram.is_file():
            # Earlier compute runs retained the shared topology for each setting.
            mapping = compute.record["scientific_artifacts"]["native_paths"]
            diagram = compute.export / mapping["baseline.bundle/reports/circuit.svg"]
        shutil.copyfile(diagram, run.export / "network.svg")
        (run.export / "numbers.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        result = run.run_id
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
