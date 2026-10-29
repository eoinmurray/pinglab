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
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp134 import recipe
from pingstore.library import source_run, stage_run
from snnlab import lang, viz


def circuit_diagram(bundle):
    """Lay out native weighted projections; omit recording and scoring aliases."""
    import matplotlib

    native = lang.diagram(bundle, view="circuit")
    known = {node.id for node in native.nodes if node.kind in ("input", "population")}
    symbols = dict(drive="X", image="X", E="E", I="I", readout="R")
    positions = dict(
        drive=(-360, 0), image=(-360, 0), E=(0, 0), I=(0, 300), readout=(360, 0)
    )
    routes = dict(
        E_to_I=viz.EdgeLayout(source_port="bottom_left", target_port="top_left"),
        I_to_E=viz.EdgeLayout(source_port="top_right", target_port="bottom_right"),
    )
    title = "PING classifier" if "I" in known else "COBA classifier"
    diagram = replace(
        native,
        title=title,
        groups=(),
        nodes=tuple(
            replace(node, detail=node.detail.splitlines()[0])
            for node in native.nodes
            if node.id in known
        ),
        edges=tuple(edge for edge in native.edges if edge.id is not None),
    )
    ports = dict(
        upper_left=(0, 0.3),
        lower_left=(0, 0.7),
        upper_right=(1, 0.3),
        lower_right=(1, 0.7),
        top_left=(0.25, 0),
        top_right=(0.75, 0),
        bottom_left=(0.25, 1),
        bottom_right=(0.75, 1),
    )
    layout = viz.DiagramLayout(
        nodes={
            identifier: viz.NodeLayout(
                position=position,
                label=f"${symbols[identifier]}$",
                min_size=(240, 160),
                ports=ports,
            )
            for identifier, position in positions.items()
            if identifier in known
        },
        edges={
            edge.id: replace(
                routes.get(edge.id, viz.EdgeLayout()),
                label=viz.DiagramLabel(
                    rf"$W_{{{symbols[edge.target]},{symbols[edge.source]}}}$"
                ),
            )
            for edge in diagram.edges
        },
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=16), gap=30
        ),
    )
    with matplotlib.rc_context({"savefig.bbox": None}):
        return viz.resolve_diagram(
            diagram, layout=layout, theme=viz.DiagramTheme(secondary_size=13)
        )


def save_circuit(bundle, path):
    """Export the resolved native architecture without inheriting plot cropping."""
    import matplotlib

    circuit = circuit_diagram(bundle)
    with matplotlib.rc_context({"savefig.bbox": None}):
        for suffix in ("svg", "png"):
            viz.render_diagram(circuit, Path(path).with_suffix("." + suffix))


def redraw_networks(presentation, *, run_id=None):
    """Preserve an explicit presentation's evidence and replace only its diagrams."""
    previous = source_run(
        REPO / ".pingstore", presentation, experiment="exp134", stage="present"
    )
    sources = {
        role: source_run(REPO / ".pingstore", pin["run_id"], reference=pin)
        for role, pin in previous.record["inputs"].items()
    }
    specs = [(sources["network_diagrams"], "baseline.bundle", "network")]
    with stage_run(
        REPO,
        previous.record["experiment"],
        "present",
        inputs={**sources, "figure_source": previous},
        configuration=previous.record["execution"]["configuration"],
        run_id=run_id,
    ) as run:
        shutil.copytree(previous.export, run.export, dirs_exist_ok=True)
        for source, prefix, stem in specs:
            bundle_path = (
                source.restore_bundle(prefix, run.scratch / (stem + ".bundle"))
                if source.record.get("scientific_artifacts", {}).get("native_paths")
                else source.unit(*prefix.split("/"))
            )
            save_circuit(lang.load_bundle(bundle_path), run.export / (stem + ".svg"))
        run.record["execution"]["snnlab_version"] = version("snnlab")
        previous.check_unchanged()
    return run.run_id


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
        bundle = lang.load_bundle(
            compute.restore_bundle("baseline.bundle", run.scratch / "network.bundle")
        )
        save_circuit(bundle, run.export / "network.svg")
        (run.export / "numbers.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        result = run.run_id
    print(result, flush=True)
    return result


if __name__ == "__main__":
    if any(arg.startswith("--redraw-from") for arg in sys.argv[1:]):
        parser = argparse.ArgumentParser(
            description="Refresh only the circuit diagrams of an explicit presentation."
        )
        parser.add_argument("--redraw-from", dest="presentation", required=True)
        parser.add_argument("--run-id")
        print(redraw_networks(**vars(parser.parse_args())), flush=True)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--source", required=True)
        present(parser.parse_args().source)
