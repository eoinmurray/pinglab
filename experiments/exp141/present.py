"""Render learning curves from a named, completed analysis run.

Run `uv run python experiments/exp141/present.py --source <analysis-run-id>`.
The analysis provides all plotted values: for each mode, an SVG/PDF pair places
cross-entropy and accuracy side by side, with online training and fixed-weight
validation curves, plus a selected-checkpoint test point.
Epoch zero represents the untrained model. Accuracy is
a fraction on a fixed zero-to-one scale, making both mode plots comparable.
Each mode also gets a house-style raster showing the original image, identical
Poisson input, hidden populations and spiking readout at the selected epoch.

Plotting runs with Matplotlib's noninteractive Agg backend and writes a new
Pingstore presentation run. Its numbers.json preserves the analysis values
and adds the figure filenames for downstream consumers. The input run is
explicit and its recipe is checked before plotting; this stage does not
load a dataset, execute a network or dispatch either upstream stage."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import numpy as np
from experiments import theme
from experiments.exp141.recipe import configuration
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from snnlab import lang, viz
from snnlab.sim import RecordingData, SignalInfo, TrainingHistory
from snnlab.viz import plot_learning_curves, plot_raster

DEFAULT_FORMATS = ("svg", "pdf")


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
        REPO / ".pingstore", presentation, experiment="exp141", stage="present"
    )
    sources = {
        role: source_run(REPO / ".pingstore", pin["run_id"], reference=pin)
        for role, pin in previous.record["inputs"].items()
    }
    pin = sources["analyse"].record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore", pin["run_id"], reference=pin, stage="compute"
    )
    specs = [
        (compute, f"{mode}.bundle", f"{mode}--circuit") for mode in ("coba", "ping")
    ]
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


def sample_raster(mode, pixels, recording, sample, cfg, export):
    """Show the image, shared input, hidden activity and digit readout in house style."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(14, 3.6),
        gridspec_kw={"width_ratios": [1, 2, 3, 2]},
        constrained_layout=True,
    )
    axes[0].imshow(pixels, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set(
        title=f"Sample {sample['index']} · digit {sample['label']}",
        xticks=[],
        yticks=[],
    )

    raster_style = dict(marker="|", s=8, linewidths=0.5, rasterized=True)
    plot_raster(
        axes[1], recording, signal="drive", color=theme.INK_BLACK, **raster_style
    )
    axes[1].set(
        title="Shared Poisson input", ylabel="Pixel", ylim=(-0.5, cfg["n_input"] - 0.5)
    )
    plot_raster(
        axes[2],
        recording,
        signal=f"{mode}--E",
        color=theme.INK_BLACK,
        label="E",
        **raster_style,
    )
    hidden_size = cfg["n_hidden"]
    if mode == "ping":
        plot_raster(
            axes[2],
            recording,
            signal=f"{mode}--I",
            color=theme.DEEP_RED,
            offset=hidden_size,
            label="I",
            **raster_style,
        )
        axes[2].axhline(hidden_size - 0.5, color=theme.MUTED, linewidth=0.5)
        hidden_size += cfg["n_inhibitory"]
    axes[2].set(
        title="Hidden populations", ylabel="Neuron", ylim=(-0.5, hidden_size - 0.5)
    )
    axes[2].legend()
    plot_raster(
        axes[3],
        recording,
        signal=f"{mode}--readout",
        color=theme.INK_BLACK,
        **raster_style,
    )
    axes[3].set(
        title="Spiking readout", ylabel="Digit", ylim=(-0.5, cfg["n_classes"] - 0.5)
    )
    axes[3].set_yticks(range(cfg["n_classes"]))
    for axis in axes[1:]:
        axis.set(xlabel="Time (ms)", xlim=(0, cfg["duration_ms"]))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    prediction = sample["modes"][mode]["prediction"]
    epoch = sample["modes"][mode]["epoch"]
    figure.suptitle(
        f"{mode.upper()} · selected epoch {epoch} · predicted {prediction} · {sample['partition']} sample (diagnostic)"
    )
    filename = f"{mode}--sample-raster.svg"
    save_figure(figure, export / filename)
    plt.close(figure)
    return filename


def present(source, *, run_id=None):
    cfg = configuration()
    # Read one explicit analysis run, keeping the figure provenance unambiguous.
    upstream = source_run(
        REPO / ".pingstore",
        source,
        stage="analyse",
        experiment=cfg["experiment"],
    )
    numbers = load_json(upstream.file("numbers.json"))
    if numbers["configuration"] != cfg:
        raise ValueError("analysis configuration differs from this recipe")
    # Keep the font cache in a writable temporary location and select a backend
    # that can render on machines without a display. Respect an existing cache setting.
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "exp141-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    theme.apply()

    # Presentation outputs belong to a new run; the input numbers remain unchanged.
    with stage_run(
        REPO,
        cfg["experiment"],
        "present",
        inputs={"analyse": upstream},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        # Circuit figures come from the executed compute bundles, not a new graph.
        reference = upstream.record["inputs"]["compute"]
        compute = source_run(
            REPO / ".pingstore",
            reference["run_id"],
            stage="compute",
            experiment=cfg["experiment"],
            reference=reference,
        )
        for mode in cfg["models"]:
            bundle = lang.load_bundle(
                compute.restore_bundle(f"{mode}.bundle", run.scratch / f"{mode}.bundle")
            )
            save_circuit(bundle, run.export / f"{mode}--circuit.svg")
        figures = {}
        for mode in cfg["models"]:
            history = TrainingHistory.from_metrics(numbers["training"][mode])
            figure, axes = plt.subplots(
                1,
                2,
                figsize=(8, 3),
                constrained_layout=True,
            )
            # Training is an online minibatch aggregate; validation uses fixed
            # epoch-end weights. Neither invent a training baseline at epoch zero
            # nor join the separate held-out test score into the online curve.
            artists = plot_learning_curves(
                {"loss": axes[0], "accuracy": axes[1]},
                history,
                include_test=True,
                colors={
                    "train": theme.INK_BLACK,
                    "validation": theme.DEEP_RED,
                    "test": theme.DEEP_RED,
                },
                marker="o" if cfg["epochs"] == 1 else None,
            )
            figure.legend(
                handles=[
                    artists[split, "accuracy"][0]
                    for split in ("train", "validation", "test")
                ],
                labels=("Training (online)", "Validation", "Held-out test (selected)"),
                loc="outside lower center",
                ncols=3,
                frameon=False,
                fontsize=8,
            )
            for axis, label in zip(axes, ("Cross-entropy", "Accuracy"), strict=True):
                axis.set(
                    xlabel="Epoch",
                    ylabel=label,
                )
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            # Accuracy is stored as a fraction, not a percentage.
            axes[1].set_ylim(0, 1)
            figure.suptitle(mode.upper())
            figures[mode] = f"{mode}.svg"
            save_figure(figure, run.export / figures[mode])
            # Release each figure before rendering the next mode.
            plt.close(figure)
        # Reuse saved post-training spikes; rendering never executes a network.
        raster_figures = {}
        with np.load(upstream.file("rasters.npz"), allow_pickle=False) as arrays:
            recording = RecordingData(
                signals={name: arrays[name] for name in numbers["recording_info"]},
                info={
                    name: SignalInfo(**values)
                    for name, values in numbers["recording_info"].items()
                },
            )
            for mode in cfg["models"]:
                raster_figures[mode] = sample_raster(
                    mode,
                    arrays["pixels"],
                    recording,
                    numbers["sample"],
                    cfg,
                    run.export,
                )
        # Carry the numeric evidence alongside the filenames used to find each SVG.
        write_json_atomic(
            run.export / "numbers.json",
            {**numbers, "figures": figures, "raster_figures": raster_figures},
        )
    return run.run_id


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
        parser.add_argument("--run-id")
        present(**vars(parser.parse_args()))
