"""Draw one four-panel figure from an explicit validated Task 1 analysis.

Run present.py --source ANALYSE_ID. Voltage, gate voltages, ionic conductances
and the exact current drive share a time axis. SVG and PNG encode the same
figure; the compiled circuit is retained separately for the report appendix.
"""

import argparse
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
from experiments.exp146.recipe import validate_configuration
from pingstore.library import load_json, source_run, stage_run
from snnlab import lang, viz


def network_diagram(bundle):
    """Resolve the saved circuit with explicit routes and target-first weight labels."""
    native = lang.diagram(bundle, view="circuit")
    symbols = {"drive": "D", "cell": "N"}
    diagram = replace(
        native,
        title="Current-driven ionic neuron",
        nodes=tuple(
            replace(
                node,
                detail=node.detail.splitlines()[0].replace("1 neurons", "1 neuron"),
            )
            for node in native.nodes
            if node.id in symbols
        ),
        edges=tuple(edge for edge in native.edges if edge.role != "output"),
        groups=(),
    )
    positions = {"drive": (0, 0), "cell": (500, 0)}
    routes = dict(injection=viz.EdgeLayout(source_port="right", target_port="left"))
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
        },
        edges={
            edge.id: replace(
                routes[edge.id],
                label=viz.DiagramLabel(
                    rf"$W_{{{symbols[edge.target]} , {symbols[edge.source]}}}$"
                ),
            )
            for edge in diagram.edges
        },
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=16), gap=30
        ),
    )
    # Embedded math must keep its measured size independently of plot cropping.
    with matplotlib.rc_context({"savefig.bbox": None}):
        return viz.resolve_diagram(
            diagram, layout=layout, theme=viz.DiagramTheme(secondary_size=13)
        )


def figure(data, cfg):
    theme.apply()
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, layout="constrained")
    time = np.r_[0.0, data["time_ms"]]
    voltage = np.r_[cfg["initial_voltage_mv"], data["voltage_mv"]]
    axes[0].plot(time, voltage, color=theme.INK_BLACK)
    spikes = data["spikes"].astype(bool)
    axes[0].scatter(
        data["time_ms"][spikes],
        data["voltage_mv"][spikes],
        color=theme.DEEP_RED,
        s=12,
        zorder=3,
        label="Upward −20 mV crossing",
    )
    axes[0].set(ylabel="Voltage (mV)")
    axes[0].legend(loc="upper right")
    for channel, label, color in (
        ("na", "Na", theme.INK_BLACK),
        ("k", "K", theme.DEEP_RED),
    ):
        gate = np.r_[cfg["initial_voltage_mv"], data[f"z_{channel}_mv"]]
        axes[1].plot(time, gate, color=color, label=label)
        initial_g = cfg[f"g_{channel}_us"] / (
            1
            + np.exp(
                -cfg[f"gamma_{channel}_per_mv"]
                * (cfg["initial_voltage_mv"] - cfg[f"bias_{channel}_mv"])
            )
        )
        axes[2].plot(
            time, np.r_[initial_g, data[f"g_{channel}_us"]], color=color, label=label
        )
    axes[1].set(ylabel="Gate voltage (mV)")
    axes[2].set(ylabel="Conductance (µS)")
    axes[1].legend(loc="upper right")
    axes[2].legend(loc="upper right")
    axes[3].stairs(
        data["current_na"],
        np.r_[data["input_time_ms"], cfg["duration_ms"]],
        color=theme.INK_BLACK,
    )
    axes[3].set(
        ylabel="Injected current (nA)",
        xlabel="Time (ms)",
        ylim=(-0.05, 1.0),
        xlim=(0, cfg["duration_ms"]),
    )
    for letter, ax in zip("ABCD", axes, strict=True):
        ax.text(
            -0.09,
            1.0,
            letter,
            transform=ax.transAxes,
            weight="bold",
            fontsize=theme.SIZE_PANEL,
            va="top",
        )
        ax.axvline(
            cfg["pulse_start_ms"], color=theme.GREY_LIGHT, linewidth=0.7, linestyle="--"
        )
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def present(*, source, run_id=None):
    analysed = source_run(
        REPO / ".pingstore", source, experiment="exp146", stage="analyse"
    )
    ref = analysed.record["inputs"]["compute"]
    computed = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        experiment="exp146",
        stage="compute",
        reference=ref,
    )
    cfg = analysed.record["execution"]["configuration"]
    validate_configuration(cfg)
    if cfg != computed.record["execution"]["configuration"]:
        raise ValueError("analysis and compute configurations differ")
    numbers = load_json(analysed.file("results.json"))
    if not numbers["independent_euler_passed"]:
        raise ValueError("presentation requires validated numerical evidence")
    with stage_run(
        REPO,
        "exp146",
        "present",
        inputs={"analyse": analysed, "compute": computed},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        with np.load(analysed.file("traces.npz")) as data:
            fig = figure(data, cfg)
        for extension in ("svg", "png"):
            fig.savefig(run.export / f"single-neuron.{extension}", dpi=180)
        plt.close(fig)
        bundle = lang.load_bundle(
            computed.restore_bundle(
                "simulation/network.bundle", run.scratch / "diagram.bundle"
            )
        )
        circuit = network_diagram(bundle)
        with matplotlib.rc_context({"savefig.bbox": None}):
            for extension in ("svg", "png"):
                viz.render_diagram(circuit, run.export / f"circuit.{extension}")
        run.record["execution"]["snnlab_version"] = version("snnlab")
        shutil.copyfile(analysed.file("results.json"), run.export / "numbers.json")
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    present(**vars(parser.parse_args()))
