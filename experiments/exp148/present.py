"""Render the raster, controls and window PSDs from explicit analysis ancestry.

Run present.py --source ANALYSE_ID. PNG, SVG and PDF exports use the same saved
spikes and recorded settings; spectra.json retains the plotted spectral estimates
and their native metadata. A native circuit diagram accompanies the report.
"""

import argparse
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp148.analyse import validate_events, window_spectra
from experiments.exp148.recipe import drive_gain, windows
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from snnlab import lang, viz


def network_diagram(bundle):
    """Resolve the saved circuit with explicit routes and target-first weight labels."""
    native = lang.diagram(bundle, view="circuit")
    symbols = {"X": "X", "E": "E", "I": "I"}
    diagram = replace(
        native,
        title="Shared drive and recurrent E/I pathways",
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
    positions = {"X": (-400, 150), "E": (0, 0), "I": (0, 300)}
    routes = dict(
        X_to_E=viz.EdgeLayout(source_port="upper_right", target_port="left"),
        X_to_I=viz.EdgeLayout(source_port="lower_right", target_port="left"),
        E_to_I=viz.EdgeLayout(source_port="bottom_left", target_port="top_left"),
        I_to_E=viz.EdgeLayout(source_port="top_right", target_port="bottom_right"),
        E_to_E=viz.EdgeLayout(
            source_port="top_right", target_port="right", waypoints=((180, -130),)
        ),
        I_to_I=viz.EdgeLayout(
            source_port="bottom_right", target_port="right", waypoints=((180, 430),)
        ),
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


def transition_figure(events, cfg, spectra):
    theme.apply()
    plt.rcParams["svg.fonttype"] = "none"
    fig = plt.figure(figsize=(16, 10), layout="none")
    heights = [4, 1.15, 1.8]
    grid = fig.add_gridspec(
        len(heights),
        1,
        height_ratios=heights,
        left=0.075,
        right=0.99,
        bottom=0.08,
        top=0.95,
        hspace=0.45,
    )
    raster = fig.add_subplot(grid[0, 0])
    control = fig.add_subplot(grid[1, 0])
    for p, color, offset in (
        ("E", theme.INK_BLACK, 0),
        ("I", theme.DEEP_RED, cfg["n_e"]),
    ):
        times, cells = events[p]
        selected = times >= cfg["burn_in_ms"]
        raster.scatter(
            times[selected] - cfg["burn_in_ms"],
            cells[selected] + offset,
            s=1,
            color=color,
            linewidths=0,
            rasterized=True,
            label=p,
        )
    raster.axhline(cfg["n_e"], color=theme.GREY_LIGHT, lw=0.7)
    raster.set(
        ylim=(0, cfg["n_e"] + cfg["n_i"]),
        ylabel="Neuron index",
        title="Reducing only direct external excitation of I",
    )

    raster.legend(loc="lower right", bbox_to_anchor=(1, 1.01), ncol=2, markerscale=3)
    schedule = cfg["control"]
    burn = cfg["burn_in_ms"]
    duration = cfg["duration_ms"] - burn
    labels = dict(
        baseline="AI TARGET",
        ramp="AI-TO-PING TRANSITION",
        hold="PING TARGET",
    )
    for name, left, right in windows(cfg):
        left, right = left - burn, right - burn
        label = labels[name]
        raster.text(
            (left + right) / 2,
            0.985,
            label,
            transform=raster.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=theme.SIZE_ANNOTATION,
            fontweight="bold",
            bbox=dict(facecolor=theme.PAPER, edgecolor="none", pad=3),
        )
    times = np.arange(0, duration + schedule["update_ms"] / 2, schedule["update_ms"])
    control.step(
        times, 100 * drive_gain(times + burn, cfg), where="post", color=theme.DEEP_RED
    )
    control.set(
        ylim=(-5, 115),
        yticks=[0, 50, 100],
        ylabel="Weight multiplier\n(% reference)",
        xlabel="Time after burn-in (ms)",
        title="External Input→I weight multiplier",
    )
    for axis in (raster, control):
        axis.set_xlim(0, duration)
        axis.set_xticks([0, *[right - burn for _, _, right in windows(cfg)]])
        for boundary in [row[2] - burn for row in windows(cfg)[:-1]]:
            if axis is raster:
                axis.axvline(
                    boundary,
                    color=theme.INK_BLACK,
                    ls=(0, (5, 3)),
                    lw=2,
                    zorder=8,
                    path_effects=[
                        path_effects.Stroke(linewidth=4, foreground=theme.PAPER),
                        path_effects.Normal(),
                    ],
                )
            else:
                axis.axvline(boundary, color=theme.GREY_MID, ls="--", lw=1)
    raster.tick_params(labelbottom=False)
    spectral_grid = grid[2, 0].subgridspec(1, 3, wspace=0.25)
    spectral_axes = []
    for index, (name, window_number) in enumerate(
        (("baseline", 1), ("ramp", 2), ("hold", 3))
    ):
        axis = fig.add_subplot(
            spectral_grid[0, index],
            sharey=spectral_axes[0] if spectral_axes else None,
        )
        spectral_axes.append(axis)
        for population, color in (("E", theme.INK_BLACK), ("I", theme.DEEP_RED)):
            spectrum = spectra[name][population]
            frequencies = spectrum["frequencies_hz"]
            power = spectrum["power"]
            selected = (frequencies > 0) & (frequencies <= 150)
            axis.plot(
                frequencies[selected],
                power[selected],
                color=color,
                label=population + (" (silent)" if not np.any(power) else ""),
            )
        axis.set(
            xlim=(0, 150),
            xlabel="Frequency (Hz)",
            ylabel="Rate PSD\n((spikes/s/neuron)²/Hz)" if index == 0 else "",
            title=f"Window {window_number}: {labels[name]}",
        )
        axis.legend()
    spectral_axes[0].set_ylim(bottom=0)
    theme.label_panels([raster, control, *spectral_axes])
    input_summary = f"{cfg['n_external']:,} shared Poisson inputs · fixed {100 * cfg['input_connection_probability']:g}% input connectivity"
    fig.supxlabel(
        f"{cfg['n_e']:,} E / {cfg['n_i']:,} I · {input_summary} · seed {cfg['seed']}\n"
        f"{burn:g} ms burn-in excluded · ramps updated every {schedule['update_ms']:g} ms",
        fontsize=theme.SIZE_CAPTION,
        y=0.015,
    )
    return fig


def mean_field_figure(numbers, trajectory):
    """Equilibrium stability and matched protocol, without fitting spiking data."""
    model = numbers["mean_field"]
    rows = model["continuation"]
    cfg = model["configuration"]["spiking_configuration"]
    alpha = np.array([row["alpha"] for row in rows])
    states = np.array([row["state"] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout="constrained")
    for index, (p, color) in enumerate((("E", theme.INK_BLACK), ("I", theme.DEEP_RED))):
        axes[0].plot(alpha, states[:, index], color=color, label=p)
        mask = trajectory["time_ms"] >= cfg["burn_in_ms"]
        axes[2].plot(
            trajectory["time_ms"][mask] - cfg["burn_in_ms"],
            trajectory["state"][mask, index],
            color=color,
            label=p + " model",
            lw=1,
        )
        for row in numbers["windows"]:
            axes[2].plot(
                [
                    row["start_ms"] - cfg["burn_in_ms"],
                    row["end_ms"] - cfg["burn_in_ms"],
                ],
                [row[p + "_rate_hz"]] * 2,
                color=color,
                ls="--",
                lw=1,
                label=p + " spiking mean" if row["window"] == "baseline" else None,
            )
    axes[0].set(
        xlabel="External Input→I weight multiplier α",
        ylabel="Equilibrium rate (Hz/neuron)",
        title="Population-rate equilibrium",
        xlim=(1, 0),
    )
    axes[1].plot(alpha, [row["max_real_per_ms"] for row in rows], color=theme.INK_BLACK)
    axes[1].axhline(0, color=theme.GREY_MID, ls="--", lw=1)
    axes[1].set(
        xlabel="External Input→I weight multiplier α",
        ylabel="Largest Re(λ) (ms⁻¹)",
        title="Local stability",
        xlim=(1, 0),
    )
    for row in model["crossings"]:
        for axis in axes[:2]:
            axis.axvline(row["alpha"], color=theme.GREY_MID, ls="--", lw=1)
    for _, _, right in windows(cfg)[:-1]:
        axes[2].axvline(right - cfg["burn_in_ms"], color=theme.GREY_MID, ls="--", lw=1)
    axes[2].set(
        xlabel="Time after burn-in (ms)",
        ylabel="Rate (Hz/neuron)",
        title="Matched protocol; no fitted parameters",
        xlim=(0, cfg["duration_ms"] - cfg["burn_in_ms"]),
    )
    axes[0].legend()
    axes[2].legend(fontsize=8)
    theme.label_panels(axes)
    return fig


def present(*, source, run_id=None):
    analysed = source_run(
        REPO / ".pingstore", source, experiment="exp148", stage="analyse"
    )
    ref = analysed.record["inputs"]["compute"]
    computed = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        experiment="exp148",
        stage="compute",
        reference=ref,
    )
    cfg = computed.record["execution"]["configuration"]
    if cfg != analysed.record["execution"]["configuration"]:
        raise ValueError("analysis and compute settings differ")
    inputs = {"analyse": analysed, "compute": computed}
    numbers = load_json(analysed.file("results.json"))
    model = None
    if "mean_field" in analysed.record["inputs"]:
        reference = analysed.record["inputs"]["mean_field"]
        model = source_run(
            REPO / ".pingstore",
            reference["run_id"],
            experiment="exp148",
            stage="compute",
            reference=reference,
        )
        inputs["mean_field"] = model
    with stage_run(
        REPO,
        "exp148",
        "present",
        inputs=inputs,
        configuration=cfg,
        run_id=run_id,
    ) as run:
        with np.load(computed.file("simulation/events.npz")) as saved:
            validate_events(saved, cfg)
            events = {
                p: (saved[p + "_time_ms"], saved[p + "_cell"]) for p in ("E", "I")
            }
            spectra = window_spectra(saved, cfg)
            fig = transition_figure(events, cfg, spectra)
        write_json_atomic(
            run.export / "spectra.json",
            {
                name: {
                    population: {
                        key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in spectrum.items()
                    }
                    for population, spectrum in populations.items()
                }
                for name, populations in spectra.items()
            },
        )
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(run.export / f"transition.{suffix}")
        plt.close(fig)
        if model is not None:
            with np.load(model.file("mean-field.npz")) as trajectory:
                fig = mean_field_figure(numbers, trajectory)
            for suffix in ("png", "svg", "pdf"):
                fig.savefig(run.export / f"mean-field.{suffix}")
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
        write_json_atomic(run.export / "numbers.json", numbers)
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    present(**vars(parser.parse_args()))
