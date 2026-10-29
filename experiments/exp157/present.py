"""Render Figure 1: the complete nine-window E-only competition protocol.

Drive, E/I activity, gamma frequency and relative phase share one time axis.
The figure and its uncoupled receiver reference consume explicit validated
analysis ancestry; presentation never executes the model.
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
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp157 import analyse, recipe
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from snnlab import lang, viz

# Explicit circuit colours requested for this experiment; other styling uses
# the shared house theme. I traces retain the circuit hue at lower opacity.
COLORS = dict(A="#0072b2", B=theme.DEEP_RED, C=theme.INK_BLACK)
I_ALPHA = 0.35


def save(figure, export, name):
    for extension in ("png", "svg"):
        figure.savefig(export / f"{name}.{extension}")
    plt.close(figure)


def network_diagram(bundle):
    """Lay out the saved architecture; output aliases are not synaptic pathways."""
    native = lang.diagram(bundle, view="circuit")
    nodes = tuple(
        replace(node, detail=node.detail.splitlines()[0])
        for node in native.nodes
        if node.kind != "output"
    )
    diagram = replace(
        native,
        title="E-only drive and feedforward competition",
        nodes=nodes,
        edges=tuple(edge for edge in native.edges if edge.role != "output"),
    )
    # Equal E/I spacing in each circuit; A and B converge on C from the left.
    positions = dict(
        X_A=(0, 0),
        A_E=(360, 0),
        A_I=(360, 260),
        X_B=(0, 780),
        B_E=(360, 780),
        B_I=(360, 1040),
        X_C=(1060, 100),
        C_E=(1060, 390),
        C_I=(1060, 650),
    )
    node_layouts = {}
    for node in nodes:
        first, second = node.id.split("_")
        symbol = f"X_{{{second}}}" if first == "X" else f"{second}_{{{first}}}"
        node_layouts[node.id] = viz.NodeLayout(
            position=positions[node.id],
            label=f"${symbol}$",
            min_size=(210, 140),
            ports=dict(
                down_left=(0.25, 1),
                up_left=(0.25, 0),
                down_right=(0.75, 1),
                up_right=(0.75, 0),
                cross_out=(1, 0.7),
                cross_upper=(0, 0.25),
                cross_lower=(0, 0.75),
            ),
        )
    edges = {}
    for edge in diagram.edges:
        source = edge.source.split("_")
        target = edge.target.split("_")
        source_tex = (
            f"X_{source[1]}" if source[0] == "X" else f"{source[1]}_{source[0]}"
        )
        target_tex = f"{target[1]}_{target[0]}"
        # Matrix convention: target first, source second, W_{target,source}.
        label = viz.DiagramLabel(rf"$W_{{{target_tex},{source_tex}}}$")
        if edge.source == edge.target:
            x, y = positions[edge.source]
            excitatory = source[1] == "E"
            edges[edge.id] = viz.EdgeLayout(
                label=label,
                source_port="up_right" if excitatory else "down_right",
                target_port="right",
                waypoints=((x + 165, y - 120 if excitatory else y + 120),),
            )
        elif source[0] == "X":
            edges[edge.id] = viz.EdgeLayout(
                label=label,
                source_port="bottom" if source[1] == "C" else "right",
                target_port="top" if source[1] == "C" else "left",
            )
        elif source[0] == target[0]:
            excitatory = source[1] == "E"
            edges[edge.id] = viz.EdgeLayout(
                label=label,
                source_port="down_left" if excitatory else "up_right",
                target_port="up_left" if excitatory else "down_right",
            )
        else:
            upper = source[0] == "A"
            edges[edge.id] = viz.EdgeLayout(
                label=label,
                source_port="cross_out",
                target_port="cross_upper" if upper else "cross_lower",
                waypoints=((740, positions[edge.source][1] + 28),),
            )
    # Tight plot cropping changes the measured bounds of embedded math labels.
    with matplotlib.rc_context({"savefig.bbox": None}):
        return viz.resolve_diagram(
            diagram,
            layout=viz.DiagramLayout(
                nodes=node_layouts,
                edges=edges,
                defaults=viz.LayoutDefaults(
                    node=viz.NodeLayout(content_align="center", padding=16),
                    gap=30,
                ),
            ),
            theme=viz.DiagramTheme(secondary_size=13),
        )


def moving_rhythm_validity(time, state, cfg):
    width = cfg["analysis"]["frequency_window_ms"]
    centers = np.arange(
        width / 2, time[-1] - width / 2 + 0.01, cfg["analysis"]["frequency_step_ms"]
    )
    # A 200 ms window contains only five cycles at the 25 Hz lower band edge.
    # Adapt the peak requirement to this shorter figure window; the 400 ms
    # analysis windows retain their original eight-peak requirement.
    settings = dict(cfg["analysis"])
    settings["min_peaks"] = max(
        3,
        min(
            settings["min_peaks"], int(width * settings["gamma_band_hz"][0] / 1000) - 1
        ),
    )
    result = {}
    for circuit in recipe.CIRCUITS:
        values = []
        for center in centers:
            mask = (time >= center - width / 2) & (time < center + width / 2)
            metrics, _, _ = analyse.peak_phase(
                time[mask], state[f"{circuit}_E.rate"][mask], settings
            )
            values.append(metrics["gamma"])
        result[circuit] = np.asarray(values)
    return centers, result


def relative_phases(time, state, cfg, centers, rhythm_validity):
    phases = {
        c: analyse.peak_phase(time, state[c + "_E.rate"], cfg["analysis"])[1]
        for c in recipe.CIRCUITS
    }
    valid = {
        c: np.interp(
            time,
            centers,
            rhythm_validity[c].astype(float),
            left=0,
            right=0,
        )
        >= 0.999
        for c in recipe.CIRCUITS
    }
    result = {}
    for sender in ("A", "B"):
        delta = phases["C"] - phases[sender]
        delta[~(valid["C"] & valid[sender])] = np.nan
        supported = np.isfinite(delta)
        if supported.any():
            delta -= delta[supported][0]
        result[sender] = delta
    return result


def rate_panel(axis, time, state, circuit, mask):
    for kind in ("I", "E"):
        axis.plot(
            time[mask],
            state[f"{circuit}_{kind}.rate"][mask],
            color=COLORS[circuit],
            alpha=1.0 if kind == "E" else I_ALPHA,
            label=f"{circuit} {kind}",
            linewidth=1.3 if kind == "E" else 1.1,
            zorder=3 if kind == "E" else 2,
        )
    maximum = max(
        float(np.max(state[f"{circuit}_{kind}.rate"][mask])) for kind in ("E", "I")
    )
    axis.set_ylim(0, max(1.0, maximum * 1.07))
    axis.set_ylabel(f"{circuit} rate (Hz)")
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[::-1], labels[::-1], loc="upper left", ncols=2)


def frequency_panel(axis, time, state, cfg):
    for circuit in recipe.CIRCUITS:
        for kind in ("E", "I"):
            centers, frequency = analyse.interpeak_frequency(
                time, state[f"{circuit}_{kind}.rate"], cfg["analysis"]
            )
            axis.plot(
                centers,
                frequency,
                color=COLORS[circuit],
                alpha=1.0 if kind == "E" else I_ALPHA,
                label=f"{circuit} {kind}",
                linewidth=1.6 if kind == "E" else 1.1,
                zorder=3 if kind == "E" else 2,
            )
    axis.set_ylabel("Peak-interval\nfrequency (Hz)")


def overview(time, state, control_state, controls, cfg, export):
    centers, validity = moving_rhythm_validity(time, state, cfg)
    phases = relative_phases(time, state, cfg, centers, validity)
    figure, axes = plt.subplots(
        6,
        1,
        figsize=(12, 11),
        layout="constrained",
        sharex=True,
        height_ratios=[1, 1.2, 1.2, 1.2, 1, 1],
    )
    for circuit in recipe.CIRCUITS:
        axis_control = controls[f"X_{circuit}.rate"]
        axes[0].plot(
            time,
            [axis_control(t) for t in time],
            color=COLORS[circuit],
            label=f"X_{circuit}",
        )
    axes[0].set(
        ylabel="Input rate\n(Hz/source)",
        ylim=(-0.1, max(cfg["b_final_drive_hz"] * 1.12, 3)),
    )
    axes[0].legend(loc="upper left", ncols=3)
    for row, circuit in enumerate(recipe.CIRCUITS, 1):
        rate_panel(axes[row], time, state, circuit, np.ones(time.shape, dtype=bool))
    frequency_panel(axes[4], time, state, cfg)
    control_centers, control_frequency = analyse.interpeak_frequency(
        time, control_state["C_E.rate"], cfg["analysis"]
    )
    axes[4].plot(
        control_centers,
        control_frequency,
        color=COLORS["C"],
        alpha=0.6,
        linestyle="--",
        linewidth=1.5,
        label="C E uncoupled",
    )
    axes[4].legend(loc="upper left", ncols=4)
    if not any(np.isfinite(line.get_ydata()).any() for line in axes[4].lines):
        axes[4].text(
            0.5,
            0.5,
            "No supported peak intervals",
            ha="center",
            transform=axes[4].transAxes,
        )
    for sender in ("A", "B"):
        axes[5].plot(
            time,
            phases[sender],
            color=COLORS[sender],
            label=f"C − {sender}",
            linewidth=1.5,
        )
    axes[5].set(
        ylabel="Relative phase\n(cycles)",
        xlabel="Time (ms)",
        xlim=(0, cfg["duration_ms"]),
    )
    axes[5].legend(loc="upper left", ncols=2)
    if not any(np.isfinite(value).any() for value in phases.values()):
        axes[5].text(
            0.5,
            0.5,
            "No overlapping sustained gamma: phase undefined",
            ha="center",
            transform=axes[5].transAxes,
        )
    labels = (
        "1 · Off",
        "2 · C ramps",
        "3 · C holds",
        "4 · A ramps",
        "5 · A holds",
        "6 · B ramps",
        "7 · B holds",
        "8 · B ramps",
        "9 · All hold",
    )
    for window, label in zip(recipe.windows(cfg), labels, strict=True):
        for axis in axes:
            if window["ramp"]:
                axis.axvspan(
                    window["start_ms"],
                    window["stop_ms"],
                    color=theme.RULE,
                    alpha=0.5,
                    zorder=-1,
                    linewidth=0,
                )
            axis.axvline(
                window["start_ms"], color=theme.MUTED, linewidth=0.6, alpha=0.5
            )
        axes[0].text(
            (window["start_ms"] + window["stop_ms"]) / 2,
            1.07,
            label,
            ha="center",
            va="bottom",
            fontsize=theme.SIZE_ANNOTATION,
            transform=axes[0].get_xaxis_transform(),
        )
    axes[5].set_xticks(np.arange(0, cfg["duration_ms"] + 1, cfg["window_ms"]))
    theme.label_panels(axes)
    save(figure, export, "protocol")


def present(source, *, run_id=None):
    upstream = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="analyse"
    )
    reference = upstream.record["inputs"]["compute"]
    computed = source_run(
        REPO / ".pingstore",
        reference["run_id"],
        experiment=recipe.SLUG,
        stage="compute",
        reference=reference,
    )
    numbers = load_json(upstream.file("numbers.json"))
    cfg = numbers["configuration"]
    cfg["figure_style"] = dict(circuit_colors=COLORS, i_alpha=I_ALPHA)
    cfg["figure_frequency_estimator"] = dict(
        method="reciprocal_interpeak_interval",
        placement="interval_midpoint",
        connection="linear_between_adjacent_supported_intervals",
        detection_window_ms=cfg["analysis"]["frequency_window_ms"],
        minimum_prominence_hz=0.05,
        relative_prominence=0.1,
        rhythm_gate=False,
        extrapolation=False,
    )
    time, state, _ = analyse.read_trajectory(computed, "protocol", cfg)
    theme.apply()
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs=dict(analyse=upstream),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        _, control_state, _ = analyse.read_trajectory(computed, "uncoupled", cfg)
        overview(
            time, state, control_state, recipe.drive_controls(cfg), cfg, run.export
        )
        circuit = network_diagram(lang.load_bundle(computed.file("network.bundle")))
        with matplotlib.rc_context({"savefig.bbox": None}):
            for extension in ("svg", "png"):
                viz.render_diagram(circuit, run.export / f"circuit.{extension}")
        run.record["execution"]["snnlab_version"] = version("snnlab")
        write_json_atomic(run.export / "numbers.json", numbers)
        upstream.check_unchanged()
        computed.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    present(**vars(parser.parse_args()))
