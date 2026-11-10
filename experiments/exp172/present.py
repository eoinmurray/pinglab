"""Render spatial-relay and timing-mechanism figures from pinned analysis."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.special import softmax
from snnlab import lang, viz
from snnlab.sim import NetworkSnapshot

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
from experiments.exp172.analyse import (
    cycle_spatial_fits,
    materialise_compute,
)
from experiments.exp172.recipe import phase_plot_configuration
from pingstore.library import source_run, stage_run

sys.path.insert(0, str(REPO))
from experiments import theme


def apply_report_style():
    """Use house typography at sizes readable in the report’s figure column."""
    theme.set_paper_mode(False)
    theme.apply()
    plt.rcParams.update(
        {
            "font.size": theme.SIZE_PANEL,
            "axes.titlesize": theme.SIZE_PANEL,
            "axes.labelsize": theme.SIZE_PANEL,
            "xtick.labelsize": theme.SIZE_TITLE,
            "ytick.labelsize": theme.SIZE_TITLE,
            "legend.fontsize": theme.SIZE_TITLE,
        }
    )


def circuit_diagram(bundle, parameters):
    """Style the saved native architecture; recording aliases are not circuit nodes."""
    native = lang.diagram(bundle, view="circuit")
    symbols = {f"{c}_{p}": f"{p}_{c}" for c in "ABC" for p in "EI"}
    symbols.update({f"X_{c}": f"X_{c}" for c in "ABC"})
    symbols.update({f"U_{c}": f"U_{c}" for c in "AB"})
    replay = any(node.id == "replay_B" for node in native.nodes)
    symbols["replay_B"] = r"B_{\mathrm{replay}}"
    input_sizes = {p["id"]: p["shape"][-1] for p in bundle.graph["inputs"]}
    nodes = tuple(
        replace(
            n,
            detail=f"{input_sizes[n.id]:,} channels"
            if n.id in input_sizes
            else n.detail.splitlines()[0],
            badge="",
        )
        for n in native.nodes
        if n.id in symbols
    )
    edges = tuple(
        replace(e, role="signal", pen_width=0.7)
        if not parameters[e.id + ".weight"].count_nonzero()
        else e
        for e in native.edges
        if e.id is not None
    )
    diagram = replace(
        native,
        title="Spatial relay · packet replay" if replay else "Spatial relay",
        nodes=nodes,
        edges=edges,
        groups=(),
    )
    positions = {
        "A_E": (0, 0),
        "A_I": (0, -240),
        "B_E": (0, 720),
        "B_I": (0, 480),
        "C_E": (800, 360),
        "C_I": (800, 120),
        "X_A": (-500, 0),
        "U_A": (-300, 220),
        "X_B": (-500, 720),
        "U_B": (-300, 940),
        "X_C": (800, 820),
    }
    if replay:
        positions["replay_B"] = (380, 930)
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
    routes = {}
    for edge in edges:
        label = rf"W_{{{symbols[edge.target]},{symbols[edge.source]}}}"
        if edge.id.startswith("cue_"):
            label = rf"g_{edge.target[0]}(t)\,{label}"
        if not parameters[edge.id + ".weight"].count_nonzero():
            label += "=0"
        route = viz.EdgeLayout(label=viz.DiagramLabel(f"${label}$"))
        if edge.source == "replay_B":
            # The saved replay bundle retains the original projection identifier.
            label = r"W_{E_C,E_B}"
            route = replace(
                route,
                label=viz.DiagramLabel(f"${label}$"),
                source_port="top",
                target_port="lower_left",
                waypoints=((380, 400), (650, 400)),
            )
        elif edge.id.startswith("cue_"):
            x, y = positions[edge.source]
            route = replace(
                route,
                source_port="right",
                target_port="bottom_left",
                waypoints=((-45, y),),
            )
        elif edge.source == edge.target:
            x, y = positions[edge.source]
            route = replace(
                route,
                source_port="top_right",
                target_port="right",
                waypoints=((x + 145, y - 105), (x + 145, y)),
            )
        elif edge.id.endswith("E_to_I") and edge.source[0] == edge.target[0]:
            route = replace(route, source_port="top_right", target_port="bottom_right")
        elif edge.id.endswith("I_to_E"):
            route = replace(route, source_port="bottom_left", target_port="top_left")
        elif edge.source == "X_C":
            route = replace(route, source_port="top", target_port="bottom")
        elif edge.id == "A_E_to_C_E":
            route = replace(
                route,
                source_port="lower_right",
                target_port="upper_left",
                waypoints=((300, 20), (300, 340), (650, 340)),
            )
        elif edge.id == "B_E_to_C_E":
            route = replace(
                route,
                source_port="lower_right",
                target_port="lower_left",
                waypoints=((360, 740), (360, 380), (650, 380)),
            )
        elif edge.id == "A_E_to_C_I":
            route = replace(
                route,
                source_port="upper_right",
                target_port="upper_left",
                waypoints=((500, -20), (500, 100), (650, 100)),
            )
        elif edge.id == "B_E_to_C_I":
            route = replace(
                route,
                source_port="upper_right",
                target_port="lower_left",
                waypoints=((440, 700), (440, 140), (650, 140)),
            )
        routes[edge.id] = route
    positions = {key: (x * 0.9, y * 0.85) for key, (x, y) in positions.items()}
    routes = {
        key: replace(
            route, waypoints=tuple((x * 0.9, y * 0.85) for x, y in route.waypoints)
        )
        if route.waypoints
        else route
        for key, route in routes.items()
    }
    layout = viz.DiagramLayout(
        nodes={
            identifier: viz.NodeLayout(
                position=position,
                label=f"${symbols[identifier]}$",
                min_size=(180, 100),
                ports=ports,
            )
            for identifier, position in positions.items()
        },
        edges=routes,
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=14), gap=36
        ),
    )
    style = viz.DiagramTheme(
        ink=theme.INK_BLACK,
        inhibitory=theme.DEEP_RED,
        muted=theme.MUTED,
        line=theme.RULE,
        signal=theme.GREY_MID,
        background=theme.PAPER,
        population_fill=theme.PAPER_TINT,
        population_border=theme.INK_BLACK,
        input_fill=theme.PAPER_TINT,
        input_border=theme.GREY_MID,
        font_name=matplotlib.font_manager.FontProperties(family="monospace").get_name(),
        label_size=20,
        secondary_size=15,
    )
    with matplotlib.rc_context({"savefig.bbox": None}):
        return viz.resolve_diagram(diagram, layout=layout, theme=style)


def save_circuit(bundle, parameters, output, stem="circuit"):
    circuit = circuit_diagram(bundle, parameters)
    with matplotlib.rc_context({"savefig.bbox": None}):
        for suffix in ("svg", "png"):
            viz.render_diagram(circuit, output / f"{stem}.{suffix}")


def phase_evidence(spikes, cfg):
    """Population rates, detected peaks and continuous peak-count phases."""
    dt = cfg["dt_ms"]
    t = np.arange(len(spikes["A_E"])) * dt
    phases = {}
    rates = {}
    detected = {}
    for c in "ABC":
        rate = gaussian_filter1d(spikes[c + "_E"].mean(1) * 1000 / dt, 1 / dt)
        peaks, _ = find_peaks(
            rate,
            distance=round(15 / dt),
            prominence=max(0.05, 0.1 * np.ptp(rate[t > 500])),
        )
        rates[c] = rate
        detected[c] = peaks
        phases[c] = (
            np.interp(
                t,
                t[peaks],
                2 * np.pi * np.arange(len(peaks)),
                left=np.nan,
                right=np.nan,
            )
            if len(peaks) >= 2
            else np.full(t.shape, np.nan)
        )
    return t, phases, rates, detected


def relative_phases(spikes, cfg):
    """Unwrapped sender-minus-C phase sampled once per detected C E cycle."""
    t, phases, _, detected = phase_evidence(spikes, cfg)
    peaks = detected["C"]
    return t[peaks], {c: (phases[c] - phases["C"])[peaks] for c in "AB"}


def amplitude_panel(ax, course, cfg, *, softmax_scale_hz=None):
    rows = course["points"]
    amplitudes = np.array([[r[c + "_amplitude_hz"] for c in "AB"] for r in rows])
    normalised = softmax_scale_hz is not None
    values = (
        softmax(amplitudes / softmax_scale_hz, axis=1) if normalised else amplitudes
    )
    for j, (c, color) in enumerate(
        zip("AB", [theme.INK_BLACK, theme.DEEP_RED], strict=False)
    ):
        ax.plot(
            [r["time_ms"] for r in rows],
            values[:, j],
            color=color,
            lw=1.2,
            label=f"{c}-like",
        )
    ax.axvline(cfg["switch_ms"], color=theme.MUTED, ls="--", lw=1)
    ax.set(
        title="Spatial selection" if normalised else "C spatial amplitudes",
        xlabel="Time (ms)",
        ylabel="Softmax weight" if normalised else "Gaussian amplitude (Hz)",
        xlim=(0, cfg["duration_ms"]),
        ylim=(-0.1, 1.1) if normalised else (0, None),
    )
    ax.legend(loc="upper right", fontsize=theme.SIZE_TITLE)


def select_trial(data, cfg, args):
    representative_seed = args.seed if args.seed is not None else cfg["seeds"][0]
    args.cue = cfg["representative_cue_hz"] if args.cue is None else args.cue
    metrics = json.loads(
        (
            data
            / f"seed{representative_seed}-swap{int(args.swap)}-cue{args.cue:g}"
            / "metrics.json"
        ).read_text()
    )
    selection_metadata = dict(
        seed=representative_seed,
        swap=args.swap,
        cue_hz=args.cue,
        trial_id=metrics["id"],
        spatial_window_ms=100.0,
        spatial_hop_ms=5.0,
        softmax_scale_hz=1.0,
        criterion="Recorded representative cue and seed; originally selected on the displayed seed.",
        selection="recorded representative"
        if args.cue == cfg["representative_cue_hz"]
        else "explicit override",
        post_selection=True,
        averaging="none; one seed and one spatial assignment",
    )
    return metrics, selection_metadata, representative_seed


def raster_panel(raster, spikes, c, cfg):
    """Draw one population raster with the shared E/I layout and cue markers."""
    ne = spikes[c + "_E"].shape[1]
    ni = spikes[c + "_I"].shape[1]
    for p, offset, color in [("E", 0, theme.INK_BLACK), ("I", ne, theme.DEEP_RED)]:
        t, n = np.nonzero(spikes[c + "_" + p])
        raster.scatter(
            t * cfg["dt_ms"],
            n + offset + 1,
            marker=".",
            s=2.5,
            linewidths=0,
            color=color,
            rasterized=True,
        )
    raster.axhline(ne + 0.5, color=theme.GREY_LIGHT, lw=0.5)
    raster.axvline(cfg["switch_ms"], color=theme.MUTED, ls="--", lw=1)
    for start, end in cfg["windows_ms"]:
        raster.axvline(start, color=theme.GREY_MID, ls=":", lw=0.65)
        raster.axvline(end, color=theme.GREY_MID, ls=":", lw=0.65)
    raster.set(
        title=f"{c} · E/I raster",
        xlim=(0, cfg["duration_ms"]),
        ylim=(0.5, ne + ni + 0.5),
        yticks=[1, ne, ne + ni],
        ylabel="Neuron index",
        xlabel="Time (ms)",
    )
    raster.text(
        0.98,
        0.96,
        "I",
        color=theme.DEEP_RED,
        ha="right",
        va="top",
        transform=raster.transAxes,
        fontweight="bold",
    )
    raster.text(
        0.98,
        0.72,
        "E",
        color=theme.INK_BLACK,
        ha="right",
        va="top",
        transform=raster.transAxes,
        fontweight="bold",
    )


def attention_figures(
    spikes, cfg, swap=False, phase_inhibition=None, probe_window=None, used_window=None
):
    """Separate raster and per-cycle relative-phase figures for one trial."""
    apply_report_style()
    colors = [theme.INK_BLACK, theme.DEEP_RED]
    attention_fig = plt.figure(figsize=(12, 4.2), layout="constrained")
    attention_grid = attention_fig.add_gridspec(1, 3)
    attention_axes = []
    for col, population in enumerate("ABC"):
        ax = attention_fig.add_subplot(attention_grid[0, col])
        raster_panel(ax, spikes, population, cfg)
        attention_axes.append(ax)
    phase_fig = plt.figure(figsize=(12, 4.3), layout="constrained")
    phase_grid = phase_fig.add_gridspec(1, 2)
    phase_axes = []
    t, relative = relative_phases(spikes, cfg)
    dominant = {
        cycle["peak_ms"]: cycle["dominant"]
        for cycle in cycle_spatial_fits(spikes, cfg, swap)
    }
    for column, wrapped in enumerate((False, True)):
        ax = phase_fig.add_subplot(phase_grid[0, column])
        phase_axes.append(ax)
        for c, color in zip("AB", colors, strict=False):
            values = relative[c] / (2 * np.pi)
            if wrapped:
                values = (values + 0.5) % 1 - 0.5
                # Break the trace at the phase seam without dropping cycle samples.
                breaks = np.flatnonzero(np.abs(np.diff(values)) > 0.5) + 1
                trace_t = np.insert(t, breaks, np.nan)
                trace_values = np.insert(values, breaks, np.nan)
            else:
                trace_t, trace_values = t, values
            ax.plot(trace_t, trace_values, color=color, lw=1, label=f"{c} − C")
            selected = np.array([dominant.get(time) == c for time in t], dtype=bool)
            ax.plot(
                t[selected],
                values[selected],
                color=color,
                linestyle="none",
                marker="o",
                markersize=3,
            )
        ax.axhline(0, color=theme.GREY_MID, ls=":", lw=0.7)
        ax.axvline(cfg["switch_ms"], color=theme.MUTED, ls="--", lw=1)
        ax.set(
            title=("Wrapped" if wrapped else "Unwrapped") + " phase per C cycle",
            xlabel="Time (ms)",
            ylabel="Sender − C phase (cycles)",
            xlim=(0, cfg["duration_ms"]),
        )
        if wrapped:
            ax.set_ylim(-0.5, 0.5)
            shading = (
                used_window
                if used_window is not None
                else (probe_window if probe_window is not None else phase_inhibition)
            )
            if shading is not None:
                label = (
                    "Used transmission phases"
                    if used_window is not None
                    else (
                        "Probe-response window"
                        if probe_window is not None
                        else "Low-inhibition window"
                    )
                )
                for index, bounds in enumerate(shading["bounds_cycles"]):
                    ax.axhspan(
                        *bounds,
                        color=theme.GREY_LIGHT,
                        alpha=0.25,
                        linewidth=0,
                        zorder=0,
                        label=label if index == 0 else None,
                    )
        if not wrapped:
            ax.legend(loc="upper right", fontsize=theme.SIZE_TITLE)
    return [
        ("rasters", attention_fig, attention_axes),
        ("relative-phase", phase_fig, phase_axes),
    ]


def render(data, output, argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cue",
        type=float,
        help="Override the recorded representative cue (Hz/channel)",
    )
    parser.add_argument(
        "--seed", type=int, help="One input seed; default is the first saved seed"
    )
    parser.add_argument(
        "--swap", action="store_true", help="Use swapped spatial assignments throughout"
    )
    args = parser.parse_args(argv)
    cfg = json.loads((data / "configuration.json").read_text())
    metrics, selection_metadata, representative_seed = select_trial(data, cfg, args)
    directory = data / metrics["id"]
    with np.load(directory / "spikes.npz") as f:
        spikes = dict(f)
    attention_figures_to_save = attention_figures(spikes, cfg, args.swap)
    weight_fig = plt.figure(figsize=(11, 5), layout="constrained")
    weight_grid = weight_fig.add_gridspec(1, 2)
    parameters = NetworkSnapshot.read(
        directory / "start/snapshot.json"
    ).load_parameters()
    save_circuit(lang.load_bundle(directory / "network.bundle"), parameters, output)
    matrices = [parameters[c + "_E_to_C_E.weight"].detach().cpu().numpy() for c in "AB"]
    vmax = max(float(matrix.max()) for matrix in matrices)
    matrix_axes = []
    for col, (c, matrix) in enumerate(zip("AB", matrices, strict=False)):
        ax = weight_fig.add_subplot(weight_grid[0, col])
        matrix_axes.append(ax)
        heat = ax.imshow(
            matrix,
            origin="lower",
            cmap="Greys",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
            extent=(0.5, matrix.shape[1] + 0.5, 0.5, matrix.shape[0] + 0.5),
            rasterized=True,
        )
        ax.set(
            title=f"{c}_E → C_E · saved weights",
            xlabel="Target C_E neuron index",
            ylabel=f"Source {c}_E neuron index",
            xticks=[1, 100, 200, 300, 400],
            yticks=[1, 100, 200, 300, 400],
        )
    weight_fig.colorbar(
        heat, ax=matrix_axes, label="Synaptic weight (µS)", shrink=0.85, pad=0.02
    )
    for name, figure, panels in attention_figures_to_save + [
        ("weights", weight_fig, matrix_axes),
    ]:
        theme.label_panels(panels)
        figure.savefig(output / f"{name}.png", dpi=240)
        figure.savefig(output / f"{name}.svg")
        plt.close(figure)
    return selection_metadata


def render_receptive_window(path, measured, output):
    """Show arrival order relative to measured inhibition, without a hard gate."""
    colors = dict(A=theme.DEEP_RED, B="#2457c5", C="#23844f")
    with np.load(path) as z:
        arrays = dict(z)
    apply_report_style()
    reference = measured["attended_arrival_peak_ms"]
    time = arrays["time_ms"] - reference
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12, 10),
        sharex=True,
        sharey="row",
        layout="constrained",
        gridspec_kw={"height_ratios": [1.2, 1, 1.7]},
    )
    upper = 1.25 * max(
        arrays["original_inhibition_us"].max(), arrays["delayed_inhibition_us"].max()
    )
    for col, (case, title) in enumerate(
        [
            ("original", "Original · B first at lower inhibition"),
            ("delayed", "Delayed B · A first at lower inhibition"),
        ]
    ):
        points = measured["receptive_window"]["cases"][case]
        gate = axes[0, col]
        inhibition = arrays[case + "_inhibition_us"]
        gate.plot(time, inhibition, color=colors["C"], lw=1.4)
        gate.fill_between(time, 0, inhibition, color=colors["C"], alpha=0.10)
        gate.set(title=title, ylim=(0, upper))
        for order, point in enumerate(points, 1):
            sender = point["sender"]
            when = point["time_ms"] - reference
            value = point["inhibition_us"]
            gate.scatter([when], [value], color=colors[sender], s=40, zorder=4)
            label = f"{order}. {sender} · {value:.2f} µS"
            gate.annotate(
                label,
                (when, value),
                xytext=(-72, 30) if order == 1 else (12, 30),
                textcoords="offset points",
                color=colors[sender],
                fontsize=theme.SIZE_TITLE,
                arrowprops=dict(arrowstyle="->", color=colors[sender], lw=0.8),
            )
            for ax in axes[:, col]:
                ax.axvline(when, color=colors[sender], ls=":", lw=0.8, alpha=0.65)
        gate.text(
            0.98,
            0.94,
            "Lower curve = less inhibition",
            ha="right",
            va="top",
            transform=gate.transAxes,
            fontsize=theme.SIZE_TITLE,
            color=colors["C"],
        )
        arrivals = axes[1, col]
        for sender in "AB":
            arrivals.plot(
                time,
                arrays[case + "_arrival_" + sender],
                color=colors[sender],
                label=sender + (" · attended" if sender == "A" else " · distractor"),
            )
        arrivals.set(ylim=(0, None))
        arrivals.legend(loc="upper left", fontsize=theme.SIZE_TITLE)
        raster = axes[2, col]
        ii, jj = np.nonzero(arrays[case + "_C_E"])
        raster.scatter(
            time[ii],
            jj + 1,
            s=4,
            marker=".",
            linewidths=0,
            color=colors["C"],
            rasterized=True,
        )
        neurons = arrays[case + "_C_E"].shape[1]
        for sender, centre in zip(
            "AB", measured["configuration"]["centres"], strict=True
        ):
            position = centre * neurons + 1
            raster.axhline(position, color=theme.GREY_LIGHT, lw=0.7, ls=":")
            raster.text(
                0.98,
                position,
                "Pattern " + sender,
                ha="right",
                va="bottom",
                color=colors[sender],
                fontsize=theme.SIZE_TITLE,
                transform=raster.get_yaxis_transform(),
            )
        dominant = (
            "A"
            if measured["outcomes"][case]["A_amplitude_hz"]
            > measured["outcomes"][case]["B_amplitude_hz"]
            else "B"
        )
        raster.set(
            title=f"C represents {dominant} more strongly",
            ylim=(0.5, neurons + 0.5),
            yticks=[1, 100, 200, 300, 400],
            xlabel="Time from original A arrival (ms)",
        )
        for ax in axes[:, col]:
            ax.set_xlim(time[0], time[-1])
    axes[0, 0].set_ylabel("Mean C inhibition (µS)")
    axes[1, 0].set_ylabel("Arrival rate (Hz)")
    axes[2, 0].set_ylabel("C E neuron index")
    theme.label_panels(list(axes.flat))
    for suffix in ["png", "svg"]:
        fig.savefig(output / ("receptive-window." + suffix), dpi=240)
    plt.close(fig)


def render_candidate(computed, measured, output):
    """Match Figure 1’s raster row using the same renderer and geometry."""
    with np.load(computed.file(measured["trial_id"], "spikes.npz")) as z:
        spikes = dict(z)
    apply_report_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    for ax, population in zip(axes, "ABC", strict=True):
        raster_panel(ax, spikes, population, measured["configuration"])
    theme.label_panels(list(axes))
    for suffix in ["png", "svg"]:
        fig.savefig(output / ("candidate-rasters." + suffix), dpi=240)
    plt.close(fig)


def render_background_sweep(
    measured,
    output,
    *,
    control_key="background_hz",
    xlabel="C independent background rate (Hz/channel)",
    stem="background-sweep",
    reference_rate=None,
):
    """Five input-seed means with seed SD; no pooling of cycle denominators."""
    colors = dict(A=theme.DEEP_RED, B="#2457c5", C="#23844f")
    apply_report_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex="col", layout="constrained")

    def series(rows, key, field):
        return np.array(
            [np.nan if r[key][field] is None else r[key][field] for r in rows],
            dtype=float,
        )

    for col, att in enumerate("AB"):
        rows = [r for r in measured["summary"] if r["attended"] == att]
        x = np.array([r[control_key] for r in rows])
        ax = axes[0, col]
        speed = ax.twinx()
        speed.grid(False)
        for key, style, marker, label, alpha in [
            ("S", "-", "o", "Mean S (solid)", 0.14),
            ("F", (0, (6, 3)), "^", "Mean F (dashed)", 0.06),
        ]:
            mean = series(rows, key, "mean")
            sd = series(rows, key, "sd")
            ax.plot(
                x,
                mean,
                color=colors[att],
                ls=style,
                marker=marker,
                markersize=4,
                label=label,
            )
            ax.fill_between(
                x, mean - sd, mean + sd, color=colors[att], alpha=alpha, linewidth=0
            )
            for xx, yy, r in zip(x, mean, rows, strict=True):
                if r[key]["n"] < r[key]["total"]:
                    ax.annotate(
                        f"{key}: {r[key]['n']}/{r[key]['total']}",
                        (xx, yy),
                        xytext=(2, 5),
                        textcoords="offset points",
                        fontsize=theme.SIZE_TITLE,
                    )
        mean = series(rows, "latency_cycles", "mean")
        sd = series(rows, "latency_cycles", "sd")
        speed.errorbar(
            x,
            mean,
            yerr=sd,
            color=colors["C"],
            ecolor=matplotlib.colors.to_rgba(colors["C"], alpha=0.18),
            elinewidth=0.7,
            capthick=0.7,
            ls=":",
            marker="s",
            markersize=4,
            capsize=2,
            label="Latency (dotted)",
        )
        top = (
            max(
                [
                    2.0,
                    *[
                        float(m + (0 if np.isnan(d) else d))
                        for m, d in zip(mean, sd, strict=True)
                        if np.isfinite(m)
                    ],
                ]
            )
            * 1.15
        )
        speed.set(ylim=(0, top), ylabel="Latency (PING cycles)")
        speed.tick_params(axis="y", labelcolor=colors["C"])
        speed.yaxis.label.set_color(colors["C"])
        for xx, yy, r in zip(x, mean, rows, strict=True):
            metric = r["latency_cycles"]
            if metric["n"] < metric["total"]:
                speed.annotate(
                    f"{metric['n']}/{metric['total']}",
                    (xx, yy if np.isfinite(yy) else top * 0.04),
                    xytext=(2, 5),
                    textcoords="offset points",
                    fontsize=theme.SIZE_TITLE,
                    color=colors["C"],
                )
        ax.set(title=f"Attend {att}", ylabel="S / F", ylim=(-0.05, 1.05))
        handles, labels = ax.get_legend_handles_labels()
        more, extra = speed.get_legend_handles_labels()
        ax.legend(
            handles + more,
            labels + extra,
            loc="lower left",
            bbox_to_anchor=(0, 1.08),
            ncol=1,
            handlelength=3.6,
            numpoints=2,
            columnspacing=1.0,
            fontsize=theme.SIZE_TITLE,
        )
        rates = axes[1, col]
        for key, style, marker, label in [
            ("C_E_hz", "-", "o", "C E (solid)"),
            ("C_I_hz", (0, (6, 3)), "^", "C I (dashed)"),
        ]:
            mean = series(rows, key, "mean")
            sd = series(rows, key, "sd")
            rates.plot(
                x,
                mean,
                color=colors["C"],
                ls=style,
                marker=marker,
                markersize=4,
                label=label,
            )
            rates.fill_between(
                x, mean - sd, mean + sd, color=colors["C"], alpha=0.12, linewidth=0
            )
        rates.set(
            xlabel=xlabel,
            ylabel="C firing rate (Hz)",
            ylim=(0, None),
        )
        rates.legend(
            loc="best", handlelength=3.6, numpoints=2, fontsize=theme.SIZE_TITLE
        )
        for panel in axes[:, col]:
            panel.axvline(
                measured["configuration"]["background_hz"]
                if reference_rate is None
                else reference_rate,
                color=theme.GREY_MID,
                ls=":",
                lw=0.8,
            )
    theme.label_panels(list(axes.flat))
    for suffix in ["png", "svg"]:
        fig.savefig(output / (stem + "." + suffix), dpi=240)
    plt.close(fig)


def render_contrast_examples(computed, measured, output):
    """Actual saved pre-switch input profiles for lower and higher contrast cues."""
    apply_report_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True, layout="constrained")
    profiles = []
    for example in measured["examples"]:
        with np.load(computed.file(example["trial_id"], "input_rates.npz")) as z:
            profiles.append({sender: z["X_" + sender][0] for sender in "AB"})
    upper = (
        max(float(value.max()) for profile in profiles for value in profile.values())
        * 1.08
    )
    for ax, example, profile, title in zip(
        axes,
        measured["examples"],
        profiles,
        ("Lower contrast cue", "Higher contrast cue"),
        strict=True,
    ):
        x = np.arange(len(profile["A"])) / len(profile["A"])
        ax.plot(x, profile["A"], color=theme.DEEP_RED, label="A · attended", lw=1.6)
        ax.plot(
            x, profile["B"], color="#2457c5", ls="--", label="B · unattended", lw=1.6
        )
        ax.axhline(
            measured["configuration"]["spatial_baseline_hz"],
            color=theme.GREY_MID,
            ls=":",
            lw=0.8,
        )
        ax.set(
            title=f"{title} · {example['contrast_ratio']:g}×",
            xlabel="Spatial position (unit circle)",
            xlim=(0, 1),
            ylim=(0, upper),
        )
        ax.legend(loc="upper right", fontsize=theme.SIZE_TITLE)
    axes[0].set_ylabel("External input rate (Hz/channel)")
    theme.label_panels(list(axes))
    for suffix in ("png", "svg"):
        fig.savefig(output / f"contrast-examples.{suffix}", dpi=240)
    plt.close(fig)


def render_arrival_distribution(measured, output):
    """Two networks, four sender/attention distributions, and the native gate."""
    colors = dict(A=theme.DEEP_RED, B="#2457c5", C="#23844f")
    apply_report_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 7.5), sharex=True, sharey=True, layout="constrained"
    )
    density_max = max(
        max(d["density"])
        for p in measured["panels"]
        for d in p["distributions"]
        if d["density"] is not None
    )
    inhibition_max = max(max(p["mean_inhibition_us"]) for p in measured["panels"])
    for ax, panel in zip(axes, measured["panels"], strict=True):
        for dist in panel["distributions"]:
            edges = np.array(dist["edges"])
            centres = (edges[:-1] + edges[1:]) / 2
            width = edges[1] - edges[0]
            label = f"{dist['sender']} {dist['role']} (n={dist['n']})"
            style = "-" if dist["role"] == "attended" else (0, (5, 3))
            marker = "o" if dist["role"] == "attended" else "^"
            if dist["density"] is None:
                ax.plot(
                    [],
                    [],
                    color=colors[dist["sender"]],
                    ls=style,
                    marker=marker,
                    label=label,
                )
                continue
            density = np.array(dist["density"])
            ax.plot(
                np.r_[centres[0] - width, centres, centres[-1] + width],
                np.r_[density[-1], density, density[0]],
                color=colors[dist["sender"]],
                ls=style,
                marker=marker,
                markersize=3,
                markevery=3,
                label=label,
                lw=1.4,
            )
        gate = ax.twinx()
        gate.grid(False)
        gate.plot(
            panel["phase_grid"],
            panel["mean_inhibition_us"],
            color=colors["C"],
            lw=1.2,
            alpha=0.8,
            label="Mean C inhibition",
        )
        gate.set(ylabel="Mean C inhibition (µS)", ylim=(0, inhibition_max * 1.1))
        gate.yaxis.label.set_color(colors["C"])
        gate.tick_params(axis="y", labelcolor=colors["C"])
        ax.axvline(0, color=colors["C"], lw=0.8, ls=":", alpha=0.6)
        ax.set(
            title=f"{'Baseline' if panel['figure'] == 1 else 'Higher background'} · {panel['background_hz']:g} Hz/channel",
            xlabel="Arrival phase (cycles)",
            xlim=(-0.5, 0.5),
            ylim=(0, density_max * 1.1),
            xticks=[-0.5, -0.25, 0, 0.25, 0.5],
        )
        handles, labels = ax.get_legend_handles_labels()
        more, extra = gate.get_legend_handles_labels()
        ax.legend(
            handles + more,
            labels + extra,
            loc="lower left",
            bbox_to_anchor=(0, 1.08),
            ncol=1,
            handlelength=3.2,
            numpoints=2,
            fontsize=theme.SIZE_TITLE,
        )
    axes[0].set_ylabel("Arrival density (cycles⁻¹)")
    theme.label_panels(list(axes))
    for suffix in ["png", "svg"]:
        fig.savefig(output / ("arrival-distribution." + suffix), dpi=240)
    plt.close(fig)


def render_phase_example(
    computed,
    output,
    seed,
    reference_cfg,
    phase_inhibition=None,
    probe_window=None,
    used_window=None,
):
    """A saved trial selected for smoother detected phase, at fixed parameters."""
    if computed is None:
        raise ValueError("--figure2-seed requires --background-source")
    plan = json.loads(computed.file("background-plan.json").read_text())
    item = next(
        i
        for i in plan["trials"]
        if i["seed"] == seed and i["background_hz"] == reference_cfg["background_hz"]
    )
    cfg = json.loads(computed.file(item["id"], "configuration.json").read_text())
    assert cfg["cue_hz"] == reference_cfg["representative_cue_hz"]
    with np.load(computed.file(item["id"], "spikes.npz")) as z:
        spikes = dict(z)
    if phase_inhibition is not None:
        assert phase_inhibition["reference_trial"] == item["id"]
        assert phase_inhibition["seed"] == seed
    if probe_window is not None:
        assert (
            probe_window["reference_trial"] == item["id"]
            and probe_window["seed"] == seed
        )
    if used_window is not None:
        assert (
            used_window["reference_trial"] == item["id"] and used_window["seed"] == seed
        )
    for name, fig, axes in attention_figures(
        spikes,
        cfg,
        phase_inhibition=phase_inhibition,
        probe_window=probe_window,
        used_window=used_window,
    ):
        theme.label_panels(axes)
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{name}.{suffix}", dpi=240)
        plt.close(fig)
    _, _, _, peaks = phase_evidence(spikes, cfg)
    return dict(
        seed=seed,
        cue_hz=item["cue_hz"],
        trial_id=item["id"],
        background_hz=cfg["background_hz"],
        source_role="background_compute",
        spatial_window_ms=100.0,
        spatial_hop_ms=5.0,
        selection="Post hoc selection for smooth detected relative phase among seeds 185–189 at unchanged parameters",
        maximum_detected_interval_ms={
            p: float(np.diff(v).max() * cfg["dt_ms"]) for p, v in peaks.items()
        },
    )


def load_cue_comparisons(
    inputs, analysis, cue_source, background_source, contrast_source
):
    """Resolve exact cue/contrast branches without changing baseline evidence."""
    contrast_compute = None
    cue_sweep = analysis.get("cue_sweep")
    if cue_source is not None:
        if background_source is None:
            raise ValueError("--cue-source requires its --background-source")
        cue_analysis = source_run(
            REPO / ".pingstore", cue_source, experiment="exp172", stage="analyse"
        )
        ref = cue_analysis.record["inputs"]["compute"]
        cue_compute = source_run(
            REPO / ".pingstore",
            ref["run_id"],
            experiment="exp172",
            stage="compute",
            reference=ref,
        )
        assert (
            cue_compute.record["inputs"]["background_analyse"]["run_id"]
            == background_source
        )
        assert (
            cue_compute.record["inputs"]["reference_compute"]
            == inputs["background_analyse"].record["inputs"]["compute"]
        )
        cue_sweep = json.loads(cue_analysis.file("results.json").read_text())[
            "cue_sweep"
        ]
        inputs.update(cue_analyse=cue_analysis, cue_compute=cue_compute)
    contrast_sweep = analysis.get("contrast_sweep")
    if contrast_source is not None:
        if cue_source is None:
            raise ValueError("--contrast-source requires its --cue-source")
        contrast_analysis = source_run(
            REPO / ".pingstore", contrast_source, experiment="exp172", stage="analyse"
        )
        ref = contrast_analysis.record["inputs"]["compute"]
        contrast_compute = source_run(
            REPO / ".pingstore",
            ref["run_id"],
            experiment="exp172",
            stage="compute",
            reference=ref,
        )
        assert contrast_compute.record["inputs"]["cue_analyse"]["run_id"] == cue_source
        assert (
            contrast_compute.record["inputs"]["reference_compute"]
            == cue_analysis.record["inputs"]["compute"]
        )
        contrast_sweep = json.loads(contrast_analysis.file("results.json").read_text())[
            "contrast_sweep"
        ]
        inputs.update(
            contrast_analyse=contrast_analysis, contrast_compute=contrast_compute
        )
    return cue_sweep, contrast_sweep, contrast_compute


def load_phase_comparisons(
    inputs,
    figure2_seed,
    background_source,
    phase_inhibition_source,
    probe_source,
    used_window_source,
):
    background_analysis = inputs.get("background_analyse")
    phase_inhibition = None
    if phase_inhibition_source is not None:
        if figure2_seed is None or background_source is None:
            raise ValueError(
                "--phase-inhibition-source requires --figure2-seed and --background-source"
            )
        phase_analysis = source_run(
            REPO / ".pingstore",
            phase_inhibition_source,
            experiment="exp172",
            stage="analyse",
        )
        phase_ref = phase_analysis.record["inputs"]["compute"]
        phase_compute = source_run(
            REPO / ".pingstore",
            phase_ref["run_id"],
            experiment="exp172",
            stage="compute",
            reference=phase_ref,
        )
        assert (
            phase_compute.record["inputs"]["reference_compute"]
            == background_analysis.record["inputs"]["compute"]
        )
        phase_inhibition = json.loads(phase_analysis.file("results.json").read_text())[
            "phase_inhibition"
        ]
        assert (
            phase_inhibition["source_reference"]
            == background_analysis.record["inputs"]["compute"]
        )
        inputs.update(
            phase_inhibition_analyse=phase_analysis,
            phase_inhibition_compute=phase_compute,
        )
    probe_window = None
    probe_compute = None
    if probe_source is not None:
        if figure2_seed is None or background_source is None:
            raise ValueError(
                "--probe-source requires --figure2-seed and --background-source"
            )
        probe_analysis = source_run(
            REPO / ".pingstore", probe_source, experiment="exp172", stage="analyse"
        )
        ref = probe_analysis.record["inputs"]["compute"]
        probe_compute = source_run(
            REPO / ".pingstore",
            ref["run_id"],
            experiment="exp172",
            stage="compute",
            reference=ref,
        )
        probe_window = json.loads(probe_analysis.file("results.json").read_text())[
            "probe_window"
        ]
        assert (
            probe_window["reference_background"]
            == background_analysis.record["inputs"]["compute"]
        )
        inputs.update(probe_analyse=probe_analysis, probe_compute=probe_compute)
    used_window = None
    used_compute = None
    if used_window_source is not None:
        if figure2_seed is None or background_source is None:
            raise ValueError(
                "--used-window-source requires --figure2-seed and --background-source"
            )
        used_analysis = source_run(
            REPO / ".pingstore",
            used_window_source,
            experiment="exp172",
            stage="analyse",
        )
        ref = used_analysis.record["inputs"]["compute"]
        used_compute = source_run(
            REPO / ".pingstore",
            ref["run_id"],
            experiment="exp172",
            stage="compute",
            reference=ref,
        )
        used_window = json.loads(used_analysis.file("results.json").read_text())[
            "used_window"
        ]
        assert (
            used_window["reference_background"]
            == background_analysis.record["inputs"]["compute"]
        )
        inputs.update(
            used_window_analyse=used_analysis, used_window_compute=used_compute
        )
    return phase_inhibition, probe_window, probe_compute, used_window, used_compute


def present(
    source,
    run_id=None,
    cue=None,
    seed=None,
    mechanism_source=None,
    candidate_source=None,
    background_source=None,
    arrival_source=None,
    cue_source=None,
    contrast_source=None,
    figure2_seed=None,
    phase_inhibition_source=None,
    probe_source=None,
    used_window_source=None,
):
    started = perf_counter()
    analysed = source_run(
        REPO / ".pingstore", source, experiment="exp172", stage="analyse"
    )
    computed = source_run(
        REPO / ".pingstore",
        analysed.record["inputs"]["compute"]["run_id"],
        experiment="exp172",
        stage="compute",
        reference=analysed.record["inputs"]["compute"],
    )
    inputs = {"analyse": analysed, "compute": computed}
    analysis = json.loads(analysed.file("results.json").read_text())
    mechanism_compute = computed
    mechanism = analysis.get("mechanism")
    if mechanism_source is not None:
        diagnostic = source_run(
            REPO / ".pingstore", mechanism_source, experiment="exp172", stage="analyse"
        )
        diagnostic_ref = diagnostic.record["inputs"]["compute"]
        diagnostic_compute = source_run(
            REPO / ".pingstore",
            diagnostic_ref["run_id"],
            experiment="exp172",
            stage="compute",
            reference=diagnostic_ref,
        )
        baseline_ref = diagnostic_compute.record["inputs"]["reference_compute"]
        assert baseline_ref == analysed.record["inputs"]["compute"]
        mechanism_compute = diagnostic_compute
        mechanism = json.loads(diagnostic.file("results.json").read_text())["mechanism"]
        inputs.update(
            mechanism_analyse=diagnostic, mechanism_compute=diagnostic_compute
        )
    candidate = analysis.get("candidate")
    candidate_compute = computed
    if candidate_source is not None:
        candidate_analysis = source_run(
            REPO / ".pingstore", candidate_source, experiment="exp172", stage="analyse"
        )
        ref = candidate_analysis.record["inputs"]["compute"]
        candidate_compute = source_run(
            REPO / ".pingstore",
            ref["run_id"],
            experiment="exp172",
            stage="compute",
            reference=ref,
        )
        assert (
            candidate_compute.record["inputs"]["reference_compute"]
            == analysed.record["inputs"]["compute"]
        )
        candidate = json.loads(candidate_analysis.file("results.json").read_text())[
            "candidate"
        ]
        inputs.update(
            candidate_analyse=candidate_analysis, candidate_compute=candidate_compute
        )
    background_sweep = analysis.get("background_sweep")
    background_compute = computed
    if background_source is not None:
        background_analysis = source_run(
            REPO / ".pingstore", background_source, experiment="exp172", stage="analyse"
        )
        ref = background_analysis.record["inputs"]["compute"]
        background_compute = source_run(
            REPO / ".pingstore",
            ref["run_id"],
            experiment="exp172",
            stage="compute",
            reference=ref,
        )
        assert (
            background_compute.record["inputs"]["reference_compute"]
            == analysed.record["inputs"]["compute"]
        )
        background_sweep = json.loads(
            background_analysis.file("results.json").read_text()
        )["background_sweep"]
        inputs.update(
            background_analyse=background_analysis,
            background_compute=background_compute,
        )
    arrival_distribution = analysis.get("arrival_distribution")
    if arrival_source is not None:
        arrival_analysis = source_run(
            REPO / ".pingstore", arrival_source, experiment="exp172", stage="analyse"
        )
        ref = arrival_analysis.record["inputs"]["compute"]
        arrival_compute = source_run(
            REPO / ".pingstore",
            ref["run_id"],
            experiment="exp172",
            stage="compute",
            reference=ref,
        )
        assert (
            arrival_compute.record["inputs"]["figure1_compute"]
            == analysed.record["inputs"]["compute"]
        )
        assert (
            arrival_compute.record["inputs"]["figure6_compute"]["run_id"]
            == candidate_compute.record["run_id"]
        )
        arrival_distribution = json.loads(
            arrival_analysis.file("results.json").read_text()
        )["arrival_distribution"]
        inputs.update(arrival_analyse=arrival_analysis, arrival_compute=arrival_compute)
    cue_sweep, contrast_sweep, contrast_compute = load_cue_comparisons(
        inputs, analysis, cue_source, background_source, contrast_source
    )
    phase_inhibition, probe_window, probe_compute, used_window, used_compute = (
        load_phase_comparisons(
            inputs,
            figure2_seed,
            background_source,
            phase_inhibition_source,
            probe_source,
            used_window_source,
        )
    )
    with stage_run(
        REPO,
        "exp172",
        "present",
        inputs=inputs,
        run_id=run_id,
        configuration=dict(
            cue=cue,
            seed=seed,
            figure2_seed=figure2_seed,
            phase_plot=phase_plot_configuration(),
            spatial_window_ms=100.0,
            spatial_hop_ms=5.0,
        ),
    ) as run:
        work = run.scratch / "evidence"
        materialise_compute(computed, work)
        args = []
        if cue is not None:
            args += ["--cue", str(cue)]
        if seed is not None:
            args += ["--seed", str(seed)]
        selected = render(work, run.export, args)
        figure2_selection = dict(
            selected,
            background_hz=computed.record["execution"]["configuration"][
                "background_hz"
            ],
        )
        if figure2_seed is not None:
            figure2_selection = render_phase_example(
                inputs.get("background_compute"),
                run.export,
                figure2_seed,
                computed.record["execution"]["configuration"],
                phase_inhibition=phase_inhibition,
                probe_window=probe_window,
                used_window=used_window,
            )
        if used_compute is not None:
            (run.export / "used-window-circuit.svg").write_bytes(
                used_compute.file("used-window-circuit.svg").read_bytes()
            )
        if probe_compute is not None:
            (run.export / "probe-circuit.svg").write_bytes(
                probe_compute.file("probe-circuit.svg").read_bytes()
            )
        if mechanism is not None:
            assert selected["seed"] == mechanism["settings"]["source_seed"]
            assert selected["cue_hz"] == mechanism["settings"]["cue_hz"]
            replay_parameters = NetworkSnapshot.read(
                mechanism_compute.file("timing-rescue--start/snapshot.json")
            ).load_parameters()
            save_circuit(
                lang.load_bundle(
                    mechanism_compute.file("timing-rescue--network.bundle")
                ),
                replay_parameters,
                run.export,
                stem="flicker-circuit",
            )
        if candidate is not None:
            render_candidate(candidate_compute, candidate, run.export)
        if background_sweep is not None:
            render_background_sweep(background_sweep, run.export)
        if cue_sweep is not None:
            render_background_sweep(
                cue_sweep,
                run.export,
                control_key="cue_hz",
                xlabel="Cue drive (Hz/channel)",
                stem="cue-sweep",
                reference_rate=cue_sweep["settings"]["reference_cue_hz"],
            )
        if contrast_sweep is not None:
            render_contrast_examples(contrast_compute, contrast_sweep, run.export)
            render_background_sweep(
                contrast_sweep,
                run.export,
                control_key="contrast_ratio",
                xlabel="Attended / unattended bump amplitude (ratio)",
                stem="contrast-sweep",
                reference_rate=1.0,
            )
        trials = json.loads((work / "summary.json").read_text())["trials"]
        trial = next(r for r in trials if r["id"] == selected["trial_id"])
        cfg = computed.record["execution"]["configuration"]
        parameters = NetworkSnapshot.read(
            work / trial["id"] / "start/snapshot.json"
        ).load_parameters()
        matrix_stats = {
            k: dict(
                shape=list(v.shape),
                nonzero_fraction=float((v != 0).float().mean()),
                weight_min_us=float(v.min()),
                weight_max_us=float(v.max()),
                incoming_mean_us=float(v.sum(0).mean()),
            )
            for k, v in parameters.items()
            if k.endswith(".weight")
        }
        stability_selection = (
            json.loads(computed.file("stability-selection.json").read_text())
            if computed.file("stability-selection.json").exists()
            else None
        )
        analysis = json.loads(analysed.file("results.json").read_text())
        (run.export / "numbers.json").write_text(
            json.dumps(
                dict(
                    configuration=cfg,
                    selection=selected,
                    figure2_selection=figure2_selection,
                    stability_selection=stability_selection,
                    metrics=trial,
                    analysis=analysis,
                    cycle_selection=analysis.get("cycle_selection", {}).get(
                        trial["id"]
                    ),
                    matrix_stats=matrix_stats,
                    mechanism=mechanism,
                    candidate=candidate,
                    background_sweep=background_sweep,
                    cue_sweep=cue_sweep,
                    contrast_sweep=contrast_sweep,
                    arrival_distribution=arrival_distribution,
                    phase_inhibition=phase_inhibition,
                    probe_window=probe_window,
                    used_window=used_window,
                    elapsed_seconds=perf_counter() - started,
                ),
                indent=2,
            )
            + "\n"
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    print(f"Presentation {perf_counter() - started:.2f} s; no simulation")
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--cue", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mechanism-source")
    parser.add_argument("--candidate-source")
    parser.add_argument("--background-source")
    parser.add_argument("--arrival-source")
    parser.add_argument("--cue-source")
    parser.add_argument("--contrast-source")
    parser.add_argument("--figure2-seed", type=int)
    parser.add_argument("--phase-inhibition-source")
    parser.add_argument("--probe-source")
    parser.add_argument("--used-window-source")
    present(**vars(parser.parse_args()))
