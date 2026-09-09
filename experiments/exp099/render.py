"""Six-panel white/black/red presentation of retained private-afferent evidence."""

from dataclasses import replace
from pathlib import Path

import matplotlib
from experiments.helpers import theme
from tools import snnlang as snn  # noqa: TID251

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from tools.snnviz import (  # noqa: TID251
    DiagramGroup,
    FigureGrid,
    FrameTimeline,
    grid_layout,
    render_diagram,
    save_animation,
)

from . import recipe

BLACK, RED, GREY = theme.INK_BLACK, theme.DEEP_RED, theme.GREY_MID


def _style():
    theme.apply()
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "font.size": 9,
            "axes.edgecolor": BLACK,
            "axes.labelcolor": GREY,
            "xtick.color": GREY,
            "ytick.color": GREY,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def network_diagram(cfg, weights, path, *, bundle):
    """SNNLang lowers the graph; SNNViz renders its physical parameter labels."""
    diagram = snn.diagram(bundle, view="expanded")
    nodes = []
    for node in diagram.nodes:
        if node.id.startswith("private"):
            pop = node.id[-1]
            rates = (
                f"{cfg['baseline_e_hz']:g} → {cfg['stimulus_e_hz']:g} Hz"
                if pop == "e"
                else (
                    f"{cfg['baseline_i_hz']:g} → {cfg['stimulus_i_hz']:g} Hz"
                    if "stimulus_i_hz" in cfg
                    else f"{cfg['baseline_i_hz']:g} Hz"
                )
            )
            nodes.append(
                replace(
                    node,
                    title=f"{pop.upper()} PRIVATE",
                    detail="400 independent afferents per cell",
                    badge=rates,
                )
            )
        else:
            nodes.append(
                replace(
                    node,
                    title=f"{node.id} POPULATION",
                    detail=f"{cfg['n_' + node.id.lower()]:,} LIF cells · {'AMPA' if node.id == 'E' else 'GABA'} output {cfg['recurrent_' + node.id.lower() + '_weight_us'] * 1000:g} nS",
                    badge=f"{cfg['capacitance_nf'] * 1000:g} pF · {cfg['leak_us'] * 1000:g} nS leak",
                )
            )
    edges = []
    for edge in diagram.edges:
        w = weights[edge.id + ".weight"]
        nonzero = w[w > 0]
        receptor = "GABA" if edge.role == "inhibitory" else "AMPA"
        edges.append(
            replace(
                edge,
                label=(
                    f"{receptor} · {float(nonzero[0]) * 1000:g} nS"
                    if edge.connection == "feedforward"
                    else ""
                ),
                constraint=edge.connection == "feedforward",
            )
        )
    diagram = replace(
        diagram,
        nodes=tuple(nodes),
        edges=tuple(edges),
        groups=(
            DiagramGroup(
                "recurrent",
                "RECURRENT E/I · p = 0.10 · DELAY 1.5 ms",
                ("E", "I"),
                same_rank=True,
            ),
        ),
        title="PRIVATE AFFERENT E/I NETWORK",
    )
    render_diagram(diagram, path)


def frame_grid():
    grid = FigureGrid(
        rows=(0.37, 0.395),
        columns=(0.465, 0.215, 0.22),
        bounds=(0.025, 0.075, 0.95, 0.855),
        row_gap=0.09,
        column_gap=0.025,
    )
    grid.place("A", row=0, column=0, rowspan=2)
    grid.place("B", row=0, column=1)
    grid.place("C", row=0, column=2)
    grid.place("response", row=1, column=1)
    grid.place("F", row=1, column=2)
    response = grid.subgrid("response", rows=(0.155, 0.145), columns=1, row_gap=0.095)
    response.place("D", row=0, column=0)
    response.place("E", row=1, column=0)
    return grid, response


def render(
    recording,
    weights,
    measurements,
    cfg,
    output: Path,
    *,
    preview_only=False,
    view=None,
):
    _style()
    cfg = dict(cfg)
    if view is not None:
        cfg.update(view_start_ms=view["start_ms"], view_end_ms=view["end_ms"])
    data = recording.signals
    grid, response = frame_grid()
    fig = grid.figure(figsize=(14.4, 8.5), dpi=180)
    plt.rcParams["savefig.bbox"] = None
    rects = {k: grid.rect(k).mpl for k in ("A", "B", "F")}
    rects["C"] = grid.rect("C", padding=(0.02 / 0.22, 0, 0, 0)).mpl
    rects.update({k: response.rect(k).mpl for k in ("D", "E")})
    titles = {
        "A": "NETWORK FLOW",
        "B": "POPULATION MEANS",
        "C": "CONDUCTANCE PHASE",
        "D": "E/I FIRING RATE",
        "E": "PRIVATE AFFERENT RATE",
        "F": "RECURRENT WEIGHTS",
    }
    axes = {}
    for key, rect in rects.items():
        ax = fig.add_axes(rect)
        axes[key] = ax
        fig.text(
            rect[0],
            rect[1] + rect[3] + 0.018,
            f"{key} · {titles[key]}",
            fontsize=13,
            color=BLACK,
        )
        ax.tick_params(labelsize=6.5, length=2)
    net = axes["A"]
    net.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    boxes = {
        "private_e": (0.045, 0.66, 0.30, 0.21),
        "private_i": (0.045, 0.12, 0.30, 0.13),
        "e": (0.60, 0.30, 0.35, 0.59),
        "i": (0.60, 0.065, 0.35, 0.1475),
    }
    coordinates, selections, scatters = {}, {}, {}
    for key, (x, y, w, h) in boxes.items():
        pop = key[-1]
        color = BLACK if pop == "e" else RED
        count = min(cfg[f"n_{pop}"], 400 if pop == "e" else 100)
        cols = 20 if pop == "e" else 10
        xy = grid_layout(
            count,
            columns=cols,
            x_range=(x + 0.012, x + w - 0.012),
            y_range=(y + 0.012, y + h - 0.012),
        )
        coordinates[key] = xy
        selections[key] = np.linspace(0, cfg[f"n_{pop}"] - 1, count, dtype=int)
        net.add_patch(
            Rectangle(
                (x, y), w, h, facecolor="white", edgecolor=color, lw=0.75, zorder=0
            )
        )
        title = (
            f"{pop.upper()} PRIVATE"
            if key.startswith("private")
            else f"{pop.upper()} POPULATION"
        )
        net.text(x, y + h + 0.024, title, fontsize=11, color=color)
        scatters[key] = net.scatter(
            xy[:, 0],
            xy[:, 1],
            s=3 if key.startswith("private") else 7,
            color=color,
            alpha=0.3,
            zorder=4,
            linewidths=0,
        )
    net.text(0.045, 0.605, "400 sources / cell", fontsize=9, color=GREY)
    net.text(0.045, 0.576, "independent Poisson", fontsize=9, color=GREY)
    net.text(0.045, 0.547, "AMPA · 4 nS", fontsize=9, color=BLACK)
    net.text(0.045, 0.066, "400 sources / cell", fontsize=9, color=GREY)
    net.text(0.60, 0.277, f"{cfg['n_e']:,} cells", fontsize=9, color=GREY)
    net.text(0.60, 0.025, f"{cfg['n_i']:,} cells", fontsize=9, color=GREY)
    net.text(
        0.045,
        0.465,
        f"E: {cfg['baseline_e_hz']:g} → {cfg['stimulus_e_hz']:g}"
        + (f" → {cfg['baseline_e_hz']:g}" if cfg["offset_ms"] < cfg["t_ms"] else "")
        + " Hz",
        fontsize=10,
        color=BLACK,
    )
    net.text(
        0.045,
        0.430,
        f"I: {cfg['baseline_i_hz']:g}"
        + (f" → {cfg['stimulus_i_hz']:g}" if "stimulus_i_hz" in cfg else "")
        + (
            f" → {cfg['baseline_i_hz']:g}"
            if "stimulus_i_hz" in cfg and cfg["offset_ms"] < cfg["t_ms"]
            else ""
        )
        + " Hz",
        fontsize=10,
        color=RED,
    )
    net.text(0.045, 0.373, "No shared drive", fontsize=9, color=GREY)
    clock = net.text(0.045, 0.94, "", fontsize=11, color=BLACK)
    net.text(
        0.045, 0.025, "Cells and edges subsampled for display", fontsize=6.5, color=GREY
    )
    rng = np.random.default_rng(99)
    edges = []
    for src in ("e", "i"):
        for dst in ("e", "i"):
            w = weights[f"{src.upper()}_to_{dst.upper()}.weight"]
            si, ti = np.nonzero(w[np.ix_(selections[src], selections[dst])])
            choose = rng.choice(len(si), min(100, len(si)), replace=False)
            si, ti = si[choose], ti[choose]
            segments = np.stack([coordinates[src][si], coordinates[dst][ti]], axis=1)
            color = BLACK if src == "e" else RED
            collection = LineCollection(
                segments, colors=[to_rgba(color, 0.045)], linewidths=0.35, zorder=1
            )
            net.add_collection(collection)
            edges.append((collection, src, selections[src][si], color))
    private_lines = {}
    for pop in ("e", "i"):
        choose = np.arange(0, len(coordinates[pop]), 4)
        segments = np.stack(
            [coordinates[f"private_{pop}"][choose], coordinates[pop][choose]], axis=1
        )
        collection = LineCollection(
            segments, colors=[to_rgba(BLACK, 0.10)], linewidths=0.4, zorder=2
        )
        net.add_collection(collection)
        private_lines[pop] = (collection, selections[pop][choose])

    meanax = axes["B"]
    meanax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    pistons = []
    visible = (measurements["time_ms"] >= cfg["view_start_ms"]) & (
        measurements["time_ms"] <= cfg["view_end_ms"]
    )
    gmax = max(
        0.001,
        float(
            max(
                measurements["mean_g_e"][visible].max(),
                measurements["mean_g_i"][visible].max(),
            )
        )
        * 1.15,
    )
    for x, label, key, color, lo, hi, unit in (
        (0.14, r"$g_E$", "mean_g_e", BLACK, 0, gmax, "µS"),
        (0.38, r"$g_I$", "mean_g_i", RED, 0, gmax, "µS"),
        (0.64, r"$V_E$", "mean_v_e", BLACK, -80, -50, "mV"),
        (0.88, r"$V_I$", "mean_v_i", RED, -80, -50, "mV"),
    ):
        meanax.text(x, 0.905, label, ha="center", fontsize=16, color=color)
        meanax.add_patch(
            Rectangle(
                (x - 0.07, 0.18),
                0.14,
                0.60,
                facecolor="white",
                edgecolor="#bdbdbd",
                lw=0.8,
            )
        )
        fill = meanax.add_patch(
            Rectangle(
                (x - 0.06, 0.18),
                0.12,
                0.001,
                facecolor=color,
                alpha=0.2,
                edgecolor="none",
            )
        )
        head = meanax.add_patch(
            Rectangle(
                (x - 0.09, 0.18),
                0.18,
                0.025,
                facecolor=color,
                edgecolor="white",
                lw=0.5,
            )
        )
        text = meanax.text(x, 0.10, "", ha="center", fontsize=8, color=color)
        meanax.text(x, 0.04, unit, ha="center", fontsize=8, color=GREY)
        meanax.text(
            x,
            0.80,
            f"{hi:.3f}" if unit == "µS" else f"{hi:g}",
            ha="center",
            fontsize=6,
            color=GREY,
        )
        pistons.append((fill, head, text, key, lo, hi))
    meanax.text(
        0.5, 0.98, "Conductances onto E", ha="center", va="top", fontsize=7, color=GREY
    )
    phase = axes["C"]
    phase.set(
        xlim=(0, gmax),
        ylim=(-0.00004, max(0.001, float(measurements["mean_g_i"].max()) * 1.15)),
        xlabel=r"$g_E$ (µS)",
        ylabel=r"$g_I$ (µS)",
    )
    phase.xaxis.set_major_locator(MaxNLocator(4))
    phase.yaxis.set_major_locator(MaxNLocator(4))
    phase.grid(alpha=0.12, linestyle=":")
    (phase_line,) = phase.plot([], [], color=RED, lw=1)
    (phase_point,) = phase.plot([], [], "o", color=RED, ms=3)
    times = measurements["time_ms"]
    viewstart = round(cfg["view_start_ms"] / cfg["dt_ms"])
    viewstop = min(len(times) - 1, round(cfg["view_end_ms"] / cfg["dt_ms"]) - 1)
    ticktimes = np.round(
        np.linspace(cfg["view_start_ms"], cfg["view_end_ms"], 10)
    ).astype(int)
    traces = []
    for key, series, ylabel in (
        ("D", ("rate_e", "rate_i"), "Hz / neuron"),
        ("E", ("rate_private_e", "rate_private_i"), "Hz / source"),
    ):
        ax = axes[key]
        ax.set(
            xlim=(cfg["view_start_ms"], cfg["view_end_ms"]),
            xticks=ticktimes,
            xlabel="ms",
            ylabel=ylabel,
        )
        ax.set_xticklabels([f"{t - cfg.get('burn_in_ms', 0):.0f}" for t in ticktimes])
        ax.grid(alpha=0.12, linestyle=":")
        ax.set_ylabel(ylabel, fontsize=7, labelpad=1)
        if key == "D":
            ax.set_ylim(
                -0.04,
                max(1.0, max(measurements[s][visible].max() for s in series) * 1.1),
            )
        else:
            ax.set_ylim(0, cfg["stimulus_e_hz"] * 1.35)
        for color, s, label in zip(
            (BLACK, RED),
            series,
            ("E", "I") if key == "D" else ("E PRIVATE", "I PRIVATE"),
        ):
            (line,) = ax.plot([], [], color=color, lw=1.1, label=label)
            traces.append((line, s))
        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=7,
            ncol=2,
            handlelength=1.2,
            columnspacing=0.8,
        )
    axes["D"].text(
        0.99,
        0.95,
        "20 ms trailing",
        ha="right",
        va="top",
        transform=axes["D"].transAxes,
        fontsize=6,
        color=GREY,
    )
    cursors = [
        axes[k].axvline(cfg["view_start_ms"], color=GREY, lw=0.6) for k in ("D", "E")
    ]
    wax = axes["F"]
    values = sorted(
        {float(w[w > 0][0]) for k, w in weights.items() if not k.startswith("private")}
    )
    xmin, xmax = values[0] * 0.6, values[-1] * 1.7
    wax.set(xscale="log", xlim=(xmin, xmax), ylim=(0, 4), yticks=[], xlabel="µS")
    ticks = [xmin, *values, xmax]
    wax.set_xticks(ticks, [f"{v:.4g}" for v in ticks])
    wax.minorticks_off()
    wax.grid(axis="x", alpha=0.2, linestyle=":")
    for y, src, dst in (
        (3.20, "E", "E"),
        (2.25, "E", "I"),
        (1.30, "I", "I"),
        (0.35, "I", "E"),
    ):
        w = weights[f"{src}_to_{dst}.weight"]
        nonzero = w[w > 0]
        color = BLACK if src == "E" else RED
        wax.hlines(y, xmin, xmax, color=GREY, lw=0.5)
        value = float(nonzero[0])
        wax.vlines(value, y, y + 0.40, color=color, lw=1.8)
        wax.plot(value, y + 0.40, "o", ms=3, color=color)
        wax.text(
            0.03,
            y + 0.48,
            f"{src} → {dst}",
            fontsize=9,
            color=color,
            transform=wax.get_yaxis_transform(),
        )
        wax.text(
            0.97,
            y + 0.48,
            f"{value:.5g} µS",
            ha="right",
            fontsize=8,
            color=color,
            transform=wax.get_yaxis_transform(),
        )
        wax.text(
            0.97,
            y - 0.12,
            f"{np.count_nonzero(w) / w.size:.1%} connected",
            transform=wax.get_yaxis_transform(),
            ha="right",
            fontsize=6.5,
            color=GREY,
        )
    wax.text(
        0.03,
        0.985,
        "Fixed nonzero weights",
        va="top",
        transform=wax.transAxes,
        fontsize=7,
        color=GREY,
    )
    frames = (
        view["frames"]
        if view is not None
        else int(np.ceil((cfg["view_end_ms"] - cfg["view_start_ms"]) / 8.0))
    )
    timeline = FrameTimeline.compose(
        [(viewstart, viewstop, frames)], dt_ms=recording.dt_ms
    )
    frame_steps = timeline.steps

    def update(frame):
        step = int(frame_steps[frame])
        t = times[step]
        clock.set_text(f"t = {t - cfg.get('burn_in_ms', 0):.0f} ms")
        recent = slice(max(0, step - round(1 / cfg["dt_ms"])), step + 1)
        for key, scatter in scatters.items():
            pop = key[-1]
            source = data[key] if key.startswith("private") else data[f"spk_{pop}"]
            strength = np.minimum(source[recent][:, selections[key]].sum(0), 2) / 2
            color = BLACK if pop == "e" else RED
            rgba = np.tile(to_rgba(color), (len(strength), 1))
            rgba[:, 3] = 0.15 + 0.85 * strength
            scatter.set_facecolors(rgba)
            scatter.set_sizes((3 if key.startswith("private") else 7) + strength * 14)
        # Edge flashes use delayed arrivals, with a short visible persistence.
        arrived = step - round(cfg["delay_ms"] / cfg["dt_ms"])
        arrival_slice = slice(
            max(0, arrived - round(3 / cfg["dt_ms"])), max(0, arrived + 1)
        )
        for collection, src, indices, color in edges:
            active = data[f"spk_{src}"][arrival_slice][:, indices].any(0)
            rgba = np.tile(to_rgba(color), (len(active), 1))
            rgba[:, 3] = 0.035 + 0.5 * active
            collection.set_colors(rgba)
        for pop, (collection, indices) in private_lines.items():
            active = data[f"private_{pop}"][arrival_slice][:, indices].any(0)
            rgba = np.tile(to_rgba(BLACK), (len(active), 1))
            rgba[:, 3] = 0.045 + 0.22 * active
            collection.set_colors(rgba)
        for fill, head, text, key, lo, hi in pistons:
            value = measurements[key][step]
            height = 0.60 * np.clip((value - lo) / (hi - lo), 0, 1)
            fill.set_height(height)
            head.set_y(0.18 + height - 0.0125)
            text.set_text(
                f"{value:.4f}" if key.startswith("mean_g") else f"{value:.1f}"
            )
        trail = slice(max(viewstart, step - round(40 / cfg["dt_ms"])), step + 1)
        phase_line.set_data(
            measurements["mean_g_e"][trail], measurements["mean_g_i"][trail]
        )
        phase_point.set_data(
            [measurements["mean_g_e"][step]], [measurements["mean_g_i"][step]]
        )
        # One plotted point per millisecond is sufficient for the display traces.
        sl = slice(viewstart, step + 1, max(1, round(1 / cfg["dt_ms"])))
        for line, key in traces:
            line.set_data(times[sl], measurements[key][sl])
        for cursor in cursors:
            cursor.set_xdata([t, t])
        return []

    poster_frame = int(
        np.argmin(
            abs(times[frame_steps] - (cfg["peak_ms"] + cfg["plateau_end_ms"]) / 2)
        )
    )
    update(poster_frame)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / recipe.POSTER, dpi=180)
    if not preview_only:
        save_animation(
            fig, update, output / recipe.VIDEO, frames=frames, fps=25, bitrate=6000
        )
    plt.close(fig)
