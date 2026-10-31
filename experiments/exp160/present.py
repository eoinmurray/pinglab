"""Render the matched pair in two columns using one pinned analysis."""

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import matplotlib

matplotlib.use("Agg")
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from experiments.exp160 import recipe
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from scipy.signal import find_peaks
from snnlab import lang, viz


def network_diagram(bundle):
    """Align each saved E/I pair in a column with explicitly routed projections."""
    native = lang.diagram(bundle, view="circuit")
    known = {node.id for node in native.nodes if node.kind in ("input", "population")}
    pair = ("A", "B")
    a, b = pair
    symbols = {f"drive_{c}": f"X_{c}" for c in pair}
    symbols.update({f"PING_{c}_{p}": f"{p}_{c}" for c in pair for p in ("E", "I")})
    positions = {
        f"drive_{a}": (-360, 0),
        f"drive_{b}": (1010, 0),
        f"PING_{a}_E": (0, 0),
        f"PING_{b}_E": (650, 0),
        f"PING_{a}_I": (0, 360),
        f"PING_{b}_I": (650, 360),
    }
    routes = {
        f"PING_{a}_input_to_E": viz.EdgeLayout(source_port="right", target_port="left"),
        f"PING_{b}_input_to_E": viz.EdgeLayout(source_port="left", target_port="right"),
        f"PING_{a}_E_to_I": viz.EdgeLayout(
            source_port="bottom_left", target_port="top_left"
        ),
        f"PING_{a}_I_to_E": viz.EdgeLayout(
            source_port="top_right", target_port="bottom_right"
        ),
        f"PING_{b}_E_to_I": viz.EdgeLayout(
            source_port="bottom_right", target_port="top_right"
        ),
        f"PING_{b}_I_to_E": viz.EdgeLayout(
            source_port="top_left", target_port="bottom_left"
        ),
        f"PING_{a}_E_to_PING_{b}_E_K_EE": viz.EdgeLayout(
            source_port="upper_right", target_port="upper_left"
        ),
        f"PING_{a}_E_to_PING_{b}_I_K_EI": viz.EdgeLayout(
            source_port="cross_right",
            target_port="upper_left",
            routing="polyline",
            waypoints=((325, 180),),
        ),
    }
    for circuit, x in ((a, 0), (b, 650)):
        # Compact half-ellipses meet each card vertically, clear of its inputs.
        angles = np.linspace(0, np.pi, 33)
        routes[f"PING_{circuit}_E_to_E"] = viz.EdgeLayout(
            source_port="top_right",
            target_port="top_left",
            routing="polyline",
            waypoints=tuple(
                (x + 60 * np.cos(angle), -92 - 88 * np.sin(angle)) for angle in angles
            ),
        )
        routes[f"PING_{circuit}_I_to_I"] = viz.EdgeLayout(
            source_port="bottom_left",
            target_port="bottom_right",
            routing="polyline",
            waypoints=tuple(
                (x - 60 * np.cos(angle), 452 + 88 * np.sin(angle)) for angle in angles
            ),
        )
    for key, route in tuple(routes.items()):
        if "_K_E" in key:
            routes[key.rsplit("_K_", 1)[0]] = route
    title = "One-way coupled PING circuits"
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
        cross_left=(0, 0.9),
        cross_right=(1, 0.9),
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
                routes[edge.id],
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


def entrainment_figure(solutions, diagnostics, metrics, cfg):
    t = solutions["coupled"].time_ms
    last = diagnostics["coupled"]["peaks"]["B"]
    left, right = last[-(cfg["closeup_cycles"] + 1)], last[-1]
    visible = (t >= left) & (t <= right)
    inhibition_scale = max(
        solutions["coupled"].state["PING_B_I_to_E.conductance"][visible]
    )
    excitation_scale = max(
        solutions["coupled"].state["PING_A_E_to_PING_B_E_K_EE.conductance"][visible]
    )
    max_rate = max(
        max(s.state[f"PING_{c}_{p}.rate"])
        for s in solutions.values()
        for c, p in [("A", "E"), ("B", "E"), ("B", "I")]
    )
    max_output = max(max(s.state["PING_B_E.rate"][visible]) for s in solutions.values())
    max_inhibition = max(
        max(s.state["PING_B_I_to_E.conductance"][visible]) / inhibition_scale
        for s in solutions.values()
    )
    phase_max = max(np.nanmax(d["phase"]) for d in diagnostics.values())
    phase_min = min(np.nanmin(d["phase"]) for d in diagnostics.values())
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(15.5, 9.3),
        layout="constrained",
        gridspec_kw=dict(height_ratios=[1, 0.8, 1.25]),
    )
    for col, name in enumerate(("control", "coupled")):
        solution = solutions[name]
        d = diagnostics[name]
        m = metrics[name]
        role = "Uncoupled" if col == 0 else "Coupled: A → B from 500 ms"
        ax = axes[0, col]
        for key, color, label in [
            ("PING_A_E.rate", "#168047", "A E"),
            ("PING_B_E.rate", "#222222", "B E"),
            ("PING_B_I.rate", "#bc2942", "B I"),
        ]:
            ax.plot(t, solution.state[key], color=color, lw=0.75, label=label)
        ax.set(
            xlim=(0, 1500),
            ylim=(0, max_rate * 1.18),
            ylabel="population rate (Hz)",
            title=f"{chr(65 + col)}. {role}\nPopulation activity",
        )
        ax.legend(ncol=3, loc="upper right", fontsize=8)
        ax.axvline(500, color="#536477", ls=":", lw=1.1)
        ax = axes[1, col]
        ax.plot(t, d["phase"], color="#222222", lw=1.2)
        ax.axvline(500, color="#536477", ls=":", lw=1.1)
        ax.set(
            xlim=(0, 1500),
            ylim=(phase_min - 0.05, phase_max + 0.1),
            ylabel="A − B phase (cycles)",
            xlabel="time (ms)",
            title=f"{chr(67 + col)}. Relative phase",
        )
        ax.text(
            0.03,
            0.95,
            f"PLV {m['plv']:.4f} | drift {m['drift_hz']:.3f} cycles/s\nMeasured over 800–1,500 ms",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.92),
        )
        ax = axes[2, col]
        rate_axis = ax.twinx()
        if col:
            inhibition = solution.state["PING_B_I_to_E.conductance"]
            start, end = cfg["timing"]["window_ms"]
            reference = inhibition[(t >= start) & (t <= end)]
            threshold = reference.min() + cfg["timing"][
                "low_inhibition_fraction"
            ] * np.ptp(reference)
            ax.fill_between(
                t[visible],
                0,
                1,
                where=inhibition[visible] <= threshold,
                transform=ax.get_xaxis_transform(),
                color="#e5bc48",
                alpha=0.25,
                linewidth=0,
                zorder=-1,
                label="Low inhibition (excitability proxy)",
            )
            low = inhibition <= threshold
            closing = np.flatnonzero(low[:-1] & ~low[1:])
            for boundary in t[closing]:
                if left <= boundary <= right:
                    ax.axvline(boundary, color="#bb901f", ls=":", lw=1.0, alpha=0.85)
        ax.plot(
            t[visible],
            solution.state["PING_B_I_to_E.conductance"][visible] / inhibition_scale,
            color="#bc2942",
            lw=1.4,
            label="B inhibition",
        )
        ax.plot(
            t[visible],
            solution.state["PING_A_E_to_PING_B_E_K_EE.conductance"][visible]
            / excitation_scale,
            color="#168047",
            lw=1.5,
            label="A excitation (dots: peaks)" if col else "A excitation: zero",
        )
        if col:
            incoming = solution.state["PING_A_E_to_PING_B_E_K_EE.conductance"]
            reference_window = (t >= start) & (t <= end)
            peaks, _ = find_peaks(
                incoming,
                distance=round(
                    cfg["phase"]["min_peak_distance_ms"] / cfg["network"]["dt_ms"]
                ),
                prominence=0.1 * np.ptp(incoming[reference_window]),
            )
            peaks = peaks[(t[peaks] >= left) & (t[peaks] <= right)]
            ax.plot(
                t[peaks],
                incoming[peaks] / excitation_scale,
                "o",
                color="#168047",
                markeredgecolor="white",
                markeredgewidth=0.8,
                markersize=5.5,
                zorder=5,
            )
        rate_axis.plot(
            t[visible],
            solution.state["PING_B_E.rate"][visible],
            color="#222222",
            lw=1.4,
            label="B E output",
        )
        if col:
            for arrival in d["arrivals"]:
                if left <= arrival <= right:
                    ax.axvline(arrival, color="#168047", alpha=0.25, lw=0.8)
        ax.set(
            xlim=(left, right),
            ylim=(0, max_inhibition * 1.4),
            ylabel="conductance / respective coupled peak",
            xlabel="time (ms)",
            title=f"{chr(69 + col)}. Same four-cycle time interval",
        )
        rate_axis.set(ylim=(0, max_output * 1.4), ylabel="B E rate (Hz)")
        h, l = ax.get_legend_handles_labels()
        rh, rl = rate_axis.get_legend_handles_labels()
        ax.legend(
            h + rh,
            l + rl,
            fontsize=8,
            ncol=2 if col else 3,
            loc="upper left",
            framealpha=0.95,
        )
    fig.suptitle(
        f"Spiking one-way entrainment | A {25 + cfg['difference_hz'] / 2:g} / B {25 - cfg['difference_hz'] / 2:g} Hz/channel | coupling {cfg['coupling_us']:g} µS\nMatched initialization and drives; identical axes across columns",
        fontsize=13,
    )
    return fig


def present(*, source, run_id=None):
    started = perf_counter()
    analysed = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="analyse"
    )
    cfg = analysed.record["execution"]["configuration"]
    pin = analysed.record["inputs"]["compute"]
    computed = source_run(
        REPO / ".pingstore",
        pin["run_id"],
        experiment=recipe.SLUG,
        stage="compute",
        reference=pin,
    )
    if cfg != computed.record["execution"]["configuration"]:
        raise ValueError("lineage settings differ")
    results = load_json(analysed.file("results.json"))
    with np.load(analysed.file("traces.npz"), allow_pickle=False) as data:
        solutions = {
            name: SimpleNamespace(
                time_ms=data["time_ms"],
                state={
                    key.split("__", 1)[1]: data[key]
                    for key in data.files
                    if key.startswith(name + "__PING_")
                },
            )
            for name in ("control", "coupled")
        }
        diagnostics = {
            name: dict(
                phase=data[name + "__phase"],
                arrivals=data[name + "__arrivals"],
                peaks={"B": data[name + "__B_peaks"]},
            )
            for name in ("control", "coupled")
        }
    figure = entrainment_figure(solutions, diagnostics, results["metrics"], cfg)
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs=dict(analyse=analysed, compute=computed),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        for extension in ("png", "svg", "pdf"):
            figure.savefig(run.export / f"entrainment.{extension}", dpi=180)
        plt.close(figure)
        render_network(computed, run)
        write_json_atomic(run.export / "numbers.json", dict(results, configuration=cfg))
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
        analysed.check_unchanged()
        computed.check_unchanged()
    return run.run_id


def render_network(computed, run):
    bundle = lang.load_bundle(
        computed.restore_bundle("network.bundle", run.scratch / "diagram.bundle")
    )
    circuit = network_diagram(bundle)
    with matplotlib.rc_context({"savefig.bbox": None}):
        for suffix in ("svg", "png"):
            viz.render_diagram(circuit, run.export / f"circuit.{suffix}")


def redraw_network(*, presentation, run_id=None):
    started = perf_counter()
    previous = source_run(
        REPO / ".pingstore", presentation, experiment=recipe.SLUG, stage="present"
    )
    sources = {
        role: source_run(
            REPO / ".pingstore",
            previous.record["inputs"][role]["run_id"],
            reference=previous.record["inputs"][role],
            experiment=recipe.SLUG,
        )
        for role in ("compute", "analyse")
    }
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs=dict(sources, figure_source=previous),
        configuration=previous.record["execution"]["configuration"],
        run_id=run_id,
    ) as run:
        for path in previous.presentation.iterdir():
            shutil.copyfile(path, run.export / path.name)
        render_network(sources["compute"], run)
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
        for item in (*sources.values(), previous):
            item.check_unchanged()
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--source")
    source.add_argument("--redraw-from", dest="presentation")
    p.add_argument("--run-id")
    args = p.parse_args()
    if args.presentation:
        redraw_network(presentation=args.presentation, run_id=args.run_id)
    else:
        present(source=args.source, run_id=args.run_id)
