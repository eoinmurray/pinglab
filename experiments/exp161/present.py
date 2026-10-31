"""Render one cue-switch trajectory from explicitly pinned native evidence."""

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
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp161 import recipe
from experiments.exp161.analyse import read_trajectory
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from snnlab import lang, viz


def network_diagram(bundle):
    """Align A, C and B E/I pairs with compact, mirrored feedforward routes."""
    native = lang.diagram(bundle, view="circuit")
    positions = {
        "A_E": (-650, 0),
        "A_I": (-650, 360),
        "C_E": (0, 0),
        "C_I": (0, 360),
        "B_E": (650, 0),
        "B_I": (650, 360),
        "X_A": (-1010, 0),
        "U_A": (-1010, -400),
        "X_B": (1010, 0),
        "U_B": (1010, -400),
        "X_C": (0, -400),
    }
    symbols = {c + "_" + p: p + "_" + c for c in "ABC" for p in "EI"}
    symbols.update({key: key for key in positions if key.startswith(("X_", "U_"))})
    routes = {
        "sensory_A": viz.EdgeLayout(source_port="right", target_port="left"),
        "sensory_B": viz.EdgeLayout(source_port="left", target_port="right"),
        # The offset vertical feed clears C's recurrent loop.
        "sensory_C": viz.EdgeLayout(source_port="feed_bottom", target_port="feed_top"),
        "cue_A": viz.EdgeLayout(
            source_port="right",
            target_port="upper_left",
            routing="polyline",
            waypoints=((-830, -200),),
        ),
        "cue_B": viz.EdgeLayout(
            source_port="left",
            target_port="upper_right",
            routing="polyline",
            waypoints=((830, -200),),
        ),
        "A_E_to_C_E": viz.EdgeLayout(
            source_port="upper_right", target_port="upper_left"
        ),
        "B_E_to_C_E": viz.EdgeLayout(
            source_port="upper_left", target_port="upper_right"
        ),
        "A_E_to_C_I": viz.EdgeLayout(
            source_port="cross_right",
            target_port="upper_left",
            routing="polyline",
            waypoints=((-325, 180),),
        ),
        "B_E_to_C_I": viz.EdgeLayout(
            source_port="cross_left",
            target_port="upper_right",
            routing="polyline",
            waypoints=((325, 180),),
        ),
    }
    for c in "ABC":
        x, y = positions[c + "_E"]
        iy = positions[c + "_I"][1]
        outer, inner = ("right", "left") if c == "B" else ("left", "right")
        routes[c + "_E_to_I"] = viz.EdgeLayout(
            source_port="bottom_" + outer, target_port="top_" + outer
        )
        routes[c + "_I_to_E"] = viz.EdgeLayout(
            source_port="top_" + inner, target_port="bottom_" + inner
        )
        angles = np.linspace(0, np.pi, 33)
        routes[c + "_E_to_E"] = viz.EdgeLayout(
            source_port="top_right",
            target_port="top_left",
            routing="polyline",
            waypoints=tuple(
                (x + 60 * np.cos(a), y - 92 - 88 * np.sin(a)) for a in angles
            ),
        )
        routes[c + "_I_to_I"] = viz.EdgeLayout(
            source_port="bottom_left",
            target_port="bottom_right",
            routing="polyline",
            waypoints=tuple(
                (x - 60 * np.cos(a), iy + 92 + 88 * np.sin(a)) for a in angles
            ),
        )
    known = {node.id for node in native.nodes if node.kind in ("input", "population")}
    if known != set(positions) or {e.id for e in native.edges} != set(routes):
        raise ValueError("saved architecture differs from the authored diagram layout")
    diagram = replace(
        native,
        title="Competing PING senders and a shared receiver",
        groups=(),
        nodes=tuple(
            replace(node, detail=node.detail.splitlines()[0])
            for node in native.nodes
            if node.id in known
        ),
    )
    ports = dict(
        cross_left=(0, 0.9),
        cross_right=(1, 0.9),
        feed_bottom=(0.95, 1),
        feed_top=(0.95, 0),
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
            key: viz.NodeLayout(
                position=value,
                label=f"${symbols[key]}$",
                min_size=(240, 160),
                ports=ports,
            )
            for key, value in positions.items()
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


def render_network(computed, run):
    bundle = lang.load_bundle(
        computed.restore_bundle("network.bundle", run.scratch / "diagram.bundle")
    )
    diagram = network_diagram(bundle)
    with matplotlib.rc_context({"savefig.bbox": None}):
        for ext in ("svg", "png"):
            viz.render_diagram(diagram, run.export / ("circuit." + ext))


def competition_figure(sol, cfg, metrics, diag):
    t = sol.time_ms
    s = sol.state
    fig, axs = plt.subplots(3, 1, figsize=(13, 9), layout="constrained")
    colors = dict(A="#168047", B="#5367c4", C="#222222")
    for c in "ABC":
        axs[0].plot(t, s[c + "_E.rate"], color=colors[c], lw=0.85, label=c + " E")
    axs[0].set(ylabel="population rate (Hz)", title="A. Competing senders and receiver")
    axs[0].legend(ncol=3, loc="upper right")
    for c in "AB":
        axs[1].plot(
            t,
            diag["phases"][c] - diag["phases"]["C"],
            color=colors[c],
            label=c + " − C",
            lw=1.0,
        )
    axs[1].set(
        ylabel="phase difference (cycles)",
        title="B. Flat trace: locking; drifting trace: phase slips",
    )
    axs[1].legend(ncol=2)
    left = cfg["switch_ms"] - cfg["timing"]["closeup_before_ms"]
    right = cfg["switch_ms"] + cfg["timing"]["closeup_after_ms"]
    v = (t >= left) & (t <= right)
    gi = s["C_I_to_E.conductance"]
    late = t > cfg["timing"]["reference_start_ms"]
    threshold = gi[late].min() + cfg["low_inhibition_fraction"] * np.ptp(gi[late])
    axs[2].fill_between(
        t[v],
        0,
        1,
        where=gi[v] <= threshold,
        transform=axs[2].get_xaxis_transform(),
        color="#e5bc48",
        alpha=0.25,
        label="C low inhibition (proxy)",
    )
    scale = max(np.max(s[c + "_E_to_C_E.conductance"][late]) for c in "AB")
    for c in "AB":
        axs[2].plot(
            t[v],
            s[c + "_E_to_C_E.conductance"][v] / scale,
            color=colors[c],
            label=c + " arriving excitation",
        )
    axs[2].plot(t[v], gi[v] / np.max(gi[late]), color="#bc2942", label="C inhibition")
    axs[2].set(
        xlim=(left, right),
        ylabel="conductance / peak",
        xlabel="time (ms)",
        title="C. Cue-switch close-up; excitation shares one scale",
    )
    axs[2].legend(ncol=2, fontsize=8, loc="upper right")
    for ax in axs:
        ax.axvline(cfg["switch_ms"], color="#777777", ls="--", lw=1)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axs[:2]:
        ax.set_xlim(0, cfg["duration_ms"])
        ax.set_xlabel("time (ms)")
        for a, b in cfg["windows_ms"]:
            ax.axvspan(a, b, color="#888888", alpha=0.07)
    text = " | ".join(
        f"cue {r['cued']}: PLV A {r['senders']['A']['plv']:.3f}, B {r['senders']['B']['plv']:.3f}"
        for r in metrics["epochs"]
    )
    fig.suptitle(
        f"Mean-field competition | equal sensory 24 Hz/channel | constant cue {cfg['cue_hz']:g} Hz/channel, A → B at 1200 ms\ncoupling {cfg['coupling_us']:g} µS per pathway | C background {cfg['background_hz']:g} Hz/channel\n{text}",
        fontsize=11,
    )
    return fig


def present(*, source, run_id=None):
    started = perf_counter()
    analysed = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="analyse"
    )
    pin = analysed.record["inputs"]["compute"]
    computed = source_run(
        REPO / ".pingstore",
        pin["run_id"],
        reference=pin,
        experiment=recipe.SLUG,
        stage="compute",
    )
    cfg = analysed.record["execution"]["configuration"]
    if cfg != computed.record["execution"]["configuration"]:
        raise ValueError("lineage configuration differs")
    sol = read_trajectory(computed, cfg)
    results = load_json(analysed.file("results.json"))
    with np.load(analysed.file("phase.npz"), allow_pickle=False) as d:
        np.testing.assert_array_equal(d["time_ms"], sol.time_ms)
        diag = dict(phases={c: d[c + "_phase"] for c in "ABC"})
    fig = competition_figure(sol, cfg, results["metrics"], diag)
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs=dict(analyse=analysed, compute=computed),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        for ext in ("png", "svg", "pdf"):
            fig.savefig(run.export / ("competition." + ext), dpi=170)
        plt.close(fig)
        render_network(computed, run)
        write_json_atomic(run.export / "numbers.json", dict(results, configuration=cfg))
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
        analysed.check_unchanged()
        computed.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


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
        for role in ("analyse", "compute")
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
        for item in [previous, *sources.values()]:
            item.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--source")
    selection.add_argument("--redraw-from", dest="presentation")
    p.add_argument("--run-id")
    args = p.parse_args()
    if args.presentation:
        redraw_network(presentation=args.presentation, run_id=args.run_id)
    else:
        present(source=args.source, run_id=args.run_id)
