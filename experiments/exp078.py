"""Experiment 078 — authored graph for a two-PING Arnold-tongue reproduction.

This first stage compiles and visualises the fixed network topology only. It
does not generate inputs, execute the simulator, or report scientific results.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import snnlang as snn  # noqa: E402, TID251

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp078"
DT_MS = 0.1
N_INPUT = 80
N_E = 80
N_I = 20
INPUT_WEIGHT = 0.2
COUPLING_REFERENCE = 1.0
DELAY_MS = 0.1

SCALE = {
    "stage": "graph definition and visualisation only",
    "dt_ms": DT_MS,
    "n_input_per_circuit": N_INPUT,
    "n_e_per_circuit": N_E,
    "n_i_per_circuit": N_I,
}


def author_network(*, coupling: float = COUPLING_REFERENCE) -> snn.Bundle:
    """Compile the fixed Lowet-style two-circuit topology."""
    net = snn.Network("lowet_two_ping", dt=DT_MS * snn.ms)
    drives = {
        name: net.input(
            f"drive_{name}",
            shape=("time", "batch", N_INPUT),
            signal_type="spikes",
            unit="spike",
        )
        for name in ("a", "b")
    }
    circuits = {}
    for name in ("a", "b"):
        circuit = snn.components.ping(
            net,
            name=name,
            n_e=N_E,
            n_i=N_I,
            source=None,
        )
        net.connect(
            drives[name],
            circuit.E.excitatory,
            name=f"drive_{name}_to_{name}_E",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.Constant(INPUT_WEIGHT),
            constraint=snn.NonNegative(),
            delay=DELAY_MS * snn.ms,
        )
        circuits[name] = circuit

    for source, target in (("a", "b"), ("b", "a")):
        net.connect(
            circuits[source].E.spikes,
            circuits[target].E.excitatory,
            name=f"{source}_E_to_{target}_E",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.Constant(coupling),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=DELAY_MS * snn.ms,
        )
        net.connect(
            circuits[source].E.spikes,
            circuits[target].I.excitatory,
            name=f"{source}_E_to_{target}_I",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.Constant(coupling),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=DELAY_MS * snn.ms,
        )

    net.expose(
        circuits["a"].E.spikes,
        circuits["a"].I.spikes,
        circuits["b"].E.spikes,
        circuits["b"].I.spikes,
        name="population",
    )
    return snn.compile(net, target="tools/snn")


def render_network_diagram(bundle: snn.Bundle, svg_path: Path, png_path: Path) -> None:
    """Render the authored graph as a legible two-circuit scientific schematic."""
    graph = bundle.graph
    population_sizes = {row["id"]: row["size"] for row in graph["populations"]}
    projection_ids = {row["id"] for row in graph["projections"]}
    expected = {
        "a_E_to_I", "a_I_to_E", "b_E_to_I", "b_I_to_E",
        "a_E_to_b_E", "a_E_to_b_I", "b_E_to_a_E", "b_E_to_a_I",
    }
    if not expected <= projection_ids:
        raise ValueError(f"diagram graph is missing projections: {sorted(expected - projection_ids)}")

    theme.apply()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 6)
    ax.axis("off")

    ink = theme.INK_BLACK
    red = theme.DEEP_RED
    muted = theme.GREY_MID
    border = "#B9C3CF"
    panel = "#F7F8FA"
    coupling = "#53657D"
    positions = {
        "drive_a": (2.25, 5.15), "a_I": (1.15, 2.65), "a_E": (3.35, 2.65),
        "drive_b": (7.75, 5.15), "b_E": (6.65, 2.65), "b_I": (8.85, 2.65),
    }

    def card(name: str, title: str, subtitle: str, *, width: float = 1.45) -> None:
        x, y = positions[name]
        box = FancyBboxPatch(
            (x - width / 2, y - 0.55), width, 1.1,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor="white", edgecolor=border, linewidth=1.5, zorder=4,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.13, title, ha="center", va="center", fontsize=10.5,
                fontweight="semibold", color=ink, zorder=5)
        ax.text(x, y - 0.22, subtitle, ha="center", va="center", fontsize=7.5,
                color=muted, zorder=5)

    for center, label in ((2.25, "CIRCUIT A"), (7.75, "CIRCUIT B")):
        ax.add_patch(FancyBboxPatch(
            (center - 2.05, 1.15), 4.1, 3.0,
            boxstyle="round,pad=0.1,rounding_size=0.16",
            facecolor=panel, edgecolor="#D8DEE7", linewidth=1.0, zorder=0,
        ))
        ax.text(center, 3.88, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color=muted)

    card("drive_a", "drive A", "independent Poisson\n80 channels", width=2.25)
    card("drive_b", "drive B", "independent Poisson\n80 channels", width=2.25)
    for name, label in (("a_E", "E_A"), ("a_I", "I_A"), ("b_E", "E_B"), ("b_I", "I_B")):
        card(
            name,
            f"${label}$",
            f"{population_sizes[name]} neurons",
            width=1.55,
        )

    def arrow(start, end, *, color=ink, style="-", rad=0.0, heads="-|>", width=1.8, z=2):
        patch = FancyArrowPatch(
            start, end, arrowstyle=heads, mutation_scale=13,
            connectionstyle=f"arc3,rad={rad}", color=color, linewidth=width,
            linestyle=style, shrinkA=5, shrinkB=5, zorder=z,
        )
        ax.add_patch(patch)

    def routed_arrow(points, *, color=coupling, style="--", width=1.8):
        path = MplPath(
            points,
            [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1),
        )
        ax.add_patch(FancyArrowPatch(
            path=path, arrowstyle="-|>", mutation_scale=13, color=color,
            linewidth=width, linestyle=style, shrinkA=4, shrinkB=4, zorder=1,
            joinstyle="round",
        ))

    # Independent input and local PING loops.
    arrow((2.25, 4.58), (3.15, 3.23), width=1.6)
    arrow((7.75, 4.58), (6.85, 3.23), width=1.6)
    arrow((2.50, 2.91), (2.00, 2.91), width=2.0)
    arrow((2.00, 2.39), (2.50, 2.39), color=red, heads="-[", width=2.2)
    arrow((7.50, 2.91), (8.00, 2.91), width=2.0)
    arrow((8.00, 2.39), (7.50, 2.39), color=red, heads="-[", width=2.2)

    # Lowet-style reciprocal cross-circuit excitation.
    arrow((4.15, 2.95), (5.85, 2.95), color=coupling, style="--",
          heads="<|-|>", width=2.0)
    ax.text(5.0, 3.13, "reciprocal E→E", ha="center", va="bottom",
            fontsize=8.5, color=coupling)
    routed_arrow([(3.35, 2.02), (3.35, 0.72), (8.85, 0.72), (8.85, 2.02)])
    routed_arrow([(6.65, 2.02), (6.65, 0.22), (1.15, 0.22), (1.15, 2.02)])
    ax.text(6.1, 0.88, "$E_A$ → $I_B$", ha="center", va="bottom",
            fontsize=8.0, color=coupling)
    ax.text(3.9, 0.38, "$E_B$ → $I_A$", ha="center", va="bottom",
            fontsize=8.0, color=coupling)

    ax.plot([], [], color=ink, linewidth=2, label="excitatory")
    ax.plot([], [], color=red, linewidth=2.2, label="inhibitory")
    ax.plot([], [], color=coupling, linewidth=2, linestyle="--", label="cross-circuit")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              frameon=False, fontsize=8.5, handlelength=2.4)

    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = parse_meta(sys.argv)
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id}")

    with published_run(
        SLUG,
        run_id,
        scale=SCALE,
        plot_only=meta.plot_only,
    ) as (scratch, staging):
        bundle = author_network()
        scratch_bundle = scratch / "network.bundle"
        if scratch_bundle.exists():
            shutil.rmtree(scratch_bundle)
        bundle.write(scratch_bundle, visualise=True)

        shutil.copytree(scratch_bundle, staging / "network.bundle")
        render_network_diagram(
            bundle,
            staging / "network.svg",
            staging / "network.png",
        )

        graph = bundle.graph
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload={
                "stage": "graph definition and visualisation only",
                "graph": {
                    "name": graph["name"],
                    "digest": bundle.manifest["graph_digest"],
                    "populations": len(graph["populations"]),
                    "projections": len(graph["projections"]),
                    "observables": len(graph["observables"]),
                },
                "config": {
                    **SCALE,
                    "input_weight": INPUT_WEIGHT,
                    "coupling_reference": COUPLING_REFERENCE,
                    "delay_ms": DELAY_MS,
                },
                "simulation_executed": False,
            },
        )


if __name__ == "__main__":
    main()
