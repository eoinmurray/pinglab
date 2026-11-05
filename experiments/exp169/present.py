"""Render per-condition rasters, membrane/conductance traces and native circuits."""

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "libraries")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp169 import recipe
from experiments.exp169.analyse import read_evidence
from pingstore.library import source_run, stage_run
from snnlab import lang, viz


def diagnostic_figure(row, cfg, *, voltage_limits, conductance_limit, private=False):
    """Show the unchanged raster and one fixed E neuron's recorded dynamics."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.5, 1, 1]},
    )
    raster, voltage, conductance = axes
    e, i = row["e"], row["i"]
    for spikes, offset, colour in [
        (e, 0, theme.INK_BLACK),
        (i, e.shape[1], theme.DEEP_RED),
    ]:
        times, cells = np.nonzero(spikes)
        raster.scatter(
            times * cfg["dt_ms"],
            cells + offset,
            marker=".",
            s=3,
            color=colour,
            linewidths=0,
            rasterized=True,
        )
    raster.axhline(e.shape[1] - 0.5, color=theme.MUTED, lw=0.5)
    seconds = cfg["duration_ms"] / 1000
    rate_e = e.sum() / e.shape[1] / seconds
    rate_i = i.sum() / i.shape[1] / seconds
    source_label = "private input" if private else "I"
    raster.set(
        ylim=(-0.5, e.shape[1] + i.shape[1] - 0.5),
        ylabel="Neuron / input index" if private else "Neuron index",
        title=f"{row['name']}  |  E {rate_e:.1f} Hz  |  {source_label} {rate_i:.1f} Hz/channel",
    )
    raster.set_yticks([0, 512, e.shape[1] - 1, e.shape[1] + i.shape[1] - 1])
    time = np.arange(len(e)) * cfg["dt_ms"]
    voltage.plot(time, row["voltage_mv"], color=theme.INK_BLACK, lw=1)
    voltage.axhline(
        cfg["threshold_mv"], color=theme.MUTED, lw=0.8, ls="--", label="Threshold"
    )
    voltage.set(
        ylabel="Membrane potential (mV)",
        ylim=voltage_limits,
        title=f"E neuron {row['neuron_index']} (zero-based index)",
    )
    voltage.legend(loc="upper right")
    conductance.plot(
        time,
        row["g_excitatory_us"],
        color=theme.INK_BLACK,
        lw=1,
        label="Excitatory (AMPA)",
    )
    conductance.plot(
        time,
        row["g_inhibitory_us"],
        color=theme.DEEP_RED,
        lw=1,
        label="Inhibitory (GABA)",
    )
    conductance.set(
        ylabel="Conductance (µS)", xlabel="Time (ms)", ylim=(0, conductance_limit)
    )
    conductance.legend(loc="upper right")
    for ax in axes:
        ax.set_xlim(0, cfg["duration_ms"])
        ax.spines[["top", "right"]].set_visible(False)
    theme.label_panels(axes)
    return fig


def diagnostic_rows(computed, bank, rows, traces):
    """Bind every diagnostic to the same immutable trial as its raster."""
    pin = traces.record["inputs"]["compute"]
    if (
        pin["run_id"] != computed.record["run_id"]
        or pin["payload_digest"] != computed.record["payload_digest"]
    ):
        raise ValueError("Membrane traces belong to a different compute run")
    summary = json.loads(traces.file("summary.json").read_text())
    if not summary["original_outputs_exact"] or not summary["frozen_parameters_exact"]:
        raise ValueError("Diagnostic replay was not exact")
    neuron = summary["neuron_index"]
    for row, cell in zip(
        rows[:2], ("coba__off__seed42", "ping__off__seed42"), strict=True
    ):
        with np.load(bank.file("snapshots", cell, "recording.npz")) as z:
            row.update(
                voltage_mv=z["v_e_1"][:, neuron].copy(),
                g_excitatory_us=z["ge_e_1"][:, neuron].copy(),
                g_inhibitory_us=z["gi_e_1"][:, neuron].copy(),
                neuron_index=neuron,
            )
    with np.load(traces.file("traces.npz")) as z:
        rows[-1].update({key: z[key].copy() for key in z.files}, neuron_index=neuron)
    for row in rows:
        for key in ("voltage_mv", "g_excitatory_us", "g_inhibitory_us"):
            if row[key].shape != (len(row["e"]),) or not np.isfinite(row[key]).all():
                raise ValueError(f"Invalid {key} trace")
    return rows


def circuit(bundle, destination, *, reference=False):
    native = lang.diagram(bundle, view="circuit")
    positions = {"drive": (-440, 0), "E": (0, 0), "I": (0, 360), "readout": (440, 0)}
    if not reference:
        positions["artificial"] = (-440, 360)
    native = replace(
        native,
        title="TR02 reference topology"
        if reference
        else "Frozen COBA with private GABA inputs",
        groups=(),
        nodes=tuple(
            replace(
                node,
                detail=f"{next(row for row in bundle.graph['inputs'] if row['id'] == node.id)['shape'][-1]:,} spike channels",
            )
            if node.kind == "input"
            else node
            for node in native.nodes
            if node.id in positions
        ),
        edges=tuple(
            edge
            for edge in native.edges
            if edge.source in positions and edge.target in positions
        ),
    )
    labels = {
        "input_to_E": r"$W_{EX}^{C/P}$" if reference else r"$W_{EX}^{C}$",
        "E_to_I": r"$W_{IE}^{C/P}$" if reference else "$0$",
        "I_to_E": r"$W_{EI}^{C/P}$" if reference else "$0$",
        "readout_projection": r"$W_{RE}^{C/P}$" if reference else r"$W_{RE}^{C}$",
        "artificial_to_E": r"$q_j\delta_{jk}$",
    }
    routes = {
        "input_to_E": dict(source_port="right", target_port="left"),
        "E_to_I": dict(source_port="bottom_right", target_port="top_right"),
        "I_to_E": dict(source_port="top_left", target_port="bottom_left"),
        "readout_projection": dict(source_port="right", target_port="left"),
        "artificial_to_E": dict(
            source_port="top",
            target_port="lower_left",
            routing="polyline",
            waypoints=((-300, 180), (-190, 180)),
        ),
    }
    ports = {
        "top_left": (0.25, 0),
        "top_right": (0.75, 0),
        "bottom_left": (0.25, 1),
        "bottom_right": (0.75, 1),
        "lower_left": (0, 0.8),
    }
    layout = viz.DiagramLayout(
        nodes={
            key: viz.NodeLayout(position=pos, min_size=(200, 130), ports=ports)
            for key, pos in positions.items()
        },
        edges={
            edge.id: viz.EdgeLayout(
                **routes[edge.id], label=viz.DiagramLabel(labels[edge.id])
            )
            for edge in native.edges
        },
    )
    viz.render_diagram(native, destination, layout=layout)


def present(identity, *, traces_identity, run_id=None):
    started = perf_counter()
    analysis = source_run(
        ROOT / ".pingstore", identity, experiment=recipe.SLUG, stage="analyse"
    )
    pin = analysis.record["inputs"]["compute"]
    computed = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        experiment=recipe.SLUG,
        stage="compute",
        reference=pin,
    )
    results = json.loads(analysis.file("results.json").read_text())
    cfg, bank, rows, _, _, _ = read_evidence(computed)
    traces = source_run(
        ROOT / ".pingstore",
        traces_identity,
        experiment=recipe.SLUG,
        stage="compute",
        phase="membrane-traces",
    )
    cfg = dict(cfg)
    cfg["threshold_mv"] = traces.record["execution"]["configuration"]["threshold_mv"]
    rows = diagnostic_rows(computed, bank, rows, traces)
    results["diagnostic_neuron_index"] = rows[0]["neuron_index"]
    voltage_limits = (
        min(row["voltage_mv"].min() for row in rows) - 2,
        max(cfg["threshold_mv"], max(row["voltage_mv"].max() for row in rows)) + 2,
    )
    conductance_limit = 1.1 * max(
        row[key].max() for row in rows for key in ("g_excitatory_us", "g_inhibitory_us")
    )
    with stage_run(
        ROOT,
        recipe.SLUG,
        "present",
        inputs={"analyse": analysis, "compute": computed, "traces": traces},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        with matplotlib.rc_context():
            theme.set_paper_mode(True)
            theme.apply()
            for index, (stem, row) in enumerate(
                zip(("coba", "ping", "artificial"), rows, strict=True)
            ):
                fig = diagnostic_figure(
                    row,
                    cfg,
                    voltage_limits=voltage_limits,
                    conductance_limit=conductance_limit,
                    private=index == 2,
                )
                for ext in ("png", "svg"):
                    fig.savefig(run.export / (stem + "." + ext), dpi=200)
                plt.close(fig)
        bundle = lang.load_bundle(
            computed.restore_bundle("network.bundle", run.scratch / "control.bundle")
        )
        for ext in ("svg", "png"):
            circuit(bundle, run.export / ("circuit." + ext))
        authored = recipe.author_network(recipe.configuration())
        if authored.graph != bundle.graph:
            raise ValueError("Current recipe differs from retained native architecture")
        reference = recipe.author_network(recipe.configuration(), include_private=False)
        reference.write(run.scratch / "reference.bundle", visualise=False)
        shutil.make_archive(
            str(run.export / "reference.bundle"),
            "zip",
            run.scratch / "reference.bundle",
        )
        for ext in ("svg", "png"):
            circuit(
                reference, run.export / ("reference-circuit." + ext), reference=True
            )
        (run.export / "numbers.json").write_text(json.dumps(results, indent=2) + "\n")
        run.record["execution"].update(
            no_simulation=True, elapsed_seconds=perf_counter() - started
        )
        for source in (analysis, computed, bank, traces):
            source.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--traces", required=True, help="Exact membrane-traces compute replay"
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    present(args.source, traces_identity=args.traces, run_id=args.run_id)
