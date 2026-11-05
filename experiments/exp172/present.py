"""Render four spatial-relay figures from explicitly pinned analysis; no simulation."""

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
from snnlab import lang, viz
from snnlab.sim import NetworkSnapshot

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
from experiments.exp172.analyse import (
    fit_profile,
    materialise_compute,
    spatial_timecourse,
)
from experiments.exp172.recipe import gaussian
from pingstore.library import source_run, stage_run

sys.path.insert(0, str(REPO))
from experiments import theme


def circuit_diagram(bundle, parameters):
    """Style the saved native architecture; recording aliases are not circuit nodes."""
    native = lang.diagram(bundle, view="circuit")
    symbols = {f"{c}_{p}": f"{p}_{c}" for c in "ABC" for p in "EI"}
    symbols.update({f"X_{c}": f"X_{c}" for c in "ABC"})
    symbols.update({f"U_{c}": f"U_{c}" for c in "AB"})
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
        native, title="Spatial relay", nodes=nodes, edges=edges, groups=()
    )
    positions = {
        "A_E": (0, 0),
        "A_I": (0, -240),
        "B_E": (0, 720),
        "B_I": (0, 480),
        "C_E": (800, 360),
        "C_I": (800, 120),
        "X_A": (-420, 0),
        "U_A": (-420, 200),
        "X_B": (-420, 720),
        "U_B": (-420, 920),
        "X_C": (800, 820),
    }
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
        if edge.id.endswith("E_to_I") and edge.source[0] == edge.target[0]:
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
        population_fill=theme.PAPER,
        population_border=theme.INK_BLACK,
        input_fill=theme.PAPER_TINT,
        input_border=theme.GREY_MID,
        label_size=18,
        secondary_size=13,
    )
    with matplotlib.rc_context({"savefig.bbox": None}):
        return viz.resolve_diagram(diagram, layout=layout, theme=style)


def save_circuit(bundle, parameters, output):
    circuit = circuit_diagram(bundle, parameters)
    with matplotlib.rc_context({"savefig.bbox": None}):
        for suffix in ("svg", "png"):
            viz.render_diagram(circuit, output / f"circuit.{suffix}")


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
    """Continuous sender-minus-C phase; no modulo wrapping or phase reset."""
    t, phases, _, _ = phase_evidence(spikes, cfg)
    return t, {c: phases[c] - phases["C"] for c in "AB"}


def amplitude_panel(ax, course, cfg):
    rows = course["points"]
    for c, color in zip("AB", [theme.INK_BLACK, theme.DEEP_RED], strict=False):
        ax.plot(
            [r["time_ms"] for r in rows],
            [r[c + "_amplitude_hz"] for r in rows],
            color=color,
            lw=1.2,
            label=f"{c}-like amplitude",
        )
    ax.axvline(cfg["switch_ms"], color=theme.MUTED, ls="--", lw=1)
    for start, end in cfg["windows_ms"]:
        ax.axvspan(start, end, color=theme.GREY_LIGHT, alpha=0.25)
    ax.set(
        title=f"C spatial components · {course['window_ms']:g} ms windows, {course['hop_ms']:g} ms step",
        xlabel="Time (ms)",
        ylabel="Fitted Gaussian amplitude (Hz)",
        xlim=(0, cfg["duration_ms"]),
        ylim=(0, None),
    )
    ax.legend(loc="upper right", fontsize=theme.SIZE_ANNOTATION)


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
        criterion="Recorded representative cue and seed; originally selected on the displayed seed.",
        selection="recorded representative"
        if args.cue == cfg["representative_cue_hz"]
        else "explicit override",
        post_selection=True,
        averaging="none; one seed and one spatial assignment",
    )
    return metrics, selection_metadata, representative_seed


def plot_populations(
    spikes,
    profiles,
    cfg,
    swap,
    colors,
    raster_fig,
    raster_grid,
    raster_axes,
    profile_fig,
    profile_grid,
    profile_axes,
):
    e_axes = []
    i_axes = []
    for row, c in enumerate("ABC"):
        raster = raster_fig.add_subplot(raster_grid[0, row])
        raster_axes.append(raster)
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
        for col, p in [(1, "E"), (2, "I")]:
            ax = profile_fig.add_subplot(profile_grid[row, col - 1])
            profile_axes.append(ax)
            (e_axes if p == "E" else i_axes).append(ax)
            for epoch in range(2):
                y = profiles[f"epoch{epoch}_{c}_{p}"]
                x = np.arange(len(y)) + 1
                ax.plot(x, y, color=colors[epoch], lw=0.5, alpha=0.22)
                ax.plot(
                    x,
                    gaussian_filter1d(y, 2, mode="wrap"),
                    color=colors[epoch],
                    lw=1.3,
                    label=f"Attend {'AB'[epoch]}",
                )
                if p == "E":
                    sigma = np.sqrt(
                        cfg["spatial_sigma"] ** 2
                        + cfg["sensory_kernel_sigma"] ** 2
                        + (cfg["cross_kernel_sigma"] ** 2 if c == "C" else 0)
                    )
                    fit, pred, templates = fit_profile(
                        y,
                        (cfg["centres"][::-1] if swap else cfg["centres"]),
                        sigma,
                        cfg["minimum_total_amplitude_hz"],
                    )
                    ax.plot(x, pred, color=colors[epoch], ls="--", lw=1.7)
                    if c == "C":
                        for j in range(2):
                            ax.plot(
                                x,
                                fit["background_hz"]
                                + fit["amplitudes_hz"][j] * templates[:, j],
                                color=colors[epoch],
                                ls=":" if j == 0 else "-.",
                                lw=0.9,
                            )
            ax.set(
                title=f"{c}_{p} · spatial rate",
                xlabel="Neuron index",
                ylabel="Rate (Hz)",
                xlim=(1, len(y)),
                ylim=(0, None),
            )
            if row == 0:
                ax.legend(loc="upper right", fontsize=theme.SIZE_ANNOTATION)
    for group in [e_axes, i_axes]:
        upper = max(a.get_ylim()[1] for a in group)
        for a in group:
            a.set_ylim(0, upper)


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
    with np.load(directory / "profiles.npz") as f:
        profiles = dict(f)
    theme.apply()
    colors = [theme.INK_BLACK, theme.DEEP_RED]
    course = spatial_timecourse(spikes, cfg, args.swap)
    raster_fig = plt.figure(figsize=(17, 5), layout="constrained")
    raster_grid = raster_fig.add_gridspec(1, 3)
    profile_fig = plt.figure(figsize=(11, 10), layout="constrained")
    profile_grid = profile_fig.add_gridspec(3, 2)
    signal_fig = plt.figure(figsize=(17, 5), layout="constrained")
    signal_grid = signal_fig.add_gridspec(1, 3)
    weight_fig = plt.figure(figsize=(11, 5), layout="constrained")
    weight_grid = weight_fig.add_gridspec(1, 2)
    raster_axes = []
    profile_axes = []
    signal_axes = []
    plot_populations(
        spikes,
        profiles,
        cfg,
        args.swap,
        colors,
        raster_fig,
        raster_grid,
        raster_axes,
        profile_fig,
        profile_grid,
        profile_axes,
    )
    reference = signal_fig.add_subplot(signal_grid[0, 0])
    signal_axes.append(reference)
    x = (
        np.arange(cfg["network"]["populations"]["E"]["size"])
        / cfg["network"]["populations"]["E"]["size"]
    )
    for j, c in enumerate("AB"):
        reference.plot(
            x * len(x) + 1,
            gaussian(
                x,
                (cfg["centres"][::-1] if args.swap else cfg["centres"])[j],
                cfg["spatial_sigma"],
            ),
            color=colors[j],
            lw=1.8,
            label=f"{c} input reference",
        )
        broadened = np.sqrt(
            cfg["spatial_sigma"] ** 2
            + cfg["sensory_kernel_sigma"] ** 2
            + cfg["cross_kernel_sigma"] ** 2
        )
        reference.plot(
            x * len(x) + 1,
            gaussian(
                x, (cfg["centres"][::-1] if args.swap else cfg["centres"])[j], broadened
            ),
            color=colors[j],
            ls="--",
            lw=1,
            label=f"{c} C-fit template",
        )
    reference.set(
        title="Gaussian references / fit templates",
        xlabel="Equivalent E neuron index",
        ylabel="Normalised shape (unit peak)",
        xlim=(1, len(x)),
        ylim=(0, 1.3),
    )
    reference.legend(loc="upper center", ncol=2, fontsize=theme.SIZE_ANNOTATION)
    relative_ax = signal_fig.add_subplot(signal_grid[0, 1])
    signal_axes.append(relative_ax)
    t, relative = relative_phases(spikes, cfg)
    for c, color in zip("AB", colors, strict=False):
        relative_ax.plot(
            t, relative[c] / (2 * np.pi), color=color, lw=1, label=f"{c} − C"
        )
    relative_ax.axhline(0, color=theme.GREY_MID, ls=":", lw=0.7)
    relative_ax.axvline(cfg["switch_ms"], color=theme.MUTED, ls="--", lw=1)
    for start, end in cfg["windows_ms"]:
        relative_ax.axvspan(start, end, color=theme.GREY_LIGHT, alpha=0.25)
    relative_ax.set(
        title="Unwrapped E-population phase difference",
        xlabel="Time (ms)",
        ylabel="Sender phase − C phase (cycles)",
        xlim=(0, cfg["duration_ms"]),
    )
    relative_ax.legend(loc="upper right", fontsize=theme.SIZE_ANNOTATION)
    amplitude_ax = signal_fig.add_subplot(signal_grid[0, 2])
    signal_axes.append(amplitude_ax)
    amplitude_panel(amplitude_ax, course, cfg)
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
    for name, figure, panels in [
        ("rasters", raster_fig, raster_axes),
        ("profiles", profile_fig, profile_axes),
        ("selection", signal_fig, signal_axes),
        ("weights", weight_fig, matrix_axes),
    ]:
        theme.label_panels(panels)
        figure.savefig(output / f"{name}.png", dpi=240)
        figure.savefig(output / f"{name}.svg")
        plt.close(figure)
    return selection_metadata


def present(source, run_id=None, cue=None, seed=None):
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
    with stage_run(
        REPO,
        "exp172",
        "present",
        inputs={"analyse": analysed, "compute": computed},
        run_id=run_id,
        configuration=dict(
            cue=cue, seed=seed, spatial_window_ms=100.0, spatial_hop_ms=5.0
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
        (run.export / "numbers.json").write_text(
            json.dumps(
                dict(
                    configuration=cfg,
                    selection=selected,
                    stability_selection=stability_selection,
                    metrics=trial,
                    analysis=json.loads(analysed.file("results.json").read_text()),
                    matrix_stats=matrix_stats,
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
    present(**vars(parser.parse_args()))
