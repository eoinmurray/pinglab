"""EXP099: richer-input PING simulation, retained metrics, and video."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(REPO),
    str(REPO / "tools"),
    str(REPO / "tools" / "snnsim"),
    str(Path(__file__).parent),
]

import infer  # noqa: E402
import models as snnsim_models  # noqa: E402, TID251
from bundle import load_graph_bundle, translate_cobanet_v1  # noqa: E402
from tools import snnlang as snn  # noqa: E402, TID251
from tools.snnsim.metrics import (  # noqa: E402, TID251
    rhythmicity_metrics,
    rolling_conductance_loop_score,
)
from tools.snnviz import (  # noqa: E402, TID251
    FrameTimeline,
    exponential_trace,
    grid_layout,
    load_snnsim_recording,
    save_animation,
)

from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp099"
DT_MS, DURATION_MS, SEED = 0.25, 2_000.0, 7
N_E, N_I = 400, 100
VIEW_START_MS, VIEW_END_MS = 300.0, 1_800.0
ONSET_MS, PEAK_MS, OFFSET_MS = 600.0, 850.0, 1_100.0
VIDEO = "richer-input-ai-to-intermittent-ping.mp4"
POSTER = "richer-input-ai-to-intermittent-ping.png"
SCALE = {"dt_ms": DT_MS, "t_ms": DURATION_MS, "n_e": N_E, "n_i": N_I, "seed": SEED}


# SNNLANG --------------------------------------------------------------------


def recurrent(mean: float, std: float) -> snn.LowerClampedNormal:
    return snn.LowerClampedNormal(
        mean, std, initial_zero_fraction=0.975, zeroing="exact_k"
    )


def background(tau: float, group: int, private: float, shared: float):
    return snn.BackgroundChannel(
        private=snn.ShotNoise(500, private, tau),
        shared=snn.GroupedShotNoise(80, shared, tau, group),
        heterogeneity=snn.CellDistribution(
            rate=snn.LowerClampedNormal(1.0, 0.1),
            amplitude=snn.LowerClampedNormal(1.0, 0.1),
        ),
    )


def author_network() -> snn.Bundle:
    """Author the latest canvas condition without depending on canvas files."""
    net = snn.Network("exp099_richer_input_ping", dt=DT_MS * snn.ms)
    source_e = net.input(
        "afferent_e", shape=("time", "batch", N_E), signal_type="spikes", unit="spike"
    )
    source_i = net.input(
        "afferent_i", shape=("time", "batch", N_E), signal_type="spikes", unit="spike"
    )
    cell = snn.components.ping(
        net,
        name="balanced_circuit",
        n_e=N_E,
        n_i=N_I,
        source_e=source_e,
        source_i=source_i,
        tau_gaba=9 * snn.ms,
        w_in_e=snn.Normal(0.08, 0.008),
        w_in_i=snn.Normal(0.02, 0.002),
        w_ee=recurrent(0.85, 0.255),
        w_ei=recurrent(0.6, 0.18),
        w_ie=recurrent(3.0, 0.9),
        w_ii=recurrent(0.4, 0.12),
    )
    readout = snn.readouts.MeanVoltage(
        source=cell.E.spikes,
        classes=10,
        name="state_readout",
        tau=2 * snn.ms,
        weight=snn.Normal(5.1, 3.8),
    )
    net.output("state_logits", readout)
    net.expose(cell.E.spikes, cell.I.spikes, name="populations")
    simulation = snn.SimulationSpec(
        spike_sources=[snn.CorrelatedPoissonAfferents(source_e, source_i, 10, 15, 15)],
        backgrounds=[
            snn.ConductanceBackground(
                cell.E, background(2, 25, 0.06, 0.02), background(9, 25, 0.03, 0.01)
            ),
            snn.ConductanceBackground(
                cell.I, background(2, 10, 0.03, 0.01), background(9, 10, 0.03, 0.01)
            ),
        ],
        weather=snn.StationaryRateWeather(tau_ms=250, std_fraction=0.12),
        afferent_wave=snn.TransientAfferentWave(
            ONSET_MS, PEAK_MS, OFFSET_MS, 1.2, 6.5, plateau_end_ms=PEAK_MS
        ),
    )
    return snn.compile(net, simulation=simulation, target="tools/snnsim")


# SNNSIM ---------------------------------------------------------------------


def simulate(root: Path) -> None:
    bundle = root / "network.bundle"
    run = root / "simulation"
    author_network().write(bundle, visualise=True)
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools/snnsim/tool.py"),
            "sim",
            "--bundle",
            str(bundle),
            "--t-ms",
            str(DURATION_MS),
            "--seed",
            str(SEED),
            "--out-dir",
            str(run),
            "--wipe-dir",
        ],
        cwd=REPO,
        check=True,
    )
    export_initialized_weights(bundle, run)
    shutil.copy2(bundle / "reports/expanded.svg", root / "network.svg")


def export_initialized_weights(bundle: Path, run: Path) -> None:
    """Recreate and retain the exact seeded matrices used by SNNSIM."""
    _, graph = load_graph_bundle(bundle)
    spec = translate_cobanet_v1(graph)
    infer._pin_run(spec.dt, DURATION_MS, seed=SEED)
    snnsim_models.N_IN = spec.input_size
    snnsim_models.N_INH = spec.hidden_size // 4
    snnsim_models.EXACT_K_INITIALIZATION = spec.exact_k_initialization
    network = infer.build_net(
        "ping",
        w_in=spec.w_in,
        w_in_i=spec.w_in_i,
        w_ee=spec.w_ee,
        w_ei=spec.w_ei,
        w_ie=spec.w_ie,
        w_ii=spec.w_ii,
        w_in_initial_zero_fraction=0.0,
        recurrent_initial_zero_fraction=spec.recurrent_initial_zero_fraction,
        device=infer._auto_device(),
        randomize_init=True,
        dales_law=True,
        hidden_sizes=[spec.hidden_size],
        readout_mode=spec.readout_mode,
        signed_readout=False,
        readout_bias=False,
        n_inh_per_layer=None,
        train_leak=False,
        adaptive_threshold=False,
    )
    np.savez_compressed(
        run / "recurrent-weights.npz",
        w_in_e=network.W_ff[0].detach().cpu().numpy(),
        w_in_i=network.W_in_i.detach().cpu().numpy(),
        w_ee=network.W_ee["1"].detach().cpu().numpy(),
        w_ei=network.W_ei["1"].detach().cpu().numpy(),
        w_ie=network.W_ie["1"].detach().cpu().numpy(),
        w_ii=network.W_ii["1"].detach().cpu().numpy(),
    )


def rhythm_trace(spikes: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    centres = np.arange(200.0, len(spikes) * dt - 200.0, 10.0)
    values = np.zeros(len(centres))
    half = round(200 / dt)
    for index, centre_ms in enumerate(centres):
        centre = round(centre_ms / dt)
        result = rhythmicity_metrics(spikes[centre - half : centre + half], dt)
        values[index] = float(result["contrast"] or 0.0)
    return centres, values


def summarize(root: Path) -> dict:
    record = load_snnsim_recording(root / "simulation")
    e = record.signals["spk_e"].astype(bool)
    i = record.signals["spk_i"].astype(bool)
    times, rhythm = rhythm_trace(e, record.dt_ms)
    return {
        "condition": "richer-input",
        "e_spikes": int(e.sum()),
        "i_spikes": int(i.sum()),
        "peak_rhythmicity": float(rhythm.max()),
        "peak_rhythmicity_time_ms": float(times[np.argmax(rhythm)]),
    }


# SNNVIZ ---------------------------------------------------------------------


def external_trace(data: dict, name: str, dt: float, tau: float, steps: int):
    events = data.get(name)
    if events is None:
        return np.zeros(steps)
    return exponential_trace(events, dt_ms=dt, tau_ms=tau).mean(axis=1)


def render_approximation(root: Path) -> None:
    run = root / "simulation"
    record = load_snnsim_recording(run)
    data, dt = record.signals, record.dt_ms
    weights = np.load(run / "recurrent-weights.npz")
    wave = json.loads((run / "config.json").read_text())["_simulation_recipe"][
        "afferent_wave"
    ]
    e, i = data["spk_e"].astype(bool), data["spk_i"].astype(bool)
    v_e, v_i = data["v_e_1"], data["v_i_1"]
    g_e, g_i = data["ge_e_1"], data["gi_e_1"]
    shared = data["input_afferent_shared"].astype(bool)
    aff_e = data["input_afferent_e_private"].astype(bool)
    aff_i = data["input_afferent_i_private"].astype(bool)
    weather = data["input_weather_scale"]
    shared_scale = data["input_afferent_shared_scale"]
    steps, times = len(e), np.arange(len(e)) * dt
    mean_v_e, mean_v_i = v_e.mean(1), v_i.mean(1)
    mean_g_e, mean_g_i = g_e.mean(1), g_i.mean(1)
    ext = {
        "E AMPA": external_trace(data, "input_excitatory_e_executed", dt, 2, steps),
        "E GABA": external_trace(data, "input_inhibitory_e_executed", dt, 9, steps),
        "I AMPA": external_trace(data, "input_excitatory_i_executed", dt, 2, steps),
        "I GABA": external_trace(data, "input_inhibitory_i_executed", dt, 9, steps),
    }
    rhythm_t, rhythm = rhythm_trace(e, dt)
    view = slice(round(VIEW_START_MS / dt), round(VIEW_END_MS / dt))
    loop = rolling_conductance_loop_score(
        mean_g_e[view], mean_g_i[view], dt, window_ms=40, stride_ms=5
    )
    loop_t = VIEW_START_MS + loop["times_ms"]
    loop_smooth = np.convolve(loop["raw"], np.ones(15) / 15, mode="same")
    lo, hi = np.percentile(loop_smooth, [10, 95])
    loop_score = np.clip((loop_smooth - lo) / max(float(hi - lo), 1e-12), 0, 1)

    bg, black, red, grey = "#f3efe6", "#20201e", "#a62a24", "#77726a"
    fig = plt.figure(figsize=(14.4, 8.1), dpi=120, facecolor=bg)
    main = fig.add_axes([0.035, 0.31, 0.93, 0.64])
    main.set(xlim=(0, 1), ylim=(0, 1))
    main.axis("off")
    main.text(0, 1.03, "A · AFFERENT INPUT → E–I CORE", fontsize=17, weight="medium")
    main.text(
        1, 1.03, "AI → intermittent PING → AI", ha="right", fontsize=11, color=red
    )
    status = main.text(1, 0.975, "", ha="right", va="top", fontsize=9, color=grey)

    shared_xy = grid_layout(N_E, columns=40, x_range=(0.34, 0.66), y_range=(0.82, 0.90))
    aff_e_xy = grid_layout(N_E, columns=20, x_range=(0.02, 0.20), y_range=(0.34, 0.68))
    aff_i_xy = grid_layout(N_E, columns=20, x_range=(0.80, 0.98), y_range=(0.34, 0.68))
    e_xy = grid_layout(N_E, columns=20, x_range=(0.27, 0.57), y_range=(0.26, 0.70))
    i_xy = grid_layout(N_I, columns=10, x_range=(0.64, 0.76), y_range=(0.36, 0.62))
    for xy, label, x, y, color in (
        (shared_xy, "SHARED → E + I", 0.50, 0.915, grey),
        (aff_e_xy, "E-TARGETING", 0.11, 0.705, black),
        (aff_i_xy, "I-TARGETING", 0.89, 0.705, red),
    ):
        main.scatter(
            xy[:, 0], xy[:, 1], s=2.3, facecolors=bg, edgecolors=color, linewidths=0.25
        )
        main.text(x, y, label, ha="center", fontsize=8, color=color)
    main.text(0.42, 0.73, f"EXCITATORY · n={N_E}", ha="center", fontsize=10)
    main.text(0.70, 0.65, f"INHIBITORY · n={N_I}", ha="center", fontsize=10, color=red)
    e_nodes = main.scatter(
        e_xy[:, 0], e_xy[:, 1], s=5, color=black, alpha=0.68, zorder=3
    )
    i_nodes = main.scatter(i_xy[:, 0], i_xy[:, 1], s=7, color=red, alpha=0.78, zorder=3)
    active = [
        main.scatter([], [], s=20, color=grey, edgecolors=bg, zorder=7),
        main.scatter([], [], s=20, color=black, edgecolors=bg, zorder=7),
        main.scatter([], [], s=20, color=red, edgecolors=bg, zorder=7),
        main.scatter(
            [], [], s=46, facecolors="none", edgecolors=black, linewidths=1.2, zorder=8
        ),
        main.scatter(
            [], [], s=56, facecolors="none", edgecolors=red, linewidths=1.3, zorder=8
        ),
    ]

    trace_e = exponential_trace(e, dt_ms=dt, tau_ms=2)
    trace_i = exponential_trace(i, dt_ms=dt, tau_ms=9)
    trace_s = exponential_trace(shared, dt_ms=dt, tau_ms=2)
    trace_ae = exponential_trace(aff_e, dt_ms=dt, tau_ms=2)
    trace_ai = exponential_trace(aff_i, dt_ms=dt, tau_ms=2)

    def projection(matrix, source_xy, target_xy, trace, color):
        source, target = np.nonzero(matrix)
        return (
            source,
            np.stack([source_xy[source], target_xy[target]], axis=1),
            matrix[source, target],
            trace,
            color,
        )

    projections = [
        projection(weights["w_ee"], e_xy, e_xy, trace_e, black),
        projection(weights["w_ei"], e_xy, i_xy, trace_e, black),
        projection(weights["w_ie"], i_xy, e_xy, trace_i, red),
        projection(weights["w_ii"], i_xy, i_xy, trace_i, red),
        projection(weights["w_in_e"], shared_xy, e_xy, trace_s, grey),
        projection(weights["w_in_i"], shared_xy, i_xy, trace_s, grey),
        projection(weights["w_in_e"], aff_e_xy, e_xy, trace_ae, grey),
        projection(weights["w_in_i"], aff_i_xy, i_xy, trace_ai, grey),
    ]
    lines: list[LineCollection] = []

    def small_panel(index: int, title: str):
        axis = fig.add_axes([0.035 + index * 0.133, 0.045, 0.115, 0.205])
        axis.set_title(title, fontsize=8.0, weight="medium", pad=4)
        axis.tick_params(labelsize=6.2, colors=grey, length=2, pad=2)
        axis.spines[["top", "right"]].set_visible(False)
        return axis

    means_ax = small_panel(0, "B · POPULATION MEANS")
    means_ax.plot(times, mean_v_e, color=black, lw=0.7, label="V E")
    means_ax.plot(times, mean_v_i, color=red, lw=0.7, label="V I")
    means_ax.set_xlim(VIEW_START_MS, VIEW_END_MS)
    means_ax.legend(frameon=False, fontsize=5.4)
    means_cursor = means_ax.axvline(0, color=red, lw=0.8)

    phase_ax = small_panel(1, "C · CONDUCTANCE LOOP")
    phase_ax.set(
        xlim=(mean_g_e.min(), mean_g_e.max()), ylim=(mean_g_i.min(), mean_g_i.max())
    )
    (phase_point,) = phase_ax.plot([], [], "o", ms=4, color=red)
    phase_lines: list[LineCollection] = []

    input_ax = small_panel(2, "D · INPUT CONDUCTANCE")
    for (label, values), color, style in zip(
        ext.items(), (black, red, grey, red), ("-", "-", "--", "--")
    ):
        input_ax.plot(times, values, color=color, ls=style, lw=0.65, label=label)
    input_ax.set_xlim(VIEW_START_MS, VIEW_END_MS)
    input_ax.legend(frameon=False, fontsize=4.4, ncol=2)
    input_cursor = input_ax.axvline(0, color=red, lw=0.8)

    metric_ax = small_panel(3, "E · PING METRICS")
    metric_ax.axvspan(ONSET_MS, OFFSET_MS, color=grey, alpha=0.08)
    metric_ax.plot(rhythm_t, rhythm, color=black, lw=1, label="R")
    metric_ax.plot(loop_t, loop_score, color=red, lw=0.9, label="L")
    metric_ax.set(xlim=(VIEW_START_MS, VIEW_END_MS), ylim=(0, 1))
    metric_ax.legend(frameon=False, fontsize=5.5)
    metric_cursor = metric_ax.axvline(0, color=red, lw=0.8)

    raster_ax = small_panel(4, "F · COMPLETE RASTER")
    offset = 0
    for spikes, color in (
        (shared, grey),
        (aff_e, black),
        (aff_i, red),
        (e, black),
        (i, red),
    ):
        t, cell = np.nonzero(spikes)
        raster_ax.scatter(
            t * dt, cell + offset, s=0.35, color=color, marker="s", linewidths=0
        )
        offset += spikes.shape[1]
    raster_ax.set(xlim=(VIEW_START_MS, VIEW_END_MS), ylim=(offset, 0), yticks=[])
    raster_cursor = raster_ax.axvline(0, color=red, lw=0.8)

    weight_ax = small_panel(5, "G · RECURRENT WEIGHTS")
    for name, color in (("w_ee", black), ("w_ei", black), ("w_ii", red), ("w_ie", red)):
        values = weights[name]
        weight_ax.hist(
            values[values > 0],
            bins=25,
            density=True,
            histtype="step",
            color=color,
            lw=0.8,
            label=name.upper(),
        )
    weight_ax.set_xscale("log")
    weight_ax.legend(frameon=False, fontsize=4.4)

    weather_ax = small_panel(6, "H · INPUT WEATHER")
    weather_ax.plot(times, weather, color=grey, lw=0.7, label="weather")
    weather_ax.plot(times, shared_scale, color=black, lw=0.9, label="shared wave")
    weather_ax.set_xlim(VIEW_START_MS, VIEW_END_MS)
    weather_ax.legend(frameon=False, fontsize=4.8)
    weather_cursor = weather_ax.axvline(0, color=red, lw=0.8)

    timeline = FrameTimeline.compose(
        [
            (round(VIEW_START_MS / dt), round(ONSET_MS / dt) - 1, 140),
            (round(ONSET_MS / dt), round(PEAK_MS / dt) - 1, 170),
            (round(PEAK_MS / dt), round(OFFSET_MS / dt) - 1, 170),
            (round(OFFSET_MS / dt), round(VIEW_END_MS / dt) - 1, 120),
        ],
        dt_ms=dt,
    )
    frame_steps, trail = timeline.steps, round(40 / dt)
    v_min, v_max = (
        min(float(v_e.min()), float(v_i.min())),
        max(float(v_e.max()), float(v_i.max())),
    )

    def update(frame: int):
        step = int(frame_steps[frame])
        while lines:
            lines.pop().remove()
        for index, (source, segments, weight, trace, color) in enumerate(projections):
            value = weight * trace[step, source]
            ids = np.flatnonzero(value > 1e-8)
            limit = 100 if index >= 4 else 240
            if len(ids) > limit:
                ids = ids[np.argpartition(value[ids], -limit)[-limit:]]
            if len(ids):
                strength = value[ids] / max(float(value[ids].max()), 1e-12)
                rgba = np.tile(np.asarray(to_rgba(color)), (len(ids), 1))
                rgba[:, 3] = (0.015 if index >= 4 else 0.03) + (
                    0.08 if index >= 4 else 0.22
                ) * np.sqrt(strength)
                artist = LineCollection(
                    segments[ids],
                    colors=rgba,
                    linewidths=0.08 + (0.6 if index >= 4 else 1.8) * strength,
                    zorder=2,
                )
                main.add_collection(artist)
                lines.append(artist)
        for artist, xy, spikes in zip(
            active,
            (shared_xy, aff_e_xy, aff_i_xy, e_xy, i_xy),
            (shared, aff_e, aff_i, e, i),
        ):
            artist.set_offsets(xy[spikes[step]])
        e_nodes.set_sizes(
            3 + 24 * np.clip((v_e[step] - v_min) / (v_max - v_min), 0, 1) ** 1.3
        )
        i_nodes.set_sizes(
            4 + 28 * np.clip((v_i[step] - v_min) / (v_max - v_min), 0, 1) ** 1.3
        )
        for cursor in (
            means_cursor,
            input_cursor,
            metric_cursor,
            raster_cursor,
            weather_cursor,
        ):
            cursor.set_xdata([step * dt, step * dt])
        while phase_lines:
            phase_lines.pop().remove()
        start = max(0, step - trail)
        points = np.c_[mean_g_e[start : step + 1], mean_g_i[start : step + 1]]
        if len(points) > 1:
            segments = np.stack([points[:-1], points[1:]], axis=1)
            colors = np.tile(to_rgba(black), (len(segments), 1))
            colors[:, 3] = np.linspace(0.05, 0.75, len(segments))
            artist = LineCollection(segments, colors=colors, linewidths=1)
            phase_ax.add_collection(artist)
            phase_lines.append(artist)
        phase_point.set_data([mean_g_e[step]], [mean_g_i[step]])
        time_ms = step * dt
        state = (
            "stationary AI"
            if time_ms < wave["onset_ms"]
            else "input rising"
            if time_ms < wave["peak_ms"]
            else "input falling"
            if time_ms < wave["offset_ms"]
            else "recovery"
        )
        status.set_text(
            f"t={time_ms:6.1f} ms · {state} · shared ×{shared_scale[step]:.2f} · {int(e[step].sum())} E · {int(i[step].sum())} I spikes"
        )
        return (*active, e_nodes, i_nodes, status, *lines)

    representative = int(
        np.argmin(np.abs(frame_steps - round(rhythm_t[np.argmax(rhythm)] / dt)))
    )
    update(representative)
    fig.savefig(root / POSTER, dpi=160, facecolor=bg)
    save_animation(
        fig, update, root / VIDEO, frames=len(frame_steps), fps=25, bitrate=3_800
    )
    plt.close(fig)


# Exact canvas visual migration ----------------------------------------------


def render(root: Path) -> None:
    """Render with the exact production visual grammar used by the canvas."""
    subprocess.run(
        [sys.executable, str(REPO / ".canvas/ping-ai-state/render_emergence_style.py")],
        cwd=REPO,
        env=os.environ
        | {
            "PINGLAB_STATE": "input",
            "PINGLAB_PACING": "story",
            "PINGLAB_CONDITION": "inside-band",
            "PINGLAB_RUN": str((root / "simulation").resolve()),
            "PINGLAB_VIDEO": str((root / VIDEO).resolve()),
            "PINGLAB_POSTER": str((root / POSTER).resolve()),
        },
        check=True,
    )


# EXPERIMENT ORCHESTRATION ---------------------------------------------------


def child(action: str, root: Path) -> None:
    subprocess.run(
        [sys.executable, __file__],
        cwd=REPO,
        env=os.environ | {"EXP099_ACTION": action, "EXP099_ROOT": str(root)},
        check=True,
    )


def refresh_web_assets(published: Path) -> None:
    """Refresh Demolab's static copies after the managed run has published."""
    destination = REPO / "assets" / SLUG
    destination.mkdir(parents=True, exist_ok=True)
    for filename in (VIDEO, POSTER, "network.svg"):
        temporary = destination / f".{filename}.staging"
        shutil.copy2(published / filename, temporary)
        os.replace(temporary, destination / filename)


def orchestrate() -> None:
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"experiment_run_id = {run_id}")
    with published_run(SLUG, run_id, scale=SCALE) as (_scratch, staging):
        child("simulate", staging)
        child("render", staging)
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload={
                "question": "Does richer input preserve or destabilize a reference PING state?",
                "results": {"richer-input": summarize(staging)},
                "disposition": "draft",
            },
        )
    refresh_web_assets(REPO / "artifacts" / "data" / SLUG)
    print(f"{SLUG} complete: {run_id}")


ACTION = os.environ.get("EXP099_ACTION")
if ACTION is None:
    orchestrate()
elif ACTION == "simulate":
    simulate(Path(os.environ["EXP099_ROOT"]))
elif ACTION == "render":
    render(Path(os.environ["EXP099_ROOT"]))
else:
    raise ValueError(f"unknown EXP099_ACTION: {ACTION}")
