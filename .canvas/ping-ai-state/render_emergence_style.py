"""Render matched AI/PING recordings in the visual grammar of ping-emergence."""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))

from tools.snnviz import (  # noqa: E402
    FrameTimeline,
    exponential_trace,
    grid_layout,
    load_snnsim_recording,
    save_animation,
)
from tools.snnviz import (  # noqa: E402
    representative_frame as select_representative_frame,
)

STATE = os.environ.get("PINGLAB_STATE", "ai").lower()
if STATE not in {"ai", "ping", "transition"}:
    raise ValueError("PINGLAB_STATE must be 'ai', 'ping', or 'transition'")
RUN_DIR = ROOT / os.environ.get("PINGLAB_RUN", f"run-{STATE}-v1")
OUT = ROOT / os.environ.get(
    "PINGLAB_VIDEO",
    "ping-ai-state-v5-weight-ridgelines.mp4" if STATE == "ai" else f"{STATE}-state-v1-weight-ridgelines.mp4",
)
POSTER = ROOT / os.environ.get(
    "PINGLAB_POSTER",
    "ping-ai-state-v5-representative-frame.png" if STATE == "ai" else f"{STATE}-state-v1-representative-frame.png",
)
recording = load_snnsim_recording(RUN_DIR)
data = recording.signals
weights = np.load(RUN_DIR / "recurrent-weights.npz")
run_config = json.loads((RUN_DIR / "config.json").read_text())
transition_start_ms = float(run_config.get("transition_start_ms") or 1000.0)
transition_end_ms = float(run_config.get("transition_end_ms") or 2000.0)

dt = recording.dt_ms
e_spikes = data["spk_e"].astype(bool)
i_spikes = data["spk_i"].astype(bool)
v_e, v_i = data["v_e_1"], data["v_i_1"]
g_e, g_i = data["ge_e_1"], data["gi_e_1"]
n_steps, n_e = e_spikes.shape
n_i = i_spikes.shape[1]
weight_scales = {
    name: data[f"weight_scale_{name}"] if f"weight_scale_{name}" in data else np.ones(n_steps)
    for name in ("w_ee", "w_ei", "w_ie", "w_ii")
}

# Recreate the exact independent E/I drive event streams used by tools/snnsim.
drive_e_rate, drive_e_conductance = map(float, run_config["independent_drive"])
drive_i_rate, _drive_i_conductance = map(float, run_config["independent_drive_i"])
seed = int(run_config["seed"])
drive_e = (torch.rand(n_steps, n_e, generator=torch.Generator().manual_seed(seed + 1)) < drive_e_rate * dt / 1000).numpy()
drive_i = (torch.rand(n_steps, n_i, generator=torch.Generator().manual_seed(seed + 2)) < drive_i_rate * dt / 1000).numpy()

mean_v_e, mean_v_i = v_e.mean(1), v_i.mean(1)
mean_g_e, mean_g_i = g_e.mean(1), g_i.mean(1)
v_min = min(float(mean_v_e.min()), float(mean_v_i.min()))
v_max = max(float(mean_v_e.max()), float(mean_v_i.max()))
g_max = max(float(mean_g_e.max()), float(mean_g_i.max()))


drive_e_xy = grid_layout(n_e, columns=20, x_range=(0.285, 0.420), y_range=(0.425, 0.710))
e_xy = grid_layout(n_e, columns=20, x_range=(0.500, 0.700), y_range=(0.410, 0.725))
i_xy = grid_layout(n_i, columns=10, x_range=(0.805, 0.935), y_range=(0.470, 0.675))

# Each recurrent conductance factorises into the presynaptic spike trace and
# the fixed synaptic matrix. This retains exact source→target identity without
# materialising a multi-gigabyte T×source×target tensor.
trace_e = exponential_trace(e_spikes, dt_ms=dt, tau_ms=2.0)
trace_i = exponential_trace(i_spikes, dt_ms=dt, tau_ms=9.0)
trace_drive_e = exponential_trace(drive_e, dt_ms=dt, tau_ms=2.0)

drive_segments = np.stack([drive_e_xy, e_xy], axis=1)
drive_peak = max(float(trace_drive_e.max() * drive_e_conductance), 1e-12)


def projection(name, weight, source_xy, target_xy, source_trace, color):
    source, target = np.nonzero(weight)
    peak = float((weight[source, target] * source_trace.max(axis=0)[source]).max())
    peak *= float(weight_scales[name].max())
    return {
        "name": name,
        "source": source,
        "target": target,
        "weight": weight[source, target],
        "matrix": weight,
        "segments": np.stack([source_xy[source], target_xy[target]], axis=1),
        "trace": source_trace,
        "color": color,
        "peak": max(peak, 1e-12),
    }

BG, BLACK, RED, GREY = "#f3efe6", "#20201e", "#a62a24", "#77726a"
projections = (
    projection("w_ee", weights["w_ee"], e_xy, e_xy, trace_e, BLACK),
    projection("w_ei", weights["w_ei"], e_xy, i_xy, trace_e, BLACK),
    projection("w_ie", weights["w_ie"], i_xy, e_xy, trace_i, RED),
    projection("w_ii", weights["w_ii"], i_xy, i_xy, trace_i, RED),
)
fig, ax = plt.subplots(figsize=(12.8, 5.625), dpi=120)
fig.subplots_adjust(left=0.0, right=0.78, bottom=0.0, top=1.0)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Additive ridgeline: activity remains in the network view; one shared
# absolute log axis makes projection-scale differences spatially explicit.
ridge_ax = fig.add_axes([0.805, 0.175, 0.180, 0.625])
ridge_ax.set_facecolor("#ebe7df")
ridge_ax.text(0.0, 1.13, "D · recurrent weights", transform=ridge_ax.transAxes,
              color=BLACK, fontsize=13.0, weight="medium", ha="left", va="bottom")
ridge_ax.text(0.0, 1.065, "current nonzero weights · shared log axis",
              transform=ridge_ax.transAxes, color="#69655e", fontsize=8.8,
              ha="left", va="bottom")
ridge_ax.set_xscale("log")
ridge_ax.set_xlim(0.005, 2.2)
ridge_ax.set_ylim(-0.48, 3.58)
ridge_ax.set_xticks([0.01, 0.1, 1.0, 2.0], labels=["0.01", "0.1", "1", "2"])
ridge_ax.tick_params(axis="x", colors="#69655e", labelsize=8.0, length=3, pad=3)
ridge_ax.set_xlabel("synaptic weight (µS) · log scale", color="#4f4c47", fontsize=8.2, labelpad=5)
ridge_ax.set_yticks([3, 2, 1, 0], labels=[r"$W_{EE}$", r"$W_{EI}$", r"$W_{II}$", r"$W_{IE}$"])
ridge_ax.tick_params(axis="y", colors="#4f4c47", labelsize=10.0, length=0, pad=7)
for spine in ridge_ax.spines.values():
    spine.set_color("#aaa59c")
    spine.set_linewidth(0.55)
for tick in [0.01, 0.1, 1.0]:
    ridge_ax.axvline(tick, color="#c9c3b9", linewidth=0.55, linestyle=(0, (2, 3)), zorder=0)

log_edges = np.logspace(np.log10(0.005), np.log10(2.2), 45)
centres = np.sqrt(log_edges[:-1] * log_edges[1:])
ridge_order = (projections[0], projections[1], projections[3], projections[2])
dynamic_ridges = {}
for base, p in zip([3, 2, 1, 0], ridge_order):
    matrix = p["matrix"]
    nonzero = matrix[matrix > 0]
    counts, _ = np.histogram(nonzero, bins=log_edges)
    smooth = np.convolve(counts.astype(float), np.array([1, 2, 3, 2, 1]) / 9, mode="same")
    height = smooth / max(float(smooth.max()), 1.0) * 0.48
    initial_scale = float(weight_scales[p["name"]][0])
    x_values = centres * initial_scale
    fill = ridge_ax.fill_between(x_values, base, base + height, color=p["color"], alpha=0.28, zorder=2)
    line, = ridge_ax.plot(x_values, base + height, color=p["color"], linewidth=1.05, zorder=3)
    lo, median, hi = np.percentile(nonzero, [5, 50, 95])
    bar, = ridge_ax.plot([lo * initial_scale, hi * initial_scale], [base - 0.10, base - 0.10],
                         color=p["color"], linewidth=2.0, solid_capstyle="round", zorder=4)
    dot = ridge_ax.scatter([median * initial_scale], [base - 0.10], s=14, color=p["color"],
                           edgecolors=BG, linewidths=0.55, zorder=5)
    label = ridge_ax.annotate(f"{median * initial_scale:.2g} µS", (median * initial_scale, base + 0.50),
                              xytext=(0, 0), textcoords="offset points", ha="center", va="bottom",
                              color=p["color"], fontsize=7.4, weight="medium")
    dynamic_ridges[p["name"]] = dict(
        fill=fill, line=line, bar=bar, dot=dot, label=label, base=base,
        height=height, lo=lo, median=median, hi=hi,
    )
ridge_ax.text(0.98, 0.02, "area = distribution\nbar = 5–95% · dot = median",
              transform=ridge_ax.transAxes, ha="right", va="bottom",
              color="#4f4c47", fontsize=7.0)

state_heading = (
    "Balanced E–I network · asynchronous-irregular state"
    if STATE == "ai" else "Balanced E–I network · PING state"
    if STATE == "ping" else "Balanced E–I network · AI → PING transition"
)
state_subtitle = (
    "synthetic four-coupling circuit · fixed fan-in K≈10 · independent E/I drive"
    if STATE == "ai"
    else rf"{drive_e_rate:g} Hz independent drive · weight-only transition · $W_{{EE}}$ ×10.85"
)
ax.text(0.05, 0.945, state_heading,
        ha="left", va="top", color=BLACK, fontsize=17.0, weight="medium")
ax.text(0.05, 0.898, state_subtitle,
        ha="left", va="top", color="#69655e", fontsize=9.8, weight="medium")
time_text = ax.text(0.95, 0.945, "", ha="right", va="top", color=BLACK,
                    fontsize=12.5, family="monospace")
count_text = ax.text(0.95, 0.895, "", ha="right", va="top", color="#4f4c47", fontsize=9.0)

ax.text(0.135, 0.785, "A · population means", ha="center", color="#4f4c47", fontsize=10.0, weight="medium")
ax.text(0.352, 0.785, "independent drive · E cells", ha="center", color="#625f59", fontsize=10.0, weight="medium")
ax.text(0.600, 0.785, f"Excitatory · n={n_e}", ha="center", color=BLACK, fontsize=10.5, weight="medium")
ax.text(0.870, 0.785, f"Inhibitory · n={n_i}", ha="center", color=RED, fontsize=10.5, weight="medium")

drive_nodes = ax.scatter(drive_e_xy[:, 0], drive_e_xy[:, 1], s=4, facecolors=BG,
                         edgecolors=GREY, linewidths=0.35, zorder=2)
e_nodes = ax.scatter(e_xy[:, 0], e_xy[:, 1], s=5, color=BLACK, alpha=0.72, zorder=2)
i_nodes = ax.scatter(i_xy[:, 0], i_xy[:, 1], s=7, color=RED, alpha=0.80, zorder=2)
active_drive = ax.scatter([], [], s=34, facecolors=GREY, edgecolors=BG, linewidths=0.7, zorder=5)
active_e = ax.scatter([], [], s=48, facecolors="none", edgecolors=BLACK, linewidths=1.25, zorder=5)
active_i = ax.scatter([], [], s=58, facecolors="none", edgecolors=RED, linewidths=1.4, zorder=5)


def piston(x, label, color, scale_min, scale_max, digits):
    y, w, h = 0.445, 0.016, 0.245
    ax.add_patch(plt.Rectangle((x - w / 2, y), w, h, facecolor="#ebe7df",
                               edgecolor="#aaa59c", linewidth=0.75))
    fill = ax.add_patch(plt.Rectangle((x - w / 2 + 0.003, y), w - 0.006, 0.001,
                                      facecolor=color, alpha=0.22, edgecolor="none"))
    head = ax.add_patch(plt.Rectangle((x - w / 2 - 0.005, y), w + 0.010, 0.010,
                                      facecolor=color, edgecolor=BG, linewidth=0.55))
    ax.text(x, 0.722, label, ha="center", va="center", color=color, fontsize=8.5, weight="medium")
    ax.text(x, 0.696, f"{scale_max:.{digits}f}", ha="center", va="bottom",
            color="#77726a", fontsize=6.0, family="monospace")
    ax.text(x, 0.438, f"{scale_min:.{digits}f}", ha="center", va="top",
            color="#77726a", fontsize=6.0, family="monospace")
    value = ax.text(x, 0.405, "", ha="center", va="center", color=color,
                    fontsize=7.4, family="monospace", weight="medium")
    return fill, head, value, y, h - 0.010


p_ge = piston(0.060, r"$g_E$ µS", BLACK, 0.0, g_max, 2)
p_gi = piston(0.105, r"$g_I$ µS", RED, 0.0, g_max, 2)
p_ve = piston(0.165, r"$V_E$ mV", BLACK, v_min, v_max, 0)
p_vi = piston(0.210, r"$V_I$ mV", RED, v_min, v_max, 0)


def set_piston(p, level, text):
    fill, head, value, y, travel = p
    y1 = y + np.clip(level, 0, 1) * travel
    fill.set_height(max(y1 - y, 0.001))
    head.set_y(y1)
    value.set_text(text)


phase = ax.inset_axes([0.050, 0.075, 0.315, 0.275])
phase.set_facecolor("none")
phase.spines["top"].set_visible(False)
phase.spines["right"].set_visible(False)
pad_e, pad_i = np.ptp(mean_g_e) * 0.08, np.ptp(mean_g_i) * 0.08
phase.set_xlim(mean_g_e.min() - pad_e, mean_g_e.max() + pad_e)
phase.set_ylim(mean_g_i.min() - pad_i, mean_g_i.max() + pad_i)
phase.set_xticks(np.linspace(mean_g_e.min(), mean_g_e.max(), 3))
phase.set_yticks(np.linspace(mean_g_i.min(), mean_g_i.max(), 3))
phase.set_xticklabels([f"{value:.2f}" for value in np.linspace(mean_g_e.min(), mean_g_e.max(), 3)])
phase.set_yticklabels([f"{value:.2f}" for value in np.linspace(mean_g_i.min(), mean_g_i.max(), 3)])
phase.tick_params(colors="#69655e", labelsize=7.0, length=2, pad=2)
phase.grid(color="#c9c3b9", linewidth=0.45, linestyle=(0, (2, 3)), alpha=0.65)
phase.set_xlabel(r"mean $g_E$ (µS)", color="#625f59", fontsize=8.5, labelpad=3)
phase.set_ylabel(r"mean $g_I$ (µS)", color=RED, fontsize=8.5, labelpad=3)
phase.set_title("B · conductance trajectory · previous 40 ms", color="#4f4c47",
                fontsize=10.0, weight="medium", pad=4)
phase_point, = phase.plot([], [], "o", ms=4.5, color=RED, mec=BG, mew=0.7, zorder=5)
phase_segments = []
phase_direction = []

# Complete raster, matching ping-emergence: drive, E and I retain all spikes.
raster = ax.inset_axes([0.430, 0.075, 0.520, 0.275])
raster.set_facecolor("none")
for side in ("top", "right", "left"):
    raster.spines[side].set_visible(False)
raster.set_xlim(0, n_steps * dt)
raster.set_ylim(n_e * 2 + n_i + 5, -5)
raster.axhspan(0, n_e, color=GREY, alpha=0.035, zorder=0)
raster.axhspan(n_e, n_e * 2, color=BLACK, alpha=0.035, zorder=0)
raster.axhspan(n_e * 2, n_e * 2 + n_i, color=RED, alpha=0.055, zorder=0)
raster.axhline(n_e, color="#c9c3b9", linewidth=0.55)
raster.axhline(n_e * 2, color="#c9c3b9", linewidth=0.55)
raster.set_yticks([n_e / 2, n_e + n_e / 2, 2 * n_e + n_i / 2], ["Drive", "E", "I"])
raster.tick_params(axis="x", colors=GREY, labelsize=7.5, length=2)
raster.tick_params(axis="y", colors="#625f59", labelsize=8.0, length=0, pad=4)
raster.set_xlabel("simulation time (ms)", color=GREY, fontsize=8.5, labelpad=3)
raster.set_title(f"C · complete spike raster · {n_steps * dt:,.0f} ms", color="#4f4c47",
                 fontsize=10.0, weight="medium", pad=4)
for spikes, offset, color, size in ((drive_e, 0, GREY, 0.6), (e_spikes, n_e, BLACK, 0.7),
                                    (i_spikes, 2 * n_e, RED, 0.8)):
    t, c = np.nonzero(spikes)
    raster.scatter(t * dt, c + offset, s=size, color=color, marker="s", linewidths=0)
raster_cursor = raster.axvline(0, color=RED, lw=0.9, alpha=0.72)

trail_steps = int(round(40 / dt))
timeline = FrameTimeline.sample(n_steps, frames=300, dt_ms=dt)
frame_steps = timeline.steps
transmission_artists = []


def draw_transmissions(step):
    while transmission_artists:
        transmission_artists.pop().remove()

    # Independent E-cell drive is a private one-to-one projection. Its event
    # adds 2.10 µS to that E cell's AMPA conductance; the line persists with
    # the same decay as the simulated conductance state.
    drive_values = trace_drive_e[step] * drive_e_conductance
    drive_active = np.flatnonzero(drive_values > 1e-8)
    if drive_active.size:
        drive_strength = np.clip(drive_values[drive_active] / drive_peak, 0, 1)
        drive_rgba = np.tile(np.asarray(to_rgba(GREY)), (drive_active.size, 1))
        drive_rgba[:, 3] = 0.06 + 0.44 * np.sqrt(drive_strength)
        drive_lines = LineCollection(
            drive_segments[drive_active], colors=drive_rgba,
            linewidths=0.20 + 2.5 * drive_strength,
            capstyle="round", zorder=3,
        )
        ax.add_collection(drive_lines)
        transmission_artists.append(drive_lines)
    drive_events = np.flatnonzero(drive_e[step])
    if drive_events.size:
        starts = drive_segments[drive_events, 0]
        delta = drive_segments[drive_events, 1] - starts
        drive_arrows = ax.quiver(
            starts[:, 0], starts[:, 1], delta[:, 0], delta[:, 1],
            angles="xy", scale_units="xy", scale=1,
            width=0.00050, headwidth=4.4, headlength=5.2,
            headaxislength=4.6, color=to_rgba(GREY, 0.86),
            pivot="tail", zorder=4,
        )
        transmission_artists.append(drive_arrows)

    for p in projections:
        scale = float(weight_scales[p["name"]][step])
        values = p["weight"] * scale * p["trace"][step, p["source"]]
        active = np.flatnonzero(values > 1e-8)
        if active.size > 260:
            active = active[np.argpartition(values[active], -260)[-260:]]
        if active.size:
            strength = np.clip(values[active] / p["peak"], 0, 1)
            rgba = np.tile(np.asarray(to_rgba(p["color"])), (active.size, 1))
            rgba[:, 3] = 0.025 + 0.24 * np.sqrt(strength)
            lines = LineCollection(p["segments"][active], colors=rgba,
                                   linewidths=0.10 + 2.2 * strength,
                                   capstyle="round", zorder=3)
            ax.add_collection(lines)
            transmission_artists.append(lines)

        # Direction-bearing arrowheads mark transmissions initiated by spikes
        # at this timestep; line weight still follows the synaptic conductance.
        event = np.flatnonzero(e_spikes[step] if p["trace"] is trace_e else i_spikes[step])
        mask = np.isin(p["source"], event)
        ids = np.flatnonzero(mask)
        if ids.size:
            starts = p["segments"][ids, 0]
            delta = p["segments"][ids, 1] - starts
            arrows = ax.quiver(starts[:, 0], starts[:, 1], delta[:, 0], delta[:, 1],
                               angles="xy", scale_units="xy", scale=1,
                               width=0.00028, headwidth=4.0, headlength=4.8,
                               headaxislength=4.2, color=to_rgba(p["color"], 0.34),
                               pivot="tail", zorder=4)
            transmission_artists.append(arrows)


def voltage_sizes(v):
    return 3.0 + 25.0 * np.clip((v - v_min) / max(v_max - v_min, 1e-9), 0, 1) ** 1.35


def update_weight_ridges(step):
    for name, artists in dynamic_ridges.items():
        scale = float(weight_scales[name][step])
        x_values = centres * scale
        base = artists["base"]
        height = artists["height"]
        artists["line"].set_data(x_values, base + height)
        polygon = np.c_[
            np.r_[x_values, x_values[::-1]],
            np.r_[base + height, np.full_like(height, base)[::-1]],
        ]
        artists["fill"].set_verts([polygon])
        artists["bar"].set_data(
            [artists["lo"] * scale, artists["hi"] * scale],
            [base - 0.10, base - 0.10],
        )
        artists["dot"].set_offsets([[artists["median"] * scale, base - 0.10]])
        artists["label"].xy = (artists["median"] * scale, base + 0.50)
        artists["label"].set_text(f"{artists['median'] * scale:.2g} µS")


def update(frame):
    step = frame_steps[frame]
    draw_transmissions(step)
    update_weight_ridges(step)
    active_drive.set_offsets(drive_e_xy[drive_e[step]])
    active_e.set_offsets(e_xy[e_spikes[step]])
    active_i.set_offsets(i_xy[i_spikes[step]])
    e_nodes.set_sizes(voltage_sizes(v_e[step]))
    i_nodes.set_sizes(voltage_sizes(v_i[step]))

    set_piston(p_ve, (mean_v_e[step] - v_min) / (v_max - v_min), f"{mean_v_e[step]:.1f}")
    set_piston(p_vi, (mean_v_i[step] - v_min) / (v_max - v_min), f"{mean_v_i[step]:.1f}")
    set_piston(p_ge, mean_g_e[step] / g_max, f"{mean_g_e[step]:.3f}")
    set_piston(p_gi, mean_g_i[step] / g_max, f"{mean_g_i[step]:.3f}")

    while phase_segments:
        phase_segments.pop().remove()
    while phase_direction:
        phase_direction.pop().remove()
    start = max(0, step - trail_steps)
    points = np.c_[mean_g_e[start:step + 1], mean_g_i[start:step + 1]]
    if len(points) > 1:
        seg = np.stack([points[:-1], points[1:]], axis=1)
        rgba = np.tile(to_rgba(BLACK), (len(seg), 1))
        rgba[:, 3] = np.linspace(0.04, 0.70, len(seg))
        line = LineCollection(seg, colors=rgba, linewidths=1.1)
        phase.add_collection(line)
        phase_segments.append(line)
        if len(points) >= 3:
            arrow = phase.annotate("", xy=points[-1], xytext=points[-3],
                                   arrowprops=dict(arrowstyle="-|>", color=RED,
                                                   linewidth=1.0, mutation_scale=8),
                                   zorder=6)
            phase_direction.append(arrow)
    phase_point.set_data([mean_g_e[step]], [mean_g_i[step]])
    raster_cursor.set_xdata([step * dt, step * dt])

    n_de = int(drive_e[step].sum())
    n_di = int(drive_i[step].sum())
    n_es = int(e_spikes[step].sum())
    n_is = int(i_spikes[step].sum())
    time_text.set_text(f"t = {step * dt:06.2f} ms   frame {frame:03d}")
    if STATE == "transition":
        time_ms = step * dt
        phase_name = (
            "AI" if time_ms < transition_start_ms
            else "weight ramp" if time_ms < transition_end_ms
            else "PING"
        )
        count_text.set_text(
            f"{phase_name} · W_EE ×{weight_scales['w_ee'][step]:.2f} · "
            f"{n_de + n_di} drive · {n_es} E · {n_is} I spikes"
        )
    else:
        count_text.set_text(f"{n_de + n_di} drive · {n_es} E · {n_is} I spikes")
    return active_drive, active_e, active_i, e_nodes, i_nodes, raster_cursor, phase_point, time_text, count_text, *transmission_artists


# Export the sampled frame with the greatest simultaneous recurrent activity.
# This gives design iteration a representative view of both active neurons and
# source→target transmissions instead of an arbitrary final frame.
representative_frame = select_representative_frame(
    e_spikes[frame_steps], i_spikes[frame_steps]
)
update(representative_frame)
fig.savefig(POSTER, dpi=160, facecolor=BG)
save_animation(fig, update, OUT, frames=300, fps=25, bitrate=3800)
plt.close(fig)
print(OUT)
print(POSTER)
