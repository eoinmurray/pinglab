"""Production canvas layout restored from 27e9ba5^; see README for changes.

Only retained recordings and analysis are accepted. No simulation or estimator
runs here; exponential traces below encode animated transmission intensity.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from tools.snnviz import (  # noqa: TID251
    FrameTimeline,
    exponential_trace,
    grid_layout,
    save_animation,
)
from tools.snnviz import (  # noqa: TID251
    representative_frame as select_representative_frame,
)

from . import recipe

# Native browser video controls overlay the frame rather than extending the
# player. Keep this band free of plots, labels, and other evidence.
VIDEO_CONTROL_SAFE_BOTTOM = 0.15
DIAGNOSTIC_PANEL_BOTTOM = 0.19
DIAGNOSTIC_PANEL_HEIGHT = 0.20
NETWORK_FOOTER_Y = 0.415


def render(
    retained_recording,
    retained_weights: dict,
    measurements: dict,
    settings: dict,
    output: Path,
) -> None:
    theme.apply()
    STATE, PACING, CONDITION = "input", "story", "inside-band"
    OUT, POSTER = output / recipe.VIDEO, output / recipe.POSTER
    recording = retained_recording
    data = recording.signals
    weights = retained_weights
    run_config = recording.metadata["config"]
    transition_start_ms = float(run_config.get("transition_start_ms") or 1000.0)
    transition_end_ms = float(run_config.get("transition_end_ms") or 2000.0)
    wave_config = run_config["_simulation_recipe"]["afferent_wave"]
    input_onset_ms = float(wave_config["onset_ms"])
    input_peak_ms = float(wave_config["peak_ms"])
    input_offset_ms = float(wave_config["offset_ms"])
    view_start_ms = settings["view_start_ms"]
    view_end_ms = settings["view_end_ms"]

    dt = recording.dt_ms
    e_spikes = data["spk_e"].astype(bool)
    i_spikes = data["spk_i"].astype(bool)
    v_e, v_i = data["v_e_1"], data["v_i_1"]
    n_steps, n_e = e_spikes.shape
    n_i = i_spikes.shape[1]
    weight_scales = {
        name: data[f"weight_scale_{name}"]
        if f"weight_scale_{name}" in data
        else np.ones(n_steps)
        for name in ("w_ee", "w_ei", "w_ie", "w_ii")
    }
    shared_afferent_scale = data["input_afferent_shared_scale"]

    weather_inputs = combined_inputs = True
    afferent_shared = data["input_afferent_shared"].astype(bool)
    afferent_e_private = data["input_afferent_e_private"].astype(bool)
    afferent_i_private = data["input_afferent_i_private"].astype(bool)
    drive_e = data["input_structured_spikes_e"].astype(bool)
    drive_i = data["input_structured_spikes_i"].astype(bool)
    weather_scale = data["input_weather_scale"]
    drive_e_rate = float(np.count_nonzero(drive_e) / drive_e.size * 1000 / dt)
    drive_e_conductance = 1.0

    def retained_events(name: str, size: int) -> np.ndarray:
        return data[name] > 0

    background = {
        "e_exc_private": retained_events("input_excitatory_e_private", n_e),
        "e_inh_private": retained_events("input_inhibitory_e_private", n_e),
        "i_exc_private": retained_events("input_excitatory_i_private", n_i),
        "i_inh_private": retained_events("input_inhibitory_i_private", n_i),
        "e_exc_shared": retained_events("input_excitatory_e_shared", n_e),
        "e_inh_shared": retained_events("input_inhibitory_e_shared", n_e),
        "i_exc_shared": retained_events("input_excitatory_i_shared", n_i),
        "i_inh_shared": retained_events("input_inhibitory_i_shared", n_i),
    }

    external_conductance = {
        name: measurements[name] for name in ("E AMPA", "E GABA", "I AMPA", "I GABA")
    }
    mean_v_e, mean_v_i = measurements["mean_v_e"], measurements["mean_v_i"]
    mean_g_e, mean_g_i = measurements["mean_g_e"], measurements["mean_g_i"]
    v_min = min(float(mean_v_e.min()), float(mean_v_i.min()))
    v_max = max(float(mean_v_e.max()), float(mean_v_i.max()))
    g_max = max(float(mean_g_e.max()), float(mean_g_i.max()))

    drive_e_xy = grid_layout(
        n_e, columns=20, x_range=(0.06, 0.24), y_range=(0.45, 0.72)
    )
    shared_xy = grid_layout(n_e, columns=40, x_range=(0.38, 0.72), y_range=(0.76, 0.84))
    afferent_e_xy = grid_layout(
        n_e, columns=20, x_range=(0.06, 0.24), y_range=(0.45, 0.72)
    )
    afferent_i_xy = grid_layout(
        n_e, columns=20, x_range=(0.82, 0.98), y_range=(0.45, 0.72)
    )
    e_xy = grid_layout(n_e, columns=20, x_range=(0.29, 0.60), y_range=(0.43, 0.74))
    i_xy = grid_layout(n_i, columns=10, x_range=(0.66, 0.80), y_range=(0.47, 0.69))

    # Each recurrent conductance factorises into the presynaptic spike trace and
    # the fixed synaptic matrix. This retains exact source→target identity without
    # materialising a multi-gigabyte T×source×target tensor.
    trace_e = exponential_trace(e_spikes, dt_ms=dt, tau_ms=2.0)
    trace_i = exponential_trace(i_spikes, dt_ms=dt, tau_ms=9.0)
    trace_drive_e = exponential_trace(drive_e, dt_ms=dt, tau_ms=2.0)
    trace_afferent_shared = exponential_trace(afferent_shared, dt_ms=dt, tau_ms=2.0)
    trace_afferent_e_private = exponential_trace(
        afferent_e_private, dt_ms=dt, tau_ms=2.0
    )
    trace_afferent_i_private = exponential_trace(
        afferent_i_private, dt_ms=dt, tau_ms=2.0
    )

    drive_segments = np.stack([drive_e_xy, e_xy], axis=1)
    drive_peak = max(float(trace_drive_e.max() * drive_e_conductance), 1e-12)

    def projection(name, weight, source_xy, target_xy, source_trace, color):
        source, target = np.nonzero(weight)
        peak = float((weight[source, target] * source_trace.max(axis=0)[source]).max())
        peak *= float(weight_scales.get(name, np.ones(1)).max())
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

    BG = theme.PAPER
    BLACK = theme.INK_BLACK
    RED = theme.DEEP_RED
    GREY = theme.GREY_MID
    projections = (
        projection("w_ee", weights["w_ee"], e_xy, e_xy, trace_e, BLACK),
        projection("w_ei", weights["w_ei"], e_xy, i_xy, trace_e, BLACK),
        projection("w_ie", weights["w_ie"], i_xy, e_xy, trace_i, RED),
        projection("w_ii", weights["w_ii"], i_xy, i_xy, trace_i, RED),
    )
    input_e_projection = (
        projection("w_in_e", weights["w_in_e"], drive_e_xy, e_xy, trace_drive_e, GREY)
        if combined_inputs and "w_in_e" in weights
        else None
    )
    input_weather_projections = (
        {
            "shared_e": projection(
                "w_in_e",
                weights["w_in_e"],
                shared_xy,
                e_xy,
                trace_afferent_shared,
                GREY,
            ),
            "shared_i": projection(
                "w_in_i",
                weights["w_in_i"],
                shared_xy,
                i_xy,
                trace_afferent_shared,
                GREY,
            ),
            "private_e": projection(
                "w_in_e",
                weights["w_in_e"],
                afferent_e_xy,
                e_xy,
                trace_afferent_e_private,
                GREY,
            ),
            "private_i": projection(
                "w_in_i",
                weights["w_in_i"],
                afferent_i_xy,
                i_xy,
                trace_afferent_i_private,
                GREY,
            ),
        }
        if weather_inputs and {"w_in_e", "w_in_i"} <= set(weights)
        else {}
    )
    fig, ax = plt.subplots(figsize=(14.4, 8.1), dpi=120)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Mask at figure level so artists from any axes cannot leak into the area
    # reserved for native playback controls.
    fig.add_artist(
        plt.Rectangle(
            (0.0, 0.0),
            1.0,
            VIDEO_CONTROL_SAFE_BOTTOM,
            transform=fig.transFigure,
            facecolor=BG,
            edgecolor="none",
            zorder=10_000,
            clip_on=False,
        )
    )
    # Additive ridgeline: activity remains in the network view; one shared
    # absolute log axis makes projection-scale differences spatially explicit.
    ridge_ax = fig.add_axes(
        [0.845, DIAGNOSTIC_PANEL_BOTTOM, 0.135, DIAGNOSTIC_PANEL_HEIGHT]
    )
    ridge_ax.set_facecolor(BG)
    ridge_ax.text(
        0.0,
        1.04,
        "G · recurrent weights",
        transform=ridge_ax.transAxes,
        color=BLACK,
        fontsize=8.8,
        weight="medium",
        ha="left",
        va="bottom",
    )
    ridge_ax.text(
        0.0,
        0.925,
        "current nonzero weights",
        transform=ridge_ax.transAxes,
        color=theme.MUTED,
        fontsize=6.5,
        ha="left",
        va="bottom",
    )
    ridge_ax.set_xscale("log")
    ridge_ax.set_xlim(0.005, 2.2)
    ridge_ax.set_ylim(-0.48, 3.90)
    ridge_ax.set_xticks([0.01, 0.1, 1.0, 2.0], labels=["0.01", "0.1", "1", "2"])
    ridge_ax.tick_params(axis="x", colors=theme.MUTED, labelsize=7.0, length=2, pad=3)
    ridge_ax.set_xlabel(
        "synaptic weight (µS)", color=theme.DIM, fontsize=7.5, labelpad=3
    )
    ridge_ax.set_yticks(
        [3, 2, 1, 0], labels=[r"$W_{EE}$", r"$W_{EI}$", r"$W_{II}$", r"$W_{IE}$"]
    )
    ridge_ax.tick_params(axis="y", colors=theme.DIM, labelsize=8.0, length=0, pad=5)
    for spine in ridge_ax.spines.values():
        spine.set_color(theme.GREY_LIGHT)
        spine.set_linewidth(0.55)
    for tick in [0.01, 0.1, 1.0]:
        ridge_ax.axvline(
            tick, color=theme.RULE_WARM, linewidth=0.55, linestyle=(0, (2, 3)), zorder=0
        )

    log_edges = np.logspace(np.log10(0.005), np.log10(2.2), 45)
    centres = np.sqrt(log_edges[:-1] * log_edges[1:])
    ridge_order = (projections[0], projections[1], projections[3], projections[2])
    dynamic_ridges = {}
    for base, p in zip([3, 2, 1, 0], ridge_order):
        matrix = p["matrix"]
        nonzero = matrix[matrix > 0]
        counts, _ = np.histogram(nonzero, bins=log_edges)
        smooth = np.convolve(
            counts.astype(float), np.array([1, 2, 3, 2, 1]) / 9, mode="same"
        )
        height = smooth / max(float(smooth.max()), 1.0) * 0.48
        initial_scale = float(weight_scales[p["name"]][0])
        x_values = centres * initial_scale
        fill = ridge_ax.fill_between(
            x_values, base, base + height, color=p["color"], alpha=0.28, zorder=2
        )
        (line,) = ridge_ax.plot(
            x_values, base + height, color=p["color"], linewidth=1.05, zorder=3
        )
        lo, median, hi = np.percentile(nonzero, [5, 50, 95])
        (bar,) = ridge_ax.plot(
            [lo * initial_scale, hi * initial_scale],
            [base - 0.10, base - 0.10],
            color=p["color"],
            linewidth=2.0,
            solid_capstyle="round",
            zorder=4,
        )
        dot = ridge_ax.scatter(
            [median * initial_scale],
            [base - 0.10],
            s=14,
            color=p["color"],
            edgecolors=BG,
            linewidths=0.55,
            zorder=5,
        )
        label = ridge_ax.annotate(
            f"{median * initial_scale:.2g} µS",
            (median * initial_scale, base + 0.30),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=p["color"],
            fontsize=7.8,
            weight="medium",
        )
        dynamic_ridges[p["name"]] = dict(
            fill=fill,
            line=line,
            bar=bar,
            dot=dot,
            label=label,
            base=base,
            height=height,
            lo=lo,
            median=median,
            hi=hi,
        )
    state_heading = (
        "Balanced E–I network · asynchronous-irregular state"
        if STATE == "ai"
        else "Balanced E–I network · PING state"
        if STATE == "ping"
        else "Balanced E–I network · transient afferent input"
        if STATE == "input" and CONDITION == "inside-band"
        else "Balanced E–I network · AI → transient PING → AI"
        if STATE == "input"
        else "Balanced E–I network · AI → PING transition"
    )
    input_description = (
        "stationary weather · shared/private afferents · local AMPA/GABA"
        if weather_inputs
        else "authenticated afferent spikes + private/shared AMPA/GABA"
        if combined_inputs
        else f"{drive_e_rate:g} Hz independent drive"
    )
    if STATE == "input":
        w_ee_nonzero = weights["w_ee"][weights["w_ee"] > 0]
        w_ee_typical = float(np.median(w_ee_nonzero))
        state_subtitle = (
            f"fixed recurrent weights · median W_EE={w_ee_typical:.2f} µS · "
            "smooth coherent afferent bout over stationary weather"
        )
    elif STATE == "ai":
        state_subtitle = (
            f"synthetic four-coupling circuit · fixed fan-in K≈10 · {input_description}"
        )
    else:
        state_subtitle = (
            rf"{input_description} · weight-only transition · $W_{{EE}}$ ×10.85"
        )
    ax.text(
        0.05,
        0.945,
        state_heading,
        ha="left",
        va="top",
        color=BLACK,
        fontsize=17.0,
        weight="medium",
    )
    ax.text(
        0.05,
        0.898,
        state_subtitle,
        ha="left",
        va="top",
        color=theme.MUTED,
        fontsize=8.8,
        weight="medium",
    )
    time_text = ax.text(
        0.95,
        0.945,
        "",
        ha="right",
        va="top",
        color=BLACK,
        fontsize=12.5,
        family="monospace",
    )
    count_text = ax.text(
        0.95, 0.890, "", ha="right", va="top", color=theme.DIM, fontsize=8.4
    )

    if weather_inputs:
        ax.text(
            0.535,
            NETWORK_FOOTER_Y,
            "sampled active paths · + private AMPA · × private GABA · translucent wash = locally shared",
            ha="center",
            color=GREY,
            fontsize=7.2,
        )
    input_heading = (
        "input weather · excitatory afferents"
        if weather_inputs
        else "authenticated afferent spikes · E + I"
        if combined_inputs
        else "independent drive · E cells"
    )
    ax.text(
        0.035,
        0.850,
        "A · " + input_heading + " → E–I core",
        ha="left",
        color=theme.MUTED,
        fontsize=10.8,
        weight="medium",
    )
    ax.text(
        0.445,
        0.745,
        f"Excitatory · n={n_e}",
        ha="center",
        color=BLACK,
        fontsize=10.5,
        weight="medium",
    )
    ax.text(
        0.730,
        0.690,
        f"Inhibitory · n={n_i}",
        ha="center",
        color=RED,
        fontsize=10.5,
        weight="medium",
    )

    if weather_inputs:
        for label, xy, x, y in (
            ("shared → E + I", shared_xy, 0.550, 0.845),
            ("E-targeting", afferent_e_xy, 0.150, 0.715),
            ("I-targeting", afferent_i_xy, 0.900, 0.715),
        ):
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=2.6,
                facecolors=BG,
                edgecolors=GREY,
                linewidths=0.28,
                zorder=2,
            )
            ax.text(x, y, label, ha="center", va="bottom", color=GREY, fontsize=8.2)
        _drive_nodes = None
    else:
        _drive_nodes = ax.scatter(
            drive_e_xy[:, 0],
            drive_e_xy[:, 1],
            s=4,
            facecolors=BG,
            edgecolors=GREY,
            linewidths=0.35,
            zorder=2,
        )
    e_nodes = ax.scatter(e_xy[:, 0], e_xy[:, 1], s=5, color=BLACK, alpha=0.72, zorder=2)
    i_nodes = ax.scatter(i_xy[:, 0], i_xy[:, 1], s=7, color=RED, alpha=0.80, zorder=2)
    active_drive = ax.scatter(
        [], [], s=34, facecolors=GREY, edgecolors=BG, linewidths=0.7, zorder=5
    )
    active_shared = ax.scatter(
        [], [], s=18, facecolors=GREY, edgecolors=BG, linewidths=0.5, zorder=5
    )
    active_afferent_e = ax.scatter(
        [], [], s=18, facecolors=BLACK, edgecolors=BG, linewidths=0.5, zorder=5
    )
    active_afferent_i = ax.scatter(
        [], [], s=18, facecolors=RED, edgecolors=BG, linewidths=0.5, zorder=5
    )
    active_e = ax.scatter(
        [], [], s=48, facecolors="none", edgecolors=BLACK, linewidths=1.25, zorder=5
    )
    active_i = ax.scatter(
        [], [], s=58, facecolors="none", edgecolors=RED, linewidths=1.4, zorder=5
    )

    # Background conductance is shown at its target, not as fictitious source edges.
    private_e_exc = ax.scatter(
        [], [], s=18, marker="+", color=GREY, linewidths=0.7, zorder=6
    )
    private_e_inh = ax.scatter(
        [], [], s=18, marker="x", color=RED, linewidths=0.7, zorder=6
    )
    private_i_exc = ax.scatter(
        [], [], s=18, marker="+", color=GREY, linewidths=0.7, zorder=6
    )
    private_i_inh = ax.scatter(
        [], [], s=18, marker="x", color=RED, linewidths=0.7, zorder=6
    )
    shared_e_exc = ax.scatter(
        [], [], s=38, facecolors=GREY, edgecolors="none", alpha=0.18, zorder=1
    )
    shared_e_inh = ax.scatter(
        [], [], s=38, facecolors=RED, edgecolors="none", alpha=0.16, zorder=1
    )
    shared_i_exc = ax.scatter(
        [], [], s=46, facecolors=GREY, edgecolors="none", alpha=0.18, zorder=1
    )
    shared_i_inh = ax.scatter(
        [], [], s=46, facecolors=RED, edgecolors="none", alpha=0.16, zorder=1
    )

    input_g_ax = fig.add_axes(
        [0.347, DIAGNOSTIC_PANEL_BOTTOM, 0.135, DIAGNOSTIC_PANEL_HEIGHT]
    )
    for label, values, color, linestyle in (
        ("E AMPA", external_conductance["E AMPA"], BLACK, "-"),
        ("E GABA", external_conductance["E GABA"], RED, "-"),
        ("I AMPA", external_conductance["I AMPA"], GREY, "--"),
        ("I GABA", external_conductance["I GABA"], RED, "--"),
    ):
        input_g_ax.plot(
            np.arange(n_steps) * dt,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=0.65,
            alpha=0.82,
            label=label,
        )
    input_g_ax.set_xlim(view_start_ms, view_end_ms or n_steps * dt)
    input_g_ax.set_ylim(bottom=0)
    input_g_top = input_g_ax.get_ylim()[1]
    weather_display = (
        weather_scale / max(float(weather_scale.max()), 1e-9) * input_g_top
    )
    input_g_ax.plot(
        np.arange(n_steps) * dt,
        weather_display,
        color=GREY,
        linestyle=(0, (2, 2)),
        linewidth=0.55,
        alpha=0.55,
        label="weather (scaled)",
    )
    if STATE == "input":
        input_g_ax.plot(
            np.arange(n_steps) * dt,
            shared_afferent_scale / shared_afferent_scale.max() * input_g_top,
            color=BLACK,
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            alpha=0.72,
            label="shared afferent wave (scaled)",
        )
    input_g_ax.tick_params(colors=GREY, labelsize=7.0, length=2, pad=3)
    input_g_ax.set_xlabel("simulation time (ms)", color=GREY, fontsize=7.5, labelpad=3)
    input_g_ax.set_ylabel(
        "mean conductance (a.u.)", color=GREY, fontsize=7.5, labelpad=3
    )
    for side in ("top", "right"):
        input_g_ax.spines[side].set_visible(False)
    input_g_ax.set_facecolor(BG)
    input_g_ax.set_title(
        "D · external conductance",
        color=GREY,
        fontsize=8.0,
        weight="medium",
        pad=4,
        loc="left",
    )
    input_g_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=6.0,
        handlelength=1.6,
        columnspacing=1.2,
    )
    input_g_cursor = input_g_ax.axvline(0, color=RED, linewidth=0.7, alpha=0.75)

    means_ax = fig.add_axes(
        [0.015, DIAGNOSTIC_PANEL_BOTTOM, 0.135, DIAGNOSTIC_PANEL_HEIGHT]
    )
    means_ax.set_xlim(0, 1)
    means_ax.set_ylim(0, 1)
    means_ax.axis("off")
    means_ax.set_title(
        "B · population means",
        color=theme.DIM,
        fontsize=8.0,
        weight="medium",
        pad=4,
        loc="left",
    )

    def piston(x, label, color, scale_min, scale_max, digits):
        y, w, h = 0.18, 0.10, 0.52
        means_ax.add_patch(
            plt.Rectangle(
                (x - w / 2, y),
                w,
                h,
                facecolor=BG,
                edgecolor=theme.GREY_LIGHT,
                linewidth=0.75,
            )
        )
        fill = means_ax.add_patch(
            plt.Rectangle(
                (x - w / 2 + 0.02, y),
                w - 0.04,
                0.001,
                facecolor=color,
                alpha=0.22,
                edgecolor="none",
            )
        )
        head = means_ax.add_patch(
            plt.Rectangle(
                (x - w / 2 - 0.02, y),
                w + 0.04,
                0.035,
                facecolor=color,
                edgecolor=BG,
                linewidth=0.55,
            )
        )
        means_ax.text(
            x,
            0.82,
            label,
            ha="center",
            va="center",
            color=color,
            fontsize=8.0,
            weight="medium",
        )
        means_ax.text(
            x,
            0.74,
            f"{scale_max:.{digits}f}",
            ha="center",
            va="bottom",
            color=theme.MUTED_SOFT,
            fontsize=6.8,
            family="monospace",
        )
        means_ax.text(
            x,
            0.15,
            f"{scale_min:.{digits}f}",
            ha="center",
            va="top",
            color=theme.MUTED_SOFT,
            fontsize=6.8,
            family="monospace",
        )
        value = means_ax.text(
            x,
            0.06,
            "",
            ha="center",
            va="center",
            color=color,
            fontsize=8.2,
            family="monospace",
            weight="medium",
        )
        return fill, head, value, y, h - 0.010

    p_ge = piston(0.14, r"$g_E$ µS", BLACK, 0.0, g_max, 2)
    p_gi = piston(0.38, r"$g_I$ µS", RED, 0.0, g_max, 2)
    p_ve = piston(0.64, r"$V_E$ mV", BLACK, v_min, v_max, 0)
    p_vi = piston(0.88, r"$V_I$ mV", RED, v_min, v_max, 0)

    def set_piston(p, level, text):
        fill, head, value, y, travel = p
        y1 = y + np.clip(level, 0, 1) * travel
        fill.set_height(max(y1 - y, 0.001))
        head.set_y(y1)
        value.set_text(text)

    phase = fig.add_axes(
        [0.181, DIAGNOSTIC_PANEL_BOTTOM, 0.135, DIAGNOSTIC_PANEL_HEIGHT]
    )
    phase.set_facecolor(BG)
    phase.spines["top"].set_visible(False)
    phase.spines["right"].set_visible(False)
    pad_e, pad_i = np.ptp(mean_g_e) * 0.08, np.ptp(mean_g_i) * 0.08
    phase.set_xlim(mean_g_e.min() - pad_e, mean_g_e.max() + pad_e)
    phase.set_ylim(mean_g_i.min() - pad_i, mean_g_i.max() + pad_i)
    phase.set_xticks(np.linspace(mean_g_e.min(), mean_g_e.max(), 3))
    phase.set_yticks(np.linspace(mean_g_i.min(), mean_g_i.max(), 3))
    phase.set_xticklabels(
        [f"{value:.2f}" for value in np.linspace(mean_g_e.min(), mean_g_e.max(), 3)]
    )
    phase.set_yticklabels(
        [f"{value:.2f}" for value in np.linspace(mean_g_i.min(), mean_g_i.max(), 3)]
    )
    phase.tick_params(colors=theme.MUTED, labelsize=7.5, length=2, pad=3)
    phase.grid(color=theme.RULE_WARM, linewidth=0.45, linestyle=(0, (2, 3)), alpha=0.65)
    phase.set_xlabel(r"mean $g_E$ (µS)", color=theme.MUTED, fontsize=8.5, labelpad=3)
    phase.set_ylabel(r"mean $g_I$ (µS)", color=RED, fontsize=8.5, labelpad=3)
    phase.set_title(
        "C · conductance phase",
        color=theme.DIM,
        fontsize=8.0,
        weight="medium",
        pad=4,
        loc="left",
    )
    (phase_point,) = phase.plot(
        [], [], "o", ms=4.5, color=RED, mec=BG, mew=0.7, zorder=5
    )
    phase_segments = []
    phase_direction = []

    # Complete raster retains every afferent and network spike even though the
    # network view samples visually excessive active edges.
    raster = fig.add_axes(
        [0.679, DIAGNOSTIC_PANEL_BOTTOM, 0.135, DIAGNOSTIC_PANEL_HEIGHT]
    )
    raster.set_facecolor(BG)
    for side in ("top", "right", "left"):
        raster.spines[side].set_visible(False)
    raster.set_xlim(view_start_ms, view_end_ms or n_steps * dt)
    if weather_inputs:
        raster_series = (
            (afferent_shared, 0, GREY, 0.45, "shared"),
            (afferent_e_private, n_e, BLACK, 0.45, "E aff."),
            (afferent_i_private, 2 * n_e, RED, 0.45, "I aff."),
            (e_spikes, 3 * n_e, BLACK, 0.65, "E"),
            (i_spikes, 4 * n_e, RED, 0.75, "I"),
        )
    else:
        raster_series = (
            (drive_e, 0, GREY, 0.6, "Afferent" if combined_inputs else "Drive"),
            (e_spikes, n_e, BLACK, 0.7, "E"),
            (i_spikes, 2 * n_e, RED, 0.8, "I"),
        )
    raster_height = sum(spikes.shape[1] for spikes, *_ in raster_series)
    raster.set_ylim(raster_height + 5, -5)
    ticks = []
    labels = []
    for spikes, offset, color, size, label in raster_series:
        height = spikes.shape[1]
        raster.axhspan(offset, offset + height, color=color, alpha=0.035, zorder=0)
        if offset:
            raster.axhline(offset, color=theme.RULE_WARM, linewidth=0.45)
        ticks.append(offset + height / 2)
        labels.append(label)
    raster.set_yticks(ticks, labels)
    raster.tick_params(axis="x", colors=GREY, labelsize=7.5, length=2, pad=3)
    raster.tick_params(axis="y", colors=theme.MUTED, labelsize=8.0, length=0, pad=5)
    raster.set_xlabel("simulation time (ms)", color=GREY, fontsize=7.5, labelpad=3)
    raster.set_title(
        "F · spike raster",
        color=theme.DIM,
        fontsize=8.0,
        weight="medium",
        pad=4,
        loc="left",
    )
    for spikes, offset, color, size, _label in raster_series:
        t, c = np.nonzero(spikes)
        raster.scatter(
            t * dt, c + offset, s=size, color=color, marker="s", linewidths=0
        )
    raster_cursor = raster.axvline(0, color=RED, lw=0.9, alpha=0.72)

    rhythm_centres = measurements["rhythm_centres"]
    rhythm_contrast = measurements["rhythm_contrast"]
    rhythm_ax = fig.add_axes(
        [0.513, DIAGNOSTIC_PANEL_BOTTOM, 0.135, DIAGNOSTIC_PANEL_HEIGHT]
    )
    rhythm_ax.set_facecolor(BG)
    rhythm_ax.axvspan(input_onset_ms, input_offset_ms, color=GREY, alpha=0.08, zorder=0)
    rhythm_ax.plot(
        rhythm_centres,
        rhythm_contrast,
        color=BLACK,
        linewidth=1.15,
        label="R · rhythmicity",
        zorder=2,
    )
    loop_times = measurements["loop_times"]
    loop_display = measurements["loop_display"]
    rhythm_ax.plot(
        loop_times,
        loop_display,
        color=RED,
        linewidth=1.0,
        alpha=0.88,
        label="L · loop (normalized, 75 ms smooth)",
        zorder=3,
    )
    rhythm_ax.set_xlim(view_start_ms, view_end_ms or n_steps * dt)
    rhythm_ax.set_ylim(0, 1)
    for side in ("top", "right"):
        rhythm_ax.spines[side].set_visible(False)
    rhythm_ax.tick_params(colors=GREY, labelsize=7.0, length=2, pad=3)
    rhythm_ax.set_xlabel("simulation time (ms)", color=GREY, fontsize=7.5, labelpad=3)
    rhythm_ax.set_ylabel("normalized score", color=GREY, fontsize=7.5, labelpad=3)
    rhythm_ax.set_title(
        "E · PING metrics",
        color=theme.DIM,
        fontsize=8.0,
        weight="medium",
        pad=4,
        loc="left",
    )
    rhythm_ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.96),
        ncol=1,
        frameon=False,
        fontsize=5.6,
        handlelength=1.5,
        borderaxespad=0.2,
    )
    rhythm_top = rhythm_ax.get_ylim()[1]
    for x, label in (
        ((view_start_ms + input_onset_ms) / 2, "baseline"),
        (input_peak_ms, "afferent bout"),
        (((view_end_ms or n_steps * dt) + input_offset_ms) / 2, "recovery"),
    ):
        rhythm_ax.text(
            x,
            rhythm_top * 0.82,
            label,
            ha="center",
            va="top",
            color=theme.MUTED,
            fontsize=6.4,
            weight="medium",
        )
    rhythm_cursor = rhythm_ax.axvline(0, color=RED, linewidth=0.8, alpha=0.72)

    trail_steps = int(round(40 / dt))
    if PACING == "story" and STATE == "input":
        timeline = FrameTimeline.compose(
            [
                (
                    int(round(view_start_ms / dt)),
                    int(round(input_onset_ms / dt)) - 1,
                    140,
                ),
                (
                    int(round(input_onset_ms / dt)),
                    int(round(input_peak_ms / dt)) - 1,
                    170,
                ),
                (
                    int(round(input_peak_ms / dt)),
                    int(round(input_offset_ms / dt)) - 1,
                    170,
                ),
                (
                    int(round(input_offset_ms / dt)),
                    int(round((view_end_ms or n_steps * dt) / dt)) - 1,
                    120,
                ),
            ],
            dt_ms=dt,
        )
    elif PACING == "story":
        cycle_start = int(round(1380 / dt))
        cycle_end = min(n_steps - 1, int(round(1418 / dt)))
        timeline = FrameTimeline.compose(
            [
                (0, int(round(500 / dt)) - 1, 100),
                (int(round(500 / dt)), int(round(1000 / dt)) - 1, 150),
                (int(round(1000 / dt)), n_steps - 1, 200),
                (cycle_start, cycle_start, 5),
                (cycle_start, cycle_end, 60),
                (cycle_end, cycle_end, 10),
                (cycle_start, cycle_start, 5),
                (cycle_start, cycle_end, 60),
                (cycle_end, cycle_end, 10),
            ],
            dt_ms=dt,
        )
    else:
        timeline = FrameTimeline.sample(n_steps, frames=300, dt_ms=dt)
    frame_steps = timeline.steps
    frame_count = len(frame_steps)
    transmission_artists = []

    def draw_afferent_projection(
        p, events, step, *, color, line_limit=120, arrow_limit=60
    ):
        values = p["weight"] * p["trace"][step, p["source"]]
        active = np.flatnonzero(values > 1e-8)
        if active.size > line_limit:
            active = active[np.argpartition(values[active], -line_limit)[-line_limit:]]
        if active.size:
            strength = np.clip(values[active] / p["peak"], 0, 1)
            rgba = np.tile(np.asarray(to_rgba(color)), (active.size, 1))
            # Afferents provide context, while recurrent E→I / I→E transmission is
            # the visual subject. Keep input paths present but deliberately quiet.
            rgba[:, 3] = 0.012 + 0.09 * np.sqrt(strength)
            lines = LineCollection(
                p["segments"][active],
                colors=rgba,
                linewidths=0.06 + 0.65 * strength,
                capstyle="round",
                zorder=3,
            )
            ax.add_collection(lines)
            transmission_artists.append(lines)
        paths = np.flatnonzero(np.isin(p["source"], np.flatnonzero(events)))
        if paths.size > arrow_limit:
            strongest = np.argpartition(p["weight"][paths], -arrow_limit)[-arrow_limit:]
            paths = paths[strongest]
        if paths.size:
            starts = p["segments"][paths, 0]
            delta = p["segments"][paths, 1] - starts
            arrows = ax.quiver(
                starts[:, 0],
                starts[:, 1],
                delta[:, 0],
                delta[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.00018,
                headwidth=4.0,
                headlength=4.8,
                headaxislength=4.2,
                color=to_rgba(color, 0.28),
                pivot="tail",
                zorder=4,
            )
            transmission_artists.append(arrows)

    def draw_transmissions(step):
        while transmission_artists:
            transmission_artists.pop().remove()

        # Authenticated structured spikes use the authored afferent projection.
        # Background conductance remains edge-free because it is not a projection.
        drive_values = trace_drive_e[step] * drive_e_conductance
        drive_active = np.flatnonzero(drive_values > 1e-8)
        if drive_active.size and not combined_inputs:
            drive_strength = np.clip(drive_values[drive_active] / drive_peak, 0, 1)
            drive_rgba = np.tile(np.asarray(to_rgba(GREY)), (drive_active.size, 1))
            drive_rgba[:, 3] = 0.06 + 0.44 * np.sqrt(drive_strength)
            drive_lines = LineCollection(
                drive_segments[drive_active],
                colors=drive_rgba,
                linewidths=0.20 + 2.5 * drive_strength,
                capstyle="round",
                zorder=3,
            )
            ax.add_collection(drive_lines)
            transmission_artists.append(drive_lines)
        drive_events = np.flatnonzero(drive_e[step])
        if drive_events.size and not combined_inputs:
            starts = drive_segments[drive_events, 0]
            delta = drive_segments[drive_events, 1] - starts
            drive_arrows = ax.quiver(
                starts[:, 0],
                starts[:, 1],
                delta[:, 0],
                delta[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.00050,
                headwidth=4.4,
                headlength=5.2,
                headaxislength=4.6,
                color=to_rgba(GREY, 0.86),
                pivot="tail",
                zorder=4,
            )
            transmission_artists.append(drive_arrows)

        if weather_inputs:
            for key, events, color in (
                ("shared_e", afferent_shared[step], GREY),
                ("shared_i", afferent_shared[step], GREY),
                ("private_e", afferent_e_private[step], BLACK),
                ("private_i", afferent_i_private[step], RED),
            ):
                if key in input_weather_projections:
                    draw_afferent_projection(
                        input_weather_projections[key], events, step, color=color
                    )
        elif combined_inputs and input_e_projection is not None:
            p = input_e_projection
            values = p["weight"] * p["trace"][step, p["source"]]
            active = np.flatnonzero(values > 1e-8)
            if active.size > 260:
                active = active[np.argpartition(values[active], -260)[-260:]]
            if active.size:
                strength = np.clip(values[active] / p["peak"], 0, 1)
                rgba = np.tile(np.asarray(to_rgba(GREY)), (active.size, 1))
                rgba[:, 3] = 0.035 + 0.30 * np.sqrt(strength)
                lines = LineCollection(
                    p["segments"][active],
                    colors=rgba,
                    linewidths=0.12 + 2.0 * strength,
                    capstyle="round",
                    zorder=3,
                )
                ax.add_collection(lines)
                transmission_artists.append(lines)

            event_paths = np.flatnonzero(np.isin(p["source"], drive_events))
            if event_paths.size > 120:
                strongest = np.argpartition(p["weight"][event_paths], -120)[-120:]
                event_paths = event_paths[strongest]
            if event_paths.size:
                starts = p["segments"][event_paths, 0]
                delta = p["segments"][event_paths, 1] - starts
                arrows = ax.quiver(
                    starts[:, 0],
                    starts[:, 1],
                    delta[:, 0],
                    delta[:, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.00042,
                    headwidth=4.2,
                    headlength=5.0,
                    headaxislength=4.4,
                    color=to_rgba(GREY, 0.78),
                    pivot="tail",
                    zorder=4,
                )
                transmission_artists.append(arrows)

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
                lines = LineCollection(
                    p["segments"][active],
                    colors=rgba,
                    linewidths=0.10 + 2.2 * strength,
                    capstyle="round",
                    zorder=3,
                )
                ax.add_collection(lines)
                transmission_artists.append(lines)

            # Direction-bearing arrowheads mark transmissions initiated by spikes
            # at this timestep; line weight still follows the synaptic conductance.
            event = np.flatnonzero(
                e_spikes[step] if p["trace"] is trace_e else i_spikes[step]
            )
            mask = np.isin(p["source"], event)
            ids = np.flatnonzero(mask)
            if ids.size:
                starts = p["segments"][ids, 0]
                delta = p["segments"][ids, 1] - starts
                arrows = ax.quiver(
                    starts[:, 0],
                    starts[:, 1],
                    delta[:, 0],
                    delta[:, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.00028,
                    headwidth=4.0,
                    headlength=4.8,
                    headaxislength=4.2,
                    color=to_rgba(p["color"], 0.34),
                    pivot="tail",
                    zorder=4,
                )
                transmission_artists.append(arrows)

    def voltage_sizes(v):
        return (
            3.0 + 25.0 * np.clip((v - v_min) / max(v_max - v_min, 1e-9), 0, 1) ** 1.35
        )

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
            artists["label"].xy = (artists["median"] * scale, base + 0.30)
            artists["label"].set_text(f"{artists['median'] * scale:.2g} µS")

    def update(frame):
        step = frame_steps[frame]
        draw_transmissions(step)
        update_weight_ridges(step)
        if weather_inputs:
            active_drive.set_offsets(np.empty((0, 2)))
            active_shared.set_offsets(shared_xy[afferent_shared[step]])
            active_afferent_e.set_offsets(afferent_e_xy[afferent_e_private[step]])
            active_afferent_i.set_offsets(afferent_i_xy[afferent_i_private[step]])
        else:
            active_drive.set_offsets(drive_e_xy[drive_e[step]])
            active_shared.set_offsets(np.empty((0, 2)))
            active_afferent_e.set_offsets(np.empty((0, 2)))
            active_afferent_i.set_offsets(np.empty((0, 2)))
        active_e.set_offsets(e_xy[e_spikes[step]])
        active_i.set_offsets(i_xy[i_spikes[step]])
        e_nodes.set_sizes(voltage_sizes(v_e[step]))
        i_nodes.set_sizes(voltage_sizes(v_i[step]))
        private_e_exc.set_offsets(e_xy[background["e_exc_private"][step]])
        private_e_inh.set_offsets(e_xy[background["e_inh_private"][step]])
        private_i_exc.set_offsets(i_xy[background["i_exc_private"][step]])
        private_i_inh.set_offsets(i_xy[background["i_inh_private"][step]])
        shared_e_exc.set_offsets(e_xy[background["e_exc_shared"][step]])
        shared_e_inh.set_offsets(e_xy[background["e_inh_shared"][step]])
        shared_i_exc.set_offsets(i_xy[background["i_exc_shared"][step]])
        shared_i_inh.set_offsets(i_xy[background["i_inh_shared"][step]])
        input_g_cursor.set_xdata([step * dt, step * dt])
        rhythm_cursor.set_xdata([step * dt, step * dt])

        set_piston(
            p_ve, (mean_v_e[step] - v_min) / (v_max - v_min), f"{mean_v_e[step]:.1f}"
        )
        set_piston(
            p_vi, (mean_v_i[step] - v_min) / (v_max - v_min), f"{mean_v_i[step]:.1f}"
        )
        set_piston(p_ge, mean_g_e[step] / g_max, f"{mean_g_e[step]:.3f}")
        set_piston(p_gi, mean_g_i[step] / g_max, f"{mean_g_i[step]:.3f}")

        while phase_segments:
            phase_segments.pop().remove()
        while phase_direction:
            phase_direction.pop().remove()
        start = max(0, step - trail_steps)
        points = np.c_[mean_g_e[start : step + 1], mean_g_i[start : step + 1]]
        if len(points) > 1:
            seg = np.stack([points[:-1], points[1:]], axis=1)
            rgba = np.tile(to_rgba(BLACK), (len(seg), 1))
            rgba[:, 3] = np.linspace(0.04, 0.70, len(seg))
            line = LineCollection(seg, colors=rgba, linewidths=1.1)
            phase.add_collection(line)
            phase_segments.append(line)
            if len(points) >= 3:
                arrow = phase.annotate(
                    "",
                    xy=points[-1],
                    xytext=points[-3],
                    arrowprops=dict(
                        arrowstyle="-|>", color=RED, linewidth=1.0, mutation_scale=8
                    ),
                    zorder=6,
                )
                phase_direction.append(arrow)
        phase_point.set_data([mean_g_e[step]], [mean_g_i[step]])
        raster_cursor.set_xdata([step * dt, step * dt])

        n_de = int(drive_e[step].sum())
        n_di = int(drive_i[step].sum())
        n_afferent = (
            int(afferent_shared[step].sum())
            + int(afferent_e_private[step].sum())
            + int(afferent_i_private[step].sum())
            if weather_inputs
            else n_de + n_di
        )
        n_es = int(e_spikes[step].sum())
        n_is = int(i_spikes[step].sum())
        time_text.set_text(f"t = {step * dt:06.2f} ms   frame {frame:03d}")
        if STATE == "input":
            time_ms = step * dt
            phase_name = (
                "stationary baseline"
                if time_ms < input_onset_ms
                else "input rising"
                if time_ms < input_peak_ms
                else "input falling"
                if time_ms < input_offset_ms
                else "recovery"
            )
            count_text.set_text(
                f"{phase_name} · shared input ×{shared_afferent_scale[step]:.2f} · "
                f"weather ×{weather_scale[step]:.2f} · {n_afferent} afferent · "
                f"{n_es} E · {n_is} I spikes"
            )
        elif STATE == "transition":
            time_ms = step * dt
            if PACING == "story" and frame >= 450:
                replay = 1 if frame < 525 else 2
                cycle_ms = time_ms - 1380
                if cycle_ms < 8:
                    phase_name = "recovery · inhibition decays"
                elif cycle_ms < 15:
                    phase_name = "E volley · excitation rises"
                elif cycle_ms < 23:
                    phase_name = "I response · inhibition returns"
                else:
                    phase_name = "suppression · E cells reset"
                time_text.set_text(f"replay {replay}/2 · t = {time_ms:06.2f} ms")
            else:
                phase_name = (
                    "AI"
                    if time_ms < transition_start_ms
                    else "weight ramp"
                    if time_ms < transition_end_ms
                    else "PING rhythm"
                )
            count_text.set_text(
                f"{phase_name} · W_EE ×{weight_scales['w_ee'][step]:.2f} · "
                f"weather ×{weather_scale[step]:.2f} · {n_afferent} afferent · "
                f"{n_es} E · {n_is} I spikes"
            )
        else:
            count_text.set_text(
                f"weather ×{weather_scale[step]:.2f} · {n_afferent} afferent · "
                f"{n_es} E · {n_is} I spikes"
            )
        return (
            active_drive,
            active_shared,
            active_afferent_e,
            active_afferent_i,
            active_e,
            active_i,
            e_nodes,
            i_nodes,
            raster_cursor,
            input_g_cursor,
            rhythm_cursor,
            private_e_exc,
            private_e_inh,
            private_i_exc,
            private_i_inh,
            shared_e_exc,
            shared_e_inh,
            shared_i_exc,
            shared_i_inh,
            phase_point,
            time_text,
            count_text,
            *transmission_artists,
        )

    # Export the sampled frame with the greatest simultaneous recurrent activity.
    # This gives design iteration a representative view of both active neurons and
    # source→target transmissions instead of an arbitrary final frame.
    if STATE == "input":
        rhythm_peak_step = int(round(rhythm_centres[np.argmax(rhythm_contrast)] / dt))
        representative_frame = int(np.argmin(np.abs(frame_steps - rhythm_peak_step)))
    else:
        representative_frame = select_representative_frame(
            e_spikes[frame_steps], i_spikes[frame_steps]
        )
    update(representative_frame)
    fig.savefig(POSTER, dpi=240, facecolor=BG, bbox_inches=fig.bbox_inches)
    save_animation(fig, update, OUT, frames=frame_count, fps=25, bitrate=3800)
    plt.close(fig)
