"""Draw saved measurements; no simulation or scientific estimators."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme

from .recipe import (
    BURN_MS,
    DISPLAY_END_MS,
    DISPLAY_START_MS,
    DT_MS,
    T_MS,
)


def _normalise_window(trace: np.ndarray, window: slice) -> np.ndarray:
    values = trace[window]
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0 else values


def plot_uncoupled(analysis: dict[str, Any], out: Path) -> None:
    """Show readable E/I rhythm excerpts above the full phase-drift trace."""
    theme.apply()
    start = round(DISPLAY_START_MS / DT_MS)
    stop = round(DISPLAY_END_MS / DT_MS)
    window = slice(start, stop)
    local_time_ms = np.arange(stop - start) * DT_MS + DISPLAY_START_MS
    full_time_ms = np.arange(len(analysis["phase_difference"])) * DT_MS

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2))
    for ax, name, e_key, i_key in (
        (axes[0], "Network A", "rate_e_a", "rate_i_a"),
        (axes[1], "Network B", "rate_e_b", "rate_i_b"),
    ):
        ax.plot(
            local_time_ms,
            _normalise_window(np.asarray(analysis[e_key]), window),
            color=theme.INK_BLACK,
            lw=1.0,
            label="E",
        )
        ax.plot(
            local_time_ms,
            _normalise_window(np.asarray(analysis[i_key]), window),
            color=theme.DEEP_RED,
            lw=1.0,
            label="I",
        )
        ax.set(ylabel=f"{name}\nnormalized rate", ylim=(-0.05, 1.1))
        ax.legend(frameon=False, ncol=2, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("time (ms), rhythm excerpt")

    axes[2].plot(
        full_time_ms,
        analysis["phase_difference"],
        color=theme.INK_BLACK,
        lw=0.9,
    )
    axes[2].axvline(BURN_MS, color=theme.GREY_MID, ls="--", lw=0.8)
    axes[2].set(
        xlim=(BURN_MS, T_MS),
        ylim=(-np.pi, np.pi),
        xlabel="time (ms)",
        ylabel="wrapped phase\ndifference (rad)",
    )
    axes[2].set_yticks((-np.pi, 0, np.pi), labels=(r"$-\pi$", "0", r"$\pi$"))
    axes[2].spines[["top", "right"]].set_visible(False)
    theme.label_panels(axes)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_phase_response_examples(
    illustration: dict[str, Any],
    out: Path,
) -> None:
    """Show how three representative probes change the next PING volley."""
    theme.apply()
    left = int(illustration["left_step"])
    baseline_next = int(illustration["baseline_next_step"])
    start = left - round(2.0 / DT_MS)
    stop = left + round(32.0 / DT_MS)
    window = slice(start, stop)
    time_ms = (np.arange(start, stop) - left) * DT_MS
    panels = (
        ("e_late_advance", "E probe: advance"),
        ("i_early_no_doublet", "I probe: no correction"),
        ("i_early_doublet", "I probe: doublet and delay"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.2), sharex=True, sharey=True)
    for ax, (case_name, title) in zip(axes, panels, strict=True):
        case = illustration["cases"][case_name]
        rate_e = _normalise_window(np.asarray(case["rate_e"]), window)
        rate_i = _normalise_window(np.asarray(case["rate_i"]), window)
        arrival_ms = (int(case["arrival_step"]) - left) * DT_MS
        baseline_next_ms = (baseline_next - left) * DT_MS
        ax.plot(time_ms, rate_e, color=theme.INK_BLACK, lw=1.2, label="E")
        ax.plot(time_ms, rate_i, color=theme.DEEP_RED, lw=1.2, label="I")
        ax.axvline(
            arrival_ms,
            color=theme.DEEP_RED,
            lw=0.9,
            ls=":",
            label="probe arrival",
        )
        ax.axvline(
            baseline_next_ms,
            color=theme.GREY_MID,
            lw=0.9,
            ls="--",
            label="unperturbed next E",
        )
        ax.set(title=title, ylim=(-0.05, 1.12))
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    axes[1].set_ylabel("normalized population rate")
    axes[2].set_xlabel("time from reference E volley (ms)")
    theme.label_panels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_phase_response(
    phase_response: dict[str, Any],
    illustration: dict[str, Any],
    out: Path,
) -> None:
    """Summarize whole-cycle responses and the inhibitory doublet mechanism."""
    theme.apply()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.8, 6.8),
        gridspec_kw={"height_ratios": [1.35, 0.8, 1.0]},
    )
    i_rows = phase_response["responses"]["I"]
    i_phase = np.asarray([row["pulse_phase_fraction"] for row in i_rows])
    doublet = np.asarray([row["i_volleys_before_next_e"] == 2 for row in i_rows])
    doublet_indices = np.flatnonzero(doublet)
    first = int(doublet_indices[0])
    last = int(doublet_indices[-1])
    window_left = 0.5 * (i_phase[first - 1] + i_phase[first])
    window_right = 0.5 * (i_phase[last] + i_phase[last + 1])

    whole = axes[0]
    for target, color in (("E", theme.INK_BLACK), ("I", theme.DEEP_RED)):
        rows = phase_response["responses"][target]
        phase = np.asarray([row["pulse_phase_fraction"] for row in rows])
        response = np.asarray([row["next_volley_shift_ms"] for row in rows])
        whole.scatter(
            phase,
            response,
            s=22,
            color=color,
            label=f"pulse to {target}",
            zorder=3,
        )
    whole.axvspan(
        window_left,
        window_right,
        color=theme.DEEP_RED,
        alpha=0.08,
        lw=0,
    )
    whole.axhline(0.0, color=theme.GREY_MID, lw=0.8, ls="--")
    whole.text(
        0.5 * (window_left + window_right),
        -3.6,
        "I doublet window",
        ha="center",
        color=theme.DEEP_RED,
        fontsize=theme.SIZE_ANNOTATION,
    )
    whole.set(
        xlim=(0.0, 1.0),
        ylim=(-6.8, 3.3),
        title="Phase response across one cycle",
        ylabel="next E-volley shift (ms)",
    )
    whole.legend(frameon=False, ncol=2, loc="lower right")
    whole.spines[["top", "right"]].set_visible(False)

    state_case = illustration["cases"]["i_early_doublet"]
    state_left = int(illustration["left_step"])
    state_start = state_left - round(0.5 / DT_MS)
    state_stop = state_left + round(7.0 / DT_MS)
    state_window = slice(state_start, state_stop)
    state_time_ms = (np.arange(state_start, state_stop) - state_left) * DT_MS
    arrival_ms = (int(state_case["arrival_step"]) - state_left) * DT_MS

    local_g = np.asarray(state_case["local_e_to_i_conductance"])[state_window]
    probe_g = np.asarray(state_case["probe_e_to_i_conductance"])[state_window]
    conductance = axes[1]
    conductance.plot(
        state_time_ms,
        local_g,
        color=theme.GREY_MID,
        lw=1.1,
        ls="--",
        label="local E→I",
        zorder=3,
    )
    conductance.plot(
        state_time_ms,
        probe_g,
        color=theme.DEEP_RED,
        lw=1.2,
        label="probe→I",
        zorder=3,
    )
    conductance.plot(
        state_time_ms,
        local_g + probe_g,
        color=theme.INK_BLACK,
        lw=1.6,
        label="total excitation",
        zorder=2,
    )
    conductance.axvline(arrival_ms, color=theme.DEEP_RED, lw=0.9, ls=":")
    conductance.set(
        xlim=(-0.5, 7.0),
        title="Why the I probe produces a doublet",
        ylabel="mean I excitatory\nconductance (µS)",
    )
    conductance.legend(frameon=False, ncol=3, loc="upper right")
    conductance.spines[["top", "right"]].set_visible(False)

    voltage_values = np.asarray(state_case["i_voltage"])[state_window]
    voltage = axes[2]
    voltage.plot(
        state_time_ms,
        voltage_values,
        color=theme.INK_BLACK,
        lw=1.2,
    )
    voltage.axhline(-50.0, color=theme.GREY_MID, lw=0.8, ls="--")
    voltage.axvline(
        arrival_ms,
        color=theme.DEEP_RED,
        lw=0.9,
        ls=":",
    )
    voltage.text(
        6.8,
        -50.3,
        "threshold",
        ha="right",
        va="top",
        color=theme.GREY_MID,
        fontsize=theme.SIZE_ANNOTATION,
    )
    voltage.text(
        arrival_ms + 0.08,
        -64.5,
        "probe",
        ha="left",
        va="bottom",
        color=theme.DEEP_RED,
        fontsize=theme.SIZE_ANNOTATION,
    )
    voltage.set(
        xlim=(-0.5, 7.0),
        xlabel="time from reference E volley (ms)",
        ylabel="I membrane\nvoltage (mV)",
    )
    voltage.spines[["top", "right"]].set_visible(False)
    theme.label_panels(axes)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_event_aligned_mechanism(
    mechanism: dict[str, Any],
    traces: dict[str, np.ndarray],
    out: Path,
) -> None:
    """Trace the causal sequence around one incoming E-to-E volley."""
    theme.apply()
    time_ms = traces["time_from_arrival_ms"]
    baseline_next_ms = (
        mechanism["baseline_next_target_volley_ms_after_coupling"]
        - mechanism["arrival_ms_after_coupling"]
    )
    coupled_next_ms = (
        mechanism["coupled_next_target_volley_ms_after_coupling"]
        - mechanism["arrival_ms_after_coupling"]
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.8), sharex=True)

    axes[0].plot(
        time_ms,
        traces["incoming_e_to_e_conductance"],
        color=theme.DEEP_RED,
        label="A→B E conductance",
    )
    axes[0].set(
        title="Cross-network excitation arrives at t = 0",
        ylabel="cross-network\nconductance (µS)",
    )

    axes[1].plot(
        time_ms,
        traces["baseline_target_e_rate"],
        color=theme.GREY_MID,
        ls="--",
        label="no coupling",
    )
    axes[1].plot(
        time_ms,
        traces["coupled_target_e_rate"],
        color=theme.INK_BLACK,
        label="E→E only",
    )
    axes[1].axvline(
        baseline_next_ms,
        color=theme.GREY_MID,
        lw=0.9,
        ls="--",
    )
    axes[1].axvline(coupled_next_ms, color=theme.INK_BLACK, lw=0.9, ls=":")
    axes[1].set(
        title=f"The next E volley advances by {mechanism['next_target_volley_advance_ms']:.1f} ms",
        ylabel="target E rate\n(Hz per neuron)",
    )
    axes[1].legend(frameon=False, loc="upper left")

    axes[2].plot(
        time_ms,
        traces["baseline_inhibition_to_e"],
        color=theme.GREY_MID,
        ls="--",
        label="no coupling",
    )
    axes[2].plot(
        time_ms,
        traces["coupled_inhibition_to_e"],
        color=theme.DEEP_RED,
        label="E→E only",
    )
    axes[2].set(
        title="Feedback inhibition advances with it",
        xlabel="time from cross-network excitation reaching Network B (ms)",
        ylabel="inhibition onto E\n(µS)",
    )

    for ax in axes:
        ax.axvline(0.0, color=theme.DEEP_RED, lw=0.9, ls="--")
        ax.spines[["top", "right"]].set_visible(False)
    theme.label_panels(axes)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pathway_comparison(
    pathway_comparison: dict[str, Any],
    traces: dict[str, Any],
    out: Path,
) -> None:
    """Show which coupling pathways arrest relative-phase drift."""
    theme.apply()
    colors = {
        "none": theme.GREY_MID,
        "e_to_e": theme.INK_BLACK,
        "e_to_i": theme.DEEP_RED,
        "both": theme.ELECTRIC_CYAN,
    }
    all_phase = np.concatenate(
        [
            np.asarray(traces[row["id"]]["unwrapped_phase_change_cycles"])
            for row in pathway_comparison["conditions"]
        ]
    )
    lower = float(all_phase.min()) - 0.3
    upper = float(all_phase.max()) + 0.3
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 6.4), sharex=True, sharey=True)
    for ax, condition in zip(
        axes,
        pathway_comparison["conditions"],
        strict=True,
    ):
        condition_id = condition["id"]
        condition_trace = traces[condition_id]
        state = "phase locked" if condition["phase_locked"] else "phase drift"
        ax.plot(
            condition_trace["time_ms"],
            condition_trace["unwrapped_phase_change_cycles"],
            color=colors[condition_id],
            lw=1.2,
        )
        ax.axhline(0.0, color=theme.GREY_LIGHT, lw=0.7, ls="--")
        ax.set(title=f"{condition['label']}: {state}", ylim=(lower, upper))
        ax.text(
            0.99,
            0.82,
            f"{condition['final_drift_rate_cycles_per_s']:.2f} cycles/s",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=theme.SIZE_ANNOTATION,
            color=colors[condition_id],
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_ylabel("unwrapped relative phase change (cycles)")
    axes[-1].set_xlabel("time after coupling onset (ms)")
    theme.label_panels(axes)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
