"""Render saved exp086 arrays; no simulation, estimation or selection."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme

from .recipe import DISPLAY_WINDOW_MS, DT_MS


def _normalise(values: np.ndarray, window: slice) -> np.ndarray:
    selected = values[window]
    maximum = float(selected.max()) if selected.size else 0.0
    return selected / maximum if maximum > 0 else selected


def plot_uncoupled(trajectory: dict[str, object], out: Path) -> None:
    """Result 1: show the two uncoupled rhythms and their relative-phase drift."""
    theme.apply()
    rate_length = len(np.asarray(trajectory["rate_e_a"]))
    stop = rate_length
    start = max(0, stop - round(250.0 / DT_MS))
    window = slice(start, stop)
    excerpt_time = np.arange(start, stop) * DT_MS
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2))
    for ax, label, e_key, i_key in (
        (axes[0], "Network A", "rate_e_a", "rate_i_a"),
        (axes[1], "Network B", "rate_e_b", "rate_i_b"),
    ):
        ax.plot(
            excerpt_time,
            _normalise(np.asarray(trajectory[e_key]), window),
            color=theme.INK_BLACK,
            lw=1.0,
            label="E",
        )
        ax.plot(
            excerpt_time,
            _normalise(np.asarray(trajectory[i_key]), window),
            color=theme.DEEP_RED,
            lw=1.0,
            label="I",
        )
        ax.set(ylabel=f"{label}\nnormalized rate", ylim=(-0.05, 1.08))
        ax.legend(frameon=False, ncol=2, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("time after coupling decision (ms)")
    axes[2].plot(
        trajectory["time_ms"],
        trajectory["wrapped_phase"],
        color=theme.INK_BLACK,
        lw=0.9,
    )
    axes[2].set(
        xlabel="time after coupling decision (ms)",
        ylabel="relative-phase\nposition (rad)",
        ylim=(-np.pi, np.pi),
    )
    axes[2].set_yticks((-np.pi, 0, np.pi), labels=(r"$-\pi$", "0", r"$\pi$"))
    axes[2].spines[["top", "right"]].set_visible(False)
    theme.label_panels(axes)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_position_time(ax, trajectory: dict[str, object], color: str) -> None:
    time_ms = np.asarray(trajectory["time_ms"])
    keep = time_ms >= max(time_ms.min(), time_ms.max() - DISPLAY_WINDOW_MS)
    ax.plot(
        time_ms[keep] - time_ms[keep][0],
        np.asarray(trajectory["wrapped_phase"])[keep],
        color=color,
        lw=1.0,
    )
    ax.set(ylim=(-np.pi, np.pi))
    ax.set_yticks((-np.pi, 0, np.pi), labels=(r"$-\pi$", "0", r"$\pi$"))
    ax.spines[["top", "right"]].set_visible(False)


def _plot_velocity_position(ax, trajectory: dict[str, object], color: str) -> None:
    centres = np.asarray(trajectory["phase_bin_centres"])
    velocity = np.asarray(trajectory["mean_velocity_by_phase"])
    ax.plot(centres, velocity, color=color, marker="o", ms=2.5, lw=1.2)
    ax.axhline(0.0, color=theme.GREY_LIGHT, lw=0.7, ls="--")
    ax.set(xlim=(-np.pi, np.pi))
    ax.set_xticks((-np.pi, 0, np.pi), labels=(r"$-\pi$", "0", r"$\pi$"))
    ax.spines[["top", "right"]].set_visible(False)


def plot_coupling_regimes(
    strong: dict[str, object],
    intermediate: dict[str, object],
    uncoupled: dict[str, object],
    out: Path,
) -> None:
    """Result 2: reproduce the Method 2 three-regime schematic with data."""
    theme.apply()
    columns = (
        (strong, "Strong coupling", theme.INK_BLACK),
        (intermediate, "Intermediate coupling", theme.DEEP_RED),
        (uncoupled, "No coupling", theme.GREY_MID),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.5), sharey="row")
    for column, (trajectory, label, color) in enumerate(columns):
        _plot_position_time(axes[0, column], trajectory, color)
        _plot_velocity_position(axes[1, column], trajectory, color)
        axes[0, column].set_title(f"{label}\nK = {float(trajectory['k']):.3f} µS")
        axes[0, column].set_xlabel("time in displayed window (ms)")
        axes[1, column].set_xlabel("relative-phase position (rad)")
    axes[0, 0].set_ylabel("relative-phase\nposition (rad)")
    axes[1, 0].set_ylabel("mean relative-phase\nvelocity (rad/s)")
    fig.suptitle("Measured transition as reciprocal coupling weakens", y=1.01)
    theme.label_panels(axes.flat)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_intermittent_attraction(
    trajectory: dict[str, object],
    out: Path,
) -> None:
    """Result 3: reproduce the four-panel mechanism schematic with data."""
    theme.apply()
    time_ms = np.asarray(trajectory["time_ms"])
    keep = time_ms >= max(time_ms.min(), time_ms.max() - DISPLAY_WINDOW_MS)
    local_time = time_ms[keep] - time_ms[keep][0]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.2))

    axes[0, 0].plot(
        local_time,
        np.asarray(trajectory["wrapped_phase"])[keep],
        color=theme.INK_BLACK,
        lw=1.0,
    )
    axes[0, 0].set(
        title="Relative-phase position through time",
        xlabel="time in displayed window (ms)",
        ylabel="position (rad)",
        ylim=(-np.pi, np.pi),
    )
    axes[0, 0].set_yticks((-np.pi, 0, np.pi), labels=(r"$-\pi$", "0", r"$\pi$"))

    axes[0, 1].plot(
        local_time,
        np.asarray(trajectory["relative_velocity_smoothed_rad_s"])[keep],
        color=theme.DEEP_RED,
        lw=1.0,
    )
    axes[0, 1].set(
        title="Relative-phase velocity through time",
        xlabel="time in displayed window (ms)",
        ylabel="velocity (rad/s)",
    )

    _plot_velocity_position(axes[1, 0], trajectory, theme.DEEP_RED)
    axes[1, 0].set(
        title="Velocity depends on phase position",
        xlabel="relative-phase position (rad)",
        ylabel="mean velocity (rad/s)",
    )

    centres = np.asarray(trajectory["phase_bin_centres"])
    density = np.asarray(trajectory["phase_density"])
    axes[1, 1].fill_between(
        centres,
        density,
        color=theme.ELECTRIC_CYAN,
        alpha=0.18,
    )
    axes[1, 1].plot(centres, density, color=theme.ELECTRIC_CYAN, lw=1.5)
    axes[1, 1].set(
        title="Phase-position distribution",
        xlabel="relative-phase position (rad)",
        ylabel="density",
        xlim=(-np.pi, np.pi),
    )
    axes[1, 1].set_xticks((-np.pi, 0, np.pi), labels=(r"$-\pi$", "0", r"$\pi$"))
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"Intermediate condition: K = {float(trajectory['k']):.3f} µS",
        y=1.01,
    )
    theme.label_panels(axes.flat)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
