"""Rendering saved analysis only; no simulation, dataset loading or estimators."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme

from .recipe import (
    CLASS_PROBABILITY_TICKS,
    DT_MS,
    N_CLASSES,
    SINGLE_TRIAL_TRANSITION_WINDOW_MS,
)


def plot_design(path):
    """Prospective protocol diagram, not measured evidence."""
    theme.apply()
    fig, axis = plt.subplots(figsize=(6.9, 1.8), constrained_layout=True)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.axis("off")
    labels = (
        "Changing digit input",
        "Continuous PING state",
        "Output spikes → counts",
        "Prediction",
    )
    for x, label in zip((0.1, 0.37, 0.66, 0.91), labels, strict=True):
        axis.text(
            x,
            0.65,
            label,
            ha="center",
            va="center",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#eef4f8",
                "edgecolor": "#7b909f",
            },
        )
    for a, b in ((0.2, 0.25), (0.49, 0.54), (0.79, 0.86)):
        axis.annotate(
            "",
            (b, 0.65),
            (a, 0.65),
            arrowprops={"arrowstyle": "->", "color": "#7b909f"},
        )
    axis.text(
        0.5,
        0.2,
        "At digit boundaries: hidden state continues; output state and counts reset.",
        ha="center",
        fontsize=8,
    )
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def plot_stream(result: dict[str, Any], path: Path, run_id: str) -> None:
    theme.apply()
    spikes_e = result["spikes_e"]
    spikes_i = result["spikes_i"]
    probabilities = result["probabilities"]
    time_ms = np.arange(len(spikes_e)) * DT_MS
    fig, axes = plt.subplots(
        3, 1, figsize=(6.5, 5.4), sharex=True, constrained_layout=True
    )
    e_t, e_n = np.nonzero(spikes_e[:, :200])
    i_t, i_n = np.nonzero(spikes_i[:, :64])
    axes[0].scatter(e_t * DT_MS, e_n, s=1, color=theme.INK_BLACK)
    axes[1].scatter(i_t * DT_MS, i_n, s=2, color=theme.DEEP_RED)
    for label in range(N_CLASSES):
        axes[2].plot(time_ms, probabilities[:, label], lw=0.9, label=str(label))
    for boundary in result["boundaries"][1:-1]:
        for axis in axes:
            axis.axvline(boundary * DT_MS, color=theme.GREY_MID, lw=0.7, ls=":")
    axes[0].set_ylabel("E neuron")
    axes[1].set_ylabel("I neuron")
    axes[2].set(xlabel="time (ms)", ylabel="softmax count share", ylim=(0, 1))
    axes[2].legend(ncol=5, frameon=False, fontsize=7)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def plot_stream_headline(
    result: dict[str, Any],
    path: Path,
    run_id: str,
    *,
    annotate_final_counts: bool = False,
) -> None:
    """Exp048-Figure-1-style streaming headline for one or more trials."""
    theme.apply()
    conditions = result["conditions"]
    boundaries = np.asarray(result["boundaries"], dtype=int)
    starts = boundaries[:-1]
    stops = boundaries[1:]
    starts_ms = starts * DT_MS
    stops_ms = stops * DT_MS
    total_ms = stops_ms[-1]
    time_ms = np.arange(len(result["spikes_e"])) * DT_MS
    labels = result["labels"]
    predictions = result["predictions"]

    fig = plt.figure(figsize=(6.9, 5.33), dpi=150)
    grid = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.35, 2.2, 1.2, 2.0],
        hspace=0.18,
    )

    thumbnail_axis = fig.add_subplot(grid[0])
    thumbnail_axis.set(xlim=(0, total_ms), ylim=(0, 1))
    thumbnail_axis.set_xticks([])
    thumbnail_axis.set_yticks([])
    for spine in thumbnail_axis.spines.values():
        spine.set_visible(False)
    rates = np.asarray([condition[1] for condition in conditions], dtype=float)
    log_rates = np.log(rates)
    centers = (starts_ms + stops_ms) / 2
    spacing = np.diff(centers).min() if len(centers) > 1 else total_ms
    width = min(0.085, spacing / total_ms * thumbnail_axis.get_position().width * 0.8)
    width = min(
        width,
        (thumbnail_axis.get_position().height - 0.045)
        * fig.get_figheight()
        / fig.get_figwidth(),
    )
    height = width * fig.get_figwidth() / fig.get_figheight()
    for index, ((duration_ms, rate_hz), start_ms, stop_ms) in enumerate(
        zip(conditions, starts_ms, stops_ms, strict=True)
    ):
        left = (
            thumbnail_axis.get_position().x0
            + (
                (start_ms + stop_ms)
                / 2
                / total_ms
                * (thumbnail_axis.get_position().x1 - thumbnail_axis.get_position().x0)
            )
            - width / 2
        )
        inset = fig.add_axes(  # ty: ignore[no-matching-overload]
            [
                left,
                thumbnail_axis.get_position().y0 + 0.035,
                width,
                height,
            ]
        )
        alpha = 1.0
        if log_rates.max() > log_rates.min():
            alpha = 0.2 + 0.8 * (np.log(rate_hz) - log_rates.min()) / (
                log_rates.max() - log_rates.min()
            )
        inset.imshow(
            np.asarray(result["pixels"])[index].reshape(28, 28),
            cmap="Greys",
            interpolation="nearest",
            aspect="equal",
            alpha=alpha,
        )
        inset.set_xticks([])
        inset.set_yticks([])
        colour = (
            theme.INK_BLACK if predictions[index] == labels[index] else theme.DEEP_RED
        )
        inset.text(
            0.05,
            -0.05,
            f"{labels[index]}→{predictions[index]}",
            transform=inset.transAxes,
            ha="left",
            va="top",
            fontsize=theme.SIZE_LABEL,
            color="white",
            weight="bold",
            bbox=dict(
                facecolor=colour,
                edgecolor="none",
                boxstyle="round,pad=0.2",
                alpha=0.95,
            ),
        )
        thumbnail_axis.text(
            (start_ms + stop_ms) / 2,
            1.02 if index % 2 == 0 else 1.15,
            f"{duration_ms:g} ms · {rate_hz:g} Hz",
            transform=thumbnail_axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=theme.SIZE_LABEL - 2,
            color=theme.MUTED,
            clip_on=False,
        )

    raster_specs = (
        (result["spikes_e"][:, :200], "E cell", theme.INK_BLACK, 2.0, 200),
        (result["spikes_i"][:, :64], "I cell", theme.DEEP_RED, 2.0, 64),
    )
    for row, (spikes, label, colour, size, population) in enumerate(
        raster_specs, start=1
    ):
        axis = fig.add_subplot(grid[row])
        spike_times, neurons = np.nonzero(spikes)
        axis.scatter(
            spike_times * DT_MS,
            neurons,
            s=size,
            c=colour,
            marker="|",
            linewidths=0.4,
        )
        axis.set(xlim=(0, total_ms), ylim=(0, spikes.shape[1]))
        axis.set_yticks([0, spikes.shape[1]], ["0", f"{population}"])
        axis.set_ylabel(label, fontsize=theme.SIZE_LABEL)
        axis.tick_params(axis="x", labelbottom=False)
        axis.spines[["top", "right"]].set_visible(False)
        for boundary in starts_ms[1:]:
            axis.axvline(boundary, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)

    evidence_axis = fig.add_subplot(grid[3])
    probabilities = result["probabilities"]
    for class_index in range(N_CLASSES):
        evidence_axis.plot(
            time_ms,
            probabilities[:, class_index],
            color=theme.GREY_MID,
            lw=0.6,
            alpha=0.45,
        )
    final_counts = np.asarray(result["final_counts"])
    final_winner = result["winner"]
    if annotate_final_counts and final_winner not in set(labels):
        evidence_axis.plot(
            time_ms,
            probabilities[:, final_winner],
            color=theme.INK_BLACK,
            lw=1.5,
        )
    for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        evidence_axis.plot(
            time_ms[start:stop],
            probabilities[start:stop, labels[index]],
            color=theme.DEEP_RED,
            lw=2.2,
        )
    for boundary in starts_ms[1:]:
        evidence_axis.axvline(
            boundary,
            color=theme.GREY_MID,
            lw=0.5,
            ls=":",
            alpha=0.7,
        )
    evidence_axis.axhline(0.5, color=theme.GREY_MID, lw=0.5, ls="--", alpha=0.6)
    evidence_axis.set(
        xlim=(0, total_ms),
        ylim=(0, 1),
        xlabel="time (ms)",
        ylabel="softmax share\n" + r"$p_c(u)$",
    )
    evidence_axis.set_yticks(CLASS_PROBABILITY_TICKS)
    evidence_axis.spines[["top", "right"]].set_visible(False)
    if annotate_final_counts:
        true_class = int(labels[0])
        runner_up_class = result["runner_up"]
        margin = result["margin"]
        if true_class == final_winner:
            summary = (
                f"correct class {true_class}: {final_counts[true_class]} spikes · "
                f"runner-up {runner_up_class}: {final_counts[runner_up_class]} spikes · "
                f"margin {margin}"
            )
        else:
            summary = (
                f"true {true_class}: {final_counts[true_class]} spikes · "
                f"winner {final_winner}: {final_counts[final_winner]} spikes · "
                f"margin {margin}"
            )
        evidence_axis.text(
            0.01,
            0.98,
            summary,
            transform=evidence_axis.transAxes,
            ha="left",
            va="top",
            fontsize=theme.SIZE_ANNOTATION,
            color=theme.INK_BLACK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82),
        )

    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def plot_variable_headline(result: dict[str, Any], path: Path, run_id: str) -> None:
    """Plot the variable-condition stream used as the exp048 successor."""
    plot_stream_headline(result, path, run_id)


def plot_single_trial(result: dict[str, Any], path: Path, run_id: str) -> None:
    """Plot one selected presentation to explain spike-count evidence."""
    plot_stream_headline(result, path, run_id, annotate_final_counts=True)


def plot_single_trial_transition(
    result: dict[str, Any],
    path: Path,
    run_id: str,
) -> None:
    """Resolve the output spikes behind the selected evidence transition."""
    theme.apply()
    start_ms, stop_ms = SINGLE_TRIAL_TRANSITION_WINDOW_MS
    start = int(round(start_ms / DT_MS))
    stop = int(round(stop_ms / DT_MS)) + 1
    spikes_out = np.asarray(result["spikes_out"])
    counts = np.asarray(result["counts"])
    probabilities = np.asarray(result["probabilities"])
    time_ms = np.arange(len(spikes_out)) * DT_MS
    true_class = int(result["labels"][0])
    winner = result["winner"]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.9, 4.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.4, 1.7)},
    )
    spike_times, spike_classes = np.nonzero(spikes_out[start:stop])
    spike_times_ms = (spike_times + start) * DT_MS
    spike_colours = [
        theme.DEEP_RED
        if class_index == true_class
        else theme.INK_BLACK
        if class_index == winner
        else theme.GREY_MID
        for class_index in spike_classes
    ]
    axes[0].scatter(
        spike_times_ms,
        spike_classes,
        c=spike_colours,
        marker="|",
        s=48,
        linewidths=1.2,
    )
    axes[0].set_ylabel("output class")
    axes[0].set_yticks(range(N_CLASSES))

    for class_index in range(N_CLASSES):
        colour = theme.GREY_MID
        width = 0.7
        alpha = 0.45
        if class_index == winner:
            colour, width, alpha = theme.INK_BLACK, 1.6, 1.0
        if class_index == true_class:
            colour, width, alpha = theme.DEEP_RED, 2.0, 1.0
        axes[1].step(
            time_ms[start:stop],
            counts[start:stop, class_index],
            where="post",
            color=colour,
            lw=width,
            alpha=alpha,
        )
        axes[2].step(
            time_ms[start:stop],
            probabilities[start:stop, class_index],
            where="post",
            color=colour,
            lw=width,
            alpha=alpha,
        )
    axes[1].set_ylabel(r"output count $z_c(u)$")
    axes[2].set(
        xlabel="time (ms)",
        ylabel="softmax share\n" + r"$p_c(u)$",
        ylim=(0, 1),
    )
    axes[2].set_yticks(CLASS_PROBABILITY_TICKS)
    axes[2].axhline(0.5, color=theme.GREY_MID, lw=0.5, ls="--", alpha=0.6)
    for axis in axes:
        axis.set_xlim(start_ms, stop_ms)
        axis.spines[["top", "right"]].set_visible(False)
    title = (
        f"true and winning class {true_class} (red)"
        if true_class == winner
        else f"true class {true_class} (red) · eventual winner {winner} (black)"
    )
    axes[0].set_title(title, loc="left", fontsize=theme.SIZE_LABEL)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def plot_psychometric(rows: dict[str, Any], path: Path, run_id: str) -> None:
    theme.apply()
    plt.rcParams["svg.hashsalt"] = "pinglab-exp082"
    rates, means, sems = rows["rates"], rows["means"], rows["sems"]
    fig, axis = plt.subplots(figsize=(6.5, 3.66), constrained_layout=True)
    axis.errorbar(rates, means, yerr=sems, color=theme.INK_BLACK, marker="o", capsize=3)
    axis.set_xscale("log")
    axis.set_xticks(rates)
    axis.set_xticklabels([f"{rate:g}" for rate in rates])
    axis.set(xlabel="maximum-pixel input rate (Hz)", ylabel="accuracy", ylim=(0, 1))
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def plot_duration_rate_summary(
    rows: dict[str, Any],
    path: Path,
    run_id: str,
) -> None:
    """Exp048-Figure-2-style duration×rate map plus the 200-ms psychometric."""
    theme.apply()
    durations, rates = rows["durations"], rows["rates"]
    grid, sem = np.asarray(rows["grid"]), np.asarray(rows["grid_sem"])
    fig, (map_axis, curve_axis) = plt.subplots(
        1,
        2,
        figsize=(6.5, 3.25),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.15, 1)},
    )
    image = map_axis.imshow(
        grid, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis"
    )
    map_axis.set_xticks(range(len(durations)), [f"{value:g}" for value in durations])
    map_axis.set_yticks(range(len(rates)), [f"{value:g}" for value in rates])
    map_axis.set(xlabel="presentation = readout (ms)", ylabel="input rate (Hz)")
    fig.colorbar(image, ax=map_axis, label="accuracy")
    curve_axis.errorbar(
        rates, grid[:, -1], yerr=sem, color=theme.INK_BLACK, marker="o", capsize=3
    )
    curve_axis.set_xscale("log")
    curve_axis.set_xticks(rates, [f"{value:g}" for value in rates])
    curve_axis.tick_params(axis="x", labelrotation=60, labelsize=7)
    for label in curve_axis.get_xticklabels():
        label.set_horizontalalignment("right")
    curve_axis.set(xlabel="input rate (Hz)", ylabel="accuracy at 200 ms", ylim=(0, 1))
    curve_axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)
