"""Spike raster plot generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinglab.plots.styles import save_both
from pinglab.backends.types import Spikes


def save_raster(
    spikes: Spikes,
    path: Path | str,
    label: str | None = None,
    show_labels: bool = True,
    show_legend: bool = True,
    show_title: bool = True,
    info_lines: list[str] | None = None,
    vertical_lines: list[float] | None = None,
    vertical_line_kwargs: dict | None = None,
    external_input: np.ndarray | None = None,
    dt: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    layer_bounds: list[tuple[int, int, str]] | None = None,
    layer_order: list[str] | None = None,
) -> None:
    """
    Save spike raster plot with optional external input overlay.

    Parameters:
        spikes: Spike data with times, ids, and optional types
        path: Output file path (without extension)
        label: Plot title label
        vertical_lines: Optional list of times to mark with vertical lines
        external_input: Optional external input array of shape (num_steps, N)
        dt: Time step in ms (required if external_input is provided)
        layer_bounds: Optional layer delineation as (start_id, stop_id, label)
        layer_order: Optional display order of layer labels, e.g. ["I", "E1", "E2", "E3"]
    """

    if external_input is not None and dt is None:
        raise ValueError("dt must be provided if external_input is given.")

    def plot_fun() -> None:
        if external_input is not None:
            num_steps = external_input.shape[0]
            time = np.arange(num_steps) * dt
        else:
            time = None

        fig = None
        ax_in = None
        ax_ras = None

        if external_input is None:
            fig, ax_ras = plt.subplots(1, 1, figsize=(8, 8))
        else:
            fig, (ax_in, ax_ras) = plt.subplots(
                2, 1,
                figsize=(8, 8),
                sharex=True,
                height_ratios=[1, 2]
            )

        if ax_in is not None and external_input is not None:
            # --- Top: external input for one neuron (0) ---
            ax_in.plot(time, external_input[:, 0], lw=0.3)
            if show_labels:
                ax_in.set_ylabel("I_ext (neuron 0)")
            if show_title:
                ax_in.set_title(f"Raster | {label}" if label is not None else "Raster")
        else:
            if show_title:
                ax_ras.set_title(f"Raster | {label}" if label is not None else "Raster")

        # --- Bottom: spike raster ---
        times = np.asarray(spikes.times)
        ids = np.asarray(spikes.ids)
        plot_ids = ids.astype(float, copy=True)
        display_bounds: list[tuple[int, int, str]] = []

        if layer_bounds:
            normalized_bounds = [
                (int(start), int(stop), str(label)) for start, stop, label in layer_bounds
            ]
            if layer_order:
                rank = {str(label): idx for idx, label in enumerate(layer_order)}
                ordered_bounds = sorted(
                    normalized_bounds,
                    key=lambda b: rank.get(str(b[2]), len(rank)),
                )
            else:
                ordered_bounds = normalized_bounds

            cursor = 0
            for src_start, src_stop, src_label in ordered_bounds:
                size = int(src_stop - src_start)
                if size <= 0:
                    continue
                mask = (ids >= src_start) & (ids < src_stop)
                if np.any(mask):
                    plot_ids[mask] = cursor + (ids[mask] - src_start)
                display_bounds.append((cursor, cursor + size, src_label))
                cursor += size

        types = getattr(spikes, "types", None)

        if types is not None:
            # 0 = E, 1 = I by convention
            mask_E = types == 0
            mask_I = types == 1
            ax_ras.scatter(times[mask_E], plot_ids[mask_E], s=2.2, marker=".", label="E")
            ax_ras.scatter(
                times[mask_I],
                plot_ids[mask_I],
                s=2.2,
                marker=".",
                alpha=0.7,
                label="I",
            )
            if show_legend:
                ax_ras.legend(loc="upper right", fontsize=8)
        else:
            ax_ras.scatter(times, plot_ids, s=2.8, marker=".")

        if vertical_lines is not None:
            vline_style = {"linestyle": "--", "lw": 0.7}
            if vertical_line_kwargs:
                vline_style.update(vertical_line_kwargs)
            for vline in vertical_lines:
                ax_ras.axvline(vline, **vline_style)

        if info_lines:
            info_text = "\n".join(info_lines)
            face = ax_ras.get_facecolor()
            # Relative luminance for contrast decision
            luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
            text_color = "white" if luminance < 0.5 else "black"
            ax_ras.text(
                0.98,
                0.02,
                info_text,
                transform=ax_ras.transAxes,
                va="bottom",
                ha="right",
                fontsize=10,
                color=text_color,
                bbox=dict(
                    facecolor=face,
                    edgecolor=face,
                    boxstyle="round,pad=0.5",
                    alpha=1.0,
                ),
            )

        if time is not None:
            ax_ras.set_xlim(time[0], time[-1])
        if xlim is not None:
            ax_ras.set_xlim(xlim[0], xlim[1])
        x_min, x_max = ax_ras.get_xlim()

        if display_bounds:
            for start, stop, layer_label in display_bounds:
                ax_ras.hlines(
                    y=float(start),
                    xmin=float(x_min),
                    xmax=float(x_max),
                    linewidth=0.8,
                    linestyles="--",
                    alpha=0.35,
                )
                y_center = (float(start) + float(stop - 1)) * 0.5
                ax_ras.text(
                    float(x_max) * 1.005,
                    y_center,
                    layer_label,
                    fontsize=10,
                    va="center",
                    ha="left",
                    clip_on=False,
                )
            ax_ras.hlines(
                y=float(display_bounds[-1][1]),
                xmin=float(x_min),
                xmax=float(x_max),
                linewidth=0.8,
                linestyles="--",
                alpha=0.35,
            )

        if ylim is not None:
            ax_ras.set_ylim(ylim[0], ylim[1])
        elif display_bounds:
            ax_ras.set_ylim(float(display_bounds[-1][1] + 2), -2.0)

        if show_labels:
            ax_ras.set_xlabel("Time (ms)")
            ax_ras.set_ylabel("Neuron id")

        plt.tight_layout()

    save_both(path, plot_fun)
