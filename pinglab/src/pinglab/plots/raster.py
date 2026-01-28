"""Spike raster plot generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinglab.plots.styles import save_both
from pinglab.types import Spikes


def save_raster(
    spikes: Spikes,
    path: Path | str,
    label: str | None = None,
    show_labels: bool = True,
    show_legend: bool = True,
    show_title: bool = True,
    info_lines: list[str] | None = None,
    vertical_lines: list[float] | None = None,
    external_input: np.ndarray | None = None,
    dt: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
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
                ax_in.set_title(
                    f"External input and spikes ({label})"
                    if label is not None
                    else "External input and spikes"
                )
        else:
            if show_title:
                ax_ras.set_title(
                    f"Spike raster ({label})" if label is not None else "Spike raster"
                )

        # --- Bottom: spike raster ---
        times = spikes.times
        ids = spikes.ids
        types = getattr(spikes, "types", None)

        if types is not None:
            # 0 = E, 1 = I by convention
            mask_E = types == 0
            mask_I = types == 1
            ax_ras.scatter(times[mask_E], ids[mask_E], s=0.5, marker=".", label="E")
            ax_ras.scatter(
                times[mask_I],
                ids[mask_I],
                s=0.5,
                marker=".",
                alpha=0.7,
                label="I",
            )
            if show_legend:
                ax_ras.legend(loc="upper right", fontsize=8)
        else:
            ax_ras.scatter(times, ids, s=1, marker=".")

        if vertical_lines is not None:
            for vline in vertical_lines:
                ax_ras.axvline(vline, linestyle="--", lw=0.7)

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
        if ylim is not None:
            ax_ras.set_ylim(ylim[0], ylim[1])

        if show_labels:
            ax_ras.set_xlabel("Time (ms)")
            ax_ras.set_ylabel("Neuron id")

        plt.tight_layout()

    save_both(path, plot_fun)
