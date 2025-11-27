
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from pinglab.plots.styles import save_both

def save_raster(
    spikes,
    path: Path,
    label: str,
    vertical_lines: list[float] | None = None,
    external_input: np.ndarray | None = None,
    dt: float | None = None,
) -> None:
    """
    Custom figure: external input (top) + spike raster (bottom).
    `spikes` is your Spikes model: times, ids, optional types.
    """

    def plot_fun():
        if external_input is not None:
            num_steps = external_input.shape[0]
            time = np.arange(num_steps) * dt
        else:
            time = None

        fig, ax_in, ax_ras = None, None, None

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
            ax_in.set_ylabel("I_ext (neuron 0)")
            ax_in.set_title(f"External input and spikes ({label})")
        else:
            ax_ras.set_title(f"Spike raster ({label})")

        # --- Bottom: spike raster ---
        times = spikes.times
        ids = spikes.ids
        types = getattr(spikes, "types", None)

        if types is not None:
            # 0 = E, 1 = I by convention
            mask_E = types == 0
            mask_I = types == 1
            ax_ras.scatter(
                times[mask_E], ids[mask_E], s=0.5, marker=".", label="E"
            )
            ax_ras.scatter(
                times[mask_I],
                ids[mask_I],
                s=0.5,
                marker=".",
                alpha=0.7,
                label="I",
            )
            ax_ras.legend(loc="upper right", fontsize=8)
        else:
            ax_ras.scatter(times, ids, s=1, marker=".")

        if vertical_lines is not None:
            for vline in vertical_lines:
                ax_ras.axvline(vline, linestyle="--", lw=0.7)

        ax_ras.set_xlabel("Time (ms)")
        ax_ras.set_ylabel("Neuron id")

        plt.tight_layout()

    save_both(path, plot_fun)
