"""The original four panels, drawn exclusively from saved summaries."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure


def plot_controls(summary: dict, cfg: dict, out_path: Path) -> None:
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0), sharex=True)
    colors = [theme.GREY_MID, theme.DEEP_RED, theme.INK_BLACK]
    markers = ["s", "^", "o"]

    controls = [
        ("fixed_total", "Fixed summed coupling $G_{IE}$", cfg["reference_g_ie"]),
        ("fixed_synapse", "Fixed mean synapse $j_{IE}$", cfg["reference_j_ie"]),
    ]
    for col, (control, title, levels) in enumerate(controls):
        axes[0, col].set_title(title, fontsize=theme.SIZE_TITLE)
        for level, color, marker in zip(levels, colors, markers):
            cells = [
                summary[control][f"{level:.12g}"][str(n)] for n in cfg["n_i_sweep"]
            ]
            if control == "fixed_total":
                label = f"$G_{{IE}}={level:g}$ μS"
            else:
                label = f"$j_{{IE}}={level * 1000:.2f}$ nS"
            for row, metric in enumerate(("r_e_hz", "r_i_hz")):
                means = [cell[f"{metric}_mean"] for cell in cells]
                sds = [cell[f"{metric}_sd"] for cell in cells]
                axes[row, col].errorbar(
                    cfg["n_i_sweep"],
                    means,
                    yerr=sds,
                    marker=marker,
                    color=color,
                    lw=1.5,
                    ms=5.5,
                    capsize=2,
                    label=label,
                )

    for row, ylabel in enumerate(("E rate  (Hz / cell)", "I rate  (Hz / cell)")):
        axes[row, 0].set_ylabel(ylabel, fontsize=theme.SIZE_LABEL)
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(cfg["n_i_sweep"], labels=[str(n) for n in cfg["n_i_sweep"]])
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("inhibitory pool size  $N_I$", fontsize=theme.SIZE_LABEL)
    axes[0, 0].legend(frameon=False, fontsize=theme.SIZE_LEGEND - 1)
    axes[0, 1].legend(frameon=False, fontsize=theme.SIZE_LEGEND - 1)

    theme.label_panels(axes.flat)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)
