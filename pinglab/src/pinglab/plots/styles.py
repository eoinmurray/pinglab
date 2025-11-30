import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from pathlib import Path

LIGHT = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "grid.color": "#e0e0e0",
    "patch.edgecolor": "#333333",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "axes.prop_cycle": cycler(color=["#000000", "#ff0000", "#0000ff", "#00aa00"]),
}

DARK = {
    "figure.facecolor": "#18181b",
    "axes.facecolor": "#18181b",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#cccccc",
    "text.color": "#cccccc",
    "xtick.color": "#cccccc",
    "ytick.color": "#cccccc",
    "grid.color": "#333333",
    "patch.edgecolor": "#cccccc",
    "savefig.facecolor": "#18181b",
    "savefig.edgecolor": "#18181b",
    "axes.prop_cycle": cycler(color=["#ffffff", "#ff6b6b", "#4dabf7", "#51cf66"]),
}

figsize = (8, 8)

def apply_style(style):
    mpl.rcParams.update(style)

def save_both(name, plot_fn):
    name_path = Path(name)
    base_name = str(name_path).replace('.png', '')

    apply_style(LIGHT)
    plot_fn()
    plt.savefig(f"{base_name}_light.png", dpi=300)
    plt.close()

    apply_style(DARK)
    plot_fn()
    plt.savefig(f"{base_name}_dark.png", dpi=300)
    plt.close()
