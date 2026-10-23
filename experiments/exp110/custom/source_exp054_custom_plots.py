"""Preserved exp054 figure geometry; consume saved analysis only.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from contextlib import contextmanager

import matplotlib.pyplot as plt

import numpy as np

from experiments import theme

from experiments.exp110.custom import source_exp054_recipe as recipe

_DEFAULT = recipe.configuration()

DT_MS = _DEFAULT["dt_ms"]

WEI_MEAN_GRID = _DEFAULT["wei_mean"]

WIE_MEAN_GRID = _DEFAULT["wie_mean"]

GRID_WIN_MS = _DEFAULT["display_window_ms"]

TURNON_POINTS = recipe.turnon_points(_DEFAULT)


@contextmanager
def configured(cfg):
    names = {
        "DT_MS": cfg["dt_ms"],
        "BIN_MS": cfg["bin_ms"],
        "WEI_MEAN_GRID": cfg["wei_mean"],
        "WIE_MEAN_GRID": cfg["wie_mean"],
        "GRID_DISPLAY_STRIDE": cfg["display_stride"],
        "GRID_WIN_MS": cfg["display_window_ms"],
        "TURNON_POINTS": recipe.turnon_points(cfg),
    }
    old = {key: globals().get(key) for key in names}
    paper = theme.PAPER_MODE
    try:
        globals().update(names)
        with plt.rc_context():
            theme.apply()
            plt.rcParams["savefig.bbox"] = "standard"
            yield
    finally:
        for key, value in old.items():
            if value is None:
                globals().pop(key, None)
            else:
                globals()[key] = value
        theme.set_paper_mode(paper)


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def turnon_map_panels(grid):
    """The three scalar-map specs (E rate, I rate edge-clipped, contrast) for the
    turn-on grid: (values 2-D array, title, colour vmax, cell-label fmt, mark A/B/C).
    Single source of truth so exp054 Fig 1 and the onset super-compound agree."""
    e = np.array([[c["rate_hz"] for c in row] for row in grid])
    i = np.array([[c["rate_i_hz"] for c in row] for row in grid])
    ct = np.array([[c["contrast"] for c in row] for row in grid])
    i_fin = i[np.isfinite(i)]
    i_cap = float(np.nanpercentile(i_fin, 92)) if i_fin.size else None
    e_max = float(np.nanmax(e)) if np.isfinite(e).any() else 1.0
    return [
        (e, "E firing rate (Hz)", e_max, "{:.0f}", False),
        (i, "I firing rate (Hz; edge clipped)", i_cap, "{:.0f}", False),
        (ct, "lobe–trough contrast", 1.0, "{:.2f}", True),
    ]


def draw_turnon_map(
    ax,
    vals,
    *,
    title,
    vmax_color,
    fmt,
    mark=False,
    show_y=True,
    mark_labels=None,
    cell_fontsize=4.5,
):
    """Draw one W_EI × W_IE scalar heatmap into ax (per-cell value labels + optional
    sample-point markers). Shared by exp054 Figure 1 and the onset super-compound —
    restyle here and both figures update. mark_labels overrides the circle letters
    (the super-compound relabels them D/E/F to match its raster panel letters).
    cell_fontsize scales the per-cell value labels down for the small super-compound
    panels, where the default (tuned for Figure 1's wider maps) would overlap."""
    ax.imshow(
        vals, origin="lower", aspect="equal", cmap="Greys", vmin=0.0, vmax=vmax_color
    )
    xt = range(0, len(WEI_MEAN_GRID), 2)
    ax.set_xticks(list(xt))
    ax.set_xticklabels(
        [f"{WEI_MEAN_GRID[t]:.1f}" for t in xt], fontsize=theme.SIZE_TICK - 1
    )
    if show_y:
        yt = range(0, len(WIE_MEAN_GRID), 2)
        ax.set_yticks(list(yt))
        ax.set_yticklabels(
            [f"{WIE_MEAN_GRID[t]:.1f}" for t in yt], fontsize=theme.SIZE_TICK - 1
        )
        ax.set_ylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
    else:
        ax.set_yticks([])
    ax.set_xlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
    ax.set_title(
        title, loc="center", fontsize=theme.SIZE_LABEL - 1, fontweight="semibold"
    )
    for iy in range(vals.shape[0]):
        for ix in range(vals.shape[1]):
            v = vals[iy, ix]
            if np.isfinite(v):
                frac = (v / vmax_color) if vmax_color else 0.0
                ax.text(
                    ix,
                    iy,
                    fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    color="white" if frac > 0.55 else theme.INK_BLACK,
                )
    if mark:
        labels = (
            mark_labels if mark_labels is not None else [p[0] for p in TURNON_POINTS]
        )
        for (_, wei_i, wie_i), disp in zip(TURNON_POINTS, labels):
            ax.text(
                wei_i,
                wie_i,
                disp,
                ha="center",
                va="center",
                zorder=6,
                fontsize=theme.SIZE_TICK,
                fontweight="bold",
                color=theme.INK_BLACK,
                clip_on=False,
                bbox={
                    "boxstyle": "circle,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": theme.INK_BLACK,
                    "linewidth": 1.2,
                },
            )


def draw_turnon_raster(ax, cell, *, label, wei_i, wie_i, show_label=True):
    """Draw one representative sample-point raster into ax (E black below, I red
    above). Shared by exp054 Figure 1 and the onset super-compound. show_label=False
    drops the leading letter from the title (the super-compound identifies each
    raster by its prominent panel letter instead)."""
    e_ = cell["e"]
    n_e = e_.shape[1]
    T = e_.shape[0]
    t_ms = np.arange(T) * DT_MS
    e_idx, e_t = np.where(e_.T)
    ax.scatter(t_ms[e_t], e_idx, s=0.5, c=theme.INK_BLACK, marker="|", linewidths=0.35)
    total = n_e
    if cell["i"] is not None:
        ii = cell["i"]
        n_i = ii.shape[1]
        i_idx, i_t = np.where(ii.T)
        ax.scatter(
            t_ms[i_t], n_e + i_idx, s=0.5, c=theme.DEEP_RED, marker="|", linewidths=0.35
        )
        ax.axhline(n_e, color=theme.GREY_MID, lw=0.5, alpha=0.6)
        total = n_e + n_i
    ax.set_ylim(0, total)
    ax.set_xlim(0, GRID_WIN_MS)
    ax.set_yticks([])
    prefix = f"{label}   " if show_label else ""
    ax.set_title(
        f"{prefix}W_EI={WEI_MEAN_GRID[wei_i]:.1f}, W_IE={WIE_MEAN_GRID[wie_i]:.1f}"
        f"   contrast={cell['contrast']:.2f}",
        loc="left",
        fontsize=theme.SIZE_TICK - 1,
    )
    ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    _despine(ax)
