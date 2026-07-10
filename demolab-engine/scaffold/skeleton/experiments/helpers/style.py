"""Shared matplotlib style for demolab figures (HOUSESTYLE H10-H16).

Import it for the rcParams side effect, then use INK / ACCENT / BAND and the .svg-vs-.png
suffix convention. One place for the whole lab's plot look, so figures don't drift from
runner to runner.

    from helpers import style
    fig, ax = plt.subplots()                 # 16:9 at column width, retina dpi, white bg
    ax.plot(x, y, color=_style.INK)           # near-black by default
    fig.savefig(dst)                          # dst.svg for line plots, dst.png for rasters
"""
import matplotlib as mpl

# Headless backend: demolab figures are savefig-only (never plt.show()), and the default GUI
# backend breaks where Tcl/Tk is absent — notably uv-managed pythons on Windows.
mpl.use("Agg")

INK = "#1a1a1a"     # near-black: the default for any single trace (H13)
ACCENT = "#c8102e"  # signal red: only to separate two traces sharing one axis (H13)
BAND = "#cccccc"    # light grey: error bands and fills (H13)

# Sized for the ~6.5in content column, dpi crisp on retina + print, white bg, tight box.
mpl.rcParams.update({
    "figure.figsize": (6.5, 3.66),   # 16:9 at column width; override per figure (H11-H12)
    "figure.dpi": 240,
    "savefig.dpi": 240,              # 6.5in x 240 = 1560 px, sharp on retina and in print (H11)
    "savefig.bbox": "tight",         # no dead margin skewing the aspect (H11)
    "font.size": 10,                 # ~body size once shown at column width (H11)
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.6,
    "axes.prop_cycle": mpl.cycler(color=[INK, ACCENT]),  # black first, red only if needed
    "figure.facecolor": "white",     # (H14)
    "savefig.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})
