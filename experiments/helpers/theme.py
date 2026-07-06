"""Canonical color palette for pinglab figures.

Mirrors the @theme block in src/docs/src/styles/global.css. The two files
are the source of truth for their respective stacks (matplotlib here,
Tailwind/CSS there) and are kept in sync by hand — if you add or change a
color in one, update the other in the same commit. Both files list
identical token names and values.
"""

# Ink ramp — text hierarchy from strongest to faintest.
INK_STRONG = "#1a1a1a"
INK = "#222"
INK_SOFT = "#333"
DIM = "#555"
MUTED = "#666"
MUTED_SOFT = "#777"
LABEL = "#888"
FAINT = "#bbb"
UNDERLINE = "#ccc"

# Rules, backgrounds.
RULE = "#e7e5df"
RULE_WARM = "#d9d5c8"
PAPER_TINT = "#fafaf7"
PAPER = "#fff"

# Warning / accent red. One canonical red across web and plots.
DANGER = "#cc4444"

# Brutalist cyberpunk palette: black primary, deep-red accent, with a
# grey ramp plus two cyberpunk accents (cyan, amber) that open the
# palette enough for 5–6 distinct series without losing the vibe.
# Use the greys for related sub-families; reach for cyan/amber when
# series need to read as *categorically distinct*, not gradations.
# Enforced house plot pair: the two primary series colours. Use these for
# any one- or two-series plot (ink for the first/control, red for the
# second/contrast); reach for the accents below only for a 3rd+ series.
INK_BLACK = "#1a1a1a"  # ink — near-black, the default series colour
DEEP_RED = "#c8102e"   # signal red — the contrast series colour
DEEP_RED_LIGHT = "#e0566a"
ELECTRIC_CYAN = "#00b4d8"
AMBER = "#e89400"
GREY_DARK = "#3a3a3a"
GREY_MID = "#6a6a6a"
GREY_LIGHT = "#a0a0a0"

# Back-compat aliases for callers that import CAT_*.
CAT_BLUE = INK_BLACK
CAT_BLUE_LIGHT = GREY_MID
CAT_BLUE_DARK = INK_BLACK
CAT_RED = DEEP_RED
CAT_GREEN = GREY_DARK
CAT_PURPLE = GREY_MID
CAT_ORANGE = DEEP_RED
CAT_BROWN = GREY_DARK

CYCLE = [INK_BLACK, DEEP_RED, ELECTRIC_CYAN, AMBER, GREY_DARK, GREY_MID]

# Typography sizes (pt). Tokens, not magic numbers — every figure picks
# from this ladder so cross-figure comparisons line up visually. Stay
# disciplined: prefer reusing one of these over inventing a new size.
SIZE_BASE = 10.5      # default text
SIZE_TITLE = 11       # plot title (bold)
SIZE_LABEL = 10       # axis labels
SIZE_TICK = 9         # tick labels
SIZE_LEGEND = 9       # legend body
SIZE_ANNOTATION = 8   # in-plot annotations, secondary text
SIZE_CAPTION = 8      # figure captions / footers

# Paper / print profile. When PAPER_MODE is on, figures are rendered for
# a LaTeX manuscript (eLife class): smaller absolute fonts so type lands
# at print point sizes when a figure is included at its true width, thinner
# rules, and vector PDF output. Toggle with set_paper_mode() BEFORE the
# plotting code runs — it reassigns the SIZE_* ladder above (which the
# notebook plot functions read directly) and flips apply() into print rc.
PAPER_MODE = False
_SCREEN_SIZES = dict(BASE=10.5, TITLE=11, LABEL=10, TICK=9,
                     LEGEND=9, ANNOTATION=8, CAPTION=8)
_PAPER_SIZES = dict(BASE=8.0, TITLE=8.5, LABEL=8.0, TICK=7.0,
                    LEGEND=7.0, ANNOTATION=6.5, CAPTION=6.5)


def set_paper_mode(on: bool = True) -> None:
    """Switch the typography ladder between screen and print. Reassigns the
    module-level SIZE_* tokens so notebook plot code (which reads them at call
    time) picks up the print sizes, and records the mode for apply()."""
    global PAPER_MODE, SIZE_BASE, SIZE_TITLE, SIZE_LABEL
    global SIZE_TICK, SIZE_LEGEND, SIZE_ANNOTATION, SIZE_CAPTION
    PAPER_MODE = on
    s = _PAPER_SIZES if on else _SCREEN_SIZES
    SIZE_BASE, SIZE_TITLE, SIZE_LABEL = s["BASE"], s["TITLE"], s["LABEL"]
    SIZE_TICK, SIZE_LEGEND = s["TICK"], s["LEGEND"]
    SIZE_ANNOTATION, SIZE_CAPTION = s["ANNOTATION"], s["CAPTION"]


def apply() -> None:
    """Register the pinglab brutalist-cyberpunk matplotlib style.

    Black primary, deep-red accent, monospace typography, heavy spines.
    Stark contrast, limited palette, no soft visual assists. Idempotent
    — call once at the top of any plotting script.
    """
    import matplotlib as mpl
    from cycler import cycler
    from matplotlib.colors import LinearSegmentedColormap

    # apply() is idempotent and called once per plot, so only register the
    # colormap the first time — re-registering (even with force=True) emits a
    # UserWarning that pollutes the run log.
    if "pinglab_brand" not in mpl.colormaps:
        cmap_brand = LinearSegmentedColormap.from_list(
            "pinglab_brand", [PAPER, DEEP_RED, INK_BLACK]
        )
        try:
            mpl.colormaps.register(cmap_brand)
        except Exception:
            pass

    mpl.rcParams.update({
        "font.family": "monospace",
        "font.monospace": [
            "JetBrains Mono", "IBM Plex Mono", "Menlo", "Consolas",
            "Courier New", "DejaVu Sans Mono",
        ],
        "font.size": SIZE_BASE,
        "font.weight": "normal",

        "axes.titlesize": SIZE_TITLE,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.titlecolor": INK_BLACK,
        "axes.labelsize": SIZE_LABEL,
        "axes.labelcolor": INK_BLACK,
        "axes.labelweight": "normal",

        "axes.edgecolor": INK_BLACK,
        "axes.linewidth": 1.4,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.facecolor": PAPER,
        "figure.facecolor": PAPER,

        "axes.prop_cycle": cycler(color=CYCLE),

        "xtick.color": INK_BLACK,
        "ytick.color": INK_BLACK,
        "xtick.labelcolor": INK_BLACK,
        "ytick.labelcolor": INK_BLACK,
        "xtick.labelsize": SIZE_TICK,
        "ytick.labelsize": SIZE_TICK,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",

        "grid.color": INK_BLACK,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.15,
        "axes.grid": False,
        "axes.axisbelow": True,

        "legend.frameon": True,
        "legend.edgecolor": INK_BLACK,
        "legend.facecolor": PAPER,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.borderpad": 0.5,
        "legend.fontsize": SIZE_LEGEND,
        "legend.labelcolor": INK_BLACK,

        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "butt",
        "lines.solid_joinstyle": "miter",

        "patch.edgecolor": INK_BLACK,
        "patch.linewidth": 1.0,
        "patch.force_edgecolor": True,

        "image.cmap": "pinglab_brand",

        "savefig.facecolor": PAPER,
        "savefig.dpi": 240,       # H11: ≥200 floor for crisp retina + print
        "savefig.bbox": "tight",
    })

    if PAPER_MODE:
        # Print refinements: lighter rules so they don't bloat at small
        # scale, vector PDF with embedded editable text, generous DPI for
        # any rasterised inset (scatter clouds, images).
        mpl.rcParams.update({
            "font.size": SIZE_BASE,
            "axes.titlesize": SIZE_TITLE,
            "axes.labelsize": SIZE_LABEL,
            "xtick.labelsize": SIZE_TICK,
            "ytick.labelsize": SIZE_TICK,
            "legend.fontsize": SIZE_LEGEND,
            "axes.linewidth": 0.8,
            "axes.titlepad": 6,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.3,
            "patch.linewidth": 0.8,
            "legend.borderpad": 0.4,
            "legend.handlelength": 1.6,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })
