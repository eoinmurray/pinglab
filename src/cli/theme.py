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


def apply() -> None:
    """Register the pinglab brutalist-cyberpunk matplotlib style.

    Black primary, deep-red accent, monospace typography, heavy spines.
    Stark contrast, limited palette, no soft visual assists. Idempotent
    — call once at the top of any plotting script.
    """
    import matplotlib as mpl
    from cycler import cycler
    from matplotlib.colors import LinearSegmentedColormap

    cmap_brand = LinearSegmentedColormap.from_list(
        "pinglab_brand", [PAPER, DEEP_RED, INK_BLACK]
    )
    try:
        mpl.colormaps.register(cmap_brand, force=True)
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
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
