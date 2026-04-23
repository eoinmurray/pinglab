"""Canonical color palette for pinglab figures.

Mirrors the @theme block in src/docs/src/styles/global.css. The two files
are the source of truth for their respective stacks (matplotlib here,
Tailwind/CSS there) and are kept in sync by hand — if you add or change a
color in one, update the other in the same commit. Both files list
identical token names and values.
"""

# Ink ramp — text hierarchy from strongest to faintest.
INK_STRONG  = "#1a1a1a"
INK         = "#222"
INK_SOFT    = "#333"
DIM         = "#555"
MUTED       = "#666"
MUTED_SOFT  = "#777"
LABEL       = "#888"
FAINT       = "#bbb"
UNDERLINE   = "#ccc"

# Rules, backgrounds.
RULE        = "#e7e5df"
RULE_WARM   = "#d9d5c8"
PAPER_TINT  = "#fafaf7"
PAPER       = "#fff"

# Warning / accent red. One canonical red across web and plots.
DANGER      = "#cc4444"

# Categorical ramp for multi-series plots (matplotlib tab10 subset).
CAT_BLUE    = "#1f77b4"
CAT_GREEN   = "#2ca02c"
CAT_PURPLE  = "#9467bd"
CAT_RED     = "#d62728"
CAT_ORANGE  = "#ff7f0e"
