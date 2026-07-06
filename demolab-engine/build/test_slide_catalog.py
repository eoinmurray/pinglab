"""The slide-layout catalog (SLIDES.md D11) and the gallery deck must stay in sync.

The layouts are reusable as a *named catalog*, not a Typst component library: the gallery
`ar005.slide.typ` is the single source of truth — each layout is a block marked `// layout: <name>` —
and SLIDES.md D11 is the index. This asserts the two name sets are identical, so a catalog entry can
never point at a missing block, and a gallery slide can never go undocumented. Adding a layout means
adding both a marked gallery slide and a catalog entry.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GALLERY = REPO / "demolab-engine" / "scaffold" / "demo" / "writings" / "ar005.slide.typ"
GUIDE = REPO / "demolab-engine" / "guides" / "SLIDES.md"


def _gallery_names() -> list[str]:
    # Block markers: lines beginning `// layout: <name>` in the gallery deck.
    return re.findall(r"^// layout: ([a-z0-9-]+)", GALLERY.read_text(), re.M)


def _catalog_names() -> list[str]:
    # Catalog keys: `layout: <name>` code spans in the D11 index (the `// layout: <name>` mention in
    # the prose has a `//` before `layout:`, so it doesn't match).
    return re.findall(r"`layout: ([a-z0-9-]+)`", GUIDE.read_text())


def test_gallery_markers_unique():
    names = _gallery_names()
    assert names, "no `// layout:` markers found in the gallery deck"
    assert len(names) == len(set(names)), f"duplicate gallery markers: {names}"


def test_catalog_matches_gallery():
    gallery, catalog = set(_gallery_names()), set(_catalog_names())
    assert gallery == catalog, (
        "SLIDES.md D11 catalog and ar005 `// layout:` markers drifted.\n"
        f"  only in gallery deck: {sorted(gallery - catalog)}\n"
        f"  only in D11 catalog:  {sorted(catalog - gallery)}"
    )
