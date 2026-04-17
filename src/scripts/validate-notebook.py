#!/usr/bin/env python3
"""Validate notebook entries: triple-existence and figure-reference resolution.

Checks:
    1. Triple-existence — every notebook slug has all three legs:
         - src/docs/src/pages/notebook/<slug>.{md,mdx}
         - src/pinglab/notebook/<slug>.py
         - src/docs/public/figures/notebook/<slug>/
    2. Figure references resolve — every /figures/notebook/<slug>/<file> path
       referenced from an entry exists on disk; every file in an entry's figure
       dir is referenced by that entry (orphan warning); references outside the
       entry's own slug are flagged (convention 4).

Usage:
    uv run python src/scripts/validate-notebook.py

Exit 0 if no failures (warnings allowed), 1 otherwise.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRIES_DIR = ROOT / "docs" / "src" / "pages" / "notebook"
REPROS_DIR = ROOT / "pinglab" / "notebook"
FIGURES_DIR = ROOT / "docs" / "public" / "figures" / "notebook"
PUBLIC_DIR = ROOT / "docs" / "public"

FIGURE_URL_RE = re.compile(r"/figures/notebook/([A-Za-z0-9._\-]+)/([A-Za-z0-9._\-]+)")
ORPHAN_EXEMPT_SUFFIXES = {".json"}
ORPHAN_EXEMPT_NAMES = {"numbers.json"}


def collect_slugs() -> set[str]:
    slugs: set[str] = set()
    if ENTRIES_DIR.is_dir():
        for p in ENTRIES_DIR.iterdir():
            if p.suffix in {".md", ".mdx"}:
                slugs.add(p.stem)
    if REPROS_DIR.is_dir():
        for p in REPROS_DIR.iterdir():
            if p.suffix == ".py" and not p.name.startswith("_"):
                slugs.add(p.stem)
    if FIGURES_DIR.is_dir():
        for p in FIGURES_DIR.iterdir():
            if p.is_dir():
                slugs.add(p.name)
    return slugs


def entry_path(slug: str) -> Path | None:
    for ext in (".mdx", ".md"):
        p = ENTRIES_DIR / f"{slug}{ext}"
        if p.exists():
            return p
    return None


def check_triple(slug: str) -> list[str]:
    issues: list[str] = []
    if entry_path(slug) is None:
        issues.append(f"missing entry: src/docs/src/pages/notebook/{slug}.{{md,mdx}}")
    if not (REPROS_DIR / f"{slug}.py").exists():
        issues.append(f"missing repro script: src/pinglab/notebook/{slug}.py")
    if not (FIGURES_DIR / slug).is_dir():
        issues.append(f"missing figure dir: src/docs/public/figures/notebook/{slug}/")
    return issues


def parse_references(entry: Path) -> list[tuple[str, str, int]]:
    """Return list of (referenced_slug, filename, line_number) tuples."""
    refs: list[tuple[str, str, int]] = []
    for lineno, line in enumerate(entry.read_text().splitlines(), start=1):
        for m in FIGURE_URL_RE.finditer(line):
            refs.append((m.group(1), m.group(2), lineno))
    return refs


def check_figures(slug: str) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for figure references in entry <slug>."""
    errors: list[str] = []
    warnings: list[str] = []
    entry = entry_path(slug)
    if entry is None:
        return errors, warnings

    refs = parse_references(entry)
    referenced_in_own_dir: set[str] = set()

    for ref_slug, name, lineno in refs:
        if ref_slug != slug:
            errors.append(
                f"cross-slug reference (line {lineno}): "
                f"/figures/notebook/{ref_slug}/{name} — entry slug is {slug}"
            )
            continue
        target = FIGURES_DIR / ref_slug / name
        if not target.exists():
            errors.append(
                f"broken reference (line {lineno}): /figures/notebook/{ref_slug}/{name}"
            )
        else:
            referenced_in_own_dir.add(name)

    fig_dir = FIGURES_DIR / slug
    if fig_dir.is_dir():
        for f in sorted(fig_dir.iterdir()):
            if not f.is_file():
                continue
            if f.name in ORPHAN_EXEMPT_NAMES:
                continue
            if f.suffix in ORPHAN_EXEMPT_SUFFIXES:
                continue
            if f.name not in referenced_in_own_dir:
                warnings.append(f"orphan figure: {f.name} (not referenced by entry)")

    return errors, warnings


def main() -> int:
    slugs = sorted(collect_slugs())
    if not slugs:
        print("no notebook slugs found", file=sys.stderr)
        return 1

    any_fail = False
    for slug in slugs:
        triple = check_triple(slug)
        fig_errors, fig_warnings = check_figures(slug)
        errors = triple + fig_errors
        if errors:
            any_fail = True
            print(f"{slug}: FAIL")
            for e in errors:
                print(f"  - {e}")
            for w in fig_warnings:
                print(f"  ! {w}")
        elif fig_warnings:
            print(f"{slug}: OK (with warnings)")
            for w in fig_warnings:
                print(f"  ! {w}")
        else:
            print(f"{slug}: OK")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
