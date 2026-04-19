#!/usr/bin/env python3
"""Validate notebook entries against conventions in src/docs/src/pages/llm-conventions.md.

Checks:
    1. Triple-existence — every notebook slug has all three legs:
         - src/docs/src/pages/notebook/<slug>.{md,mdx}
         - src/pinglab/notebook/<slug>.py
         - src/docs/public/figures/notebook/<slug>/
    2. Figure references resolve — every /figures/notebook/<slug>/<file> path
       referenced from an entry exists on disk; every file in an entry's figure
       dir is referenced by that entry (orphan warning); references outside the
       entry's own slug are flagged (convention 4).
    3. H2 skeleton (convention 1) — the first five H2 headings of each entry
       are Introduction / Method / Findings / Implications / Next steps, in
       that order.
    4. Caption after image (convention 6) — every markdown image and <video>
       tag is followed (after optional blank lines) by a line formatted as
       italic (*...*).

Frontmatter `structure: paper` exempts an entry from both the H2 skeleton
and caption checks — used for paper-style drafts with custom sectioning.

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

FIGURE_URL_RE = re.compile(r"/figures/notebook/([A-Za-z0-9._\-]+)/([A-Za-z0-9._\-]+)")
H2_RE = re.compile(r"^##\s+(.+?)\s*$")
IMAGE_LINE_RE = re.compile(r"^!\[[^\]]*\]\([^)]+\)\s*$")
VIDEO_LINE_RE = re.compile(r"<video\b")
ITALIC_LINE_RE = re.compile(r"^\*[^*].*\*\s*$")
FRONTMATTER_RE = re.compile(r"^---\s*$")

CANONICAL_H2S = ["Introduction", "Method", "Findings", "Implications", "Next steps"]
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


def split_frontmatter(text: str) -> tuple[dict[str, str], list[str]]:
    """Return (frontmatter_dict, body_lines). Frontmatter parsing is simple:
    key: value pairs, no nesting, strings unquoted."""
    lines = text.splitlines()
    if not lines or not FRONTMATTER_RE.match(lines[0]):
        return {}, lines
    fm: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if FRONTMATTER_RE.match(line):
            return fm, lines[i + 1 :]
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm, lines  # unterminated frontmatter — treat all as body


def iter_body_lines(lines: list[str]):
    """Yield (lineno, line) skipping fenced code blocks. lineno is 1-based
    against the original file (approximate: frontmatter-offset lines not
    tracked here, caller adjusts if needed)."""
    in_code = False
    for i, line in enumerate(lines, start=1):
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        yield i, line


def check_h2_skeleton(slug: str) -> list[str]:
    entry = entry_path(slug)
    if entry is None:
        return []
    fm, body_lines = split_frontmatter(entry.read_text())
    if fm.get("structure") == "paper":
        return []
    h2s: list[str] = []
    for _, line in iter_body_lines(body_lines):
        m = H2_RE.match(line)
        if m:
            h2s.append(m.group(1).strip())
    first_five = h2s[:5]
    if first_five == CANONICAL_H2S:
        return []
    if len(h2s) < 5:
        return [
            f"H2 skeleton: only {len(h2s)} H2(s) found — "
            f"expected {CANONICAL_H2S} as the first five"
        ]
    return [
        "H2 skeleton: first five H2s are "
        f"{first_five} — expected {CANONICAL_H2S}"
    ]


def check_captions(slug: str) -> list[str]:
    entry = entry_path(slug)
    if entry is None:
        return []
    fm, body_lines = split_frontmatter(entry.read_text())
    if fm.get("structure") == "paper":
        return []
    # Preserve original line numbers from the body portion. Frontmatter
    # offset isn't reported — captions sit in the body, so body-relative
    # numbers are good enough.
    issues: list[str] = []
    in_code = False
    i = 0
    while i < len(body_lines):
        line = body_lines[i]
        if line.startswith("```"):
            in_code = not in_code
            i += 1
            continue
        if in_code:
            i += 1
            continue
        is_image = bool(IMAGE_LINE_RE.match(line))
        is_video = bool(VIDEO_LINE_RE.search(line)) and "src=" in line
        if is_image or is_video:
            j = i + 1
            while j < len(body_lines) and body_lines[j].strip() == "":
                j += 1
            caption = body_lines[j] if j < len(body_lines) else ""
            if not ITALIC_LINE_RE.match(caption):
                kind = "image" if is_image else "video"
                label = line.strip()[:60] + ("…" if len(line.strip()) > 60 else "")
                issues.append(
                    f"missing italic caption after {kind} (body line {i + 1}): {label}"
                )
        i += 1
    return issues


def parse_references(entry: Path) -> list[tuple[str, str, int]]:
    refs: list[tuple[str, str, int]] = []
    for lineno, line in enumerate(entry.read_text().splitlines(), start=1):
        for m in FIGURE_URL_RE.finditer(line):
            refs.append((m.group(1), m.group(2), lineno))
    return refs


def check_figures(slug: str) -> tuple[list[str], list[str]]:
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
        h2_errors = check_h2_skeleton(slug) if not triple else []
        caption_errors = check_captions(slug) if not triple else []
        errors = triple + fig_errors + h2_errors + caption_errors
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
