"""Renumber notebooks atomically — insert at, or delete, an entry slot.

Usage:
    uv run python -m pinglab.notebooks._renumber insert <N> [--dry-run]
    uv run python -m pinglab.notebooks._renumber delete <N> [--dry-run]

`insert N` shifts every notebook with entry >= N up by one (nbN -> nb(N+1),
nb(N+1) -> nb(N+2), ...) so slot N is free for a new notebook. `delete N`
removes nbN and shifts every entry > N down by one to close the gap.

What gets touched per renamed notebook:
  - src/docs/src/pages/notebooks/nbNNN.mdx          (file rename + frontmatter)
  - src/pinglab/notebooks/nbNNN.py                  (file rename)
  - src/docs/public/figures/notebooks/nbNNN/        (dir rename)
  - src/artifacts/notebooks/nbNNN/                  (dir rename, if present)
  - any tracked .py/.md/.mdx/.astro/.ts/.tsx/.json/.txt that mentions nbNNN

Only the slug pattern `nbNNN` (exact 3-digit) is rewritten, plus the renamed
MDX's `entry:` and `"NNN — ..."` title prefix. Free-text mentions like
"entry 002" inside docstrings are NOT auto-rewritten — fix those by hand
if you care.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PAGES_DIR = REPO / "src/docs/src/pages/notebooks"
RUNNERS_DIR = REPO / "src/pinglab/notebooks"
FIGURES_DIR = REPO / "src/docs/public/figures/notebooks"
ARTIFACTS_DIR = REPO / "src/artifacts/notebooks"

TEXT_EXTS = {".py", ".md", ".mdx", ".astro", ".ts", ".tsx", ".json", ".txt"}
SCAN_ROOTS = [
    REPO / "src/docs/src",
    REPO / "src/docs/public",
    REPO / "src/pinglab",
    REPO / "src/artifacts",
]
SKIP_DIR_NAMES = {"node_modules", "__pycache__", ".git", "dist", ".astro"}

SLUG_RE = re.compile(r"nb(\d{3})")


def slug(n: int) -> str:
    return f"nb{n:03d}"


def existing_entries() -> list[int]:
    out = []
    for p in PAGES_DIR.glob("nb*.mdx"):
        m = re.fullmatch(r"nb(\d{3})", p.stem)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def build_rename_map(action: str, n: int, entries: list[int]) -> dict[int, int]:
    """Return {old_entry: new_entry}. Only entries that move are included."""
    if action == "insert":
        # entries >= n shift up by one
        return {e: e + 1 for e in entries if e >= n}
    if action == "delete":
        if n not in entries:
            sys.exit(f"delete {n}: nb{n:03d} does not exist")
        # entries > n shift down by one
        return {e: e - 1 for e in entries if e > n}
    raise ValueError(action)


def iter_text_files():
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix not in TEXT_EXTS:
                continue
            if any(part in SKIP_DIR_NAMES for part in p.parts):
                continue
            yield p


def rewrite_text(content: str, slug_map: dict[str, str]) -> str:
    """Single-pass replace: nbXXX -> slug_map[nbXXX] when present."""

    def sub(m: re.Match) -> str:
        return slug_map.get(m.group(0), m.group(0))

    return SLUG_RE.sub(sub, content)


def rewrite_mdx_frontmatter(content: str, new_entry: int) -> str:
    """Update `entry: N` and `title: "NNN — ...` inside the leading frontmatter block."""
    parts = content.split("---", 2)
    if len(parts) < 3 or parts[0].strip():
        return content  # no frontmatter
    fm = parts[1]
    fm = re.sub(r"^entry:\s*\d+\s*$", f"entry: {new_entry}", fm, count=1, flags=re.M)
    fm = re.sub(
        r'^(title:\s*")(\d{3})(\s*[—–-])',
        lambda m: f"{m.group(1)}{new_entry:03d}{m.group(3)}",
        fm,
        count=1,
        flags=re.M,
    )
    return "---" + fm + "---" + parts[2]


def safe_move(src: Path, dst: Path, dry: bool) -> None:
    if not src.exists():
        return
    if dst.exists():
        sys.exit(f"refusing to overwrite existing path: {dst}")
    print(f"  mv {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def renumber(action: str, n: int, dry: bool) -> None:
    entries = existing_entries()
    rename_map = build_rename_map(action, n, entries)

    # If deleting, remove the target first (before remap of others).
    delete_slug = slug(n) if action == "delete" else None

    if not rename_map and not delete_slug:
        print("nothing to do")
        return

    print(f"action: {action} {n}  (dry-run={dry})")
    print(f"existing entries: {entries}")
    if delete_slug:
        print(f"will delete: {delete_slug}")
    print(f"will rename: {sorted(rename_map.items())}")

    slug_map = {slug(old): slug(new) for old, new in rename_map.items()}
    # Two-pass via tmp prefix to avoid collisions on path renames.
    tmp_prefix = "__renum_tmp__"

    # ── Step 1: delete the target (if action=delete) ──────────────────────
    if delete_slug:
        for base in [PAGES_DIR, RUNNERS_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
            for ext in (".mdx", ".py", ""):
                p = base / f"{delete_slug}{ext}"
                if p.exists():
                    print(f"  rm -r {p.relative_to(REPO)}")
                    if not dry:
                        if p.is_dir():
                            shutil.rmtree(p)
                        else:
                            p.unlink()

    # ── Step 2: rename paths via tmp prefix ───────────────────────────────
    print("renaming paths (pass 1: -> tmp):")
    for old in rename_map:
        old_slug = slug(old)
        safe_move(
            PAGES_DIR / f"{old_slug}.mdx",
            PAGES_DIR / f"{tmp_prefix}{old_slug}.mdx",
            dry,
        )
        safe_move(
            RUNNERS_DIR / f"{old_slug}.py",
            RUNNERS_DIR / f"{tmp_prefix}{old_slug}.py",
            dry,
        )
        safe_move(FIGURES_DIR / old_slug, FIGURES_DIR / f"{tmp_prefix}{old_slug}", dry)
        safe_move(
            ARTIFACTS_DIR / old_slug, ARTIFACTS_DIR / f"{tmp_prefix}{old_slug}", dry
        )

    print("renaming paths (pass 2: tmp -> final):")
    for old, new in rename_map.items():
        old_slug, new_slug = slug(old), slug(new)
        safe_move(
            PAGES_DIR / f"{tmp_prefix}{old_slug}.mdx",
            PAGES_DIR / f"{new_slug}.mdx",
            dry,
        )
        safe_move(
            RUNNERS_DIR / f"{tmp_prefix}{old_slug}.py",
            RUNNERS_DIR / f"{new_slug}.py",
            dry,
        )
        safe_move(FIGURES_DIR / f"{tmp_prefix}{old_slug}", FIGURES_DIR / new_slug, dry)
        safe_move(
            ARTIFACTS_DIR / f"{tmp_prefix}{old_slug}", ARTIFACTS_DIR / new_slug, dry
        )

    # ── Step 3: rewrite text content across the tree ──────────────────────
    print("rewriting nbNNN references in text files:")
    edits = 0
    for f in iter_text_files():
        try:
            content = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new_content = rewrite_text(content, slug_map)
        if new_content != content:
            edits += 1
            print(f"  edit {f.relative_to(REPO)}")
            if not dry:
                f.write_text(new_content, encoding="utf-8")
    print(f"  ({edits} files modified)")

    # ── Step 4: per-MDX frontmatter (entry: + title prefix) ───────────────
    print("rewriting MDX frontmatter for moved entries:")
    for old, new in rename_map.items():
        mdx = PAGES_DIR / f"{slug(new)}.mdx"
        if not mdx.exists() and dry:
            # In dry-run paths haven't moved yet; skip the live read.
            print(
                f"  (dry) would rewrite frontmatter in {mdx.relative_to(REPO)} -> entry: {new}"
            )
            continue
        if not mdx.exists():
            continue
        content = mdx.read_text(encoding="utf-8")
        updated = rewrite_mdx_frontmatter(content, new)
        if updated != content:
            print(f"  frontmatter {mdx.relative_to(REPO)} entry={new}")
            if not dry:
                mdx.write_text(updated, encoding="utf-8")

    print(
        "\ndone."
        + (
            "  (dry-run, nothing changed)"
            if dry
            else "  remember to git add -A and commit."
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="action", required=True)
    p_ins = sub.add_parser("insert", help="open slot N (shift entries >= N up by one)")
    p_ins.add_argument("n", type=int)
    p_ins.add_argument("--dry-run", action="store_true")
    p_del = sub.add_parser(
        "delete", help="remove slot N (shift entries > N down by one)"
    )
    p_del.add_argument("n", type=int)
    p_del.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    renumber(args.action, args.n, args.dry_run)


if __name__ == "__main__":
    main()
