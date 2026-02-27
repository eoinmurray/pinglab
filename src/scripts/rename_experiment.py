from __future__ import annotations

import re
import shutil
from pathlib import Path


IGNORE_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "node_modules",
    "dist",
    "__pycache__",
}

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".mdx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".txt",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".css",
    ".html",
    ".ipynb",
    ".lock",
}


def _sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"(?:study|exp)\.(\d+)-", name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def _slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "untitled"


def _prompt_index(max_index: int) -> int | None:
    while True:
        raw = input(f"Select study [1-{max_index}] (or q to cancel): ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= max_index:
                return idx - 1
        print("Invalid selection.")


def _resolve_new_slug(old_slug: str, raw_name: str) -> str:
    value = raw_name.strip().lower()
    if not value:
        raise ValueError("New name cannot be empty.")

    explicit = re.fullmatch(r"(?:study|exp)\.(\d+)-([a-z0-9-]+)", value)
    if explicit:
        return f"study.{explicit.group(1)}-{explicit.group(2)}"

    bare_slug = _slugify(value)
    prefix = old_slug.split("-", 1)[0]
    if not re.fullmatch(r"(?:study|exp)\.\d+", prefix):
        raise ValueError(f"Unexpected study prefix format: {old_slug}")
    return f"{prefix}-{bare_slug}"


def _is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    # Also catch extensionless text files like Taskfile.
    return path.suffix == ""


def _iter_repo_text_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if not _is_text_file(path):
            continue
        out.append(path)
    return out


def _replace_slug_references(repo_root: Path, old_slug: str, new_slug: str) -> int:
    changed = 0
    for path in _iter_repo_text_files(repo_root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if old_slug not in text:
            continue
        path.write_text(text.replace(old_slug, new_slug), encoding="utf-8")
        changed += 1
    return changed


def _prompt_confirm(old_slug: str, new_slug: str) -> bool:
    answer = input(
        f"Rename '{old_slug}' -> '{new_slug}' and update references? Type 'r' to confirm: "
    ).strip().lower()
    return answer == "r"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    experiments_root = root / "src" / "experiments"
    posts_root = root / "src" / "posts"
    artifacts_root = posts_root / "_artifacts"

    experiments = sorted(
        [p for p in experiments_root.iterdir() if p.is_dir() and p.name != "__pycache__"],
        key=lambda p: _sort_key(p.name),
    )
    if not experiments:
        print("No studies found.")
        return

    print("Studies:")
    for i, exp in enumerate(experiments, start=1):
        post_mdx = posts_root / f"{exp.name}.mdx"
        post_md = posts_root / f"{exp.name}.md"
        has_post = post_mdx.exists() or post_md.exists()
        status = "post:yes" if has_post else "post:no"
        print(f"{i}. {exp.name} ({status})")

    selected_idx = _prompt_index(len(experiments))
    if selected_idx is None:
        print("Cancelled.")
        return

    selected = experiments[selected_idx]
    old_slug = selected.name
    entered = input("New study name (full slug or new suffix): ").strip()
    try:
        new_slug = _resolve_new_slug(old_slug, entered)
    except ValueError as exc:
        raise SystemExit(str(exc))

    if new_slug == old_slug:
        print("New name is identical to current name. Nothing to do.")
        return

    old_exp_dir = experiments_root / old_slug
    new_exp_dir = experiments_root / new_slug
    old_post_mdx = posts_root / f"{old_slug}.mdx"
    old_post_md = posts_root / f"{old_slug}.md"
    new_post_mdx = posts_root / f"{new_slug}.mdx"
    new_post_md = posts_root / f"{new_slug}.md"
    old_artifacts_dir = artifacts_root / old_slug
    new_artifacts_dir = artifacts_root / new_slug

    if new_exp_dir.exists():
        raise SystemExit(f"Target experiment already exists: {new_exp_dir.relative_to(root)}")
    if old_post_mdx.exists() and new_post_mdx.exists():
        raise SystemExit(f"Target post already exists: {new_post_mdx.relative_to(root)}")
    if old_post_md.exists() and new_post_md.exists():
        raise SystemExit(f"Target post already exists: {new_post_md.relative_to(root)}")
    if old_artifacts_dir.exists() and new_artifacts_dir.exists():
        raise SystemExit(f"Target artifacts already exists: {new_artifacts_dir.relative_to(root)}")

    print("")
    print(f"Selected: {old_slug}")
    print(f"New slug: {new_slug}")
    print(f"Experiment: {old_exp_dir.relative_to(root)} -> {new_exp_dir.relative_to(root)}")
    if old_post_mdx.exists():
        print(f"Post: {old_post_mdx.relative_to(root)} -> {new_post_mdx.relative_to(root)}")
    elif old_post_md.exists():
        print(f"Post: {old_post_md.relative_to(root)} -> {new_post_md.relative_to(root)}")
    else:
        print(f"Post: src/posts/{old_slug}.mdx (missing)")
    if old_artifacts_dir.exists():
        print(
            f"Artifacts: {old_artifacts_dir.relative_to(root)} -> {new_artifacts_dir.relative_to(root)}"
        )
    else:
        print(f"Artifacts: {old_artifacts_dir.relative_to(root)} (missing)")
    print("")

    if not _prompt_confirm(old_slug, new_slug):
        print("Cancelled.")
        return

    shutil.move(str(old_exp_dir), str(new_exp_dir))
    if old_post_mdx.exists():
        shutil.move(str(old_post_mdx), str(new_post_mdx))
    if old_post_md.exists():
        shutil.move(str(old_post_md), str(new_post_md))
    if old_artifacts_dir.exists():
        shutil.move(str(old_artifacts_dir), str(new_artifacts_dir))

    changed_files = _replace_slug_references(root, old_slug, new_slug)

    print(f"Renamed study: {old_exp_dir.relative_to(root)} -> {new_exp_dir.relative_to(root)}")
    print(f"Updated references from '{old_slug}' to '{new_slug}' in {changed_files} file(s).")


if __name__ == "__main__":
    main()
