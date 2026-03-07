from __future__ import annotations

import re
import shutil
from pathlib import Path

import questionary


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
    ".typ",
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


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    experiments_root = root / "src" / "experiments"
    typst_root = root / "src" / "typst"
    artifacts_root = typst_root / "_artifacts"

    experiments = sorted(
        [p for p in experiments_root.iterdir() if p.is_dir() and p.name != "__pycache__"],
        key=lambda p: _sort_key(p.name),
    )
    if not experiments:
        print("No studies found.")
        return

    choices = []
    for exp in experiments:
        post = typst_root / f"{exp.name}.typ"
        status = "post:yes" if post.exists() else "post:no"
        choices.append(questionary.Choice(f"{exp.name} ({status})", value=exp.name))

    old_slug = questionary.select(
        "Select study to rename:",
        choices=choices,
    ).ask()
    if old_slug is None:
        print("Cancelled.")
        return

    entered = questionary.text("New study name (full slug or new suffix):").ask()
    if entered is None:
        raise SystemExit("Cancelled.")
    entered = entered.strip()
    try:
        new_slug = _resolve_new_slug(old_slug, entered)
    except ValueError as exc:
        raise SystemExit(str(exc))

    if new_slug == old_slug:
        print("New name is identical to current name. Nothing to do.")
        return

    old_exp_dir = experiments_root / old_slug
    new_exp_dir = experiments_root / new_slug
    old_post = typst_root / f"{old_slug}.typ"
    new_post = typst_root / f"{new_slug}.typ"
    old_artifacts_dir = artifacts_root / old_slug
    new_artifacts_dir = artifacts_root / new_slug

    if new_exp_dir.exists():
        raise SystemExit(f"Target experiment already exists: {new_exp_dir.relative_to(root)}")
    if old_post.exists() and new_post.exists():
        raise SystemExit(f"Target post already exists: {new_post.relative_to(root)}")
    if old_artifacts_dir.exists() and new_artifacts_dir.exists():
        raise SystemExit(f"Target artifacts already exists: {new_artifacts_dir.relative_to(root)}")

    print("")
    print(f"Selected: {old_slug}")
    print(f"New slug: {new_slug}")
    print(f"Experiment: {old_exp_dir.relative_to(root)} -> {new_exp_dir.relative_to(root)}")
    if old_post.exists():
        print(f"Post: {old_post.relative_to(root)} -> {new_post.relative_to(root)}")
    else:
        print(f"Post: src/typst/{old_slug}.typ (missing)")
    if old_artifacts_dir.exists():
        print(
            f"Artifacts: {old_artifacts_dir.relative_to(root)} -> {new_artifacts_dir.relative_to(root)}"
        )
    else:
        print(f"Artifacts: {old_artifacts_dir.relative_to(root)} (missing)")
    print("")

    confirmed = questionary.confirm(
        f"Rename '{old_slug}' -> '{new_slug}' and update references?",
        default=False,
    ).ask()
    if not confirmed:
        print("Cancelled.")
        return

    shutil.move(str(old_exp_dir), str(new_exp_dir))
    if old_post.exists():
        shutil.move(str(old_post), str(new_post))
    if old_artifacts_dir.exists():
        shutil.move(str(old_artifacts_dir), str(new_artifacts_dir))

    changed_files = _replace_slug_references(root, old_slug, new_slug)

    print(f"Renamed study: {old_exp_dir.relative_to(root)} -> {new_exp_dir.relative_to(root)}")
    print(f"Updated references from '{old_slug}' to '{new_slug}' in {changed_files} file(s).")


if __name__ == "__main__":
    main()
