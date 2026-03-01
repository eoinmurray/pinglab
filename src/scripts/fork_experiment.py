from __future__ import annotations

import re
import shutil
from pathlib import Path


def _slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "untitled"


def _escape_yaml_double_quoted(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _list_studies(posts_dir: Path) -> list[str]:
    slugs = []
    for path in sorted(posts_dir.glob("*.mdx")):
        if path.stem == "study.0-template":
            continue
        slugs.append(path.stem)
    return slugs


def _next_experiment_number(posts_dir: Path, experiments_dir: Path) -> int:
    numbers: list[int] = []
    for path in list(posts_dir.glob("*.mdx")) + list(experiments_dir.iterdir()):
        m = re.match(r"(?:study|exp)\.(\d+)-", path.stem if path.is_file() else path.name)
        if m and path.stem not in ("study.0-template",):
            numbers.append(int(m.group(1)))
    return 1 if not numbers else (max(numbers) + 1)


def _render(content: str, old_slug: str, new_slug: str, new_title: str) -> str:
    content = content.replace(old_slug, new_slug)
    content = re.sub(r"^title:.*$", f"title: {new_title}", content, flags=re.MULTILINE)
    content = re.sub(
        r'^description:.*$',
        f'description: "{_escape_yaml_double_quoted(new_title)}"',
        content,
        flags=re.MULTILINE,
    )
    return content


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    posts_dir = root / "src" / "posts"
    experiments_dir = root / "src" / "experiments"

    studies = _list_studies(posts_dir)
    if not studies:
        raise SystemExit("No studies found to fork.")

    print("Available studies:")
    for i, slug in enumerate(studies, 1):
        print(f"  {i}) {slug}")

    raw = input("\nSelect study to fork (number or name): ").strip()
    if raw.isdigit():
        idx = int(raw) - 1
        if not (0 <= idx < len(studies)):
            raise SystemExit("Invalid selection.")
        source_slug = studies[idx]
    else:
        if raw not in studies:
            raise SystemExit(f"Study not found: {raw!r}")
        source_slug = raw

    next_num = _next_experiment_number(posts_dir, experiments_dir)
    prefill = f"study.{next_num}-"
    entered = input(f"New study name [{prefill}]: ").strip()
    if not entered:
        entered = prefill
    if entered.startswith(prefill):
        name = entered
    elif re.match(r"(?:study|exp)\.\d+-", entered):
        name = entered
    else:
        name = f"{prefill}{entered}"

    m = re.fullmatch(r"(?:study|exp)\.(\d+)-([a-z0-9-]+)", name.lower())
    new_slug = f"study.{m.group(1)}-{m.group(2)}" if m else f"study.{next_num}-{_slugify(name)}"
    new_title = name

    src_post = posts_dir / f"{source_slug}.mdx"
    src_exp = experiments_dir / source_slug
    dst_post = posts_dir / f"{new_slug}.mdx"
    dst_exp = experiments_dir / new_slug

    if dst_post.exists():
        raise SystemExit(f"Post already exists: {dst_post}")
    if dst_exp.exists():
        raise SystemExit(f"Experiment already exists: {dst_exp}")

    # Copy and rewrite post
    dst_post.write_text(
        _render(src_post.read_text(encoding="utf-8"), source_slug, new_slug, new_title),
        encoding="utf-8",
    )

    # Copy and rewrite experiment files
    if src_exp.exists():
        shutil.copytree(src_exp, dst_exp)
        for f in dst_exp.rglob("*"):
            if not f.is_file() or f.suffix in (".pyc",) or "__pycache__" in f.parts:
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, ValueError):
                continue  # skip binary files
            f.write_text(
                _render(text, source_slug, new_slug, new_title),
                encoding="utf-8",
            )

    print(f"Forked {source_slug!r} → {new_slug!r}")
    print(f"  post: {dst_post.relative_to(root)}")
    if src_exp.exists():
        print(f"  exp:  {dst_exp.relative_to(root)}")


if __name__ == "__main__":
    main()
