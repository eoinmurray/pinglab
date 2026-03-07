from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import questionary


def _slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "untitled"


def _list_studies(typst_dir: Path) -> list[str]:
    slugs = []
    for path in sorted(typst_dir.glob("*.typ")):
        if path.stem == "study.0-template":
            continue
        slugs.append(path.stem)
    return slugs


def _next_experiment_number(typst_dir: Path, experiments_dir: Path) -> int:
    numbers: list[int] = []
    for path in list(typst_dir.glob("*.typ")) + list(experiments_dir.iterdir()):
        m = re.match(r"(?:study|exp)\.(\d+)-", path.stem if path.is_file() else path.name)
        if m and path.stem not in ("study.0-template",):
            numbers.append(int(m.group(1)))
    return 1 if not numbers else (max(numbers) + 1)


def _render_typ(content: str, old_slug: str, new_slug: str) -> str:
    """Replace old slug with new slug and update metadata date."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    content = content.replace(old_slug, new_slug)

    # Update #set document(date: datetime(...))
    content = re.sub(
        r'date:\s*datetime\(year:\s*\d+,\s*month:\s*\d+,\s*day:\s*\d+\)',
        f'date: datetime(year: {now.year}, month: {now.month}, day: {now.day})',
        content,
    )
    # Update #metadata date string
    content = re.sub(
        r'(date:\s*)"[^"]*"(,\s*\n\s*description:)',
        rf'\1"{date_str}"\2',
        content,
    )
    # Update #metadata description to new slug
    content = re.sub(
        r'(description:\s*)"[^"]*"',
        rf'\1"{new_slug}"',
        content,
    )
    return content


def _render_py(content: str, old_slug: str, new_slug: str) -> str:
    return content.replace(old_slug, new_slug)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    typst_dir = root / "src" / "typst"
    experiments_dir = root / "src" / "experiments"

    studies = _list_studies(typst_dir)
    if not studies:
        raise SystemExit("No studies found to fork.")

    source_slug = questionary.select(
        "Select study to fork:",
        choices=studies,
    ).ask()
    if source_slug is None:
        raise SystemExit("Cancelled.")

    next_num = _next_experiment_number(typst_dir, experiments_dir)
    prefill = f"study.{next_num}-"
    entered = questionary.text(
        "New study name:",
        default=prefill,
    ).ask()
    if entered is None:
        raise SystemExit("Cancelled.")
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

    src_post = typst_dir / f"{source_slug}.typ"
    src_exp = experiments_dir / source_slug
    dst_post = typst_dir / f"{new_slug}.typ"
    dst_exp = experiments_dir / new_slug

    if dst_post.exists():
        raise SystemExit(f"Post already exists: {dst_post}")
    if dst_exp.exists():
        raise SystemExit(f"Experiment already exists: {dst_exp}")

    # Copy and rewrite post (.typ)
    dst_post.write_text(
        _render_typ(src_post.read_text(encoding="utf-8"), source_slug, new_slug),
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
                continue
            f.write_text(
                _render_py(text, source_slug, new_slug),
                encoding="utf-8",
            )

    print(f"Forked {source_slug!r} → {new_slug!r}")
    print(f"  post: {dst_post.relative_to(root)}")
    if src_exp.exists():
        print(f"  exp:  {dst_exp.relative_to(root)}")


if __name__ == "__main__":
    main()
