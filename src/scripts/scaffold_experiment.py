from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import questionary

TEMPLATE_SLUG = "study.0-template"


def _slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "untitled"


def _existing_experiment_numbers(typst_dir: Path, experiments_dir: Path) -> list[int]:
    numbers: list[int] = []
    for path in typst_dir.glob("*.typ"):
        m = re.match(r"(?:study|exp)\.(\d+)-", path.stem)
        if m and path.stem != TEMPLATE_SLUG:
            numbers.append(int(m.group(1)))
    for path in experiments_dir.iterdir():
        if not path.is_dir():
            continue
        m = re.match(r"(?:study|exp)\.(\d+)-", path.name)
        if m and path.name != TEMPLATE_SLUG:
            numbers.append(int(m.group(1)))
    return numbers


def _next_experiment_number(typst_dir: Path, experiments_dir: Path) -> int:
    numbers = _existing_experiment_numbers(typst_dir, experiments_dir)
    return 1 if not numbers else (max(numbers) + 1)


def _resolve_slug(name: str, next_num: int) -> str:
    value = name.strip().lower()
    explicit = re.fullmatch(r"(?:study|exp)\.(\d+)-([a-z0-9-]+)", value)
    if explicit:
        return f"study.{explicit.group(1)}-{explicit.group(2)}"
    explicit_prefix_only = re.fullmatch(r"(?:study|exp)\.(\d+)-?", value)
    if explicit_prefix_only:
        return f"study.{explicit_prefix_only.group(1)}-new"
    return f"study.{next_num}-{_slugify(value)}"


def _render_typ(content: str, slug: str, title: str, description: str) -> str:
    """Replace template slug and update #set document / #metadata in .typ files."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")

    content = content.replace(TEMPLATE_SLUG, slug)

    # Update #set document(title: ...)
    content = re.sub(
        r'(#set document\(\s*title:\s*)"[^"]*"',
        rf'\1"{slug}"',
        content,
    )
    # Update #set document(date: datetime(...))
    content = re.sub(
        r'date:\s*datetime\(year:\s*\d+,\s*month:\s*\d+,\s*day:\s*\d+\)',
        f'date: datetime(year: {now.year}, month: {now.month}, day: {now.day})',
        content,
    )
    # Update #metadata title
    content = re.sub(
        r'(#metadata\(\(\s*title:\s*)"[^"]*"',
        rf'\1"{slug}"',
        content,
    )
    # Update #metadata date
    content = re.sub(
        r'(date:\s*)"[^"]*"(,\s*\n\s*description:)',
        rf'\1"{date_str}"\2',
        content,
    )
    # Update #metadata description
    content = re.sub(
        r'(description:\s*)"[^"]*"',
        rf'\1"{description}"',
        content,
    )
    return content


def _render_py(content: str, slug: str) -> str:
    """Replace template slug in Python files."""
    return content.replace(TEMPLATE_SLUG, slug)


def _copy_template(
    template_root: Path,
    destination_root: Path,
    slug: str,
    title: str,
    description: str,
) -> list[Path]:
    created: list[Path] = []
    for source in sorted(template_root.rglob("*")):
        if source.is_dir():
            continue
        if "__pycache__" in source.parts:
            continue
        if source.suffix == ".pyc":
            continue
        rel = source.relative_to(template_root)
        target = destination_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8")
        rendered = _render_py(text, slug)
        target.write_text(rendered, encoding="utf-8")
        created.append(target)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Scaffold a new study from {TEMPLATE_SLUG}."
    )
    parser.add_argument("--name", default=None, help="Study name; if omitted prompts interactively.")
    parser.add_argument(
        "--description",
        default=None,
        help="Post description; if omitted prompts interactively.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    typst_dir = root / "src" / "typst"
    experiments_dir = root / "src" / "experiments"
    post_template = typst_dir / f"{TEMPLATE_SLUG}.typ"
    experiment_template_dir = experiments_dir / TEMPLATE_SLUG

    if not post_template.exists():
        raise SystemExit(f"Missing post template: {post_template}")
    if not experiment_template_dir.exists():
        raise SystemExit(f"Missing experiment template: {experiment_template_dir}")

    next_num = _next_experiment_number(typst_dir, experiments_dir)
    prefill = f"study.{next_num}-"
    if args.name:
        name = args.name.strip()
    else:
        entered = questionary.text("Study name:", default=prefill).ask()
        if entered is None:
            raise SystemExit("Cancelled.")
        entered = entered.strip()
        if entered.startswith(prefill):
            name = entered
        elif entered.lower().startswith("study.") or entered.lower().startswith("exp."):
            name = entered
        else:
            name = f"{prefill}{entered}"
    description = (
        args.description.strip()
        if args.description is not None
        else (questionary.text("Description (optional):").ask() or "").strip()
    )

    slug = _resolve_slug(name, next_num)
    title = name
    if not description:
        description = slug

    post_path = typst_dir / f"{slug}.typ"
    experiment_path = experiments_dir / slug
    if post_path.exists():
        raise SystemExit(f"Post already exists: {post_path}")
    if experiment_path.exists():
        raise SystemExit(f"Experiment already exists: {experiment_path}")

    # Write post (.typ)
    post_path.parent.mkdir(parents=True, exist_ok=True)
    post_path.write_text(
        _render_typ(post_template.read_text(encoding="utf-8"), slug, title, description),
        encoding="utf-8",
    )

    # Write experiment files
    created_exp_files = _copy_template(experiment_template_dir, experiment_path, slug, title, description)

    print(f"Created post:  {post_path.relative_to(root)}")
    for p in created_exp_files:
        print(f"Created file:  {p.relative_to(root)}")
    print(f"Slug: {slug}")

    run_file = experiment_path / "run.py"
    print(f"\nRunning {run_file.relative_to(root)} ...")
    completed = subprocess.run(["uv", "run", "python", str(run_file)], cwd=root)
    if completed.returncode != 0:
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
