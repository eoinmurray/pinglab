from __future__ import annotations

import argparse
import re
from pathlib import Path


def _render(template: str, values: dict[str, str]) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def _slugify(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "untitled"


def _escape_yaml_double_quoted(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _existing_experiment_numbers(posts_dir: Path, experiments_dir: Path) -> list[int]:
    numbers: list[int] = []
    for path in posts_dir.glob("*.mdx"):
        m = re.match(r"(?:study|exp)\.(\d+)-", path.stem)
        if m:
            numbers.append(int(m.group(1)))
    for path in experiments_dir.iterdir():
        if not path.is_dir():
            continue
        m = re.match(r"(?:study|exp)\.(\d+)-", path.name)
        if m:
            numbers.append(int(m.group(1)))
    return numbers


def _next_experiment_number(posts_dir: Path, experiments_dir: Path) -> int:
    numbers = _existing_experiment_numbers(posts_dir, experiments_dir)
    return 1 if not numbers else (max(numbers) + 1)


def _prompt_with_default(label: str, default_value: str) -> str:
    value = input(label).strip()
    return value if value else default_value


def _resolve_slug(name: str, next_num: int) -> str:
    value = name.strip().lower()
    explicit = re.fullmatch(r"(?:study|exp)\.(\d+)-([a-z0-9-]+)", value)
    if explicit:
        return f"study.{explicit.group(1)}-{explicit.group(2)}"
    explicit_prefix_only = re.fullmatch(r"(?:study|exp)\.(\d+)-?", value)
    if explicit_prefix_only:
        return f"study.{explicit_prefix_only.group(1)}-new"
    return f"study.{next_num}-{_slugify(value)}"


def _copy_templates(template_root: Path, destination_root: Path, values: dict[str, str]) -> list[Path]:
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
        rendered = _render(source.read_text(encoding="utf-8"), values)
        target.write_text(rendered)
        created.append(target)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively scaffold a study (post + code) from src/template."
    )
    parser.add_argument("--name", default=None, help="Study name; if omitted prompts interactively.")
    parser.add_argument(
        "--description",
        default=None,
        help="Post description; if omitted prompts interactively.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    posts_dir = root / "src" / "posts"
    experiments_dir = root / "src" / "experiments"
    post_template_dir = root / "src" / "template" / "post"
    experiment_template_dir = root / "src" / "template" / "experiment"

    post_template = post_template_dir / "post.mdx"
    if not post_template.exists():
        raise SystemExit(f"Missing post template: {post_template}")
    if not experiment_template_dir.exists():
        raise SystemExit(f"Missing study template dir: {experiment_template_dir}")

    next_num = _next_experiment_number(posts_dir, experiments_dir)
    prefill = f"study.{next_num}-"
    if args.name:
        name = args.name.strip()
    else:
        entered = _prompt_with_default(f"Study name [{prefill}]: ", prefill)
        if entered.startswith(prefill):
            name = entered
        elif entered.lower().startswith("study.") or entered.lower().startswith("exp."):
            name = entered
        else:
            name = f"{prefill}{entered}"
    description = (
        args.description.strip()
        if args.description is not None
        else input("Description (optional): ").strip()
    )

    slug = _resolve_slug(name, next_num)
    title = name
    if not description:
        description = f"Study: {slug}"

    post_path = posts_dir / f"{slug}.mdx"
    experiment_path = experiments_dir / slug
    if post_path.exists():
        raise SystemExit(f"Post already exists: {post_path}")
    if experiment_path.exists():
        raise SystemExit(f"Experiment already exists: {experiment_path}")

    values = {
        "slug": slug,
        "title": title,
        "description": _escape_yaml_double_quoted(description),
    }

    post_path.parent.mkdir(parents=True, exist_ok=True)
    post_content = _render(post_template.read_text(), values)
    post_path.write_text(post_content)

    created_exp_files = _copy_templates(experiment_template_dir, experiment_path, values)

    print(f"Created post: {post_path.relative_to(root)}")
    for p in created_exp_files:
        print(f"Created study file: {p.relative_to(root)}")
    print(f"Study slug: {slug}")
    short_selector = slug.split("-", 1)[0]
    print(f"Run study: task run -- {short_selector}")
    print(f"Run study (exact): task run -- {slug}")


if __name__ == "__main__":
    main()
