from __future__ import annotations

import argparse
from pathlib import Path


def _render(template: str, values: dict[str, str]) -> str:
    result = template
    for key, value in values.items():
        result = result.replace("{{" + key + "}}", value)
    return result


def _write_template(src: Path, dst: Path, values: dict[str, str]) -> None:
    content = src.read_text()
    dst.write_text(_render(content, values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold a new experiment.")
    parser.add_argument("slug", help="Experiment slug, e.g. exp.4-new-idea")
    parser.add_argument("--title", default=None, help="Post title (default: slug)")
    parser.add_argument(
        "--description",
        default=None,
        help="Post description (default: 'Experiment: <slug>')",
    )
    args = parser.parse_args()

    slug = args.slug.strip()
    title = args.title or slug
    description = args.description or f"Experiment: {slug}"

    repo_root = Path.cwd()
    posts_dir = repo_root / "posts"
    notebooks_dir = repo_root / "notebooks"

    post_path = posts_dir / f"{slug}.mdx"
    nb_dir = notebooks_dir / slug
    lib_dir = nb_dir / "lib"

    if post_path.exists():
        raise SystemExit(f"Post already exists: {post_path}")
    if nb_dir.exists():
        raise SystemExit(f"Notebook folder already exists: {nb_dir}")

    templates_dir = Path(__file__).resolve().parent.parent / "assets" / "templates"
    values = {
        "slug": slug,
        "title": title,
        "description": description,
    }

    posts_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    _write_template(templates_dir / "post.mdx", post_path, values)
    _write_template(templates_dir / "config.yaml", nb_dir / "config.yaml", values)
    _write_template(templates_dir / "run.py", nb_dir / "run.py", values)
    _write_template(templates_dir / "model.py", lib_dir / "model.py", values)
    _write_template(templates_dir / "experiment.py", lib_dir / "experiment.py", values)
    _write_template(templates_dir / "plots.py", lib_dir / "plots.py", values)

    print(f"[new-experiment] Created {post_path}")
    print(f"[new-experiment] Created {nb_dir}")


if __name__ == "__main__":
    main()
