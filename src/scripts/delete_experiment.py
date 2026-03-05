from __future__ import annotations

import re
import shutil
from pathlib import Path

import questionary


def _sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"(?:study|exp)\.(\d+)-", name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    experiments_root = root / "src" / "experiments"
    posts_root = root / "src" / "posts"

    experiments = sorted(
        [p for p in experiments_root.iterdir() if p.is_dir() and p.name != "__pycache__"],
        key=lambda p: _sort_key(p.name),
    )
    if not experiments:
        print("No studies found.")
        return

    choices = []
    for exp in experiments:
        post_path = posts_root / f"{exp.name}.mdx"
        status = "post:yes" if post_path.exists() else "post:no"
        choices.append(questionary.Choice(f"{exp.name} ({status})", value=exp.name))

    slug = questionary.select(
        "Select study to delete:",
        choices=choices,
    ).ask()
    if slug is None:
        print("Cancelled.")
        return
    selected = experiments_root / slug
    post_file = posts_root / f"{slug}.mdx"
    artifacts_dir = posts_root / "_artifacts" / slug

    print("")
    print(f"Selected: {slug}")
    print(f"Study dir: {selected.relative_to(root)}")
    print(
        f"Post file: {post_file.relative_to(root) if post_file.exists() else f'{post_file.relative_to(root)} (missing)'}"
    )
    print(
        "Artifacts dir: "
        + (
            str(artifacts_dir.relative_to(root))
            if artifacts_dir.exists()
            else f"{artifacts_dir.relative_to(root)} (missing)"
        )
    )
    print("")

    confirmed = questionary.confirm(
        f"Delete '{slug}' study + post + artifacts?",
        default=False,
    ).ask()
    if not confirmed:
        print("Cancelled.")
        return

    had_post = post_file.exists()
    had_artifacts = artifacts_dir.exists()
    shutil.rmtree(selected)
    if had_post:
        post_file.unlink()
    if had_artifacts:
        shutil.rmtree(artifacts_dir)

    print(f"Deleted study: {selected.relative_to(root)}")
    if had_post:
        print(f"Deleted post: {post_file.relative_to(root)}")
    else:
        print(f"Post not found (nothing to delete): {post_file.relative_to(root)}")
    if had_artifacts:
        print(f"Deleted artifacts: {artifacts_dir.relative_to(root)}")
    else:
        print(f"Artifacts not found (nothing to delete): {artifacts_dir.relative_to(root)}")


if __name__ == "__main__":
    main()
