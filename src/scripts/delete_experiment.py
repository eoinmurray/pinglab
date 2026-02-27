from __future__ import annotations

import re
import shutil
from pathlib import Path


def _sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"exp\.(\d+)-", name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


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


def _prompt_confirm(slug: str) -> bool:
    answer = input(
        f"Delete '{slug}' study + post + artifacts? Type 'd' to confirm: "
    ).strip().lower()
    return answer == "d"


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

    print("Studies:")
    for i, exp in enumerate(experiments, start=1):
        post_path = posts_root / f"{exp.name}.mdx"
        status = "post:yes" if post_path.exists() else "post:no"
        print(f"{i}. {exp.name} ({status})")

    selected_idx = _prompt_index(len(experiments))
    if selected_idx is None:
        print("Cancelled.")
        return
    selected = experiments[selected_idx]
    slug = selected.name
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

    if not _prompt_confirm(slug):
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
