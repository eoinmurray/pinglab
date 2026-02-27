from __future__ import annotations

import re
from pathlib import Path


def _sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"(?:study|exp)\.(\d+)-", name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    experiments_root = root / "src" / "experiments"
    posts_root = root / "src" / "posts"

    if not experiments_root.exists():
        print("No studies directory found.")
        return

    entries = sorted(
        [p for p in experiments_root.iterdir() if p.is_dir() and p.name != "__pycache__"],
        key=lambda p: _sort_key(p.name),
    )
    if not entries:
        print("No studies found.")
        return

    print("Studies:")
    for exp in entries:
        post = posts_root / f"{exp.name}.mdx"
        post_mark = "post:yes" if post.exists() else "post:no"
        print(f"- {exp.name} ({post_mark})")


if __name__ == "__main__":
    main()
