"""Copy a scaffold tree into the current directory — the portable engine behind
`task scaffold` and `task add-demo-content`.

Those tasks used rsync, which Windows doesn't have (and go-task's built-in shell can't
supply). This is the same copy in stdlib Python:

    uv run python demolab-engine/build/overlay.py SRC [--keep-existing] [--exclude NAME ...]

--keep-existing never clobbers a file already present (rsync --ignore-existing: safe
re-scaffold); --exclude skips top-level directories of SRC (the demo's prebuilt site/).
"""
import argparse
import shutil
import sys
from pathlib import Path


def overlay(src: Path, dst: Path, keep_existing: bool = False, exclude: tuple[str, ...] = ()) -> int:
    """Copy src/** into dst; returns the number of files copied."""
    copied = 0
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if rel.parts[0] in exclude:
            continue
        target = dst / rel
        if p.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif not (keep_existing and target.exists()):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
            copied += 1
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("--keep-existing", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[])
    args = ap.parse_args()
    if not args.src.is_dir():
        sys.exit(f"overlay: source directory not found: {args.src}")
    overlay(args.src, Path.cwd(), keep_existing=args.keep_existing, exclude=tuple(args.exclude))


if __name__ == "__main__":
    main()
