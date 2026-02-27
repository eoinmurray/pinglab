from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def resolve_experiment_dir(experiments_root: Path, selector: str) -> Path:
    selector = selector.strip()
    if not selector:
        raise ValueError("Study selector is required (e.g. study.1 or study.1-my-name).")

    direct = experiments_root / selector
    if direct.is_dir():
        return direct

    prefix = f"{selector}-"
    matches = sorted(
        p for p in experiments_root.iterdir() if p.is_dir() and p.name.startswith(prefix)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise ValueError(f"Selector '{selector}' is ambiguous. Matches: {names}")

    available = ", ".join(sorted(p.name for p in experiments_root.iterdir() if p.is_dir()))
    raise ValueError(f"No study matching '{selector}'. Available: {available}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a study by selector, e.g. `study.1` or full slug."
    )
    parser.add_argument("selector", help="Study selector (prefix or full folder name)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    experiments_root = repo_root / "src" / "experiments"
    exp_dir = resolve_experiment_dir(experiments_root, args.selector)
    run_file = exp_dir / "run.py"
    if not run_file.exists():
        raise SystemExit(f"Missing run.py for study: {exp_dir}")

    cmd = ["uv", "run", "python", str(run_file)]
    print(f"Resolved: {exp_dir.relative_to(repo_root)}")
    print("Command: " + " ".join(cmd))
    if args.dry_run:
        return

    completed = subprocess.run(cmd, cwd=repo_root)
    if completed.returncode != 0:
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
