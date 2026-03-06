"""Regenerate plots from saved run data without retraining.

Usage:
    uv run python src/scripts/plot_experiment.py study.16          # latest run
    uv run python src/scripts/plot_experiment.py study.16 abc123   # specific run_id
"""
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
        description="Regenerate plots from saved run data."
    )
    parser.add_argument("selector", help="Study selector (prefix or full folder name)")
    parser.add_argument("run_id", nargs="?", default=None,
                        help="Run ID (default: auto-discover latest)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    experiments_root = repo_root / "src" / "experiments"
    exp_dir = resolve_experiment_dir(experiments_root, args.selector)

    plots_file = exp_dir / "plots.py"
    if not plots_file.exists():
        raise SystemExit(f"Missing plots.py for study: {exp_dir}")

    cmd = ["uv", "run", "python", str(plots_file)]
    if args.run_id:
        data_dir = exp_dir / "data" / args.run_id
        if not data_dir.is_dir():
            raise SystemExit(f"Run data not found: {data_dir}")
        cmd += ["--data-dir", str(data_dir)]

    print(f"Study: {exp_dir.relative_to(repo_root)}")
    if args.run_id:
        print(f"Run:   {args.run_id}")
    else:
        print("Run:   latest (auto-discover)")

    completed = subprocess.run(cmd, cwd=repo_root)
    if completed.returncode != 0:
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
