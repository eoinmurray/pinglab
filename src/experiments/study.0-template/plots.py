"""Plot generation — loads NPZ data from a run directory, produces artifacts.

Usage:
    uv run python plots.py --data-dir data/<run_id>
    uv run python plots.py  # auto-discovers latest run
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


def main(
    data_dir: Path | str,
    artifacts_dir: Path | str | None = None,
) -> None:
    """Generate all plots from saved NPZ data.

    Args:
        data_dir: Path to run data directory with NPZ + config.json.
        artifacts_dir: Where to write PNGs. Defaults to _artifacts/<study>/.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    data_dir = Path(data_dir)
    if artifacts_dir is None:
        from settings import ARTIFACTS_ROOT
        artifacts_dir = ARTIFACTS_ROOT / Path(__file__).parent.name
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with (data_dir / "config.json").open() as f:
        spec = json.load(f)

    # TODO: load NPZ files and generate plots
    # metrics = np.load(data_dir / "training_metrics.npz")
    # inference = np.load(data_dir / "inference_data.npz")

    # Copy metadata to artifacts
    for name in ("config.json", "results.json", "train.log"):
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, artifacts_dir / name)

    print(f"Plots saved to {artifacts_dir}")


def _find_latest_run(experiment_dir: Path) -> Path:
    """Find the most recently modified run data directory."""
    data_root = experiment_dir / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"No data directory at {data_root}")
    runs = [d for d in data_root.iterdir() if d.is_dir() and d.name != "MNIST"]
    if not runs:
        raise FileNotFoundError(f"No run directories in {data_root}")
    return max(runs, key=lambda d: d.stat().st_mtime)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots from saved run data")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to run data dir (default: auto-discover latest)")
    parser.add_argument("--artifacts-dir", type=Path, default=None,
                        help="Where to write plots")
    args = parser.parse_args()

    experiment_dir = Path(__file__).parent.resolve()
    if args.data_dir is None:
        data_dir = _find_latest_run(experiment_dir)
        print(f"Auto-discovered latest run: {data_dir}")
    else:
        data_dir = args.data_dir

    main(data_dir=data_dir, artifacts_dir=args.artifacts_dir)
