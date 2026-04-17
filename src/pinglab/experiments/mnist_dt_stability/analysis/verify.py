"""Goal 1: fresh test-set verification of calibrations.

Runs oscilloscope.py infer on each trained checkpoint and asserts the
accuracy is above threshold. Halts the pipeline on failure.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pinglab.experiments.mnist_dt_stability.config import (
    MODELS,
    TRAINING_DTS,
    Size,
    calib_dir,
)


def _fresh_eval(dir_path: Path) -> float | None:
    weights = dir_path / "weights.pth"
    config = dir_path / "config.json"
    if not weights.exists() or not config.exists():
        return None
    out = dir_path / ".verify"
    cmd = [
        "uv", "run", "python", "src/pinglab/oscilloscope.py", "infer",
        "--from-dir", str(dir_path),
        "--out-dir", str(out),
        "--wipe-dir",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  infer FAILED: {result.stderr[-500:]}")
        return None
    try:
        metrics = json.loads((out / "metrics.json").read_text())
        return float(metrics.get("best_acc", metrics.get("acc", 0)))
    except Exception as e:
        print(f"  parse metrics failed: {e}")
        return None


def run(threshold: float = 80.0, size: Size = None):
    if size is None:
        raise ValueError("size is required")
    items = [(dt, m, calib_dir(m, dt, size))
             for dt in TRAINING_DTS for m in MODELS]
    print(f"Verifying {len(items)} calibrations reach ≥ {threshold:.0f}% "
          f"on fresh test-set eval at training dt...")
    print(f"  {'dt':<8}{'model':<26}{'fresh acc':<12}{'status'}")
    print("  " + "─" * 60)

    failures = []
    for dt, model, dir_path in items:
        acc = _fresh_eval(dir_path)
        if acc is None:
            print(f"  dt={dt:<5}{model:<26}{'— n/a —':<12}  missing")
            failures.append((dt, model, "missing weights"))
        elif acc < threshold:
            print(f"  dt={dt:<5}{model:<26}{acc:>6.1f}%      ✗ FAIL")
            failures.append((dt, model, f"{acc:.1f}% < {threshold:.0f}%"))
        else:
            print(f"  dt={dt:<5}{model:<26}{acc:>6.1f}%      ✓")

    print()
    if failures:
        print(f"VERIFICATION FAILED: {len(failures)}/{len(items)} model(s)")
        for dt, m, why in failures:
            print(f"  - dt={dt} {m}: {why}")
        sys.exit(1)
    print(f"All {len(items)} calibrations passed.")
