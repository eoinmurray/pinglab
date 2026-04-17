#!/usr/bin/env python3
"""Freeze a figure from src/artifacts/ into src/docs/public/figures/ with provenance metadata.

Usage:
    src/scripts/freeze-figure.py <source_path> <dest_path>

Example:
    src/scripts/freeze-figure.py \\
        src/artifacts/mnist-dt-stability/figures/parity_sweep_dt1.0.500.10.png \\
        src/docs/public/figures/journal/2026-04-16-1200-library-parity/parity_sweep_dt1.0.500.10.png

Effect:
    1. Copies the source PNG to dest.
    2. Writes a sidecar <dest>.json with provenance metadata:
       - source_path (the artifact this was copied from)
       - copied_at (ISO 8601 timestamp of the freeze)
       - git_sha (current HEAD SHA at time of freeze, plus dirty flag)
       - run_id, started_at, model, dt, samples, epochs (extracted from the
         producing run's config.json if discoverable from the source path)

If the source path lives under a calibration dir matching the
convention src/artifacts/<exp>/calibrations/.../<model>.<samples>.<epochs>/,
the sidecar's run-level metadata is populated from that dir's config.json.

Exit 0 on success, 1 on any failure.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git_sha() -> dict[str, str | bool]:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except subprocess.CalledProcessError:
        sha = "unknown"
    try:
        status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(status)
    except subprocess.CalledProcessError:
        dirty = False
    return {"sha": sha, "dirty": dirty}


def find_calibration_dir(src: Path) -> Path | None:
    """Walk up from src looking for a config.json — that's the producing run."""
    p = src.parent
    while p != p.parent:
        if (p / "config.json").exists():
            return p
        p = p.parent
    return None


def run_metadata(src: Path) -> dict:
    cal = find_calibration_dir(src)
    if cal is None:
        return {}
    try:
        cfg = json.loads((cal / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "calibration_dir": str(cal.relative_to(ROOT)),
        "run_id": cfg.get("run_id"),
        "started_at": cfg.get("started_at"),
        "model": cfg.get("model"),
        "dt": cfg.get("dt"),
        "samples": cfg.get("max_samples"),
        "epochs": cfg.get("epochs"),
        "dataset": cfg.get("dataset"),
        "input_rate": cfg.get("input_rate"),
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 1
    src = Path(sys.argv[1]).resolve()
    dst = Path(sys.argv[2]).resolve()
    if not src.exists():
        print(f"source does not exist: {src}", file=sys.stderr)
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    sidecar = {
        "source_path": str(src.relative_to(ROOT) if str(src).startswith(str(ROOT)) else src),
        "dest_path": str(dst.relative_to(ROOT) if str(dst).startswith(str(ROOT)) else dst),
        "copied_at": datetime.now(timezone.utc).isoformat(),
        "git": git_sha(),
        "run": run_metadata(src),
    }
    sidecar_path = dst.with_suffix(dst.suffix + ".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"froze {src.relative_to(ROOT) if str(src).startswith(str(ROOT)) else src}")
    print(f"   →  {dst.relative_to(ROOT) if str(dst).startswith(str(ROOT)) else dst}")
    print(f"   sidecar: {sidecar_path.relative_to(ROOT) if str(sidecar_path).startswith(str(ROOT)) else sidecar_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
