"""Notebook runner for entry 013c — Latency to correct answer.

Third part of the split nb013 trilogy. Reloads nb013's trained
checkpoints (no retraining), runs a deterministic test-batch forward
pass with recording=True, and reports the per-model curve of
P(correct) vs fraction of trial elapsed.

Writes:
  * latency_curves.png — P(correct) vs trial fraction, per model & regime
  * numbers.json — per-(dt_train, model) latency curve points

Requires: nb013's trained checkpoints under
  src/artifacts/notebooks/nb013/<regime>/<model>/train/

Notebook entry: src/docs/src/pages/notebooks/nb015.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))
sys.path.insert(0, str(REPO / "src" / "pinglab" / "notebooks"))

import _nb013_lib as nb013  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402

SLUG = "nb015"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
NB013_ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / "nb013"


def main() -> None:
    nb013.theme.apply()
    wipe_dir = "--no-wipe-dir" not in sys.argv
    nb013.TIER = parse_tier(
        sys.argv, choices=nb013.TIER_CONFIG.keys(), default=nb013.DEFAULT_TIER,
    )
    nb013.ARTIFACTS = ARTIFACTS
    nb013.FIGURES = FIGURES

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={nb013.TIER}")

    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    regime_train_dirs: dict[float, dict[str, Path]] = {}
    for dt_train in nb013.DT_TRAINS:
        td: dict[str, Path] = {}
        for m in nb013.MODELS:
            d = NB013_ARTIFACTS / nb013._regime_key(dt_train) / m / "train"
            if not (d / "weights.pth").exists():
                raise SystemExit(
                    f"missing nb013 checkpoint at {d.relative_to(REPO)}; "
                    "run nb013 first."
                )
            td[m] = d
        regime_train_dirs[dt_train] = td

    latency = nb013.compute_latency_curves(regime_train_dirs)
    nb013.plot_latency(
        latency, FIGURES / "latency_curves.png", notebook_run_id,
    )
    print(f"wrote {(FIGURES / 'latency_curves.png').relative_to(REPO)}")

    numbers_path = FIGURES / "numbers.json"
    duration_s = time.monotonic() - t_start
    summary = nb013.write_numbers(
        regime_train_dirs,
        regime_sweep_dirs={
            dt: {m: {} for m in nb013.MODELS} for dt in regime_train_dirs
        },
        out_path=numbers_path,
        notebook_run_id=notebook_run_id,
        duration_s=duration_s,
        init_match={},
    )
    summary["latency"] = latency
    numbers_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {numbers_path.relative_to(REPO)}  duration={summary['duration']}")


if __name__ == "__main__":
    main()
    sys.exit(0)
