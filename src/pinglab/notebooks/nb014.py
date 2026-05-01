"""Notebook runner for entry 014 — biophysical ladder.

Trains the conductance-based ladder (cuba → coba → ping) on MNIST at
dt = 0.1 ms, producing per-epoch training videos for each. Recipes
mirror nb009 / nb010 / nb011 verbatim — see those for justification.

The goal is a side-by-side visual comparison of the three biophysical
rungs: the same input, the same training schedule, three forward
dynamics. nb012's matrix lives at the dt-stability level; nb014 lives
at the visual / qualitative level (one video per model).

In-progress placeholder; expect content / numbers to evolve.

Notebook entry: src/docs/src/pages/notebooks/nb014.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import sh

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import append_modal_args, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402

SLUG = "nb014"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope" / "__main__.py"

MODELS = ("cuba", "coba", "ping")
DT = 0.1
T_MS = 200.0
SEED = 42
DEFAULT_TIER = "small"
TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=1),
    "small": dict(max_samples=500, epochs=5),
    "medium": dict(max_samples=2000, epochs=10),
    "large": dict(max_samples=5000, epochs=40),
    "huge": dict(max_samples=10000, epochs=80),
}


def build_args(model: str, tier: str, out_dir: Path, modal_gpu: str | None) -> list[str]:
    """Recipes mirror nb009 (cuba), nb010 (coba), nb011 (ping) verbatim."""
    cfg = TIER_CONFIG[tier]
    common = [
        "run", "python", str(OSCILLOSCOPE), "train",
        "--dataset", "mnist",
        "--max-samples", str(cfg["max_samples"]),
        "--epochs", str(cfg["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT),
        "--seed", str(SEED),
        "--observe", "video",
        "--frame-rate", "1",
        "--readout", "mem-mean",
        "--surrogate-slope", "1",
        "--batch-size", "256",
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    if model == "cuba":
        cell = ["--model", "cuba", "--kaiming-init", "--lr", "0.04"]
    elif model == "coba":
        cell = [
            "--model", "ping",  # COBANet with ei-strength 0
            "--ei-strength", "0",
            "--v-grad-dampen", "1000",
            "--w-in", "0.3",
            "--w-in-sparsity", "0.95",
            "--readout-w-out-scale", "100",
            "--lr", "0.0004",
        ]
    elif model == "ping":
        cell = [
            "--model", "ping",
            "--ei-strength", "1",
            "--v-grad-dampen", "1000",
            "--w-in", "1.2",
            "--w-in-sparsity", "0.95",
            "--readout-w-out-scale", "500",
            "--lr", "0.0004",
        ]
    else:
        raise ValueError(f"unknown model {model!r}")
    return append_modal_args(common + cell, modal_gpu)


def train_model(model: str, tier: str, modal_gpu: str | None) -> Path:
    out_dir = ARTIFACTS / model / "train"
    print(f"[{model}] training → {out_dir.relative_to(REPO)}"
          + (f"  [modal:{modal_gpu}]" if modal_gpu else ""))
    args = build_args(model, tier, out_dir, modal_gpu)
    sh.uv(*args, _cwd=str(REPO), _out=sys.stdout, _err=sys.stderr)
    if not (out_dir / "metrics.json").exists():
        raise SystemExit(f"training did not produce {out_dir / 'metrics.json'}")
    return out_dir


def copy_video(model: str, run_dir: Path) -> Path:
    src = run_dir / "training.mp4"
    if not src.exists():
        raise SystemExit(f"no training video at {src}")
    dst = FIGURES / f"training_{model}.mp4"
    FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    print(f"  → {dst.relative_to(REPO)}")
    return dst


def write_numbers(run_dirs: dict[str, Path], notebook_run_id: str,
                  duration_s: float) -> dict:
    runs = {}
    for model, run_dir in run_dirs.items():
        m = json.loads((run_dir / "metrics.json").read_text())
        cfg = json.loads((run_dir / "config.json").read_text())
        runs[model] = {
            "best_acc": m.get("best_acc"),
            "final_acc": m["epochs"][-1]["acc"],
            "final_loss": m["epochs"][-1]["loss"],
            "rate_e": m["epochs"][-1].get("rate_e"),
            "rate_i": m["epochs"][-1].get("rate_i"),
            "git_sha": cfg.get("git_sha"),
            "run_id": cfg.get("run_id"),
        }
    return {
        "notebook_run_id": notebook_run_id,
        "duration_s": duration_s,
        "duration": f"{int(duration_s // 60)}m {int(duration_s % 60):02d}s",
        "tier": TIER,
        "config": {
            "dt": DT, "t_ms": T_MS, "seed": SEED,
            "max_samples": TIER_CONFIG[TIER]["max_samples"],
            "epochs": TIER_CONFIG[TIER]["epochs"],
            "dataset": "mnist",
        },
        "runs": runs,
        "run_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


TIER = DEFAULT_TIER


def main() -> None:
    global TIER
    TIER = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={TIER}")

    if "--no-wipe-dir" not in sys.argv:
        for path in (ARTIFACTS, FIGURES):
            if path.exists():
                print(f"[wipe] {path.relative_to(REPO)}")
                shutil.rmtree(path)

    t_start = time.monotonic()
    run_dirs: dict[str, Path] = {}
    for model in MODELS:
        run_dirs[model] = train_model(model, TIER, modal_gpu)
        copy_video(model, run_dirs[model])

    duration_s = time.monotonic() - t_start
    persist_run_id(SLUG, notebook_run_id)
    summary = write_numbers(run_dirs, notebook_run_id, duration_s)
    numbers_path = FIGURES / "numbers.json"
    numbers_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {numbers_path.relative_to(REPO)}")
    for model, info in summary["runs"].items():
        print(f"  {model:5s}: best={info['best_acc']}%  final={info['final_acc']}%"
              f"  rate_e={info['rate_e']:.1f}Hz")


if __name__ == "__main__":
    main()
