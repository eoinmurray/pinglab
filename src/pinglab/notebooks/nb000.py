"""Notebook runner for entry 000 — perf baseline.

Profiling reference workload. Trains two models per run so the perf
matrix covers both training backbones used elsewhere in the lab:

  - standard-snn (CUBANet path)
  - coba         (COBANet path with ei_strength=0)

Recipes are *fixed* so wall-time / throughput numbers reflect changes
in the training stack itself (kernel launches, sync points, compile
coverage, memory layout), not recipe drift.

Anchors the "one architecture that runs efficiently on local MPS and
Modal" goal.

Notebook entry: src/docs/src/pages/notebooks/nb000.mdx
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _per_model import run, osc_base_args  # noqa: E402

SLUG = "nb000"
MODEL = "standard-snn"


def build_osc_args(tier: str, out_dir: Path) -> list[str]:
    return osc_base_args(out_dir, tier, build_as=MODEL) + [
        "--kaiming-init",
        "--readout",
        "li",
        "--surrogate-slope",
        "1",
        "--lr",
        "0.01",
        "--batch-size",
        "256",
    ]


def build_coba_osc_args(tier: str, out_dir: Path) -> list[str]:
    """coba is dispatched via COBANet with ei_strength=0, so this exercises
    the COBANet compile path next to standard-snn's CUBANet path — both
    backbones tested per nb000 run."""
    return osc_base_args(out_dir, tier, build_as="ping") + [
        "--ei-strength",
        "0",
        "--v-grad-dampen",
        "1000",
        "--w-in",
        "0.3",
        "--w-in-sparsity",
        "0.95",
        "--lr",
        "0.0001",
    ]


if __name__ == "__main__":
    run(
        SLUG,
        MODEL,
        build_osc_args,
        track_baselines=True,
        extra_train_models=[("coba", build_coba_osc_args)],
    )
