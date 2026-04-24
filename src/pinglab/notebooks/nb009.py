"""Notebook runner for entry 009 — ping single-model training.

Trains the full PING model (conductance-based synapses with the
E→I→E feedback loop that produces gamma oscillations) on MNIST at
dt=0.1 ms and publishes training curves, hidden firing rates, a
training video, and numbers.json. Companion to the other per-model
runners (nb005–nb008) and to the five-model Δt-stability sweep in
nb010.

Notebook entry: src/docs/src/pages/notebooks/nb009.mdx
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _per_model import run, osc_base_args  # noqa: E402

SLUG = "nb009"
MODEL = "ping"
# Minimum final-epoch inhibitory rate (Hz) to count PING as having formed.
# Below this, the E→I→E loop never closed — the model is training as
# feedforward-E rather than as PING.
PING_I_RATE_MIN_HZ = 1.0


def build_osc_args(tier: str, out_dir: Path) -> list[str]:
    return osc_base_args(out_dir, tier, build_as=MODEL) + [
        "--ei-strength", "0.5",
        "--v-grad-dampen", "1000",
        "--w-in", "1.2",
        "--w-in-sparsity", "0.95",
        "--lr", "0.0001",
    ]


def ping_criterion(figures: Path, run_dir: Path) -> list[dict]:
    """ping-specific gate: I must actually fire by the final epoch.
    Without rate_i > 0 the E→I→E feedback loop never closed and the
    network is effectively running as feedforward-E (coba territory),
    not PING — the trained weights no longer represent a PING model."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return [{"label": "PING forms (rate_i > threshold at final epoch)",
                 "passed": False, "detail": "no metrics.json"}]
    metrics = json.loads(metrics_path.read_text())
    rate_i = float(metrics["epochs"][-1].get("rate_i") or 0.0)
    rate_e = float(metrics["epochs"][-1].get("rate_e") or 0.0)
    return [{
        "label": f"PING forms (rate_i ≥ {PING_I_RATE_MIN_HZ:g} Hz at final epoch)",
        "passed": bool(rate_i >= PING_I_RATE_MIN_HZ),
        "detail": f"rate_e={rate_e:.2f} Hz, rate_i={rate_i:.2f} Hz",
    }]


if __name__ == "__main__":
    run(SLUG, MODEL, build_osc_args, gpu_needs_a100=True,
        extra_criteria_fn=ping_criterion)
