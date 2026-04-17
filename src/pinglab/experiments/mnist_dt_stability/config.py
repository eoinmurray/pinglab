"""Goal 1 configuration — model ladder, sizes, hyperparameters.

The 5-model headline ladder + 2 ablation intermediates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Root for this experiment's artifacts. The harness writes training
# outputs and figure PNGs here; docs are a separate concern.
ROOT = Path("src/artifacts/mnist-dt-stability")
CALIBS_ROOT = ROOT / "calibrations" / "modal" / "mnist"
FIGURES_ROOT = ROOT / "figures"


@dataclass
class ModelConfig:
    """Per-model training-flag set."""
    name: str
    args: list[str] = field(default_factory=list)


@dataclass
class Size:
    """A calibration-size preset (train-set size + epoch count)."""
    samples: int
    epochs: int


# ── Model ladder ────────────────────────────────────────────────────────
# 7 models: 5-model headline ladder + 2 ablation intermediates between
# cuba-exp and coba. See paper @tab:ladder. All use Kaiming signed init
# + subtract reset by default (overridden per model as needed).
MODELS = {
    "snntorch": ModelConfig(
        name="snntorch",
        args=["--model", "snntorch", "--kaiming-init",
              "--no-dales-law", "--lr", "0.01", "--ei-strength", "0"],
    ),
    "snntorch-library": ModelConfig(
        name="snntorch-library",
        args=["--model", "snntorch-library", "--kaiming-init",
              "--no-dales-law", "--lr", "0.01", "--ei-strength", "0"],
    ),
    "cuba": ModelConfig(
        name="cuba",
        args=["--model", "cuba", "--kaiming-init",
              "--no-dales-law", "--lr", "0.01", "--ei-strength", "0"],
    ),
    "cuba-exp": ModelConfig(
        name="cuba-exp",
        args=["--model", "cuba-exp", "--kaiming-init",
              "--no-dales-law", "--lr", "0.01", "--ei-strength", "0"],
    ),
    "coba": ModelConfig(
        name="coba",
        args=["--model", "ping", "--ei-strength", "0",
              "--cm-back-scale", "1000", "--lr", "0.0001",
              "--w-in", "0.3", "0.03", "--w-in-sparsity", "0.95"],
    ),
    "ping": ModelConfig(
        name="ping",
        args=["--model", "ping", "--ei-strength", "0.5",
              "--cm-back-scale", "1000", "--lr", "0.0001",
              "--w-in", "1.2", "0.1", "--w-in-sparsity", "0.95"],
    ),
}

HEADLINE_LADDER = ["snntorch", "cuba", "cuba-exp", "coba", "ping"]

TRAINING_DTS = [0.1, 1.0]

SWEEP_DTS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
             0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


# ── Calibration size presets ─────────────────────────────────────────────
SIZES = {
    "instant":  Size(samples=50,    epochs=1),    # pipeline validation only, ~1-2 min local
    "smoke":    Size(samples=100,   epochs=3),    # minimal learning signal, ~12 min local
    "quick":    Size(samples=500,   epochs=10),   # local iteration with learning
    "standard": Size(samples=1000,  epochs=40),   # Modal H100 ~30 min (paper baseline)
    "large":    Size(samples=10000, epochs=40),   # sample-efficiency test
    "full":     Size(samples=60000, epochs=40),   # overnight
}


# ── Common training flags ────────────────────────────────────────────────
COMMON = dict(
    dataset="mnist",
    t_ms=200,
    input_rate=50,
    early_stopping=5,
)


def suffix(size: Size) -> str:
    """Artifact-dir suffix for a given size preset."""
    return f".{size.samples}.{size.epochs}"


def calib_dir(model: str, dt: float, size: Size) -> Path:
    """Trained-checkpoint directory for (model, dt, size)."""
    return CALIBS_ROOT / f"dt{dt}" / f"{model}{suffix(size)}"
