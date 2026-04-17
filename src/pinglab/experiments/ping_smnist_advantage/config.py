"""Goal 2 configuration — models, stages, sizes for sequential MNIST.

Central question: does PING's E→I→E coupling give it an advantage on
temporally-structured classification tasks that require recurrent memory?

Starting models are the same ladder as Goal 1; sMNIST-appropriate
hyperparameters are set here. 2-layer recurrent architecture via
--n-hidden 128 256 --w-rec.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path("src/artifacts/ping-smnist-advantage")
CALIBS_ROOT = ROOT / "calibrations" / "modal" / "smnist"
FIGURES_ROOT = ROOT / "figures"


@dataclass
class ModelConfig:
    name: str
    args: list[str] = field(default_factory=list)


@dataclass
class Size:
    samples: int
    epochs: int


@dataclass
class Stage:
    """One rung of the calibration ladder with pass/fail criteria."""
    name: str
    size: Size
    pass_criteria: dict = field(default_factory=dict)


# ── Model ladder for sMNIST ─────────────────────────────────────────────
# Same 5-model headline ladder as Goal 1; hyperparameters tuned for sMNIST.
# Architecture: 2-layer 128→256 with recurrence on both hidden layers.
MODELS = {
    "snntorch": ModelConfig(
        name="snntorch",
        args=["--model", "snntorch", "--kaiming-init",
              "--no-dales-law", "--lr", "0.01",
              "--n-hidden", "128", "256",
              "--w-rec", "0", "0.05",
              "--ei-strength", "0"],
    ),
    "cuba-exp": ModelConfig(
        name="cuba-exp",
        args=["--model", "cuba-exp", "--kaiming-init",
              "--no-dales-law", "--lr", "0.01",
              "--n-hidden", "128", "256",
              "--w-rec", "0", "0.05",
              "--ei-strength", "0"],
    ),
    "coba": ModelConfig(
        name="coba",
        args=["--model", "ping", "--ei-strength", "0",
              "--cm-back-scale", "1000", "--lr", "0.0001",
              "--n-hidden", "128", "256",
              "--w-rec", "0", "0.003",
              "--w-in", "0.15", "0.05", "--w-in-sparsity", "0.7"],
    ),
    "ping": ModelConfig(
        name="ping",
        args=["--model", "ping", "--ei-strength", "0.5",
              "--cm-back-scale", "1000", "--lr", "0.0001",
              "--n-hidden", "128", "256",
              "--w-rec", "0", "0.003",
              "--w-in", "1.2", "0.1", "--w-in-sparsity", "0.95"],
    ),
}

LADDER = list(MODELS.keys())  # eventually a "best-of-ladder" subset


# ── Size presets ────────────────────────────────────────────────────────
SIZES = {
    "instant":  Size(samples=50,    epochs=1),     # pipeline validation only, ~1-2 min local
    "smoke":    Size(samples=100,   epochs=3),     # minimal learning signal
    "quick":    Size(samples=500,   epochs=15),    # local iteration with learning
    "standard": Size(samples=1000,  epochs=40),    # Modal H100 (MVP)
    "large":    Size(samples=10000, epochs=40),    # sample-efficiency test
}


# ── Staged calibration ladder ───────────────────────────────────────────
# Each stage has pass/fail criteria. We scale up only after all models
# in the current ladder pass. Hyperparameters per model can be tuned
# between stages without promoting.
STAGES = [
    Stage(name="init-probe",
          size=Size(samples=50, epochs=0),
          pass_criteria={"rate_e": {"min": 1, "max": 80},
                         "act":    {"min": 0.05, "max": 0.80}}),
    Stage(name="learn-probe",
          size=Size(samples=100, epochs=3),
          pass_criteria={"best_acc": {"min": 12}}),      # above chance
    Stage(name="short",
          size=Size(samples=500, epochs=10),
          pass_criteria={"best_acc": {"min": 30}}),
    Stage(name="medium",
          size=Size(samples=2000, epochs=25),
          pass_criteria={"best_acc": {"min": 60}}),
    Stage(name="full",
          size=Size(samples=5000, epochs=40),
          pass_criteria={"best_acc": {"min": 80}}),
]


# ── Common training flags ───────────────────────────────────────────────
COMMON = dict(
    dataset="smnist",
    t_ms=280,             # 28 rows × 10 ms/row
    input_rate=50,        # lower than Goal 1 to avoid saturation
    dt=1.0,               # coarse dt for tractable training time
    early_stopping=8,     # sMNIST trains slower; more patience
)


def suffix(size: Size) -> str:
    return f".{size.samples}.{size.epochs}"


def calib_dir(model: str, dt: float, size: Size) -> Path:
    return CALIBS_ROOT / f"dt{dt}" / f"{model}{suffix(size)}"
