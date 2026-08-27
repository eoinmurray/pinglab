"""Notebook runner for entry 022 — Training.

The single place the gamma-gated-sparsity collection trains its canonical
networks. Every cell is trained once here, to a shared artifact root
(src/artifacts/notebooks/training/), and the analysis notebooks load those
weights with `load_cell` (imported from this module) instead of retraining
their own. This replaces the collection's older "standalone runner, no
cross-notebook helpers" rule with a train-once / reuse-many policy.

102 cells across seven families (canonical, activity ceiling, τ_GABA, Δt,
recurrent initialization, variable rate, and low-input recruitment). Exp022 owns
these training contracts and their checkpoints; downstream experiments only consume
them. Standard: 50 epochs, dt = 0.1 ms, T = 200 ms, and three seeds
(42/43/44) for every cell — including the rate target interior, so the accuracy–rate
frontier carries error bars (it was single-seed; no longer). Canonical sees all
of MNIST, the sweeps 10%. (exp044's Δt sweep is the documented exception that
varies dt.)

Outputs a per-cell accuracy / E-rate summary plus a manifest (numbers.json)
recording exactly which cells were trained and the git sha — the contract
the analysis notebooks rely on.

Writing: writings/exp022.typ · figures + numbers.json: .artifacts/exp022/
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.exp022_support import campaign  # noqa: E402

from helpers import (
    runpod,  # noqa: E402
    theme,  # noqa: E402
)
from helpers.checkpoints import (  # noqa: E402
    checkpoint_provenance,
    epoch_metrics,
    resolve_checkpoint,
)
from helpers.cli import parse_meta  # noqa: E402
from helpers.fmt import format_duration  # noqa: E402
from helpers.operating_point import TAU_GABA_GAMMA_MS  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import (
    finalize_prepared_run,  # noqa: E402
    preserve_active_view,  # noqa: E402
)
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "exp022"
RESULT_CHECKPOINT_ROLE = "final_epoch"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path

# ── Canonical training registry (the hub the collection reuses) ──────
# Analysis notebooks import `load_cell` / `cell_dir` from this module rather
# than retraining; this entry is the single producer of the shared cells.
# PINGLAB_TRAINING_ROOT overrides the location: RunPod pods set it to the shared
# network-volume mount (/shared/training) so a fan-out writes durable artifacts
# there instead of an ephemeral pod disk. Local runs use the default and are
# unaffected. cell_dir / load_cell read through this, so every consumer follows.
TRAINING_ROOT = Path(os.environ["PINGLAB_TRAINING_ROOT"]) if os.environ.get(
    "PINGLAB_TRAINING_ROOT"
) else ARTIFACTS
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"

EPOCHS_STANDARD = 50
DT_MS = 0.1
T_MS = 200.0
SEEDS_BASELINE = [42, 43, 44]
VARIABLE_RATE_TRAINING_RATES_HZ = (
    0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0,
)
VARIABLE_RATE_CONSUMER = "exp082"
RATE_TARGET_GRID_HZ: list[float | None] = [None, 25.0, 10.0, 5.0, 2.5, 1.0]
# Calibrated for the sample-wise, population-normalized Hz contract. The prior
# 1e-3 value was carried over from the neuron-summed spike-count objective and
# became 40.96x weaker at N_E=1024 and T=0.2 s. The one-seed calibration at
# {0.004, 0.016, 0.041, 0.1} selected this measured elbow (PR #159).
FR_STRENGTH_UPPER = 0.041
LOW_W_IN_VALUES = (0.05, 0.1, 0.3, 0.9)
TAU_AMPA_MS = 2.0          # AMPA decay — fixed across the collection (no CLI knob)
INPUT_RATE_HZ = 25.0
N_INPUT = 784
N_EXCITATORY = 1024
N_INHIBITORY = 256
N_OUTPUT = 10
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
DALES_LAW = True
# GABA decay that puts the loop in gamma (≈ 44 Hz); the standard for every
# family except the τ_GABA sweep. Single source of truth in helpers so the
# whole collection moves together (see helpers/operating_point.py).
TAU_GABA_GAMMA = TAU_GABA_GAMMA_MS
SHARED_W_IN_SUMMED_PARENT_MEAN = "0.9"
# Stored-weight parameters exactly corresponding to the accepted legacy
# Normal(5.1, 3.8) / 1024 × 225 recipe, now expressed directly.
SHARED_READOUT_W_INIT_MEAN = "1.12060546875"
SHARED_READOUT_W_INIT_STD = "0.8349609375"
TR06_READOUT_W_INIT_MEAN = "0.05"
TR06_READOUT_W_INIT_STD = "0.04"

# Production recipe for the next exp022 bank. COBA and PING share the input and
# readout initialization distributions selected by the matched-midpoint gate. They
# differ only in loop engagement and the loop-specific backward stabilizer.
# Gradient dampening (--v-grad-dampen) is loop-specific. Sweeping a dampening
# ladder {1, 10, 100, 1000} across both architectures shows PING needs it: its
# BPTT gradient explodes through the recurrent E→I→E loop at dampening 1 and the
# network only trains once the stabiliser is applied, whereas COBA (no loop) is
# insensitive and trains identically across the whole ladder. So COBA trains with
# NO dampening (1) and PING keeps the stabiliser (1000). Training COBA without the
# crutch keeps the two architectures honest — COBA earns its accuracy on the bare
# feedforward gradient.
MODEL_RECIPES: dict[str, dict] = {
    "coba": {
        "__build_as": "ping",
        "--ei-strength": "0",
        "--v-grad-dampen": "1",
        "--w-in": SHARED_W_IN_SUMMED_PARENT_MEAN,
        "--w-in-initial-zero-fraction": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-init-mean": SHARED_READOUT_W_INIT_MEAN,
        "--readout-w-init-std": SHARED_READOUT_W_INIT_STD,
        "--lr": "0.0004",
        "--batch-size": "256",
    },
    "ping": {
        "__build_as": "ping",
        "--ei-strength": "1",
        "--v-grad-dampen": "1000",
        "--w-in": SHARED_W_IN_SUMMED_PARENT_MEAN,
        "--w-in-initial-zero-fraction": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-init-mean": SHARED_READOUT_W_INIT_MEAN,
        "--readout-w-init-std": SHARED_READOUT_W_INIT_STD,
        "--lr": "0.0004",
        "--batch-size": "256",
    },
}

MODELS = ["coba", "ping"]
MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}
MODEL_MARKERS = {"coba": "s", "ping": "D"}

TRAINING_RUN_IDS = {
    "canonical": "TR-01",
    "activity_frontier": "TR-02",
    "tau_gaba": "TR-03",
    "dt": "TR-04",
    "init": "TR-05",
    "variable_rate": "TR-06",
    "low_w_in": "TR-07",
}

# ping recipe without the fixed --ei-strength, for the init family (exp049),
# whose whole point is to vary ei-strength + recurrent trainability per cell.
MODEL_RECIPES["ping_init"] = {
    k: v for k, v in MODEL_RECIPES["ping"].items() if k != "--ei-strength"
}
# exp049 init conditions: (ei_strength, trainable W_EI, trainable W_IE).
INIT_CONDITIONS: dict[str, tuple] = {
    "frozen_ping": ("1", False, False),
    "trainable_ping_init": ("1", True, True),
    "trainable_zero_init": ("0", True, True),
    "trainable_small_init": ("0.1", True, True),
}


TAU_GABA_SWEEP = (4.5, 6.0, 9.0, 12.0, 18.0, 27.0)   # exp041
DT_SWEEP_MS = (0.05, 0.1, 0.25, 0.5, 1.0)             # exp044 (the dt exception)
MNIST_TRAIN_SAMPLES = 60000                           # official training partition
MNIST_TEST_SAMPLES = 10000                            # untouched official test partition
CANONICAL_MAX_SAMPLES = MNIST_TRAIN_SAMPLES           # all official training data
SUBSET_MAX_SAMPLES = 7000                             # reduced sweep training pool
MAX_SAMPLES = 100                                     # plumbing cap on every cell
EPOCHS = 2                                            # plumbing depth on every cell
BATCH_SIZE = 256                                      # fixed across every recipe


def rate_target_label(target_hz: float | None) -> str:
    if target_hz is None:
        return "off"
    return "rt" + f"{target_hz:g}".replace(".", "p") + "hz"


def rate_target_display(target_hz: float | None) -> str:
    return "off" if target_hz is None else f"{target_hz:g} Hz"


def seeds_for(target_hz: float | None) -> list[int]:
    """Every target — baseline and interior — runs all three seeds, so the
    accuracy–rate frontier carries across-seed error bars. The interior used to
    be single-seed (a limitation exp109 §2.3 disclosed); this removes it."""
    return list(SEEDS_BASELINE)


def cell_name(model: str, target_hz: float | None, seed: int) -> str:
    """Activity-frontier cell name, including its rate target and seed."""
    if target_hz is None:
        return f"{model}__off__seed{seed}"
    return f"{model}__{rate_target_label(target_hz)}__seed{seed}"


def _label(x: float) -> str:
    return f"{x:g}".replace(".", "p")


# ── Cell registry: one spec per trained cell, tagged by family ───────
# Each spec carries the family-specific bits (dt override + extra flags) so a
# single build_train_args reproduces every family. Names match the existing
# per-notebook artifacts, so folding the already-trained cells in is a move,
# not a retrain.

def _activity_frontier_cells() -> list[dict]:
    cells = []
    for m in MODELS:
        for target_hz in RATE_TARGET_GRID_HZ:
            extra = ([] if target_hz is None else
                     ["--fr-reg-upper-target-hz", str(target_hz),
                      "--fr-reg-upper-strength", str(FR_STRENGTH_UPPER)])
            for s in seeds_for(target_hz):
                cells.append({
                    "name": cell_name(m, target_hz, s), "model": m, "family": "activity_frontier",
                    "tag": rate_target_display(target_hz), "seed": s, "dt_ms": DT_MS,
                    "tau_gaba": TAU_GABA_GAMMA,
                    "rate_target_hz": target_hz, "extra": extra,
                })
    return cells


def _tau_gaba_cells() -> list[dict]:
    return [
        {"name": f"ping__tg{_label(tau)}__seed{s}", "model": "ping",
         "family": "tau_gaba", "tag": f"τ={tau:g}", "seed": s, "dt_ms": DT_MS,
         "tau_gaba": tau, "extra": []}
        for tau in TAU_GABA_SWEEP for s in SEEDS_BASELINE
    ]


def _dt_cells() -> list[dict]:
    # The dt sweep is the documented exception that varies dt by design.
    return [
        {"name": f"ping__dt{_label(dt)}__seed{s}", "model": "ping",
         "family": "dt", "tag": f"dt={dt:g}", "seed": s, "dt_ms": dt,
         "tau_gaba": TAU_GABA_GAMMA, "extra": []}
        for dt in DT_SWEEP_MS for s in SEEDS_BASELINE
    ]


def _canonical_cells() -> list[dict]:
    # The canonical reference: rate target = off, trained on ALL of MNIST (not the
    # subset the other families use) once the full standard is restored.
    return [
        {"name": f"{m}__canonical__seed{s}", "model": m, "family": "canonical",
         "tag": "off · all MNIST", "seed": s, "dt_ms": DT_MS, "extra": [],
         "tau_gaba": TAU_GABA_GAMMA, "max_samples": CANONICAL_MAX_SAMPLES}
        for m in MODELS for s in SEEDS_BASELINE
    ]


def _init_cells() -> list[dict]:
    cells = []
    for cond, (ei, t_ei, t_ie) in INIT_CONDITIONS.items():
        extra = ["--ei-strength", ei]
        if t_ei:
            extra.append("--trainable-w-ei")
        if t_ie:
            extra.append("--trainable-w-ie")
        for s in SEEDS_BASELINE:
            cells.append({
                "name": f"{cond}__seed{s}", "model": "ping_init",
                "family": "init", "tag": cond, "seed": s, "dt_ms": DT_MS,
                "tau_gaba": TAU_GABA_GAMMA, "extra": extra,
            })
    return cells


def _planned_variable_rate_cells() -> list[dict]:
    """Variable-rate, output-LIF spike-count PING bank consumed by exp082."""
    return [
        {
            "name": f"ping__variable_rate__seed{s}",
            "model": "ping",
            "family": "variable_rate",
            "tag": "categorical 0.5–25 Hz · output spike count",
            "seed": s,
            "dt_ms": DT_MS,
            "tau_gaba": TAU_GABA_GAMMA,
            "max_samples": SUBSET_MAX_SAMPLES,
            "readout": "spike-count",
            "input_rates_hz": list(VARIABLE_RATE_TRAINING_RATES_HZ),
            "rate_sampling": "uniform categorical per presentation",
            "consumer": VARIABLE_RATE_CONSUMER,
            "recipe_overrides": {
                "--readout-w-init-mean": TR06_READOUT_W_INIT_MEAN,
                "--readout-w-init-std": TR06_READOUT_W_INIT_STD,
            },
            "extra": [],
            "status": "ready_to_train",
        }
        for s in SEEDS_BASELINE
    ]


def low_w_in_cell_name(w_in: float, seed: int) -> str:
    label = f"{w_in:g}".replace(".", "p")
    return f"ping__low_w_in__win{label}__seed{seed}"


def _low_w_in_cells() -> list[dict]:
    """PING recruitment controls consumed by exp025."""
    return [
        {
            "name": low_w_in_cell_name(w_in, seed),
            "model": "ping",
            "family": "low_w_in",
            "tag": f"W_in={w_in:g}",
            "seed": seed,
            "dt_ms": DT_MS,
            "tau_gaba": TAU_GABA_GAMMA,
            "w_in": w_in,
            "rate_target_hz": 1.0,
            "recipe_overrides": {"--w-in": str(w_in)},
            "extra": [
                "--fr-reg-upper-target-hz", "1.0",
                "--fr-reg-upper-strength", str(FR_STRENGTH_UPPER),
            ],
        }
        for w_in in LOW_W_IN_VALUES
        for seed in SEEDS_BASELINE
    ]


PLANNED_VARIABLE_RATE_CELLS = _planned_variable_rate_cells()
BASE_CELLS = (_canonical_cells() + _activity_frontier_cells() + _tau_gaba_cells()
              + _dt_cells() + _init_cells() + _low_w_in_cells())
CANONICAL_CELLS = BASE_CELLS + PLANNED_VARIABLE_RATE_CELLS
for _cell in CANONICAL_CELLS:
    _cell["training_run_id"] = TRAINING_RUN_IDS[_cell["family"]]


def training_run_cells(training_run_id: str) -> tuple[dict, ...]:
    """Return isolated copies of the registered cells for one public TR ID."""
    if training_run_id not in set(TRAINING_RUN_IDS.values()):
        raise ValueError(f"unknown exp022 training-run ID {training_run_id!r}")
    return tuple(
        copy.deepcopy(cell)
        for cell in CANONICAL_CELLS
        if cell["training_run_id"] == training_run_id
    )


def training_run_values(training_run_id: str, field: str) -> tuple:
    """Return unique registered values for ``field`` in registry order."""
    values = []
    for cell in training_run_cells(training_run_id):
        if field not in cell:
            raise ValueError(f"{training_run_id} cell {cell['name']} has no {field!r}")
        value = cell[field]
        if not any(value == existing for existing in values):
            values.append(value)
    return tuple(values)


def training_run_cell(training_run_id: str, **identity: object) -> dict:
    """Resolve exactly one registered cell by a small set of identity fields."""
    matches = [
        cell
        for cell in training_run_cells(training_run_id)
        if all(cell.get(field) == value for field, value in identity.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {training_run_id} cell matching {identity}, "
            f"found {len(matches)}"
        )
    return matches[0]


def require_training_run_cells(
    training_run_id: str, expected_names: list[str] | tuple[str, ...] | set[str]
) -> None:
    """Fail clearly when a consumer's expected cell bank drifts from exp022."""
    registered = {cell["name"] for cell in training_run_cells(training_run_id)}
    expected = set(expected_names)
    if expected != registered:
        missing = sorted(registered - expected)
        extra = sorted(expected - registered)
        raise ValueError(
            f"{training_run_id} cell contract mismatch; missing={missing}, extra={extra}"
        )


def scientific_contract(cell: dict, max_samples: int, epochs: int) -> dict:
    """Cold-readable scientific fields that must not hide behind CLI defaults."""
    input_rates = cell.get("input_rates_hz")
    return {
        "dataset": {
            "name": "mnist",
            "source_train_partition": "official_mnist_train",
            "source_test_partition": "official_mnist_test",
            "training_pool_size": int(max_samples),
            "split": "stratified_90_10_within_official_train",
            "split_seed": 42,
            "optimizer_train_samples": round(max_samples * 0.9),
            "validation_samples": max_samples - round(max_samples * 0.9),
            "official_test_samples": MNIST_TEST_SAMPLES,
            "checkpoint_selection_partition": "validation",
            "official_test_used_during_training": False,
        },
        "input": {
            "encoding": "poisson_max_pixel_rate",
            "channels": N_INPUT,
            "rate_hz": None if input_rates else INPUT_RATE_HZ,
            "rate_distribution_hz": list(input_rates) if input_rates else None,
            "rate_sampling": (
                "uniform_categorical_per_presentation" if input_rates else "fixed"
            ),
        },
        "topology": {
            "excitatory_neurons": N_EXCITATORY,
            "inhibitory_neurons": N_INHIBITORY,
            "output_neurons": N_OUTPUT,
            "output_population": "spiking_lif",
            "ei_loop_enabled": cell["model"] != "coba",
        },
        "dynamics": {
            "presentation_duration_ms": T_MS,
            "dt_ms": float(cell["dt_ms"]),
            "tau_ampa_ms": TAU_AMPA_MS,
            "tau_gaba_ms": float(cell["tau_gaba"]),
        },
        "constraints": {"dales_law": DALES_LAW},
        "optimizer": {
            "name": "adamw",
            "learning_rate": float(MODEL_RECIPES[cell["model"]]["--lr"]),
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "batch_size": BATCH_SIZE,
            "epochs": int(epochs),
        },
        "readout": {
            "mode": cell.get(
                "readout", MODEL_RECIPES[cell["model"]]["--readout"]
            ),
            "shape": [N_EXCITATORY, N_OUTPUT],
        },
        "seed": int(cell["seed"]),
    }

RESOURCE_TIERS = (
    "standard",
    "fine_dt",
    "canonical_coba",
    "canonical_ping",
    "variable_rate",
    "all",
)


def cell_resource_tier(cell: dict) -> str:
    """Wilkes3 scheduling tier for one scientific registry cell."""
    if cell["family"] == "canonical":
        return f"canonical_{cell['model']}"
    if cell["family"] == "variable_rate":
        return "variable_rate"
    if cell["family"] == "dt" and cell["dt_ms"] == min(DT_SWEEP_MS):
        return "fine_dt"
    return "standard"


def cells_in_resource_tier(tier: str) -> list[dict]:
    """Registry-backed cell list consumed by scheduler wrappers."""
    if tier not in RESOURCE_TIERS:
        raise ValueError(
            f"unknown resource tier {tier!r}; choose from {RESOURCE_TIERS}"
        )
    if tier == "all":
        return list(CANONICAL_CELLS)
    return [cell for cell in CANONICAL_CELLS if cell_resource_tier(cell) == tier]

# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
SCALE = {
    "dataset": "mnist",
    # Reduced-pool scale; the six canonical cells override to all of MNIST.
    "max_samples": SUBSET_MAX_SAMPLES,
    "epochs": EPOCHS_STANDARD,
    "t_ms": T_MS,
    "dt_ms": DT_MS,
    "batch_size": BATCH_SIZE,
    "seeds": len(SEEDS_BASELINE),
    "cells": len(CANONICAL_CELLS),
    "grid": "7 training-run families",
}


def cell_dir(name: str) -> Path:
    """Shared per-cell artifact directory."""
    return TRAINING_ROOT / name


def load_cell(name: str) -> Path:
    """Return a trained cell's directory, or fail loudly if this notebook has
    not been run. Analysis notebooks call this instead of training."""
    d = cell_dir(name)
    if not (d / "weights.pth").exists() or not (d / "weights_final.pth").exists():
        raise SystemExit(
            f"missing trained cell '{name}' at {_display_path(d)}; "
            "run exp022 (Training) first to produce the shared cells."
        )
    return d


def _tr06_diagnostic_root() -> Path:
    return Path(os.environ["PINGLAB_ARTIFACTS_ROOT"])


def tr06_diagnostic_done(job_id: str) -> bool:
    """Modal completion hook for one bounded TR-06 readout variant."""
    from experiments.exp022_support import tr06_diagnostic

    return (
        job_id in tr06_diagnostic.VARIANTS
        and (_tr06_diagnostic_root() / job_id / "diagnostic_summary.json").exists()
    )


def run_tr06_diagnostic(job_id: str) -> None:
    """Modal execution hook; diagnostic scale is explicit in the job environment."""
    from experiments.exp022_support import tr06_diagnostic

    def optional_number(name: str, converter):
        value = os.environ.get(name)
        return None if value is None else converter(value)

    tr06_diagnostic.run_variant(
        job_id,
        root=_tr06_diagnostic_root(),
        max_samples=int(os.environ["EXP022_TR06_DIAGNOSTIC_MAX_SAMPLES"]),
        epochs=int(os.environ["EXP022_TR06_DIAGNOSTIC_EPOCHS"]),
        seed=int(os.environ["EXP022_TR06_DIAGNOSTIC_SEED"]),
        device="auto",
        n_hidden=optional_number("EXP022_TR06_DIAGNOSTIC_N_HIDDEN", int),
        t_ms=optional_number("EXP022_TR06_DIAGNOSTIC_T_MS", float),
        dt_ms=optional_number("EXP022_TR06_DIAGNOSTIC_DT_MS", float),
    )


def build_train_args(spec: dict, out_dir: Path,
                     max_samples: int, epochs: int,
                     recipes: dict[str, dict] | None = None) -> list[str]:
    """CLI `train` args for one registry cell, across all families."""
    recipe = dict((recipes or MODEL_RECIPES)[spec["model"]])
    recipe.update(spec.get("recipe_overrides", {}))
    if spec.get("readout") is not None:
        recipe["--readout"] = spec["readout"]
    ms = spec.get("max_samples") or max_samples   # canonical cells override
    args = [
        "train",
        "--model", recipe["__build_as"],
        "--dataset", "mnist",
        "--n-hidden", str(N_EXCITATORY),
        "--input-rate", str(INPUT_RATE_HZ),
        "--max-samples", str(ms),
        "--epochs", str(epochs),
        "--t-ms", str(T_MS),
        "--dt", str(spec["dt_ms"]),
        "--tau-gaba", str(spec["tau_gaba"]),
        "--seed", str(spec["seed"]),
        "--weight-decay", str(WEIGHT_DECAY),
        "--dales-law",
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    for k, v in recipe.items():
        if k.startswith("__"):
            continue
        if v is True:
            args.append(k)
        elif v is not None:
            args += [k, v]
    args += spec["extra"]
    if spec.get("input_rates_hz"):
        args += ["--input-rates", *[str(rate) for rate in spec["input_rates_hz"]]]
    return args


# ── Runner ───────────────────────────────────────────────────────────

# This runner trains at the full per-family standard: the canonical reference
# sees all of MNIST, every sweep family sees the 10% subset, and depth is
# EPOCHS_STANDARD throughout. Set PINGLAB_NB022_PLUMBING=1 to fall back to the
# tiny wiring-check scale (MAX_SAMPLES / EPOCHS) — that trains the whole
# registry in minutes to smoke-test the fan-out without spending the real
# ~94 GPU-hours; it is the only reason the plumbing constants still exist.


def cell_samples_epochs(spec: dict) -> tuple[int, int]:
    """Per-cell (max_samples, epochs) at the full per-family standard.

    Canonical cells carry their own max_samples (all of MNIST); every other
    family falls back to the 10% subset. Depth is EPOCHS_STANDARD for all
    families — the dt sweep is the exception in *dt*, not in epochs.

    PINGLAB_NB022_PLUMBING=1 overrides both to the minutes-long wiring scale."""
    if os.environ.get("PINGLAB_NB022_PLUMBING") == "1":
        return MAX_SAMPLES, EPOCHS
    return spec.get("max_samples") or SUBSET_MAX_SAMPLES, EPOCHS_STANDARD


def _json_safe(o):
    """Replace non-finite floats (NaN/inf from untrained cells) with None so
    the manifest is valid JSON the docs loader can parse."""
    if isinstance(o, float):
        return o if (o == o and abs(o) != float("inf")) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    return o


def load_metrics(d: Path) -> dict:
    p = d / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_config(d: Path) -> dict:
    """A cell's config.json — carries git_sha / torch_version / device, which
    metrics.json's own `config` block does NOT (its git_sha is null). The
    manifest reads provenance from here."""
    p = d / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def training_root_provenance(root: Path) -> dict[str, str]:
    resolved = root.resolve()
    try:
        return {"location": "repository", "path": str(resolved.relative_to(REPO))}
    except ValueError:
        return {"location": "external", "path": str(resolved)}


def final_rates(d: Path) -> tuple[float, float]:
    """Last-epoch E / I rate (Hz) from the retained epoch record."""
    rows = epoch_metrics(d)
    if not rows:
        return float("nan"), float("nan")
    row = rows[-1]
    return (float(row.get("test_rate_e", row.get("rate_e", float("nan")))),
            float(row.get("test_rate_i", row.get("rate_i", float("nan")))))


FAMILY_COLORS = {
    "canonical": theme.GREY_DARK,
    "activity_frontier": theme.INK_BLACK,
    "activity_frontier_3seed": theme.INK_BLACK,
    "tau_gaba": theme.DEEP_RED,
    "dt": theme.ELECTRIC_CYAN,
    "init": theme.AMBER,
    "variable_rate": theme.ELECTRIC_CYAN,
    "low_w_in": theme.MUTED,
}


def training_curve(d: Path) -> tuple[list[int], list[float]]:
    """Per-epoch (epoch, test accuracy) from the retained epoch record."""
    eps, accs = [], []
    for r in epoch_metrics(d):
        if "ep" in r and "acc" in r:
            eps.append(int(r["ep"]))
            accs.append(float(r["acc"]))
    return eps, accs


FAMILY_ORDER = [
    "canonical", "activity_frontier", "tau_gaba", "dt", "init",
    "variable_rate", "low_w_in",
]
FAMILY_LABELS = {
    "canonical": "Canonical reference",
    "activity_frontier": "Hidden-E activity-ceiling sweep",
    "activity_frontier_3seed": "Hidden-E activity-ceiling sweep (3-seed)",
    "tau_gaba": "τ_GABA ladder",
    "dt": "Δt sweep",
    "init": "Init variants",
    "variable_rate": "Variable-rate streaming bank",
    "low_w_in": "Low-input recruitment sweep",
}
FAMILY_ARTIFACT_SLUGS = {
    "activity_frontier": "theta_u",
    "low_w_in": "low_w_in",
}


def plot_family_curves(family: str, cells: list[dict],
                       out_path: Path, run_id: str) -> int:
    """One figure for one family: each cell's test-accuracy learning curve,
    coloured by the swept value. Returns the number of cells actually drawn."""
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    tags = list(dict.fromkeys(c["tag"] for c in cells))  # ordered unique
    # cm.viridis exists at runtime; the matplotlib stub omits it (false positive).
    colours = {t: cm.viridis(i / max(1, len(tags) - 1))  # ty: ignore[unresolved-attribute]
               for i, t in enumerate(tags)}
    # ping (and ping-init) solid, coba dashed — distinguishes the two models
    # in families that train both (rate target, canonical).
    linestyle = {"coba": "--", "ping": "-", "ping_init": "-"}
    models = list(dict.fromkeys(c["model"] for c in cells))

    fig, ax = plt.subplots(figsize=(6.5, 3.66))   # H11–H12: column width, 16:9
    n = 0
    for c in cells:
        eps, accs = training_curve(cell_dir(c["name"]))
        if eps:
            ax.plot(eps, accs, lw=1.1, color=colours[c["tag"]],
                    ls=linestyle.get(c["model"], "-"), alpha=0.85)
            n += 1
    handles = [Line2D([0], [0], color=colours[t], lw=2.4, label=t) for t in tags]
    leg1 = ax.legend(handles=handles, frameon=False, fontsize=theme.SIZE_LEGEND,
                     ncol=2, loc="lower right", title="swept value")
    ax.add_artist(leg1)
    if len(models) > 1:
        mh = [Line2D([0], [0], color=theme.MUTED, lw=2.0,
                     ls=linestyle.get(m, "-"), label="ping" if m == "ping_init" else m)
              for m in models]
        ax.legend(handles=mh, frameon=False, fontsize=theme.SIZE_LEGEND,
                  loc="lower center", title="model")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # H11: no plot title — the Typst caption carries the family + takeaway.
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)   # H10: line plot → SVG (caller passes .svg); dpi from theme
    plt.close(fig)
    return n


# ── RunPod fan-out (the second dispatch backend, alongside local) ──
# The generic pod-fleet machinery lives in helpers/runpod.py; these functions
# are the exp022-specific glue that tells it what to train. Driven by the
# `--runpod` path in main(); see writings/exp022.typ §3 for the compute design.

def runpod_is_done(cell: dict, plumbing: bool) -> bool:
    """A cell is done iff its metrics.json exists AND was trained at the scale
    THIS run expects (max_samples, epochs, dt all matching).

    Existence alone is not enough: a training root may contain cells produced
    under a different run contract, and a bare exists() check would skip those
    and silently ship a mixed-scale, invalid dataset. Comparing the baked config
    makes the marker honest: a stale cell reads as pending and gets retrained.
    """
    p = cell_dir(cell["name"]) / "metrics.json"
    if not p.exists():
        return False
    try:
        cfg = json.loads(p.read_text()).get("config", {})
    except (json.JSONDecodeError, OSError):
        return False
    if plumbing:
        os.environ["PINGLAB_NB022_PLUMBING"] = "1"
    want_ms, want_ep = cell_samples_epochs(cell)
    return (cfg.get("max_samples") == want_ms
            and cfg.get("epochs") == want_ep
            and cfg.get("dt") == cell["dt_ms"])


def _train_one_cell(cell: dict, plumbing: bool) -> None:
    """Train ONE cell by invoking the SNN CLI — flags identical to a local run.

    Writes to cell_dir(name), which sits under TRAINING_ROOT — on a pod that is
    the shared network volume (/shared/training via PINGLAB_TRAINING_ROOT), so
    the artifact is durable the moment it lands. Used by --pod-run and --train-cell.
    """
    ms, ep = cell_samples_epochs(cell)  # honours PINGLAB_NB022_PLUMBING
    spec = cell
    if plumbing:
        # build_train_args re-applies a canonical cell's own max_samples (60000),
        # which would defeat the tiny plumbing scale. Strip it so the plumbing
        # ms=100 takes — and so runpod_is_done agrees with what was trained.
        spec = {k: v for k, v in cell.items() if k != "max_samples"}
    args = build_train_args(spec, cell_dir(cell["name"]), ms, ep)
    print(
        f"[train-cell] {cell['training_run_id']} / {cell['name']} "
        f"(n={ms}, {ep} ep) → {cell_dir(cell['name'])}"
    )
    subprocess.run([sys.executable, str(SNN_TOOL), *args], cwd=REPO, check=True)
    _stamp_training_run_identity(cell)


def _stamp_training_run_identity(cell: dict) -> None:
    """Attach the public training-run ID to one completed cell's artifacts."""
    directory = cell_dir(cell["name"])
    for filename in ("config.json", "metrics.json"):
        path = directory / filename
        if not path.exists():
            raise RuntimeError(f"completed cell is missing {path}")
        payload = json.loads(path.read_text())
        payload["training_run_id"] = cell["training_run_id"]
        payload["training_cell_name"] = cell["name"]
        nested_config = payload.get("config")
        if isinstance(nested_config, dict):
            nested_config["training_run_id"] = cell["training_run_id"]
            nested_config["training_cell_name"] = cell["name"]
        path.write_text(json.dumps(payload, indent=2) + "\n")


def _cell_by_name(name: str) -> dict | None:
    return next((c for c in CANONICAL_CELLS if c["name"] == name), None)


def pod_run() -> None:
    """Pod-side entrypoint (image start script runs `exp022.py --pod-run`).

    Trains every cell named in the CELLS env var to the shared volume, skipping
    any already done there (scale-aware marker → free resume across pods), then
    self-terminates. The loop, skip-done and always-self-terminate contract lives
    in runpod.pod_run_loop; here we only say what a cell's done-check and training
    run are.
    """
    plumbing = os.environ.get("PINGLAB_NB022_PLUMBING") == "1"
    print(f"[pod-run] plumbing={plumbing} root={TRAINING_ROOT}")

    def is_done(name: str) -> bool:
        cell = _cell_by_name(name)
        return cell is not None and runpod_is_done(cell, plumbing)

    def run_job(name: str) -> None:
        cell = _cell_by_name(name)
        assert cell is not None  # pod_run_loop only passes registered job ids
        _train_one_cell(cell, plumbing)

    runpod.pod_run_loop(
        job_ids=[c["name"] for c in CANONICAL_CELLS],
        is_done=is_done, run_job=run_job,
    )


def runpod_buckets(cells: list[dict], cells_per_pod: int) -> list[dict]:
    """Assign cells to pods: each canonical cell → its own pod (heavy, isolated);
    every other family packed cells_per_pod at a time. Returns [{name, cells}]."""
    canonical = [c["name"] for c in cells if c["family"] == "canonical"]
    sweep = [c["name"] for c in cells if c["family"] != "canonical"]
    buckets = [{"name": f"canon-{n}", "cells": [n]} for n in canonical]
    for i in range(0, len(sweep), cells_per_pod):
        buckets.append({"name": f"sweep-{i // cells_per_pod:02d}",
                        "cells": sweep[i:i + cells_per_pod]})
    return buckets


def run_via_runpod(argv: list[str]) -> None:
    """`--runpod` dispatch: fire a laptop-independent RunPod fan-out via the shared
    runpod.dispatch path.

    Pods self-run their assigned cells to the shared network volume and
    self-terminate; the laptop only fires them. Retrieve results afterwards with
    `--runpod --collect`, then build figures with `exp022.py --skip-training`.
    Dry-run by DEFAULT; --live to create pods. Exp022's only bespoke bit is
    runpod_buckets (one pod per canonical cell); everything else is the common
    fan-out in helpers/runpod.py.
    """
    meta = parse_meta(argv, allow_dispatch=True)

    cells = CANONICAL_CELLS
    if meta.only_cells:
        wanted = set(meta.only_cells)
        cells = [c for c in cells if c["name"] in wanted]
        missing = wanted - {c["name"] for c in cells}
        if missing:
            raise SystemExit(f"unknown cell(s): {sorted(missing)}")

    runpod.dispatch(
        slug=SLUG, runner=SLUG,
        buckets=runpod_buckets(cells, meta.cells_per_pod),
        gpu=meta.gpu, live=meta.live, plumbing=meta.plumbing, collect=meta.collect,
        collect_subdir=runpod.TRAINING_SUBDIR,
        local_collect_dir=str(TRAINING_ROOT),
        plumbing_env={"PINGLAB_NB022_PLUMBING": "1"},
    )


# ── Appendix: one fixed-input raster per config (visual inspection) ──

def _gamma_psd(spk_i, dt):
    """I-population power spectrum + gamma peak from a single-trial raster.

    1 ms population rate → 3 ms Gaussian smooth (suppresses the harmonics that
    sharp inhibitory bursts inject) → Hann-windowed FFT. Returns (freqs, psd,
    f_gamma); f_gamma is None when the I population is silent (COBA — no E/I
    loop) or no prominent γ is resolved. The 3 ms kernel is chosen so the visible
    peak is the FUNDAMENTAL, not the 2× harmonic — verified against the
    multi-trial τ_GABA scaling, which matches nb041 (τ=6 ms → ≈ 45 Hz)."""
    import numpy as np

    T, ni = spk_i.shape
    spm = max(1, round(1.0 / dt))          # timesteps per 1 ms bin (dt-aware:
    nb = T // spm                          # the Δt sweep cells vary dt)
    b = spk_i[: nb * spm].reshape(nb, spm, ni).sum(axis=(1, 2)).astype(float)  # pop/ms
    if b.sum() < 50:                       # essentially silent (e.g. COBA I pop)
        return None, None, None
    k = np.exp(-0.5 * ((np.arange(31) - 15) / 3.0) ** 2)
    k /= k.sum()
    x = np.convolve(b - b.mean(), k, "same") * np.hanning(nb)
    fr = np.fft.rfftfreq(nb, 1 / 1000.0)   # 1 kHz after 1 ms binning
    P = np.abs(np.fft.rfft(x)) ** 2
    band = (fr >= 20) & (fr <= 110)
    fpk = float(fr[band][np.argmax(P[band])])
    prom = P[band].max() / (np.median(P[band]) + 1e-9)
    return fr, P, (fpk if prom > 2.5 else None)


def _plot_snapshot_raster(snap_path: Path, out_png: Path) -> None:
    """Raster + population rate + I-population PSD (γ peak labelled) for a single
    fixed-image snapshot. The PSD describes the exact raster shown."""
    import numpy as np

    d = np.load(snap_path)
    se, si, dt = d["spk_e"], d["spk_i"], float(d["dt"])
    T, ne = se.shape
    ni = si.shape[1]
    tms = T * dt
    et, ec = np.nonzero(se)
    it, ic = np.nonzero(si)
    e_hz = se.sum() / ne / (tms / 1000)
    i_hz = si.sum() / max(ni, 1) / (tms / 1000)
    fr, P, fgam = _gamma_psd(si, dt)

    theme.apply()
    # H12: stacked multi-panel, column width, height capped so it fits a page.
    fig = plt.figure(figsize=(6.5, 5.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.15], width_ratios=[3, 1],
                          hspace=0.32, wspace=0.20)
    ax = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax.scatter(et * dt, ec, s=1.0, c=theme.INK_BLACK, marker="|", linewidths=0.35)
    ax.scatter(it * dt, ic + ne, s=1.0, c=theme.DEEP_RED, marker="|", linewidths=0.35)
    ax.axhline(ne, color="k", lw=0.4, alpha=0.4)
    ax.set_ylim(0, ne + ni)
    ax.set_xlim(0, tms)
    ax.set_ylabel("neuron  (E below · I above)")
    # H11: no plot title (config descriptor lives in the caption). The measured
    # values stay as a compact data annotation — they're computed at render time
    # from this raster, so they can't drift, unlike a hand-typed caption number.
    gtxt = f"f_γ ≈ {fgam:.0f} Hz" if fgam else "asynchronous (no γ)"
    ax.annotate(f"digit 0 · E {e_hz:.0f} Hz · I {i_hz:.0f} Hz · {gtxt}",
                xy=(0, 1.02), xycoords="axes fraction", fontsize=theme.SIZE_ANNOTATION,
                color=theme.MUTED, ha="left", va="bottom")

    bins = np.arange(0, tms + 1, 1.0)
    re, _ = np.histogram(et * dt, bins=bins)
    ri, _ = np.histogram(it * dt, bins=bins)
    ctr = (bins[:-1] + bins[1:]) / 2
    ax2.plot(ctr, re / ne * 1000, c=theme.INK_BLACK, lw=0.7, label="E")
    ax2.plot(ctr, ri / max(ni, 1) * 1000, c=theme.DEEP_RED, lw=0.7, label="I")
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("Hz/cell")
    ax2.set_xlim(0, tms)
    ax2.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)

    if fr is not None:
        import numpy as np
        # Normalise to the gamma-band peak (not the DC/onset bin, which otherwise
        # dwarfs the rhythm) and drop the low-freq ramp so f_γ is the visual focus.
        gband = (fr >= 20) & (fr <= 110)
        norm = P[gband].max() or 1.0
        m = (fr >= 8) & (fr <= 120)
        ax3.plot(fr[m], np.clip(P[m] / norm, 0, 1.3), c=theme.DEEP_RED, lw=1.0)
        if fgam:
            ax3.axvline(fgam, color=theme.INK_BLACK, ls="--", lw=0.9)
            ax3.annotate(f"{fgam:.0f} Hz", xy=(fgam, 1.0), xytext=(5, -3),
                         textcoords="offset points", fontsize=9, fontweight="bold")
        ax3.set_ylim(0, 1.3)
    else:
        ax3.text(0.5, 0.5, "I silent\n(no γ loop)", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=9, color=theme.MUTED)
    ax3.set_xlim(0, 120)
    ax3.set_xlabel("freq (Hz)")
    ax3.set_ylabel("I PSD (norm)")

    for a in (ax, ax2, ax3):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)   # PNG (dense raster, H10); dpi 240 from theme (H11)
    plt.close(fig)


def appendix_rasters() -> None:
    """Generate one digit-0/sample-0 raster per seed-42 config for the writeup
    appendix. The SAME fixed MNIST image is sent through every trained network
    (via `sim --infer --digit 0 --sample 0` → snapshot.npz), so the rasters are
    directly comparable across configs. Writes flat rasters__*.png presentation files."""
    import shutil

    cells = [c for c in CANONICAL_CELLS if c["seed"] == 42]
    rdir = FIGURES
    # Snapshots are pure throwaway (only needed to plot each PNG). Keep them under
    # the run-state namespace and wipe the whole scratch tree at the end,
    # so temp stays minimal — the PNGs in artifacts/ are the durable output.
    scratch_root = ARTIFACTS / "appendix-scratch"
    print(f"appendix: {len(cells)} rasters (seed 42) → {_display_path(rdir)}")
    try:
        for c in cells:
            d = cell_dir(c["name"])
            if not (d / "weights_final.pth").exists():
                print(f"[skip] {c['name']} — no weights")
                continue
            checkpoint = resolve_checkpoint(d, RESULT_CHECKPOINT_ROLE)
            scratch = scratch_root / c["name"]
            inference_args = [
                sys.executable, str(SNN_TOOL), "sim", "--infer",
                "--load-config", str(d / "config.json"),
                "--load-weights", str(checkpoint["path"]),
                "--digit", "0", "--sample", "0",
                "--out-dir", str(scratch), "--wipe-dir",
            ]
            if c["family"] == "variable_rate":
                inference_args += ["--input-rate", "5"]
            subprocess.run(
                inference_args,
                cwd=REPO, check=True, capture_output=True)
            _plot_snapshot_raster(scratch / "snapshot.npz", rdir / f"rasters__{c['name']}.png")
            print(f"  {c['name']}.png")
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


# ── Results: the data-fraction contrast (100% vs 10% MNIST) ──────────
# The canonical cells use all 60k official training images; every sweep
# (including the no-budget `off` cell) uses a 7k subset. Weights differ ONLY in
# that training-pool size, so
# the raster density gap between coba/ping canonical and off is attributable to
# the training data alone — the visual counterpart to the ≈ 2× firing-rate gap.

def _raster_panel(ax, snap_path: Path) -> tuple[float, float]:
    """Draw one raster (E black below the divider, I red above) into `ax` from a
    fixed-image snapshot and return its measured (E, I) mean rate in Hz. No PSD /
    rate sub-panels — this figure is about spike DENSITY, not rhythm."""
    import numpy as np

    d = np.load(snap_path)
    se, si, dt = d["spk_e"], d["spk_i"], float(d["dt"])
    T, ne = se.shape
    ni = si.shape[1]
    tms = T * dt
    et, ec = np.nonzero(se)
    it, ic = np.nonzero(si)
    e_hz = se.sum() / ne / (tms / 1000)
    i_hz = si.sum() / max(ni, 1) / (tms / 1000)
    ax.scatter(et * dt, ec, s=0.5, c=theme.INK_BLACK, marker="|", linewidths=0.25)
    ax.scatter(it * dt, ic + ne, s=0.5, c=theme.DEEP_RED, marker="|", linewidths=0.25)
    ax.axhline(ne, color="k", lw=0.4, alpha=0.4)
    ax.set_ylim(0, ne + ni)
    ax.set_xlim(0, tms)
    return e_hz, i_hz


def comparison_rasters() -> None:
    """One 2×2 raster figure contrasting training-data fraction. The SAME digit-0
    image runs through the no-budget coba and ping cells trained on ALL of MNIST
    (canonical) versus 10% (spike-budget = off); rows are the two architectures,
    columns the data fraction. Only the fraction differs between a row's two
    cells, so the density gap is attributable to it. Reuses the already-trained
    seed-42 weights (no retraining) → .artifacts/exp022/comparison__data_fraction.png."""
    import shutil

    # (row label, 100%-MNIST cell, 10%-MNIST cell) — seed 42 as the representative.
    grid = [
        ("COBA", "coba__canonical__seed42", "coba__off__seed42"),
        ("PING", "ping__canonical__seed42", "ping__off__seed42"),
    ]
    col_titles = ["100% MNIST (canonical)", "10% MNIST (off)"]
    scratch_root = ARTIFACTS / "comparison-scratch"
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.06),   # H11–H12: 16:9
                             gridspec_kw={"hspace": 0.30, "wspace": 0.14})
    print(
        "comparison: 4 rasters (seed 42) → "
        f"{_display_path(FIGURES / 'comparison__data_fraction.png')}"
    )
    try:
        for r, (label, full_cell, sub_cell) in enumerate(grid):
            for cc, name in enumerate((full_cell, sub_cell)):
                ax = axes[r][cc]
                d = cell_dir(name)
                if not (d / "weights_final.pth").exists():
                    ax.text(0.5, 0.5, f"{name}\n(no weights)", ha="center",
                            va="center", transform=ax.transAxes,
                            fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED)
                    continue
                checkpoint = resolve_checkpoint(d, RESULT_CHECKPOINT_ROLE)
                scratch = scratch_root / name
                subprocess.run(
                    [sys.executable, str(SNN_TOOL), "sim", "--infer",
                     "--load-config", str(d / "config.json"),
                     "--load-weights", str(checkpoint["path"]),
                     "--digit", "0", "--sample", "0",
                     "--out-dir", str(scratch), "--wipe-dir"],
                    cwd=REPO, check=True, capture_output=True)
                e_hz, i_hz = _raster_panel(ax, scratch / "snapshot.npz")
                ax.annotate(f"E {e_hz:.0f} Hz · I {i_hz:.0f} Hz",
                            xy=(0, 1.01), xycoords="axes fraction",
                            fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
                            ha="left", va="bottom")
                if r == 0:
                    ax.set_title(col_titles[cc], fontsize=theme.SIZE_LABEL)
                if cc == 0:
                    ax.set_ylabel(f"{label}\nneuron (E · I)")
                if r == 1:
                    ax.set_xlabel("time (ms)")
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
        out = FIGURES / "comparison__data_fraction.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)   # PNG (dense raster, H10); dpi 240 from theme (H11)
        plt.close(fig)
        print(f"  wrote {_display_path(out)}")
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


def _campaign_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--campaign-manifest", type=Path, metavar="ROOT")
    group.add_argument("--campaign-status", type=Path, metavar="MANIFEST")
    group.add_argument("--campaign-list", type=Path, metavar="MANIFEST")
    group.add_argument("--campaign-train-cell", metavar="NAME")
    group.add_argument("--campaign-validate", type=Path, metavar="MANIFEST")
    group.add_argument("--campaign-aggregate", type=Path, metavar="MANIFEST")
    group.add_argument(
        "--campaign-import-compatible", type=Path, metavar="MANIFEST"
    )
    parser.add_argument("--campaign", type=Path, metavar="MANIFEST")
    parser.add_argument("--from-campaign", type=Path, metavar="MANIFEST")
    parser.add_argument("--campaign-id")
    parser.add_argument("--tier", default="all")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--retry-only", action="store_true")
    parser.add_argument("--recover-stale", action="store_true")
    parser.add_argument("--plumbing", action="store_true")
    return parser


def _portable_cell_contract(row: dict) -> dict:
    """Return the scientific cell contract with destination paths removed."""
    parameters = copy.deepcopy(row["parameters"])
    parameters.get("arguments", {}).pop("--out-dir", None)
    return {
        "name": row["name"],
        "training_run_id": row["training_run_id"],
        "family": row["family"],
        "resource_tier": row["resource_tier"],
        "parameters": parameters,
    }


def _import_compatible_cells(destination: dict, source_path: Path) -> dict:
    """Copy only source cells with an identical resolved scientific contract."""
    source_path = source_path.resolve()
    source = campaign.load_manifest(source_path)
    source_root = Path(source["campaign_root"])
    if source_path != source_root / "campaign.json":
        raise SystemExit("source manifest must be <campaign-root>/campaign.json")
    source_rows = {row["name"]: row for row in source["cells"]}
    imported: list[str] = []
    incompatible: list[str] = []
    for row in destination["cells"]:
        source_row = source_rows.get(row["name"])
        if (
            source_row is None
            or _portable_cell_contract(source_row) != _portable_cell_contract(row)
        ):
            incompatible.append(row["name"])
            continue
        validation = campaign.validate_cell(source_row)
        if not validation["valid"]:
            raise SystemExit(
                f"compatible source cell is invalid: {row['name']}: "
                + "; ".join(validation["reasons"])
            )
        destination_dir = Path(row["output_directory"])
        if destination_dir.exists():
            raise SystemExit(f"import destination already exists: {destination_dir}")
        shutil.copytree(Path(source_row["output_directory"]), destination_dir)
        origin = {
            "campaign_id": source["campaign_id"],
            "campaign_manifest_sha256": source["manifest_sha256"],
            "repository_commit": source["repository"]["commit"],
            "source_directory": source_row["output_directory"],
        }
        for filename in ("config.json", "metrics.json"):
            path = destination_dir / filename
            payload = json.loads(path.read_text())
            payload["imported_cell_provenance"] = origin
            payload["training_run_id"] = row["training_run_id"]
            payload["training_cell_name"] = row["name"]
            nested = payload.get("config")
            if isinstance(nested, dict):
                nested["training_run_id"] = row["training_run_id"]
                nested["training_cell_name"] = row["name"]
            path.write_text(json.dumps(payload, indent=2) + "\n")
        _stamp_campaign_identity(destination_dir, destination, row)
        imported_validation = campaign.validate_cell(row)
        if not imported_validation["valid"]:
            raise RuntimeError(
                f"imported cell failed destination validation: {row['name']}: "
                + "; ".join(imported_validation["reasons"])
            )
        imported.append(row["name"])
    return {
        "source_campaign_id": source["campaign_id"],
        "destination_campaign_id": destination["campaign_id"],
        "imported": imported,
        "pending_incompatible": incompatible,
    }


def _checked_manifest(path: Path, *, allow_generated_dirty: bool = False) -> dict:
    manifest_path = path.resolve()
    manifest = campaign.load_manifest(manifest_path)
    root = Path(manifest["campaign_root"])
    if not root.is_absolute() or root.resolve() != root:
        raise SystemExit("campaign root must be an absolute resolved path")
    if manifest_path != root / "campaign.json":
        raise SystemExit("campaign manifest must be <campaign-root>/campaign.json")
    commit, dirty = campaign.git_identity(REPO)
    if dirty:
        dirty_paths = campaign.git_dirty_paths(REPO)
        allowed_prefixes = (".artifacts/exp022/", ".demolab/pdfs/exp022.pdf")
        if not allow_generated_dirty or any(
            not path.startswith(allowed_prefixes) for path in dirty_paths
        ):
            raise SystemExit("campaign execution requires a clean source worktree")
    if manifest["repository"] != {"commit": commit, "dirty": False}:
        raise SystemExit(
            "campaign manifest does not match the clean checked-out commit: "
            f"manifest={manifest['repository']['commit']} checkout={commit}"
        )
    if manifest.get("environment", {}).get("lockfile") != campaign.lock_identity(REPO):
        raise SystemExit("campaign lockfile identity does not match the checkout")
    tier = manifest.get("selection", {}).get("tier")
    try:
        selected_cells = cells_in_resource_tier(tier)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    manifest_names_list = [row.get("name") for row in manifest.get("cells", [])]
    if len(manifest_names_list) != len(set(manifest_names_list)):
        raise SystemExit("campaign contains duplicate cell names")
    expected_names_list = [cell["name"] for cell in selected_cells]
    if manifest_names_list != expected_names_list:
        raise SystemExit("campaign cell list does not exactly match its declared selection")
    previous_plumbing = os.environ.get("PINGLAB_NB022_PLUMBING")
    runtime_commands = {}
    try:
        if manifest.get("plumbing"):
            os.environ["PINGLAB_NB022_PLUMBING"] = "1"
        else:
            os.environ.pop("PINGLAB_NB022_PLUMBING", None)
        for row in manifest["cells"]:
            spec = _cell_by_name(row["name"])
            assert spec is not None
            samples, epochs = cell_samples_epochs(spec)
            command_spec = ({k: v for k, v in spec.items() if k != "max_samples"}
                            if manifest.get("plumbing") else spec)
            train_args = build_train_args(command_spec, root / "cells" / spec["name"], samples, epochs)
            resolved = campaign.resolved_parameters(
                spec, train_args, samples, epochs,
                scientific_contract=scientific_contract(spec, samples, epochs),
            )
            command = [campaign.python_executable(), str(SNN_TOOL), *train_args]
            output_directory = (root / "cells" / spec["name"]).resolve()
            expected = {
                "name": spec["name"],
                "training_run_id": spec["training_run_id"],
                "family": spec["family"],
                "resource_tier": cell_resource_tier(spec),
                "parameters": resolved,
                "command": command,
                "command_shell": shlex.join(command),
                "output_directory": str(output_directory),
                "required_outputs": list(campaign.REQUIRED_CELL_FILES),
            }
            if row != expected:
                raise SystemExit(f"campaign manifest registry drift for {row['name']}")
            if output_directory.parent != (root / "cells").resolve():
                raise SystemExit(f"campaign output path escapes the cells root: {row['name']}")
            runtime_commands[row["name"]] = command
    finally:
        if previous_plumbing is None:
            os.environ.pop("PINGLAB_NB022_PLUMBING", None)
        else:
            os.environ["PINGLAB_NB022_PLUMBING"] = previous_plumbing
    manifest["_runtime_commands"] = runtime_commands
    return manifest


def _stamp_campaign_identity(directory: Path, manifest: dict, row: dict) -> None:
    for filename in ("config.json", "metrics.json"):
        path = directory / filename
        payload = json.loads(path.read_text())
        payload.update({
            "campaign_id": manifest["campaign_id"],
            "campaign_manifest_sha256": manifest["manifest_sha256"],
            "resource_tier": row["resource_tier"],
            "campaign_repository_commit": manifest["repository"]["commit"],
            "campaign_resolved_parameters": row["parameters"],
        })
        nested = payload.get("config")
        if isinstance(nested, dict):
            nested.update({
                "campaign_id": manifest["campaign_id"],
                "campaign_manifest_sha256": manifest["manifest_sha256"],
                "resource_tier": row["resource_tier"],
                "campaign_repository_commit": manifest["repository"]["commit"],
                "campaign_resolved_parameters": row["parameters"],
            })
        campaign.atomic_json(path, payload)


def _gpu_metadata() -> dict:
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return {"available": False}
    if query.returncode != 0:
        return {"available": False}
    return {"available": True, "devices": [line.strip() for line in query.stdout.splitlines()]}


def _campaign_train(manifest_path: Path, name: str, *, recover_stale: bool = False) -> int:
    manifest = _checked_manifest(manifest_path)
    row = campaign.manifest_cell(manifest, name)
    directory = Path(row["output_directory"])
    existing = campaign.validate_cell(row)
    if existing["valid"]:
        print(f"[skip-valid] {name} is complete and will not be touched")
        return 0
    record, attempt_lock = campaign.acquire_attempt(
        manifest, row, recover_stale=recover_stale,
    )
    status_path = campaign.status_path(manifest, name)
    exit_code = 1
    attempt_started = time.monotonic()
    try:
        existing = campaign.validate_cell(row)
        if existing["valid"]:
            record.update({
                "ended_at_utc": campaign.utc_now(), "exit_code": 0,
                "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
                "state": "complete", "validation": existing,
                "note": "became valid before training ownership was acquired",
            })
            campaign.atomic_json(status_path, record)
            print(f"[skip-valid] {name} became complete and will not be touched")
            return 0
        preserved = campaign.preserve_partial(directory)
        if preserved:
            print(f"[preserve-partial] {directory} -> {preserved}")
        record["gpu"] = _gpu_metadata()
        campaign.atomic_json(status_path, record)
        directory.parent.mkdir(parents=True, exist_ok=True)
        command = manifest["_runtime_commands"][name]
        completed = subprocess.run(command, cwd=REPO)
        exit_code = completed.returncode
        if exit_code == 0:
            spec = _cell_by_name(name)
            assert spec is not None
            old_root = globals()["TRAINING_ROOT"]
            try:
                globals()["TRAINING_ROOT"] = Path(manifest["campaign_root"]) / "cells"
                _stamp_training_run_identity(spec)
            finally:
                globals()["TRAINING_ROOT"] = old_root
            _stamp_campaign_identity(directory, manifest, row)
        validation = campaign.validate_cell(row)
        try:
            metrics_payload = load_metrics(directory)
        except (OSError, ValueError, json.JSONDecodeError):
            metrics_payload = {}
        record.update({
            "ended_at_utc": campaign.utc_now(), "exit_code": exit_code,
            "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
            "state": "complete" if exit_code == 0 and validation["valid"] else "failed",
            "validation": validation,
            "gpu_after": _gpu_metadata(),
            "training_performance": metrics_payload.get("perf"),
            "output_bytes": sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()),
        })
        directory.mkdir(parents=True, exist_ok=True)
        campaign.atomic_json(directory / "attempt.json", record)
        campaign.atomic_json(status_path, record)
        return 0 if record["state"] == "complete" else 1
    except BaseException as exc:
        record.update({
            "ended_at_utc": campaign.utc_now(), "exit_code": exit_code,
            "elapsed_seconds": round(time.monotonic() - attempt_started, 3),
            "state": "failed", "error": f"{type(exc).__name__}: {exc}",
        })
        directory.mkdir(parents=True, exist_ok=True)
        campaign.atomic_json(directory / "attempt.json", record)
        campaign.atomic_json(status_path, record)
        raise
    finally:
        campaign.release_attempt(attempt_lock, record["attempt_id"])


def _handle_campaign_cli(argv: list[str]) -> bool:
    if not any(flag in argv for flag in (
        "--campaign-manifest", "--campaign-status", "--campaign-list",
        "--campaign-train-cell", "--campaign-validate", "--campaign-aggregate",
        "--campaign-import-compatible",
    )):
        return False
    args = _campaign_parser().parse_args(argv[1:])
    if args.campaign_import_compatible:
        if args.from_campaign is None:
            raise SystemExit("--from-campaign is required")
        destination = _checked_manifest(args.campaign_import_compatible)
        print(json.dumps(
            _import_compatible_cells(destination, args.from_campaign),
            indent=2,
            sort_keys=True,
        ))
        return True
    if args.campaign_manifest:
        if not args.campaign_id:
            raise SystemExit("--campaign-id is required")
        try:
            selected = cells_in_resource_tier(args.tier)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.plumbing:
            os.environ["PINGLAB_NB022_PLUMBING"] = "1"
        root = args.campaign_manifest.resolve()
        manifest = campaign.create_manifest(
            repo=REPO, campaign_root=root, campaign_id=args.campaign_id,
            cells=selected, tier_for=cell_resource_tier,
            samples_epochs=cell_samples_epochs, build_args=build_train_args,
            scientific_contract_for=scientific_contract,
            plumbing=args.plumbing,
            selection_tier=args.tier,
        )
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise SystemExit(
                f"campaign destination already exists and will not be modified: {root}"
            ) from exc
        for child in ("cells", "logs", "status", "submissions"):
            (root / child).mkdir()
        campaign.write_manifest(root / "campaign.json", manifest)
        print(root / "campaign.json")
        return True
    manifest_path = (args.campaign or args.campaign_status or args.campaign_list
                     or args.campaign_validate or args.campaign_aggregate)
    if manifest_path is None:
        raise SystemExit("--campaign MANIFEST is required")
    manifest = _checked_manifest(manifest_path)
    if args.campaign_train_cell:
        raise SystemExit(_campaign_train(
            manifest_path, args.campaign_train_cell,
            recover_stale=args.recover_stale,
        ))
    if args.campaign_validate:
        print(f"valid manifest {manifest['campaign_id']} {manifest['manifest_sha256']}")
        return True
    status = campaign.summarize_status(manifest)
    if args.campaign_aggregate:
        if len(manifest["cells"]) != len(CANONICAL_CELLS):
            raise SystemExit("aggregation requires the complete 102-cell registry")
        incomplete = [row["name"] for row in status["cells"] if not row["valid"]]
        if incomplete:
            raise SystemExit(
                f"aggregation refused: {len(incomplete)} cells are not valid"
            )
        environment = os.environ.copy()
        environment["PINGLAB_TRAINING_ROOT"] = str(Path(manifest["campaign_root"]) / "cells")
        environment["EXP022_VERIFIED_CAMPAIGN"] = str(manifest_path.resolve())
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--skip-training"],
            cwd=REPO, env=environment, check=True,
        )
        # The guide embeds representative rasters as required results, not as
        # optional local decorations.  Generate them from the verified bank as
        # part of campaign aggregation so a promoted campaign is self-contained.
        for figure in ("appendix-rasters", "comparison-rasters"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--plot-only",
                    figure,
                ],
                cwd=REPO,
                env=environment,
                check=True,
            )
        final_manifest = _checked_manifest(manifest_path, allow_generated_dirty=True)
        final = campaign.summarize_status(final_manifest)
        if any(not row["valid"] for row in final["cells"]):
            raise SystemExit("campaign changed during aggregation")
        return True
    if args.campaign_list:
        cells = [cell for cell in manifest["cells"] if args.tier == "all" or cell["resource_tier"] == args.tier]
        if args.retry_only:
            retry = set(status["retry_cells"])
            cells = [cell for cell in cells if cell["name"] in retry]
        print("\n".join(cell["name"] for cell in cells))
    elif args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        campaign.print_status(status)
    return True


@preserve_active_view(SLUG)
def main() -> None:
    if _handle_campaign_cli(sys.argv):
        return
    meta = parse_meta(sys.argv, allow_dispatch=True)

    if meta.list_cells is not None:
        try:
            cells = cells_in_resource_tier(meta.list_cells)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print("\n".join(cell["name"] for cell in cells))
        return

    if meta.train_cell is not None:
        cell = _cell_by_name(meta.train_cell)
        if cell is None:
            raise SystemExit(f"unknown cell: {meta.train_cell!r}")
        _train_one_cell(cell, plumbing=meta.plumbing)
        return

    # Alternate render modes (visual-inspection rasters) select via --plot-only,
    # exactly like every other figure re-render — no bespoke flags. See exp042's
    # `--plot-only compound` for the same pattern.
    if meta.plot_only and meta.plot_fig == "appendix-rasters":
        appendix_rasters()
        return
    if meta.plot_only and meta.plot_fig == "comparison-rasters":
        comparison_rasters()
        return

    # RunPod backend + kill switch are handled before the local path.
    if meta.pod_run:
        pod_run()   # runs ON a pod: train assigned cells to the volume, self-terminate
        return
    if meta.reap:
        runpod.reap_all_pods()
        return
    if meta.runpod:
        run_via_runpod(sys.argv)
        return

    # A bare --plot-only redraws figures from cached weights (no training),
    # same as --skip-training for this train-once runner.
    skip_training = meta.skip_training or meta.plot_only
    only_missing = meta.only_missing

    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id} "
          f"cells={len(CANONICAL_CELLS)}"
          + ("  [skip-training]" if skip_training else ""))
    # Wipe only this entry's figures, never the shared TRAINING_ROOT.
    prepare_run_dirs(SLUG, run_id, wipe=True, skip_training=skip_training,
                     make_artifacts=False, scale=SCALE, host="local")

    if not skip_training:
        TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
        for c in CANONICAL_CELLS:
            out = cell_dir(c["name"])
            if only_missing and (out / "metrics.json").exists():
                print(f"[skip] {c['name']} already trained")
                continue
            _train_one_cell(c, plumbing=False)

    rows = []
    for c in CANONICAL_CELLS:
        d = cell_dir(c["name"])
        _stamp_training_run_identity(c)
        m = load_metrics(d)
        re, ri = final_rates(d)
        rows.append({
            "name": c["name"], "model": c["model"], "family": c["family"],
            "training_run_id": c["training_run_id"],
            "tag": c["tag"], "seed": c["seed"],
            "acc": float(m.get("best_acc", float("nan"))),
            "best_epoch": m.get("best_epoch"), "rate_e": re, "rate_i": ri,
        })
        print(f"  {c['name']:<22} acc={rows[-1]['acc']:5.1f}%  "
              f"E={re:5.1f}Hz I={ri:5.1f}Hz")

    # One training-curve figure per family. Untrained families get no figure,
    # so the entry's <Figure> shows its "not generated yet" placeholder.
    family_status = {}
    for fam in FAMILY_ORDER:
        fcells = [c for c in CANONICAL_CELLS if c["family"] == fam]
        n_trained = sum(1 for c in fcells
                        if epoch_metrics(cell_dir(c["name"])))
        family_status[fam] = {"cells": len(fcells), "trained": n_trained}
        artifact_slug = FAMILY_ARTIFACT_SLUGS.get(fam, fam)
        out = FIGURES / f"curves__{artifact_slug}.svg"   # H10: line plots → SVG
        if n_trained:
            plot_family_curves(fam, fcells, out, run_id)
            print(f"wrote {out}")
        else:
            print(f"[not trained] {fam} — no figure (placeholder shown)")

    duration_s = time.monotonic() - t_start
    # Provenance: the git sha lives in each cell's config.json (metrics.json's
    # own config block has a null git_sha). Take the first cell that has one.
    # coerce to str for cell_dir (r["name"] widens to a union across rows).
    git_sha = next((s for s in (load_config(cell_dir(str(r["name"]))).get("git_sha")
                                for r in rows) if s), None)
    summary = {
        "notebook_run_id": run_id,
        "git_sha": git_sha,
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "standard": {"epochs": EPOCHS_STANDARD, "dt_ms": DT_MS, "t_ms": T_MS,
                     "dataset": "mnist",
                     "max_samples_canonical": CANONICAL_MAX_SAMPLES,
                     "max_samples_sweeps": SUBSET_MAX_SAMPLES},
        "training_root": training_root_provenance(TRAINING_ROOT),
        "result_checkpoint_provenance": checkpoint_provenance(
            [cell_dir(c["name"]) for c in CANONICAL_CELLS],
            RESULT_CHECKPOINT_ROLE,
        ),
        "families": FAMILY_ORDER,
        "training_run_ids": TRAINING_RUN_IDS,
        "family_status": family_status,
        "n_cells": len(CANONICAL_CELLS),
        "cells": rows,
    }
    verified_campaign = os.environ.get("EXP022_VERIFIED_CAMPAIGN")
    if verified_campaign:
        source = campaign.load_manifest(Path(verified_campaign))
        summary["campaign"] = {
            "campaign_id": source["campaign_id"],
            "manifest_sha256": source["manifest_sha256"],
            "repository_commit": source["repository"]["commit"],
            "campaign_root": source["campaign_root"],
        }
    (FIGURES / "numbers.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")
    finalize_prepared_run(SLUG, run_id)


if __name__ == "__main__":
    main()
