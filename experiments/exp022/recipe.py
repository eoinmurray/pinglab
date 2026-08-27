"""Exp022 scientific registry and read-only shared bank interface."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

from helpers import theme
from helpers.checkpoints import epoch_metrics
from helpers.operating_point import TAU_GABA_GAMMA_MS
from helpers.paths import artifacts_and_figures

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
SNN_TOOL = REPO / "tools" / "snnsim" / "tool.py"

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
            "select an explicit retained compute bank or run exp022/compute.py."
        )
    return d


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
