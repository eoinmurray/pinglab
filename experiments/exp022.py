"""Notebook runner for entry 022 — Training.

The single place the gamma-gated-sparsity collection trains its canonical
networks. Every cell is trained once here, to a shared artifact root
(src/artifacts/notebooks/training/), and the analysis notebooks load those
weights with `load_cell` (imported from this module) instead of retraining
their own. This replaces the collection's older "standalone runner, no
cross-notebook helpers" rule with a train-once / reuse-many policy (see ar016).

87 cells across five families (canonical, θ_u, τ_GABA, Δt, init) that exp025
defines and that exp024 / exp037 / exp038 each used to retrain
independently. Standard: 50 epochs, dt = 0.1 ms, T = 200 ms, and THREE seeds
(42/43/44) for every cell — including the θ_u interior, so the accuracy–rate
frontier carries error bars (it was single-seed; no longer). Canonical sees all
of MNIST, the sweeps 10%. (exp044's Δt sweep is the documented exception that
varies dt.)

Outputs a per-cell accuracy / E-rate summary plus a manifest (numbers.json)
recording exactly which cells were trained and the git sha — the contract
the analysis notebooks rely on.

Writing: writings/exp022.typ · figures + numbers.json: artifacts/data/exp022/
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import (
    runpod,  # noqa: E402
    theme,  # noqa: E402
)
from helpers.cli import parse_meta  # noqa: E402
from helpers.fmt import format_duration  # noqa: E402
from helpers.operating_point import TAU_GABA_GAMMA_MS  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "exp022"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

# ── Canonical training registry (the hub the collection reuses) ──────
# Analysis notebooks import `load_cell` / `cell_dir` from this module rather
# than retraining; this entry is the single producer of the shared cells.
# PINGLAB_TRAINING_ROOT overrides the location: RunPod pods set it to the shared
# network-volume mount (/shared/training) so a fan-out writes durable artifacts
# there instead of an ephemeral pod disk. Local runs use the default and are
# unaffected. cell_dir / load_cell read through this, so every consumer follows.
TRAINING_ROOT = runpod.training_root()
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"

# The published/calibrated bank remains at TRAINING_ROOT.  The matched rerun is
# a separate, versioned generation: incomplete matched cells can never shadow a
# calibrated cell merely because their historical names are the same.
MATCHED_GENERATION = "matched_mid_v1"
MATCHED_TRAINING_ROOT = Path(os.environ.get(
    "PINGLAB_MATCHED_TRAINING_ROOT",
    str(TRAINING_ROOT / MATCHED_GENERATION),
))

EPOCHS_STANDARD = 50           # standard depth (halved from 100 — see exp022.mdx §2)
DT_MS = 0.1
T_MS = 200.0
SEEDS_BASELINE = [42, 43, 44]
THETA_U_GRID: list[float | None] = [None, 5.0, 2.0, 1.0, 0.5, 0.2]
FR_STRENGTH_UPPER = 1e-3
TAU_AMPA_MS = 2.0          # AMPA decay — fixed across the collection (no CLI knob)
# GABA decay that puts the loop in gamma (≈ 44 Hz); the standard for every
# family except the τ_GABA sweep. Single source of truth in helpers so the
# whole collection moves together (see helpers/operating_point.py).
TAU_GABA_GAMMA = TAU_GABA_GAMMA_MS

# Canonical recipe (from exp025 — the reference training).
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
        "--w-in": "0.3",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "100",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
    "ping": {
        "__build_as": "ping",
        "--ei-strength": "1",
        "--v-grad-dampen": "1000",
        "--w-in": "1.2",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "500",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
}

# Predeclared after the local viability pilot and the full-scale seed-42 gate.
# These are the only feedforward/readout scales in the matched generation.
# Do not tune them after inspecting later seeds or sweep cells.
MATCHED_MODEL_RECIPES: dict[str, dict] = {
    model: {
        **recipe,
        "--w-in": "0.9",
        "--readout-w-out-scale": "225",
    }
    for model, recipe in MODEL_RECIPES.items()
}
MODELS = ["coba", "ping"]
MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}
MODEL_MARKERS = {"coba": "s", "ping": "D"}

# ping recipe without the fixed --ei-strength, for the init family (exp049),
# whose whole point is to vary ei-strength + recurrent trainability per cell.
MODEL_RECIPES["ping_init"] = {
    k: v for k, v in MODEL_RECIPES["ping"].items() if k != "--ei-strength"
}
MATCHED_MODEL_RECIPES["ping_init"] = {
    k: v for k, v in MATCHED_MODEL_RECIPES["ping"].items()
    if k != "--ei-strength"
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
MNIST_POOL = 70000                                   # 60k train + 10k test pool
CANONICAL_MAX_SAMPLES = MNIST_POOL                    # the canonical run sees all of it
SUBSET_MAX_SAMPLES = MNIST_POOL // 10                 # 10% of MNIST for the sweeps
MAX_SAMPLES = 100                                     # plumbing cap on every cell
EPOCHS = 2                                            # plumbing depth on every cell
BATCH_SIZE = 256                                      # fixed across every recipe


def theta_label(theta_u: float | None) -> str:
    if theta_u is None:
        return "off"
    return "tu" + f"{theta_u:g}".replace(".", "p")


def theta_display(theta_u: float | None) -> str:
    return "off" if theta_u is None else f"{theta_u:g}"


def seeds_for(theta_u: float | None) -> list[int]:
    """Every θ_u value — baseline and interior — runs all three seeds, so the
    accuracy–rate frontier carries across-seed error bars. The interior used to
    be single-seed (a limitation ar009 §2.3 disclosed); this removes it."""
    return list(SEEDS_BASELINE)


def cell_name(model: str, theta_u: float | None, seed: int) -> str:
    """θ_u cell name — always seed-suffixed now the interior is 3-seed too (was
    `{model}__{tu}` with no seed for the single-seed interior). Consumers that
    read these cells must iterate seeds_for() rather than assume a single seed."""
    if theta_u is None:
        return f"{model}__off__seed{seed}"
    return f"{model}__{theta_label(theta_u)}__seed{seed}"


def _label(x: float) -> str:
    return f"{x:g}".replace(".", "p")


# ── Cell registry: one spec per trained cell, tagged by family ───────
# Each spec carries the family-specific bits (dt override + extra flags) so a
# single build_train_args reproduces every family. Names match the existing
# per-notebook artifacts, so folding the already-trained cells in is a move,
# not a retrain.

def _theta_u_cells() -> list[dict]:
    cells = []
    for m in MODELS:
        for tu in THETA_U_GRID:
            extra = ([] if tu is None else
                     ["--fr-reg-upper-theta", str(tu),
                      "--fr-reg-upper-strength", str(FR_STRENGTH_UPPER)])
            for s in seeds_for(tu):
                cells.append({
                    "name": cell_name(m, tu, s), "model": m, "family": "theta_u",
                    "tag": theta_display(tu), "seed": s, "dt_ms": DT_MS,
                    "tau_gaba": TAU_GABA_GAMMA, "theta_u": tu, "extra": extra,
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
    # The canonical reference: θ_u = off, trained on ALL of MNIST (not the
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


CANONICAL_CELLS = (_canonical_cells() + _theta_u_cells() + _tau_gaba_cells()
                   + _dt_cells() + _init_cells())

# Same five scientific families and cell names as the calibrated bank, but a
# distinct root and a frozen shared-scale recipe.  Consumers do not switch to
# this generation until the complete rerun and downstream audit pass.
MATCHED_CELLS = [
    {**cell, "generation": MATCHED_GENERATION, "recipe_family": "matched_mid"}
    for cell in CANONICAL_CELLS
]


def matched_cell_dir(name: str) -> Path:
    """One cell in the isolated matched generation."""
    return MATCHED_TRAINING_ROOT / name


def matched_canary_cells() -> list[dict]:
    """Stage-2-as-Stage-3-canary: theta-off seeds 43/44 for both models."""
    return [
        cell for cell in MATCHED_CELLS
        if cell["family"] == "theta_u"
        and cell.get("theta_u") is None
        and cell["seed"] in (43, 44)
    ]

# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
SCALE = {
    "dataset": "mnist",
    # Sweep scale (61/67 cells); the 6 canonical cells override to all of MNIST.
    "max_samples": SUBSET_MAX_SAMPLES,
    "epochs": EPOCHS_STANDARD,
    "t_ms": T_MS,
    "dt_ms": DT_MS,
    "batch_size": BATCH_SIZE,
    "seeds": len(SEEDS_BASELINE),
    "cells": len(CANONICAL_CELLS),
    "grid": "2 architectures × 5 families",
}


def cell_dir(name: str) -> Path:
    """Shared per-cell artifact directory."""
    return TRAINING_ROOT / name


def load_cell(name: str) -> Path:
    """Return a trained cell's directory, or fail loudly if this notebook has
    not been run. Analysis notebooks call this instead of training."""
    d = cell_dir(name)
    if not (d / "weights.pth").exists():
        raise SystemExit(
            f"missing trained cell '{name}' at {d.relative_to(REPO)}; "
            "run exp022 (Training) first to produce the shared cells."
        )
    return d


def build_train_args(spec: dict, out_dir: Path,
                     max_samples: int, epochs: int,
                     recipes: dict[str, dict] | None = None) -> list[str]:
    """CLI `train` args for one registry cell, across all families."""
    recipe = (recipes or MODEL_RECIPES)[spec["model"]]
    ms = spec.get("max_samples") or max_samples   # canonical cells override
    args = [
        "train",
        "--model", recipe["__build_as"],
        "--dataset", "mnist",
        "--max-samples", str(ms),
        "--epochs", str(epochs),
        "--t-ms", str(T_MS),
        "--dt", str(spec["dt_ms"]),
        "--tau-gaba", str(spec["tau_gaba"]),
        "--seed", str(spec["seed"]),
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
    return args


def build_matched_train_args(spec: dict, out_dir: Path,
                             max_samples: int, epochs: int) -> list[str]:
    """Train args for the frozen matched_mid recipe generation."""
    return build_train_args(
        spec, out_dir, max_samples, epochs, recipes=MATCHED_MODEL_RECIPES,
    )


def _arg_value(args: list[str], flag: str, default=None):
    if flag not in args:
        return default
    return args[args.index(flag) + 1]


def matched_expected_config(spec: dict, *, plumbing: bool = False) -> dict:
    """The config fields that make a cached matched cell reproducible.

    This is deliberately stricter than checking for metrics.json: a calibrated
    cell or a stale plumbing run must never satisfy the matched completion
    marker just because it has the right basename.
    """
    if plumbing:
        sample_count, epochs = MAX_SAMPLES, EPOCHS
        build_spec = {k: v for k, v in spec.items() if k != "max_samples"}
    else:
        sample_count, epochs = cell_samples_epochs(spec)
        build_spec = spec
    args = build_matched_train_args(
        build_spec, Path("/matched-config-check"), sample_count, epochs,
    )
    mean_w_in = float(_arg_value(args, "--w-in"))
    theta = _arg_value(args, "--fr-reg-upper-theta")
    theta_strength = _arg_value(args, "--fr-reg-upper-strength")
    return {
        "model": _arg_value(args, "--model"),
        "max_samples": int(_arg_value(args, "--max-samples")),
        "epochs": int(_arg_value(args, "--epochs")),
        "seed": int(_arg_value(args, "--seed")),
        "dt": float(_arg_value(args, "--dt")),
        "t_ms": float(_arg_value(args, "--t-ms")),
        "tau_gaba_ms": float(_arg_value(args, "--tau-gaba")),
        "batch_size": int(_arg_value(args, "--batch-size")),
        "lr": float(_arg_value(args, "--lr")),
        "ei_strength": float(_arg_value(args, "--ei-strength")),
        "v_grad_dampen": float(_arg_value(args, "--v-grad-dampen")),
        "w_in": [mean_w_in, mean_w_in * 0.1],
        "w_in_sparsity": float(_arg_value(args, "--w-in-sparsity")),
        "readout_mode": _arg_value(args, "--readout"),
        "readout_w_out_scale": float(_arg_value(args, "--readout-w-out-scale")),
        "fr_reg_upper_theta": float(theta) if theta is not None else 0.0,
        "fr_reg_upper_strength": (
            float(theta_strength) if theta_strength is not None else 0.0
        ),
        "trainable_w_ei": "--trainable-w-ei" in args,
        "trainable_w_ie": "--trainable-w-ie" in args,
    }


def _config_value_matches(got, want) -> bool:
    if isinstance(want, float):
        try:
            return abs(float(got) - want) <= 1e-9 * max(1.0, abs(want))
        except (TypeError, ValueError):
            return False
    if isinstance(want, list):
        return (isinstance(got, list) and len(got) == len(want)
                and all(_config_value_matches(g, w) for g, w in zip(got, want)))
    return got == want


def matched_cell_validation(spec: dict, *, plumbing: bool = False,
                            root: Path | None = None,
                            directory: Path | None = None) -> tuple[bool, list[str]]:
    """Return whether a matched cell is complete plus every failed check."""
    d = directory or ((root or MATCHED_TRAINING_ROOT) / spec["name"])
    errors = []
    for filename in ("config.json", "metrics.json", "weights.pth"):
        if not (d / filename).exists():
            errors.append(f"missing {filename}")
    if errors:
        return False, errors
    try:
        cfg = json.loads((d / "config.json").read_text())
        metrics = json.loads((d / "metrics.json").read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return False, [f"invalid JSON: {exc}"]
    for key, want in matched_expected_config(spec, plumbing=plumbing).items():
        got = cfg.get(key)
        if not _config_value_matches(got, want):
            errors.append(f"{key}: got {got!r}, want {want!r}")
    if len(metrics.get("epochs", [])) != matched_expected_config(
        spec, plumbing=plumbing,
    )["epochs"]:
        errors.append("metrics epoch count does not match config")
    return not errors, errors


def matched_cell_is_done(spec: dict, *, plumbing: bool = False) -> bool:
    return matched_cell_validation(spec, plumbing=plumbing)[0]


def adopt_matched_stage1(*, source_root: Path | None = None) -> list[str]:
    """Copy the two validated seed-42 gate cells into matched_mid_v1.

    The gate used the final recipe and full sweep scale, so retraining it would
    waste compute.  Validation happens against the registry before either copy.
    Existing valid destinations are left untouched.
    """
    source = source_root or (TRAINING_ROOT / "matched_mid_stage1")
    adopted = []
    for model in MODELS:
        spec = next(
            c for c in MATCHED_CELLS
            if c["family"] == "theta_u" and c.get("theta_u") is None
            and c["model"] == model and c["seed"] == 42
        )
        source_dir = source / f"{model}__seed42"
        ok, errors = matched_cell_validation(spec, directory=source_dir)
        if not ok:
            raise SystemExit(
                f"cannot adopt Stage 1 {model}: " + "; ".join(errors)
            )
        dest = matched_cell_dir(spec["name"])
        if matched_cell_is_done(spec):
            print(f"[adopt skip] {spec['name']} already valid")
            continue
        if dest.exists():
            raise SystemExit(
                f"refusing to overwrite invalid matched destination {dest}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Multiple canary pods may reach adoption together. Copy to a private
        # sibling then publish with one atomic rename; the loser verifies the
        # winner instead of exposing a half-copied directory as "complete".
        token = os.environ.get("RUNPOD_POD_ID", str(os.getpid()))
        staged = dest.with_name(f".{dest.name}.adopting-{token}")
        shutil.rmtree(staged, ignore_errors=True)
        shutil.copytree(source_dir, staged)
        try:
            staged.rename(dest)
        except OSError:
            shutil.rmtree(staged, ignore_errors=True)
            if not matched_cell_is_done(spec):
                raise
        adopted.append(spec["name"])
        print(f"[adopt] {source_dir} → {dest}")
    return adopted


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


def final_rates(d: Path) -> tuple[float, float]:
    """Last-epoch E / I rate (Hz) from metrics.jsonl, if present."""
    p = d / "metrics.jsonl"
    if not p.exists():
        return float("nan"), float("nan")
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    if not lines:
        return float("nan"), float("nan")
    row = json.loads(lines[-1])
    return (float(row.get("test_rate_e", row.get("rate_e", float("nan")))),
            float(row.get("test_rate_i", row.get("rate_i", float("nan")))))


FAMILY_COLORS = {
    "canonical": theme.GREY_DARK,
    "theta_u": theme.INK_BLACK,
    "theta_u_3seed": theme.INK_BLACK,
    "tau_gaba": theme.DEEP_RED,
    "dt": theme.ELECTRIC_CYAN,
    "init": theme.AMBER,
}


def training_curve(d: Path) -> tuple[list[int], list[float]]:
    """Per-epoch (epoch, test accuracy) from a cell's metrics.jsonl."""
    p = d / "metrics.jsonl"
    if not p.exists():
        return [], []
    eps, accs = [], []
    for ln in p.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if "ep" in r and "acc" in r:
            eps.append(int(r["ep"]))
            accs.append(float(r["acc"]))
    return eps, accs


FAMILY_ORDER = ["canonical", "theta_u", "theta_u_3seed", "tau_gaba", "dt", "init"]
FAMILY_LABELS = {
    "canonical": "Canonical reference",
    "theta_u": "θ_u spike-budget sweep",
    "theta_u_3seed": "θ_u spike-budget sweep (3-seed, ar010 item 3)",
    "tau_gaba": "τ_GABA ladder",
    "dt": "Δt sweep",
    "init": "Init variants",
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
    # in families that train both (θ_u, canonical).
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

    Existence alone is not enough: TRAINING_ROOT is littered with cells from
    earlier, coarser standards (e.g. 2000 samples / 100 epochs), and a bare
    exists() check — what --only-missing does — would skip those and silently
    ship a mixed-scale, invalid dataset. Comparing the baked config makes the
    marker honest: a stale cell reads as pending and gets retrained.
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
        # build_train_args re-applies a canonical cell's own max_samples (70000),
        # which would defeat the tiny plumbing scale. Strip it so the plumbing
        # ms=100 takes — and so runpod_is_done agrees with what was trained.
        spec = {k: v for k, v in cell.items() if k != "max_samples"}
    args = build_train_args(spec, cell_dir(cell["name"]), ms, ep)
    print(f"[train-cell] {cell['name']} (n={ms}, {ep} ep) → {cell_dir(cell['name'])}")
    subprocess.run([sys.executable, str(SNN_TOOL), *args], cwd=REPO, check=True)


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
    directly comparable across configs. Writes to artifacts/data/exp022/rasters/."""
    import shutil

    cells = [c for c in CANONICAL_CELLS if c["seed"] == 42]
    rdir = FIGURES / "rasters"
    # Snapshots are pure throwaway (only needed to plot each PNG). Keep them under
    # the temp/experiments/ namespace and wipe the whole scratch tree at the end,
    # so temp stays minimal — the PNGs in artifacts/ are the durable output.
    scratch_root = REPO / "temp" / "experiments" / f"{SLUG}_appendix_scratch"
    print(f"appendix: {len(cells)} rasters (seed 42) → {rdir.relative_to(REPO)}")
    try:
        for c in cells:
            d = cell_dir(c["name"])
            if not (d / "weights.pth").exists():
                print(f"[skip] {c['name']} — no weights")
                continue
            scratch = scratch_root / c["name"]
            subprocess.run(
                [sys.executable, str(SNN_TOOL), "sim", "--infer",
                 "--load-config", str(d / "config.json"),
                 "--load-weights", str(d / "weights.pth"),
                 "--digit", "0", "--sample", "0",
                 "--out-dir", str(scratch), "--wipe-dir"],
                cwd=REPO, check=True, capture_output=True)
            _plot_snapshot_raster(scratch / "snapshot.npz", rdir / f"{c['name']}.png")
            print(f"  {c['name']}.png")
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


# ── Results: the data-fraction contrast (100% vs 10% MNIST) ──────────
# The canonical cells train on all 70k images; every sweep (including the
# no-budget `off` cell) trains on 10%. Weights differ ONLY in that fraction, so
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
    seed-42 weights (no retraining) → artifacts/data/exp022/comparison__data_fraction.png."""
    import shutil

    # (row label, 100%-MNIST cell, 10%-MNIST cell) — seed 42 as the representative.
    grid = [
        ("COBA", "coba__canonical__seed42", "coba__off__seed42"),
        ("PING", "ping__canonical__seed42", "ping__off__seed42"),
    ]
    col_titles = ["100% MNIST (canonical)", "10% MNIST (off)"]
    scratch_root = REPO / "temp" / "experiments" / f"{SLUG}_comparison_scratch"
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.06),   # H11–H12: 16:9
                             gridspec_kw={"hspace": 0.30, "wspace": 0.14})
    print(f"comparison: 4 rasters (seed 42) → "
          f"{(FIGURES / 'comparison__data_fraction.png').relative_to(REPO)}")
    try:
        for r, (label, full_cell, sub_cell) in enumerate(grid):
            for cc, name in enumerate((full_cell, sub_cell)):
                ax = axes[r][cc]
                d = cell_dir(name)
                if not (d / "weights.pth").exists():
                    ax.text(0.5, 0.5, f"{name}\n(no weights)", ha="center",
                            va="center", transform=ax.transAxes,
                            fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED)
                    continue
                scratch = scratch_root / name
                subprocess.run(
                    [sys.executable, str(SNN_TOOL), "sim", "--infer",
                     "--load-config", str(d / "config.json"),
                     "--load-weights", str(d / "weights.pth"),
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
        print(f"  wrote {out.relative_to(REPO)}")
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


def main() -> None:
    meta = parse_meta(sys.argv, allow_dispatch=True)

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
            ms, ep = cell_samples_epochs(c)
            print(f"[train] {c['name']} (n={ms}, {ep} ep) → {out.relative_to(REPO)}")
            subprocess.run(
                [sys.executable, str(SNN_TOOL), *build_train_args(c, out, ms, ep)],
                cwd=REPO, check=True,
            )

    rows = []
    for c in CANONICAL_CELLS:
        d = cell_dir(c["name"])
        m = load_metrics(d)
        re, ri = final_rates(d)
        rows.append({
            "name": c["name"], "model": c["model"], "family": c["family"],
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
                        if (cell_dir(c["name"]) / "metrics.jsonl").exists())
        family_status[fam] = {"cells": len(fcells), "trained": n_trained}
        out = FIGURES / f"curves__{fam}.svg"   # H10: line plots → SVG
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
        "training_root": str(TRAINING_ROOT.relative_to(REPO)),
        "families": FAMILY_ORDER,
        "family_status": family_status,
        "n_cells": len(CANONICAL_CELLS),
        "cells": rows,
    }
    (FIGURES / "numbers.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")


if __name__ == "__main__":
    main()
