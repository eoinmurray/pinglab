"""Notebook runner for entry 042 — rhythm vs mean-inhibition control.

Pure inference on the trained exp025 PING baseline. Three conditions
applied to the I-population spike stream at evaluation time, all
preserving per-cell mean I rate within trial:

1. baseline           — no perturbation.
2. phase_shuffled_i   — per-trial permutation of the time axis of the
                        baseline I-spike tensor (single permutation per
                        trial, applied to all I-cells together). Mean
                        per-cell I rate identical; phase relationship to
                        the gamma cycle destroyed.
3. poisson_matched_i  — replace I-spikes with a Bernoulli draw matched
                        to each (trial, cell)'s baseline spike count.

If the E rate stays clamped without the rhythm, ar008's thesis
collapses to "inhibition lowers rates." If it shoots up toward COBA's
operating point, gamma is specifically what is doing the forbidding.

Writing: writings/exp042.typ · figures + numbers.json: artifacts/data/exp042/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:  # torch is imported lazily inside the functions at runtime
    import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.cli import replot_target  # noqa: E402
from helpers.datasets import load_mnist_split  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.fmt import format_duration  # noqa: E402
from helpers.modal import parse_modal_gpu  # noqa: E402
from helpers.operating_point import (  # noqa: E402
    F_GAMMA_HZ,
    MODELS_DEFAULT_TAU_GABA_MS,
    TAU_GABA_GAMMA_MS,
)
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "exp042"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"
EVAL_SEED = 20260415  # mirror cli.encoders.EVAL_SEED (kept in sync by hand)

# θ_u = off PING baseline now lives in the shared training root (exp022
# train-once / reuse-many). Three seeds available; exp042 runs against all
# three for error bars.
NB035_ARTIFACTS = REPO / "temp" / "experiments" / "exp022"
SEEDS: tuple[int, ...] = (42, 43, 44)

CONDITIONS: tuple[str, ...] = ("baseline", "phase_shuffled_i", "poisson_matched_i")

# Jitter sweep — Gaussian timing jitter on each I-spike. σ in ms.
# 0 = baseline; well above the trained network's gamma period
# (≈ T_γ at the canonical τ_GABA) the rate should approach the phase-shuffle
# release level. Predicted transition is at σ ≈ 1 / f_γ.
JITTER_SIGMAS_MS: tuple[float, ...] = (
    0.0, 1.0, 3.0, 7.0, 14.0, 21.0, 28.0, 42.0, 60.0, 100.0,
)
# Measured PING f_γ at the canonical τ_GABA (single source of truth).
F_GAMMA_REFERENCE_HZ: float = F_GAMMA_HZ

# Per-I-cell (per-spike) jitter sweep — tests whether within-burst
# synchrony matters, by drawing an independent Gaussian offset for each
# I-spike. Predicted transition timescale is τ_GABA (synaptic decay),
# where the smeared g_i profile starts looking continuous.
CELL_JITTER_SIGMAS_MS: tuple[float, ...] = (
    0.0, 0.5, 1.0, 2.0, 5.0, 9.0, 14.0, 21.0, 50.0,
)
CELL_JITTER_RASTER_SIGMAS_MS: tuple[float, ...] = (0.0, 1.0, 5.0, 9.0, 50.0)

# Pareto sweep — does rhythmic inhibition sit on the (E rate, accuracy)
# frontier among I-stream perturbations at varied mean rate? Two knobs:
#   α ∈ [0, 1]  — per-timestep mixing fraction between baseline and Poisson
#                  (0 = pure rhythm, 1 = pure rate-matched Poisson)
#   k ∈ (0, ∞)  — independent scaling of the mean I rate
# At (α=0, k=1) the condition reduces to baseline; (α=1, k=1) is the
# existing poisson_matched_i condition.
MIX_ALPHA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
MIX_K_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)

# Cross-τ_GABA jitter sweep (formerly nb045) — loads each of exp041's 18
# trained cells and re-runs the cycle-coherent jitter sweep at each
# cell's own 1/f_γ. Outputs xtau_raw_sweeps, xtau_dimensional_collapse,
# xtau_inflection_vs_period.
NB041_ARTIFACTS = REPO / "temp" / "experiments" / "exp022"
NB041_NUMBERS = (
    REPO / "artifacts" / "data" / "exp041"
    / "numbers.json"
)
XTAU_TAU_GABAS_MS: tuple[float, ...] = (4.5, 6.0, 9.0, 12.0, 18.0, 27.0)
XTAU_SEEDS: tuple[int, ...] = (42, 43, 44)
XTAU_SIGMAS_MS: tuple[float, ...] = (
    0.0, 1.0, 3.0, 7.0, 14.0, 21.0, 28.0, 42.0, 60.0, 100.0,
)
# Inverse — annotated as the predicted inflection point on the sweep plot.
# 1 / 36 Hz ≈ 27.8 ms
CONDITION_LABELS = {
    "baseline": "baseline PING",
    "phase_shuffled_i": "phase-shuffled I",
    "poisson_matched_i": "rate-matched Poisson I",
}

# Raster panel: one trial per condition, MNIST digit 0 sample 0 — same
# convention as exp025/exp037 so the panels read against existing figures.
RASTER_SAMPLE_IDX: int = 0
RASTER_N_E_PLOT: int = 200
RASTER_N_I_PLOT: int = 64

# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
# exp042 is inference-only against the trained exp025 PING baseline, so the
# dataset / max_samples / t_ms / dt_ms are inherited from each cell's own
# config.json at run time. What this runner declares is the evaluation grid:
# how many seeds and cells it sweeps, and the perturbation grids.
SCALE = {
    "dataset": "mnist",
    "seeds": len(SEEDS),
    "cells": len(CONDITIONS),
    "grid": (
        f"jitter σ ×{len(JITTER_SIGMAS_MS)}, "
        f"cell-jitter σ ×{len(CELL_JITTER_SIGMAS_MS)}, "
        f"α×k = {len(MIX_ALPHA_GRID)}×{len(MIX_K_GRID)}, "
        f"xtau τ×σ = {len(XTAU_TAU_GABAS_MS)}×{len(XTAU_SIGMAS_MS)}"
    ),
}


# ─── trained-network loading (mirrors exp037 helper) ─────────────────


# ─── CLI-backed baseline + override (net execution runs in the CLI) ──────


def _load_eval(train_dir: Path):
    """Config + held-out MNIST test split for a trained cell (no net)."""
    cfg = json.loads((train_dir / "config.json").read_text())
    _, X_te, _, y_te = load_mnist_split(max_samples=int(cfg["max_samples"]))
    return cfg, X_te, y_te


_BASE_CACHE: dict = {}


def _run_baseline(train_dir: Path, tau_gaba=None):
    """Baseline pass via `sim --infer --outputs rasters`; return (metrics, rasters).
    Cached per (cell, τ_GABA) — the baseline I-stream is condition-independent."""
    key = f"{train_dir}|{tau_gaba}"
    if key not in _BASE_CACHE:
        out_dir = (ARTIFACTS / "baseline" / train_dir.name).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "uv", "run", "python", str(SNN_TOOL), "sim", "--infer",
            "--load-config", str((train_dir / "config.json").resolve()),
            "--load-weights", str((train_dir / "weights.pth").resolve()),
            "--outputs", "rasters", "--out-dir", str(out_dir),
        ]
        if tau_gaba is not None:
            cmd += ["--tau-gaba", str(tau_gaba)]
        subprocess.run(cmd, cwd=REPO, check=True)
        m = json.loads((out_dir / "metrics.json").read_text())
        R = dict(np.load(out_dir / "rasters.npz"))
        _BASE_CACHE[key] = (m, R)
    return _BASE_CACHE[key]


def _build_override_file(R, condition, gen, dt_ms, out_path, cycle_period_ms=None):
    """Build a sparse I-override NPZ from baseline rasters R by applying the pure
    _build_override transform per trial (per-trial independent). The transform stays
    in the notebook; the CLI only injects the result."""
    import torch
    T, n_i, n_tr = int(R["T"]), int(R["n_i"]), int(R["n_trials"])
    tr = R["i_trial"]
    order = np.argsort(tr, kind="stable")
    tr, tt, tc = tr[order], R["i_t"][order], R["i_cell"][order]
    bounds = np.searchsorted(tr, np.arange(n_tr + 1))
    kwargs = {"cycle_period_ms": cycle_period_ms} if cycle_period_ms else {}
    out_tr, out_t, out_c = [], [], []
    for b in range(n_tr):
        lo, hi = bounds[b], bounds[b + 1]
        s_i = np.zeros((T, 1, n_i), dtype=np.float32)
        s_i[tt[lo:hi], 0, tc[lo:hi]] = 1.0
        ov = _build_override(torch.from_numpy(s_i), condition, gen, dt_ms=dt_ms, **kwargs)
        ov = ov.detach().cpu().numpy()[:, 0, :]  # (T, n_i)
        ti, ci = ov.nonzero()
        out_t.append(ti.astype("int32"))
        out_c.append(ci.astype("int32"))
        out_tr.append(np.full(ti.size, b, dtype="int32"))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, "int32")  # noqa: E731
    np.savez(
        out_path, n_trials=np.int32(n_tr), T=np.int32(T), n_i=np.int32(n_i),
        i_trial=cat(out_tr), i_t=cat(out_t), i_cell=cat(out_c),
    )


def _run_with_override(train_dir: Path, override_path: Path, tau_gaba=None) -> dict:
    """Pass B via `sim --infer --i-override-file`; return metrics."""
    out_dir = (ARTIFACTS / "ovrun" / f"{train_dir.name}__{override_path.stem}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "python", str(SNN_TOOL), "sim", "--infer",
        "--load-config", str((train_dir / "config.json").resolve()),
        "--load-weights", str((train_dir / "weights.pth").resolve()),
        "--i-override-file", str(override_path), "--out-dir", str(out_dir),
    ]
    if tau_gaba is not None:
        cmd += ["--tau-gaba", str(tau_gaba)]
    subprocess.run(cmd, cwd=REPO, check=True)
    return json.loads((out_dir / "metrics.json").read_text())


def _pack_metrics(m: dict, condition: str) -> dict:
    """Shape a CLI metrics.json into exp042's per-condition row."""
    rates = m.get("rates_hz", {})
    hid = max((k for k in rates if k.startswith("hid")), default=None)
    inh = max((k for k in rates if k.startswith("inh")), default=None)
    return {
        "condition": condition,
        "acc": float(m["best_acc"]),
        "e_rate_hz": float(rates.get(hid, 0.0)) if hid else 0.0,
        "i_rate_hz": float(rates.get(inh, 0.0)) if inh else 0.0,
        "n_total": int(m.get("n_total", 0)),
    }


def _build_override(
    s_i_base: "torch.Tensor", condition: str, generator, dt_ms: float = 0.1,
    cycle_period_ms: float | None = None,
) -> "torch.Tensor":
    """Construct the I-spike override tensor for one batch.

    s_i_base: (T, B, N_I) baseline recorded I-spikes.
    Returns (T, B, N_I) override tensor preserving per-(trial, cell)
    spike counts in expectation.

    Conditions:
      - phase_shuffled_i: permute time axis per trial (all I cells share permutation)
      - poisson_matched_i: per-(trial, cell) Bernoulli at matched mean rate
      - jitter_sigma_{X}: cycle-coherent Gaussian jitter with σ = X ms.
        Uses F_GAMMA_REFERENCE_HZ unless `cycle_period_ms` is provided
        (the nb045 cross-cell experiment passes the cell's own 1/f_γ).
      - cell_jitter_sigma_{X}: per-spike Gaussian jitter with σ = X ms
        (destroys within-burst synchrony; preserves burst placement on average).
    """
    import torch

    if s_i_base.ndim == 2:  # (T, N_I) when batch size is 1
        s_i_base = s_i_base.unsqueeze(1)
    T, B, N_I = s_i_base.shape
    if condition == "phase_shuffled_i":
        out = torch.empty_like(s_i_base)
        for b in range(B):
            perm = torch.randperm(T, generator=generator)
            out[:, b, :] = s_i_base[perm, b, :]
    elif condition == "poisson_matched_i":
        counts = s_i_base.sum(dim=0)
        p = (counts / float(T)).clamp(0.0, 1.0).unsqueeze(0).expand(T, B, N_I)
        out = (torch.rand(T, B, N_I, generator=generator) < p).to(s_i_base.dtype)
    elif condition.startswith("jitter_sigma_"):
        sigma_ms = float(condition.split("_")[-1])
        kwargs = {"cycle_period_ms": cycle_period_ms} if cycle_period_ms else {}
        out = _jitter_i_stream(s_i_base, sigma_ms, dt_ms, generator, **kwargs)
    elif condition.startswith("cell_jitter_sigma_"):
        sigma_ms = float(condition.split("_")[-1])
        out = _cell_jitter_i_stream(s_i_base, sigma_ms, dt_ms, generator)
    elif condition.startswith("alpha_mix_"):
        # alpha_mix_a{α}_k{k} — per-timestep mix of baseline rhythm and
        # Poisson at scaled mean rate. See _alpha_mix_i_stream docstring.
        parts = condition.split("_")
        alpha = float(parts[2][1:])  # strip leading 'a'
        k = float(parts[3][1:])      # strip leading 'k'
        out = _alpha_mix_i_stream(s_i_base, alpha, k, generator)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return out


def _jitter_i_stream(
    s_i_base: "torch.Tensor", sigma_ms: float, dt_ms: float, generator,
    cycle_period_ms: float | None = None,
) -> "torch.Tensor":
    """Cycle-coherent jitter on the I-spike stream.

    Bins time into blocks of one gamma cycle (1 / F_GAMMA_REFERENCE_HZ
    ≈ 28 ms at the trained operating point, unless overridden by
    `cycle_period_ms`), draws one Gaussian offset Δ ~ 𝒩(0, σ²) per
    (trial, cycle), and shifts every I-spike in that block by Δ.
    Within-burst cross-cell synchrony is preserved exactly; what's
    perturbed is the *placement* of each burst relative to where the
    baseline cycle put it.

    The diagnostic prediction: rate release should be small when
    σ ≪ 1/f_γ (bursts barely move from their phase-locked slots) and
    large when σ ≳ 1/f_γ (bursts can land anywhere within the cycle,
    losing phase relation to E).

    σ in milliseconds; the conversion to timesteps uses dt_ms.
    Pass `cycle_period_ms` to override the default cycle period — needed
    for cross-cell experiments where each cell has its own measured f_γ
    (see nb045).
    """
    import torch

    T, B, N_I = s_i_base.shape
    if sigma_ms <= 0.0:
        return s_i_base.clone()

    if cycle_period_ms is None:
        cycle_period_ms = 1000.0 / F_GAMMA_REFERENCE_HZ
    cycle_period_steps = max(1, int(round(cycle_period_ms / dt_ms)))
    n_cycles = (T + cycle_period_steps - 1) // cycle_period_steps
    sigma_steps = sigma_ms / dt_ms

    # Per-(trial, cycle) Gaussian offset, in timestep units, rounded.
    offsets = torch.randn(B, n_cycles, generator=generator) * sigma_steps
    offsets_int = offsets.round().long()

    spike_positions = s_i_base.nonzero(as_tuple=False)  # (n_spikes, 3): (t, b, n)
    if spike_positions.numel() == 0:
        return s_i_base.clone()
    t_orig = spike_positions[:, 0]
    b_idx = spike_positions[:, 1]
    n_idx = spike_positions[:, 2]
    cycle_idx = (t_orig // cycle_period_steps).clamp(0, n_cycles - 1)
    # Look up the per-(b, cycle) offset for each spike, add, clamp.
    jitter = offsets_int[b_idx, cycle_idx]
    new_t = (t_orig + jitter).clamp(0, T - 1)
    out = torch.zeros_like(s_i_base)
    out.index_put_(
        (new_t, b_idx, n_idx),
        torch.ones(spike_positions.shape[0], dtype=s_i_base.dtype),
        accumulate=False,
    )
    return out


def _cell_jitter_i_stream(
    s_i_base: "torch.Tensor", sigma_ms: float, dt_ms: float, generator,
) -> "torch.Tensor":
    """Per-spike (per-I-cell) Gaussian jitter on the I-spike stream.

    Each spike gets its own independent Gaussian offset Δ ~ 𝒩(0, σ²).
    Within-burst cross-cell synchrony is destroyed — different I-cells
    that fired at the same timestep in baseline land at different times
    in the override. Burst placement is preserved on average (each
    spike's offset has zero mean), but the burst itself smears across
    a window of width ≈ σ.

    Complements `_jitter_i_stream` (cycle-coherent): the cycle-coherent
    sweep tests whether the *placement* of each burst relative to the
    gamma cycle matters; per-cell jitter tests whether the *sharpness*
    of each burst matters.

    Mean per-cell I rate is preserved exactly (every spike survives —
    we only move it in time and clamp to the valid range).
    """
    import torch

    T, B, N_I = s_i_base.shape
    if sigma_ms <= 0.0:
        return s_i_base.clone()

    sigma_steps = sigma_ms / dt_ms
    spike_positions = s_i_base.nonzero(as_tuple=False)  # (n_spikes, 3): (t, b, n)
    if spike_positions.numel() == 0:
        return s_i_base.clone()
    t_orig = spike_positions[:, 0]
    b_idx = spike_positions[:, 1]
    n_idx = spike_positions[:, 2]
    # Independent Gaussian offset per spike, rounded to timestep grid.
    n_spikes = spike_positions.shape[0]
    offsets = (
        torch.randn(n_spikes, generator=generator) * sigma_steps
    ).round().long()
    new_t = (t_orig + offsets).clamp(0, T - 1)
    out = torch.zeros_like(s_i_base)
    out.index_put_(
        (new_t, b_idx, n_idx),
        torch.ones(n_spikes, dtype=s_i_base.dtype),
        accumulate=False,
    )
    return out


def _alpha_mix_i_stream(
    s_i_base: "torch.Tensor", alpha: float, k: float, generator,
) -> "torch.Tensor":
    """Interpolate between baseline rhythm and rate-matched Poisson.

    α ∈ [0, 1] controls the per-timestep mixing fraction:
      - α = 0 reproduces baseline (passes s_i_base through, possibly rate-scaled by k)
      - α = 1 reproduces the existing poisson_matched_i condition (at k=1)
      - intermediate α swaps a random α fraction of timesteps for Poisson draws.

    k ∈ (0, ∞) is an independent mean-rate scaling. For each cell, the
    Poisson component draws at rate (k × baseline_mean_rate). The
    baseline component is itself Bernoulli-thinned at rate min(1, k)
    when k < 1, or unchanged when k ≥ 1 (with the rate top-up coming
    through the Poisson channel when k > 1).
    """
    import torch

    T, B, N_I = s_i_base.shape
    counts = s_i_base.sum(dim=0).float()  # (B, N_I)
    p_mean = (counts / float(T)).clamp(0.0, 1.0)
    p_poisson = (k * p_mean).clamp(0.0, 1.0).unsqueeze(0).expand(T, B, N_I)
    poisson_draw = (
        torch.rand(T, B, N_I, generator=generator) < p_poisson
    ).to(s_i_base.dtype)

    if k < 1.0:
        thin_mask = (
            torch.rand(T, B, N_I, generator=generator) < k
        ).to(s_i_base.dtype)
        baseline_scaled = s_i_base * thin_mask
    else:
        baseline_scaled = s_i_base

    use_poisson = torch.rand(T, B, N_I, generator=generator) < alpha
    out = torch.where(use_poisson, poisson_draw, baseline_scaled)
    return out


# ─── per-condition evaluation ───────────────────────────────────────


def evaluate_condition(
    train_dir: Path, condition: str, seed_offset: int = 0, cycle_period_ms=None
) -> dict:
    """Accuracy + E/I rate for one I-override condition, via the CLI.

    Two passes: baseline (`--outputs rasters`, cached) supplies the I-stream, the
    notebook builds the override with its pure transforms, then `--i-override-file`
    replays it. baseline condition just reads the baseline metrics.
    """
    import torch
    cfg, _, _ = _load_eval(train_dir)
    m0, R = _run_baseline(train_dir)
    if condition == "baseline":
        return _pack_metrics(m0, condition)
    gen = torch.Generator().manual_seed(EVAL_SEED + 17 + seed_offset)
    ov_path = (
        ARTIFACTS / "override" / f"{train_dir.name}_{condition}_{seed_offset}.npz"
    ).resolve()
    ov_path.parent.mkdir(parents=True, exist_ok=True)
    _build_override_file(R, condition, gen, float(cfg["dt"]), ov_path, cycle_period_ms)
    return _pack_metrics(_run_with_override(train_dir, ov_path), condition)


def _snapshot(train_dir: Path, sample_idx: int, name: str, i_override=None):
    """Single-trial snapshot via `sim --infer --sample-index N` (optional
    --i-override-file); return the loaded snapshot.npz dict."""
    out_dir = (ARTIFACTS / "condraster" / f"{train_dir.name}_{name}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "python", str(SNN_TOOL), "sim", "--infer",
        "--load-config", str((train_dir / "config.json").resolve()),
        "--load-weights", str((train_dir / "weights.pth").resolve()),
        "--sample-index", str(sample_idx), "--out-dir", str(out_dir),
    ]
    if i_override is not None:
        cmd += ["--i-override-file", str(i_override)]
    else:
        cmd += ["--outputs", "rasters"]  # baseline pass exposes the I-stream
    subprocess.run(cmd, cwd=REPO, check=True)
    return np.load(out_dir / "snapshot.npz")


def capture_condition_raster(
    train_dir: Path, condition: str, sample_idx: int,
    seed_offset: int = 0, cycle_period_ms=None,
) -> dict:
    """Single-trial raster under one I-override condition, via the CLI snapshot.

    Baseline snapshot supplies the trial's I-stream; the notebook builds the
    override and a second snapshot replays it under --i-override-file.
    """
    import torch
    cfg = json.loads((train_dir / "config.json").read_text())
    d0 = _snapshot(train_dir, sample_idx, f"base_s{sample_idx}")

    if condition == "baseline":
        d = d0
    else:
        s_i = d0["spk_i"]
        if s_i.ndim == 3:
            s_i = s_i[:, 0, :]
        gen = torch.Generator().manual_seed(EVAL_SEED + 17 + seed_offset)
        kwargs = {"cycle_period_ms": cycle_period_ms} if cycle_period_ms else {}
        ov = _build_override(
            torch.from_numpy(s_i[:, None, :].astype(np.float32)),
            condition, gen, dt_ms=float(cfg["dt"]), **kwargs,
        ).detach().cpu().numpy()[:, 0, :]  # (T, n_i)
        ti, ci = ov.nonzero()
        ov_path = (
            ARTIFACTS / "condraster" / f"{train_dir.name}_{condition}_s{sample_idx}_ov.npz"
        ).resolve()
        ov_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            ov_path, n_trials=np.int32(1), T=np.int32(ov.shape[0]),
            n_i=np.int32(ov.shape[1]),
            i_trial=np.zeros(ti.size, "int32"),
            i_t=ti.astype("int32"), i_cell=ci.astype("int32"),
        )
        d = _snapshot(train_dir, sample_idx, f"{condition}_s{sample_idx}", i_override=ov_path)

    e_full, i_full = d["spk_e"], d["spk_i"]
    if e_full.ndim == 3:
        e_full = e_full[:, 0, :]
    if i_full.ndim == 3:
        i_full = i_full[:, 0, :]
    y_b = int(d["label"])
    t_sec = float(cfg["t_ms"]) / 1000.0
    e_rate = float(e_full.sum() / (e_full.shape[1] * t_sec))
    i_rate = float(i_full.sum() / (i_full.shape[1] * t_sec))
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], RASTER_N_I_PLOT, replace=False))
    return {
        "condition": condition,
        "label": y_b,
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "e_rate_hz": e_rate,
        "i_rate_hz": i_rate,
        "dt": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


# ─── plotting ───────────────────────────────────────────────────────


def plot_bar_chart(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Three conditions on x; grouped bars for E rate (left axis) and
    accuracy (right axis). Mean I rate annotated above each E-rate bar
    so the reader can verify the control held."""
    theme.apply()
    # Aggregate across seeds: mean ± SEM per (condition, metric).
    agg: dict[str, dict[str, tuple[float, float]]] = {}
    for cond in CONDITIONS:
        sub = [r for r in rows if r["condition"] == cond]
        agg[cond] = {
            k: (
                float(np.mean([r[k] for r in sub])),
                float(np.std([r[k] for r in sub], ddof=1) / np.sqrt(max(1, len(sub))))
                if len(sub) > 1 else 0.0,
            )
            for k in ("acc", "e_rate_hz", "i_rate_hz")
        }

    fig, ax_rate = plt.subplots(figsize=(5.6, 3.15))
    xs = np.arange(len(CONDITIONS))
    width = 0.35

    e_means = [agg[c]["e_rate_hz"][0] for c in CONDITIONS]
    e_sems = [agg[c]["e_rate_hz"][1] for c in CONDITIONS]
    bars_e = ax_rate.bar(
        xs - width / 2, e_means, width, yerr=e_sems,
        color=theme.INK_BLACK, alpha=0.85, label="E rate", capsize=3,
    )
    ax_rate.set_ylabel("E rate (Hz)", fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK)
    ax_rate.tick_params(axis="y", labelcolor=theme.INK_BLACK)
    ax_rate.set_xticks(xs)
    ax_rate.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS],
                            fontsize=theme.SIZE_TICK, rotation=10)

    # Annotate mean I rate above each E-rate bar — the control sanity
    # check ("we held mean I inhibition constant").
    for bar, cond in zip(bars_e, CONDITIONS):
        i_mu, _ = agg[cond]["i_rate_hz"]
        ax_rate.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(e_means) * 0.02,
            f"I = {i_mu:.1f} Hz",
            ha="center", va="bottom",
            fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
        )

    ax_acc = ax_rate.twinx()
    acc_means = [agg[c]["acc"][0] for c in CONDITIONS]
    acc_sems = [agg[c]["acc"][1] for c in CONDITIONS]
    ax_acc.bar(
        xs + width / 2, acc_means, width, yerr=acc_sems,
        color=theme.DEEP_RED, alpha=0.85, label="accuracy", capsize=3,
    )
    ax_acc.set_ylabel("Test accuracy (%)",
                     fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED)
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)

    fig.suptitle(
        "Rhythm vs mean-inhibition — E rate and accuracy across I-stream conditions",
        fontsize=theme.SIZE_TITLE,
    )
    ax_rate.spines["top"].set_visible(False)
    ax_acc.spines["top"].set_visible(False)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_raster_strip(samples: list[dict], out_path: Path, run_id: str) -> None:
    """Stacked single-trial rasters across the three conditions."""
    theme.apply()
    n = len(samples)
    n_e = RASTER_N_E_PLOT
    n_i = RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(6.9, 3.88),
        sharex=True, gridspec_kw={"hspace": 0.22},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(
            t_axis[e_t], e_n,
            s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4,
        )
        ax.scatter(
            t_axis[i_t], i_n + n_e + gap,
            s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4,
        )
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.text(
            1.012, 0.5,
            CONDITION_LABELS[s["condition"]]
            + f"\nE = {s['e_rate_hz']:.1f} Hz"
            + f"\nI = {s['i_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i == 0:
            ax.set_title(
                "Single-trial rasters — trained PING (exp025 seed 42) "
                "under each I-stream condition"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


# σ values to render in the jitter-raster strip — diagnostic subset that
# spans the predicted transition at 1/f_γ ≈ 28 ms.
JITTER_RASTER_SIGMAS_MS: tuple[float, ...] = (0.0, 7.0, 14.0, 28.0, 100.0)


def plot_jitter_raster_strip(
    samples: list[dict], out_path: Path, run_id: str,
) -> None:
    """Stacked single-trial rasters across jitter σ values.

    Each sample carries a ``sigma_ms`` field instead of the categorical
    ``condition`` used by plot_raster_strip; layout otherwise identical.
    """
    theme.apply()
    n = len(samples)
    n_e = RASTER_N_E_PLOT
    n_i = RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(6.9, 0.69 * n + 1.035),
        sharex=True, gridspec_kw={"hspace": 0.22},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(t_axis[e_t], e_n,
                   s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4)
        ax.scatter(t_axis[i_t], i_n + n_e + gap,
                   s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4)
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.text(
            1.012, 0.5,
            f"σ = {s['sigma_ms']:g} ms"
            f"\nE = {s['e_rate_hz']:.1f} Hz"
            f"\nI = {s['i_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i == 0:
            ax.set_title(
                "Single-trial rasters — trained PING (exp025 seed 42) "
                "under cycle-coherent I-jitter"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def plot_cell_jitter_raster_strip(
    samples: list[dict], out_path: Path, run_id: str,
) -> None:
    """Stacked single-trial rasters across per-I-cell jitter σ values.

    Same layout as plot_jitter_raster_strip but for the per-spike (per-cell)
    jitter — within-burst synchrony is destroyed rather than preserved.
    """
    theme.apply()
    n = len(samples)
    n_e = RASTER_N_E_PLOT
    n_i = RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(6.9, 0.69 * n + 1.035),
        sharex=True, gridspec_kw={"hspace": 0.22},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(t_axis[e_t], e_n,
                   s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4)
        ax.scatter(t_axis[i_t], i_n + n_e + gap,
                   s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4)
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.text(
            1.012, 0.5,
            f"σ = {s['sigma_ms']:g} ms"
            f"\nE = {s['e_rate_hz']:.1f} Hz"
            f"\nI = {s['i_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i == 0:
            ax.set_title(
                "Single-trial rasters — trained PING (exp025 seed 42) "
                "under per-I-cell jitter (within-burst smearing)"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def plot_cell_jitter_sweep(
    cell_rows: list[dict], baseline_e_rate: float,
    poisson_e_rate: float, out_path: Path, run_id: str,
    tau_gaba_ms: float = TAU_GABA_GAMMA_MS,
) -> None:
    """Per-I-cell jitter sweep — E rate + accuracy on twin axes.

    Mirrors plot_jitter_sweep's layout but for the per-spike jitter family.
    Annotates the predicted transition at σ ≈ τ_GABA (the smearing width
    at which g_i looks continuous) and the Poisson asymptote.
    """
    theme.apply()
    by_sigma: dict[float, list[dict]] = {}
    for r in cell_rows:
        by_sigma.setdefault(r["sigma_ms"], []).append(r)
    sigmas_sorted = sorted(by_sigma.keys())
    e_means = [
        float(np.mean([r["e_rate_hz"] for r in by_sigma[s]])) for s in sigmas_sorted
    ]
    e_sems = [
        float(np.std([r["e_rate_hz"] for r in by_sigma[s]], ddof=1)
              / np.sqrt(max(1, len(by_sigma[s]))))
        if len(by_sigma[s]) > 1 else 0.0 for s in sigmas_sorted
    ]
    acc_means = [
        float(np.mean([r["acc"] for r in by_sigma[s]])) for s in sigmas_sorted
    ]
    acc_sems = [
        float(np.std([r["acc"] for r in by_sigma[s]], ddof=1)
              / np.sqrt(max(1, len(by_sigma[s]))))
        if len(by_sigma[s]) > 1 else 0.0 for s in sigmas_sorted
    ]

    fig, ax_rate = plt.subplots(figsize=(5.6, 3.11))
    ax_rate.errorbar(
        sigmas_sorted, e_means, yerr=e_sems,
        marker="D", markersize=6, lw=1.4, color=theme.INK_BLACK, capsize=3,
        label="E rate (Hz)",
    )
    ax_rate.set_xlabel(
        "Per-I-cell jitter σ on the I-stream (ms)",
        fontsize=theme.SIZE_LABEL,
    )
    ax_rate.set_ylabel("Hidden E rate (Hz)",
                       fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK)
    ax_rate.tick_params(axis="y", labelcolor=theme.INK_BLACK)

    # Reference horizontal lines: baseline and Poisson (the σ → ∞ asymptote).
    ax_rate.axhline(baseline_e_rate, color=theme.MUTED, lw=0.7, ls="--", alpha=0.7)
    ax_rate.text(
        ax_rate.get_xlim()[1], baseline_e_rate + 0.4,
        f"  baseline ≈ {baseline_e_rate:.1f} Hz",
        fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
        ha="right", va="bottom",
    )
    ax_rate.axhline(poisson_e_rate, color=theme.DEEP_RED, lw=0.7, ls="--",
                    alpha=0.7)
    ax_rate.text(
        ax_rate.get_xlim()[1], poisson_e_rate + 0.4,
        f"  rate-matched Poisson ≈ {poisson_e_rate:.1f} Hz",
        fontsize=theme.SIZE_ANNOTATION, color=theme.DEEP_RED,
        ha="right", va="bottom",
    )
    # Predicted transition: τ_GABA (smearing width at which g_i goes continuous).
    ax_rate.axvline(tau_gaba_ms, color=theme.GREY_MID, lw=0.7, ls=":", alpha=0.8)
    ax_rate.text(
        tau_gaba_ms, ax_rate.get_ylim()[1] * 0.95,
        f" τ_GABA = {tau_gaba_ms:g} ms",
        fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
        ha="left", va="top",
    )

    ax_acc = ax_rate.twinx()
    ax_acc.errorbar(
        sigmas_sorted, acc_means, yerr=acc_sems,
        marker="s", markersize=6, lw=1.4, color=theme.DEEP_RED, capsize=3,
    )
    ax_acc.set_ylabel("Test accuracy (%)",
                      fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED)
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)
    ax_acc.axhline(10.0, color=theme.DEEP_RED, lw=0.5, ls=":", alpha=0.4)

    fig.suptitle(
        "Per-I-cell jitter sweep — within-burst synchrony silences E",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_pareto_raster_strip(
    samples: list[dict], out_path: Path, run_id: str,
) -> None:
    """Stacked single-trial rasters at four cells of the (α, k) grid.

    Each sample carries ``alpha``, ``k``, and a short ``note`` describing
    where on the Pareto plot the cell sits.
    """
    theme.apply()
    n = len(samples)
    n_e = RASTER_N_E_PLOT
    n_i = RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(6.9, 0.69 * n + 1.035),
        sharex=True, gridspec_kw={"hspace": 0.32},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(t_axis[e_t], e_n,
                   s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4)
        ax.scatter(t_axis[i_t], i_n + n_e + gap,
                   s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4)
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.text(
            1.012, 0.5,
            f"α = {s['alpha']:g}, k = {s['k']:g}"
            f"\nE = {s['e_rate_hz']:.1f} Hz"
            f"\nI = {s['i_rate_hz']:.1f} Hz"
            f"\nacc = {s['acc']:.1f}%",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        ax.text(
            0.005, 1.04, s.get("note", ""),
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=theme.SIZE_ANNOTATION,
            color=theme.MUTED,
            fontstyle="italic",
        )
        if i == 0:
            ax.set_title(
                "Pareto-sweep rasters — trained PING (exp025 seed 42) "
                "under (α, k) I-stream perturbations"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def plot_jitter_sweep(
    jitter_rows: list[dict], baseline_e_rate: float,
    phase_shuffle_e_rate: float, out_path: Path, run_id: str,
) -> None:
    """E rate vs σ_ms with predicted inflection at 1/f_γ annotated.

    jitter_rows: list of dicts with sigma_ms, e_rate_hz, i_rate_hz, acc.
    Aggregated across seeds before plotting.
    """
    theme.apply()
    by_sigma: dict[float, list[dict]] = {}
    for r in jitter_rows:
        by_sigma.setdefault(r["sigma_ms"], []).append(r)
    sigmas_sorted = sorted(by_sigma.keys())
    e_means = [
        float(np.mean([r["e_rate_hz"] for r in by_sigma[s]])) for s in sigmas_sorted
    ]
    e_sems = [
        float(np.std([r["e_rate_hz"] for r in by_sigma[s]], ddof=1)
              / np.sqrt(max(1, len(by_sigma[s]))))
        if len(by_sigma[s]) > 1 else 0.0 for s in sigmas_sorted
    ]
    acc_means = [
        float(np.mean([r["acc"] for r in by_sigma[s]])) for s in sigmas_sorted
    ]
    acc_sems = [
        float(np.std([r["acc"] for r in by_sigma[s]], ddof=1)
              / np.sqrt(max(1, len(by_sigma[s]))))
        if len(by_sigma[s]) > 1 else 0.0 for s in sigmas_sorted
    ]

    fig, ax_rate = plt.subplots(figsize=(5.6, 3.11))
    # Use a symlog x-axis so both σ = 0 and σ = 100 are visible.
    ax_rate.errorbar(
        sigmas_sorted, e_means, yerr=e_sems,
        marker="D", markersize=6, lw=1.4, color=theme.INK_BLACK, capsize=3,
        label="E rate (Hz)",
    )
    ax_rate.set_xscale("symlog", linthresh=1.0)
    ax_rate.set_xlabel(
        "Cycle-coherent jitter σ on the I-stream (ms, symlog)",
        fontsize=theme.SIZE_LABEL,
    )
    ax_rate.set_ylabel("Hidden E rate (Hz)",
                       fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK)
    ax_rate.tick_params(axis="y", labelcolor=theme.INK_BLACK)

    # Annotate baseline and full-phase-shuffle reference levels.
    ax_rate.axhline(baseline_e_rate, color=theme.MUTED, lw=0.7, ls="--", alpha=0.7)
    ax_rate.text(
        ax_rate.get_xlim()[1], baseline_e_rate + 0.4,
        f"  baseline ≈ {baseline_e_rate:.1f} Hz",
        fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
        ha="right", va="bottom",
    )
    ax_rate.axhline(phase_shuffle_e_rate, color=theme.DEEP_RED, lw=0.7, ls="--",
                    alpha=0.7)
    ax_rate.text(
        ax_rate.get_xlim()[1], phase_shuffle_e_rate + 0.4,
        f"  full phase-shuffle ≈ {phase_shuffle_e_rate:.1f} Hz",
        fontsize=theme.SIZE_ANNOTATION, color=theme.DEEP_RED,
        ha="right", va="bottom",
    )
    # Annotate predicted inflection at 1/f_γ.
    period_ms = 1000.0 / F_GAMMA_REFERENCE_HZ
    ax_rate.axvline(period_ms, color=theme.GREY_MID, lw=0.7, ls=":", alpha=0.8)
    ax_rate.text(
        period_ms, ax_rate.get_ylim()[1] * 0.95,
        f" 1/f_γ = {period_ms:.1f} ms",
        fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
        ha="left", va="top",
    )

    ax_acc = ax_rate.twinx()
    ax_acc.errorbar(
        sigmas_sorted, acc_means, yerr=acc_sems,
        marker="s", markersize=6, lw=1.4, color=theme.DEEP_RED, capsize=3,
    )
    ax_acc.set_ylabel("Test accuracy (%)",
                      fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED)
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)

    fig.suptitle(
        "Jitter sweep — E rate release scales with σ; transition at 1/f_γ",
        fontsize=theme.SIZE_TITLE,
    )
    ax_rate.spines["top"].set_visible(False)
    ax_acc.spines["top"].set_visible(False)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_pareto(
    pareto_rows: list[dict], baseline_row: dict, out_path: Path, run_id: str,
) -> dict:
    """Scatter every (α, k) condition in (E rate, accuracy) space; mark
    the rhythmic baseline and the Pareto front of the non-baseline
    points. Returns a small dict of summary numbers (n above frontier,
    etc.) for numbers.json.
    """
    theme.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.15))

    # Plot non-baseline points coloured by α, sized by k.
    alphas_unique = sorted({r["alpha"] for r in pareto_rows})
    cmap = plt.get_cmap("viridis")
    for r in pareto_rows:
        ai = alphas_unique.index(r["alpha"])
        color = cmap(ai / max(1, len(alphas_unique) - 1))
        size = 30 + 10 * (r["k"] * 4)
        ax.scatter(
            r["e_rate_hz"], r["acc"], s=size, c=[color],
            edgecolor=theme.INK, lw=0.4, alpha=0.85,
        )

    # Baseline marker.
    ax.scatter(
        baseline_row["e_rate_hz"], baseline_row["acc"],
        s=200, marker="*", c="white", edgecolor=theme.INK, lw=1.6,
        zorder=10, label="rhythmic baseline (α = 0, k = 1)",
    )

    # Pareto frontier among non-baseline points (minimise E, maximise acc).
    pts = sorted(pareto_rows, key=lambda r: (r["e_rate_hz"], -r["acc"]))
    frontier = []
    best_acc = -1.0
    for r in pts:
        if r["acc"] > best_acc:
            frontier.append(r)
            best_acc = r["acc"]
    if frontier:
        ax.plot(
            [r["e_rate_hz"] for r in frontier],
            [r["acc"] for r in frontier],
            color=theme.GREY_MID, lw=1.0, ls="--", alpha=0.7,
            label="non-rhythmic Pareto frontier",
        )

    # Legend chips for α colour and k size.
    α_handles = []
    for ai, a in enumerate(alphas_unique):
        color = cmap(ai / max(1, len(alphas_unique) - 1))
        α_handles.append(
            plt.Line2D([], [], marker="o", linestyle="", color=color,
                       markeredgecolor=theme.INK, markersize=6, label=f"α = {a}")
        )
    ax.legend(
        handles=α_handles + [
            plt.Line2D([], [], marker="*", linestyle="",
                       markeredgecolor=theme.INK, markerfacecolor="white",
                       markersize=12, label="baseline"),
            plt.Line2D([], [], color=theme.GREY_MID, lw=1.0, ls="--",
                       label="non-rhythmic frontier"),
        ],
        fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right",
    )

    ax.set_xlabel("Mean hidden-E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    ax.set_title(
        "Rhythm-vs-Poisson (α, k) Pareto sweep — "
        "is rhythmic baseline on the frontier?",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)

    # Summary: is the baseline strictly Pareto-above every non-baseline cell?
    dominated_by_baseline = sum(
        1 for r in pareto_rows
        if r["e_rate_hz"] >= baseline_row["e_rate_hz"]
        and r["acc"] <= baseline_row["acc"]
    )
    above_baseline_acc = sum(1 for r in pareto_rows if r["acc"] > baseline_row["acc"])
    below_baseline_rate = sum(
        1 for r in pareto_rows
        if r["e_rate_hz"] < baseline_row["e_rate_hz"] and r["acc"] >= baseline_row["acc"] - 1.0
    )
    return {
        "n_cells": len(pareto_rows),
        "n_dominated_by_baseline": dominated_by_baseline,
        "n_strictly_above_baseline_acc": above_baseline_acc,
        "n_below_baseline_rate_at_matched_acc": below_baseline_rate,
    }


# ─── cross-τ_GABA jitter (formerly nb045) ───────────────────────────


def _xtau_tau_label(tau_ms: float) -> str:
    return f"tg{f'{tau_ms:g}'.replace('.', 'p')}"


def _xtau_exp041_cell_dir(tau_ms: float, seed: int) -> Path:
    return NB041_ARTIFACTS / f"ping__{_xtau_tau_label(tau_ms)}__seed{seed}"


def _xtau_load_exp041_f_gamma() -> dict[tuple[float, int], float]:
    if not NB041_NUMBERS.exists():
        raise SystemExit(
            f"missing exp041 numbers.json at {NB041_NUMBERS}; "
            "re-render exp041 (skip-training) to produce it."
        )
    data = json.loads(NB041_NUMBERS.read_text())
    out: dict[tuple[float, int], float] = {}
    for r in data.get("results", []):
        out[(float(r["tau_gaba_ms"]), int(r["seed"]))] = float(r["f_gamma_hz"])
    return out


def _xtau_evaluate_cell(
    train_dir: Path, sigma_ms: float, cycle_period_ms: float,
    seed_offset: int = 0,
) -> dict:
    """Per-cell inference under cycle-coherent jitter (the cell's own 1/f_γ as the
    binning period), via the CLI two-pass override under the cell's τ_GABA."""
    import torch
    cfg = json.loads((train_dir / "config.json").read_text())
    tau_gaba_ms = float(cfg.get("tau_gaba_ms") or MODELS_DEFAULT_TAU_GABA_MS)
    m0, R = _run_baseline(train_dir, tau_gaba=tau_gaba_ms)
    if sigma_ms <= 0.0:
        p = _pack_metrics(m0, "baseline")
    else:
        gen = torch.Generator().manual_seed(EVAL_SEED + 17 + seed_offset)
        cond = f"jitter_sigma_{sigma_ms:g}"
        ov_path = (
            ARTIFACTS / "override" / f"{train_dir.name}_{cond}_{seed_offset}.npz"
        ).resolve()
        ov_path.parent.mkdir(parents=True, exist_ok=True)
        _build_override_file(
            R, cond, gen, float(cfg["dt"]), ov_path, cycle_period_ms=cycle_period_ms,
        )
        p = _pack_metrics(_run_with_override(train_dir, ov_path, tau_gaba=tau_gaba_ms), cond)
    return {
        "tau_gaba_ms": tau_gaba_ms,
        "sigma_ms": float(sigma_ms),
        "cycle_period_ms": float(cycle_period_ms),
        "acc": p["acc"],
        "e_rate_hz": p["e_rate_hz"],
        "i_rate_hz": p["i_rate_hz"],
        "n_total": p["n_total"],
    }


def _xtau_aggregate(rows: list[dict]) -> dict:
    by_key: dict[tuple[float, float], list[dict]] = {}
    for r in rows:
        by_key.setdefault((r["tau_gaba_ms"], r["sigma_ms"]), []).append(r)
    agg: dict = {}
    for key, group in by_key.items():
        e = [g["e_rate_hz"] for g in group]
        a = [g["acc"] for g in group]
        f = [g.get("f_gamma_hz", 0.0) for g in group]
        agg[key] = {
            "e_rate_mean": float(np.mean(e)),
            "e_rate_sem": float(
                np.std(e, ddof=1) / np.sqrt(len(e)) if len(e) > 1 else 0.0
            ),
            "acc_mean": float(np.mean(a)),
            "f_gamma_mean": float(np.mean(f)),
        }
    return agg


def plot_xtau_raw_sweeps(rows: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    agg = _xtau_aggregate(rows)
    fig, ax = plt.subplots(figsize=(5.6, 3.11))
    cmap = plt.get_cmap("viridis")
    taus_sorted = sorted({k[0] for k in agg.keys()})
    for i, tau in enumerate(taus_sorted):
        color = cmap(i / max(1, len(taus_sorted) - 1))
        sigmas = sorted({k[1] for k in agg.keys() if k[0] == tau})
        e_means = [agg[(tau, s)]["e_rate_mean"] for s in sigmas]
        e_sems = [agg[(tau, s)]["e_rate_sem"] for s in sigmas]
        f_gamma = agg[(tau, sigmas[0])]["f_gamma_mean"]
        ax.errorbar(
            sigmas, e_means, yerr=e_sems, marker="o", markersize=5, lw=1.2,
            color=color, capsize=3,
            label=f"τ_GABA = {tau:g} ms  (f_γ = {f_gamma:.0f} Hz)",
        )
    ax.set_xlabel("Cycle-coherent jitter σ (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    fig.suptitle("Jitter sweep across τ_GABA — raw σ axis",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_xtau_dimensional_collapse(
    rows: list[dict], out_path: Path, run_id: str,
) -> None:
    theme.apply()
    agg = _xtau_aggregate(rows)
    fig, ax = plt.subplots(figsize=(5.6, 3.11))
    cmap = plt.get_cmap("viridis")
    taus_sorted = sorted({k[0] for k in agg.keys()})
    for i, tau in enumerate(taus_sorted):
        color = cmap(i / max(1, len(taus_sorted) - 1))
        sigmas = sorted({k[1] for k in agg.keys() if k[0] == tau})
        f_gamma = agg[(tau, sigmas[0])]["f_gamma_mean"]
        baseline = agg[(tau, sigmas[0])]["e_rate_mean"]
        rate_max = max(agg[(tau, s)]["e_rate_mean"] for s in sigmas)
        rate_range = max(rate_max - baseline, 1e-6)
        x_scaled = [s * f_gamma / 1000.0 for s in sigmas]
        y_norm = [
            (agg[(tau, s)]["e_rate_mean"] - baseline) / rate_range
            for s in sigmas
        ]
        ax.plot(x_scaled, y_norm, marker="o", markersize=5, lw=1.2,
                color=color, label=f"τ_GABA = {tau:g} ms")
    ax.axvline(1.0, color=theme.GREY_MID, lw=0.7, ls=":", alpha=0.8)
    ax.text(1.0, 0.97, " predicted inflection: σ·f_γ = 1",
            transform=ax.get_xaxis_transform(),
            fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
            ha="left", va="top")
    ax.set_xlabel("σ · f_γ  (dimensionless, units of one cycle)",
                  fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Normalised E rate (r − baseline) / (max − baseline)",
                  fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    ax.set_ylim(-0.05, 1.1)
    fig.suptitle("Dimensional collapse — σ rescaled by f_γ",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_xtau_inflection_vs_period(
    rows: list[dict], out_path: Path, run_id: str,
) -> dict:
    theme.apply()
    agg = _xtau_aggregate(rows)
    taus_sorted = sorted({k[0] for k in agg.keys()})
    inflections, periods, f_gammas = [], [], []
    for tau in taus_sorted:
        sigmas = sorted({k[1] for k in agg.keys() if k[0] == tau})
        baseline = agg[(tau, sigmas[0])]["e_rate_mean"]
        rate_max = max(agg[(tau, s)]["e_rate_mean"] for s in sigmas)
        rate_range = max(rate_max - baseline, 1e-6)
        ys = [
            (agg[(tau, s)]["e_rate_mean"] - baseline) / rate_range
            for s in sigmas
        ]
        sigma_inflection = None
        for k in range(len(sigmas) - 1):
            y0, y1 = ys[k], ys[k + 1]
            if (y0 <= 0.5 <= y1) or (y1 <= 0.5 <= y0):
                frac = (0.5 - y0) / (y1 - y0 + 1e-12)
                sigma_inflection = sigmas[k] + frac * (sigmas[k + 1] - sigmas[k])
                break
        if sigma_inflection is None:
            continue
        f_gamma = agg[(tau, sigmas[0])]["f_gamma_mean"]
        inflections.append(sigma_inflection)
        periods.append(1000.0 / f_gamma)
        f_gammas.append(f_gamma)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    alpha = r2 = float("nan")
    if inflections:
        ax.scatter(periods, inflections, s=60, color=theme.INK_BLACK, zorder=3,
                   label="measured inflection σ")
        p = np.array(periods)
        s = np.array(inflections)
        alpha = float(np.sum(p * s) / np.sum(p * p))
        ss_res = float(np.sum((s - alpha * p) ** 2))
        ss_tot = float(np.sum((s - s.mean()) ** 2)) if len(s) > 1 else 1.0
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        xs_fit = np.linspace(0, max(p) * 1.1, 100)
        ax.plot(xs_fit, alpha * xs_fit, color=theme.DEEP_RED, ls="--", lw=1.2,
                label=f"σ = α · (1/f_γ)  (α = {alpha:.2f}, R² = {r2:.3f})")
        ax.plot(xs_fit, xs_fit, color=theme.GREY_MID, ls=":", lw=0.8,
                label="predicted: σ = 1/f_γ")
    ax.set_xlabel("1/f_γ  (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Measured inflection σ  (ms)", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    fig.suptitle("Inflection σ tracks 1/f_γ across τ_GABA",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)
    return {"alpha": alpha, "r2": r2, "n_points": len(inflections)}


# ─── rhythm-vs-mean compound (the manuscript figure) ────────────────


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _compound_raster_panel(ax, s: dict, title: str, subtitle: str) -> None:
    """One raster panel — E (black) below, I (red) above, rates annotated."""
    n_e, n_i, gap = RASTER_N_E_PLOT, RASTER_N_I_PLOT, 6
    T = s["e"].shape[0]
    t_axis = np.arange(T) * s["dt"]
    e_t, e_n = np.where(s["e"])
    i_t, i_n = np.where(s["i"])
    ax.scatter(t_axis[e_t], e_n, s=1.5, c=theme.INK_BLACK, marker="|", linewidths=0.4)
    ax.scatter(t_axis[i_t], i_n + n_e + gap,
               s=1.5, c=theme.DEEP_RED, marker="|", linewidths=0.4)
    ax.set_ylim(-2, n_e + n_i + gap + 2)
    ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
    ax.set_yticklabels(["E", "I"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, s["t_ms"])
    ax.set_title(title, fontsize=theme.SIZE_LABEL)
    ax.text(
        0.98, 0.94, subtitle
        + f"\nE = {s['e_rate_hz']:.1f} Hz   I = {s['i_rate_hz']:.1f} Hz",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
    )
    _despine(ax)


def _compound_sweep_panel(
    ax, rows: list[dict], *, xlabel: str, title: str,
    baseline_e: float, symlog: bool,
) -> None:
    """One sweep panel — E rate (black) and accuracy (red, twin axis) vs σ."""
    by_sigma: dict[float, list[dict]] = {}
    for r in rows:
        by_sigma.setdefault(r["sigma_ms"], []).append(r)
    sig = sorted(by_sigma)
    e_means = [float(np.mean([r["e_rate_hz"] for r in by_sigma[s]])) for s in sig]
    a_means = [float(np.mean([r["acc"] for r in by_sigma[s]])) for s in sig]

    ax.plot(sig, e_means, marker="D", ms=5, lw=1.4, color=theme.INK_BLACK)
    if symlog:
        ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel(xlabel, fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("E rate (Hz)", color=theme.INK_BLACK, fontsize=theme.SIZE_LABEL)
    ax.tick_params(axis="y", labelcolor=theme.INK_BLACK)
    ax.axhline(baseline_e, color=theme.MUTED, lw=0.7, ls="--", alpha=0.7)
    ax.text(ax.get_xlim()[1], baseline_e, f" baseline ≈ {baseline_e:.1f} Hz",
            fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
            ha="right", va="bottom")
    ax.set_title(title, fontsize=theme.SIZE_LABEL)

    ax_acc = ax.twinx()
    ax_acc.plot(sig, a_means, marker="s", ms=5, lw=1.4, color=theme.DEEP_RED)
    ax_acc.set_ylabel("accuracy (%)", color=theme.DEEP_RED, fontsize=theme.SIZE_LABEL)
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax_acc.spines["top"].set_visible(False)


def fig_rhythm_compound(
    cyc_rows: list[dict], cell_rows: list[dict], baseline_e: float,
    raster_cyc: dict, raster_cell: dict, out_path: Path, run_id: str,
) -> None:
    """2×2 manuscript compound — matched mean I, opposite E response.

    Columns are the two manipulations that both preserve mean I rate:
      left  — cycle-coherent jitter (within-burst synchrony kept, bursts
              displaced) → E fires through the opened gaps, rate rises.
      right — per-I-cell jitter (synchrony destroyed, bursts smeared into
              a continuous shunt) → E silenced, rate falls to zero.
    Top row: example single-trial rasters; bottom row: the full sweeps.
    """
    theme.apply()
    prev_bbox = plt.rcParams["savefig.bbox"]
    plt.rcParams["savefig.bbox"] = "standard"
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.88))

    _compound_raster_panel(
        axes[0, 0], raster_cell,
        "Smear the bursts — synchrony destroyed",
        f"per-I-cell jitter σ = {raster_cell['sigma_ms']:g} ms",
    )
    _compound_raster_panel(
        axes[0, 1], raster_cyc,
        "Move the bursts — synchrony preserved",
        f"cycle-coherent jitter σ = {raster_cyc['sigma_ms']:g} ms",
    )
    _compound_sweep_panel(
        axes[1, 0], cell_rows,
        xlabel="per-I-cell jitter σ (ms)",
        title="Smear bursts → E rate falls to zero",
        baseline_e=baseline_e, symlog=False,
    )
    _compound_sweep_panel(
        axes[1, 1], cyc_rows,
        xlabel="cycle-coherent jitter σ (ms, symlog)",
        title="Displace bursts → E rate rises",
        baseline_e=baseline_e, symlog=True,
    )
    fig.suptitle(
        "Gamma gates the rate, not mean inhibition — "
        "matched mean I, opposite E response",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense rasters: PNG, not SVG
    plt.close(fig)
    plt.rcParams["savefig.bbox"] = prev_bbox


def build_rhythm_compound(run_id: str = "replot") -> None:
    """Rebuild the compound from cached numbers.json — no sweep re-runs.

    Sweep curves load from numbers.json; the two example rasters are cheap
    single-trial forward passes against the cached exp025 PING weights.
    """
    data = json.loads((FIGURES / "numbers.json").read_text())
    rows = data["results"]
    cyc_rows = data["jitter_sweep"]
    cell_rows = data["cell_jitter_sweep"]
    baseline_e = float(np.mean(
        [r["e_rate_hz"] for r in rows if r["condition"] == "baseline"]
    ))

    seed = int(data["config"]["seeds"][0])
    train_dir = NB035_ARTIFACTS / f"ping__off__seed{seed}"
    raster_cyc = capture_condition_raster(
        train_dir, "jitter_sigma_100", RASTER_SAMPLE_IDX,
        seed_offset=seed + 100,
    )
    raster_cyc["sigma_ms"] = 100.0
    raster_cell = capture_condition_raster(
        train_dir, "cell_jitter_sigma_5", RASTER_SAMPLE_IDX,
        seed_offset=seed + int(5 * 13),
    )
    raster_cell["sigma_ms"] = 5.0

    fig_rhythm_compound(
        cyc_rows, cell_rows, baseline_e, raster_cyc, raster_cell,
        FIGURES / "rhythm_compound", run_id,
    )
    print(f"wrote {FIGURES / 'rhythm_compound'}")


# ─── success criteria ───────────────────────────────────────────────

def main() -> None:
    # Publication profile: every figure this notebook writes is a print-sized
    # vector, emitted as both SVG (docs) and PDF (manuscript) by save_figure.
    theme.set_paper_mode(True)

    if replot_target(sys.argv) == "compound":
        build_rhythm_compound()
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modal-gpu", default="none")
    parser.add_argument("--no-wipe-dir", action="store_true")
    args = parser.parse_args()

    modal_gpu = parse_modal_gpu(sys.argv)
    if modal_gpu:
        raise SystemExit(
            "exp042 is inference-only on local exp025 weights; --modal-gpu unused"
        )

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} seeds={SEEDS}")

    prepare_run_dirs(
        SLUG, notebook_run_id, wipe=not args.no_wipe_dir, make_artifacts=True,
        scale=SCALE, host=f"modal:{modal_gpu}" if modal_gpu else "local",
    )

    rows: list[dict] = []
    for seed in SEEDS:
        train_dir = NB035_ARTIFACTS / f"ping__off__seed{seed}"
        if not (train_dir / "weights.pth").exists():
            raise SystemExit(
                f"missing exp025 trained PING checkpoint at {train_dir} — "
                "train exp025 baselines first"
            )
        print(f"[eval] seed={seed} from {train_dir.relative_to(REPO)}")
        for cond in CONDITIONS:
            t0 = time.monotonic()
            res = evaluate_condition(train_dir, cond, seed_offset=seed)
            res["seed"] = seed
            rows.append(res)
            print(
                f"    {cond:<22}  acc={res['acc']:5.2f}%  "
                f"E={res['e_rate_hz']:6.2f} Hz  I={res['i_rate_hz']:6.2f} Hz  "
                f"({time.monotonic() - t0:.1f}s)"
            )

    # Raster strip — single trial per condition from seed 42.
    raster_train_dir = NB035_ARTIFACTS / f"ping__off__seed{SEEDS[0]}"
    print(
        f"[raster] single-trial panels from seed {SEEDS[0]}, "
        f"sample {RASTER_SAMPLE_IDX}"
    )
    samples = [
        capture_condition_raster(
            raster_train_dir, cond, RASTER_SAMPLE_IDX,
            seed_offset=SEEDS[0],
        )
        for cond in CONDITIONS
    ]
    plot_raster_strip(samples, FIGURES / "raster_strip", notebook_run_id)
    print(f"wrote {FIGURES / 'raster_strip'}")

    plot_bar_chart(rows, FIGURES / "bar_chart", notebook_run_id)
    print(f"wrote {FIGURES / 'bar_chart'}")

    # ── Jitter sweep ───────────────────────────────────────────────
    # Adds Gaussian timing jitter σ to each I-spike at inference.
    # Predicts the rate-release transition at σ ≈ 1/f_γ ≈ 28 ms.
    print(f"[jitter] sweep σ ∈ {list(JITTER_SIGMAS_MS)} ms")
    jitter_rows: list[dict] = []
    for seed in SEEDS:
        train_dir = NB035_ARTIFACTS / f"ping__off__seed{seed}"
        for sigma_ms in JITTER_SIGMAS_MS:
            cond = f"jitter_sigma_{sigma_ms:g}"
            t0 = time.monotonic()
            # Reuse evaluate_condition — it dispatches on the condition string.
            res = evaluate_condition(train_dir, cond,
                                     seed_offset=seed + int(sigma_ms))
            res["seed"] = seed
            res["sigma_ms"] = float(sigma_ms)
            jitter_rows.append(res)
            print(
                f"    σ={sigma_ms:>5.1f}ms seed={seed}  "
                f"acc={res['acc']:5.2f}%  E={res['e_rate_hz']:6.2f} Hz  "
                f"I={res['i_rate_hz']:6.2f} Hz  ({time.monotonic() - t0:.1f}s)"
            )

    # Reference levels: baseline from the three-condition section above;
    # full phase-shuffle from the same section gives the upper asymptote
    # the jitter sweep should approach at σ ≫ 1/f_γ.
    baseline_e = float(np.mean(
        [r["e_rate_hz"] for r in rows if r["condition"] == "baseline"]
    ))
    phase_shuffle_e = float(np.mean(
        [r["e_rate_hz"] for r in rows if r["condition"] == "phase_shuffled_i"]
    ))
    plot_jitter_sweep(
        jitter_rows, baseline_e, phase_shuffle_e,
        FIGURES / "jitter_sweep", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'jitter_sweep'}")

    # Jitter raster strip — one panel per σ from the diagnostic subset,
    # all from the first seed at sample 0 so panels read against the
    # baseline raster strip directly.
    raster_seed = SEEDS[0]
    raster_train_dir = NB035_ARTIFACTS / f"ping__off__seed{raster_seed}"
    print(
        f"[jitter-raster] panels from seed {raster_seed}, "
        f"σ ∈ {list(JITTER_RASTER_SIGMAS_MS)} ms"
    )
    jitter_raster_samples = []
    for sigma_ms in JITTER_RASTER_SIGMAS_MS:
        cond = f"jitter_sigma_{sigma_ms:g}"
        sample = capture_condition_raster(
            raster_train_dir, cond, RASTER_SAMPLE_IDX,
            seed_offset=raster_seed + int(sigma_ms),
        )
        sample["sigma_ms"] = float(sigma_ms)
        jitter_raster_samples.append(sample)
    plot_jitter_raster_strip(
        jitter_raster_samples,
        FIGURES / "jitter_raster_strip",
        notebook_run_id,
    )
    print(f"wrote {FIGURES / 'jitter_raster_strip'}")

    # ── Per-cell jitter sweep ──────────────────────────────────────
    # Independent Gaussian offset per spike — destroys within-burst
    # synchrony while preserving burst placement on average. Predicts
    # the rate-release transition at σ ≈ τ_GABA (the smearing width
    # at which the integrated g_i profile starts looking continuous).
    print(f"[cell-jitter] sweep σ ∈ {list(CELL_JITTER_SIGMAS_MS)} ms")
    cell_jitter_rows: list[dict] = []
    for seed in SEEDS:
        train_dir = NB035_ARTIFACTS / f"ping__off__seed{seed}"
        for sigma_ms in CELL_JITTER_SIGMAS_MS:
            cond = f"cell_jitter_sigma_{sigma_ms:g}"
            t0 = time.monotonic()
            res = evaluate_condition(train_dir, cond,
                                     seed_offset=seed + int(sigma_ms * 13))
            res["seed"] = seed
            res["sigma_ms"] = float(sigma_ms)
            cell_jitter_rows.append(res)
            print(
                f"    σ={sigma_ms:>5.1f}ms seed={seed}  "
                f"acc={res['acc']:5.2f}%  E={res['e_rate_hz']:6.2f} Hz  "
                f"I={res['i_rate_hz']:6.2f} Hz  ({time.monotonic() - t0:.1f}s)"
            )

    poisson_e = float(np.mean(
        [r["e_rate_hz"] for r in rows if r["condition"] == "poisson_matched_i"]
    ))
    plot_cell_jitter_sweep(
        cell_jitter_rows, baseline_e, poisson_e,
        FIGURES / "cell_jitter_sweep", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'cell_jitter_sweep'}")

    # Per-cell jitter raster strip — diagnostic subset.
    print(
        f"[cell-jitter-raster] panels from seed {raster_seed}, "
        f"σ ∈ {list(CELL_JITTER_RASTER_SIGMAS_MS)} ms"
    )
    cell_jitter_raster_samples = []
    for sigma_ms in CELL_JITTER_RASTER_SIGMAS_MS:
        cond = f"cell_jitter_sigma_{sigma_ms:g}"
        sample = capture_condition_raster(
            raster_train_dir, cond, RASTER_SAMPLE_IDX,
            seed_offset=raster_seed + int(sigma_ms * 13),
        )
        sample["sigma_ms"] = float(sigma_ms)
        cell_jitter_raster_samples.append(sample)
    plot_cell_jitter_raster_strip(
        cell_jitter_raster_samples,
        FIGURES / "cell_jitter_raster_strip",
        notebook_run_id,
    )
    print(f"wrote {FIGURES / 'cell_jitter_raster_strip'}")

    # Manuscript compound: matched mean I, opposite E response. Reuse the
    # σ = 100 ms cycle-coherent and σ = 5 ms per-cell raster samples.
    raster_cyc = next(s for s in jitter_raster_samples if s["sigma_ms"] == 100.0)
    raster_cell = next(s for s in cell_jitter_raster_samples if s["sigma_ms"] == 5.0)
    fig_rhythm_compound(
        jitter_rows, cell_jitter_rows, baseline_e, raster_cyc, raster_cell,
        FIGURES / "rhythm_compound", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'rhythm_compound'}")

    # ── Pareto sweep ──────────────────────────────────────────────
    # Probe whether the rhythmic baseline sits at the (low E, high acc)
    # corner of the (α, k) grid. Single seed (first in SEEDS) — the
    # frontier shape, not error bars, is the load-bearing observation.
    pareto_seed = SEEDS[0]
    pareto_train_dir = NB035_ARTIFACTS / f"ping__off__seed{pareto_seed}"
    print(
        f"[pareto] α × k sweep on seed {pareto_seed}: "
        f"α ∈ {list(MIX_ALPHA_GRID)}, k ∈ {list(MIX_K_GRID)}"
    )
    pareto_rows: list[dict] = []
    for alpha in MIX_ALPHA_GRID:
        for k in MIX_K_GRID:
            # Skip the cell that exactly reduces to baseline (α=0, k=1).
            if alpha == 0.0 and k == 1.0:
                continue
            cond = f"alpha_mix_a{alpha:g}_k{k:g}"
            t0 = time.monotonic()
            res = evaluate_condition(
                pareto_train_dir, cond,
                seed_offset=pareto_seed + int(alpha * 100) + int(k * 10),
            )
            res["seed"] = pareto_seed
            res["alpha"] = float(alpha)
            res["k"] = float(k)
            pareto_rows.append(res)
            print(
                f"    α={alpha:>4} k={k:>4}  acc={res['acc']:5.2f}%  "
                f"E={res['e_rate_hz']:6.2f}Hz  I={res['i_rate_hz']:6.2f}Hz  "
                f"({time.monotonic() - t0:.1f}s)"
            )

    baseline_row = next(
        r for r in rows if r["condition"] == "baseline" and r["seed"] == pareto_seed
    )
    pareto_summary = plot_pareto(
        pareto_rows, baseline_row, FIGURES / "pareto_sweep", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'pareto_sweep'}")
    print(f"  pareto summary: {pareto_summary}")

    # ── Cross-τ_GABA jitter sweep (formerly nb045) ─────────────────
    # Replay the cycle-coherent jitter sweep on each of exp041's 18
    # trained cells, with each cell's own 1/f_γ as the binning period.
    # Tests whether the σ ≈ 1/f_γ inflection scales when f_γ varies.
    print(f"[xtau] τ_GABA × seed × σ sweep: {len(XTAU_TAU_GABAS_MS)} × "
          f"{len(XTAU_SEEDS)} × {len(XTAU_SIGMAS_MS)} = "
          f"{len(XTAU_TAU_GABAS_MS) * len(XTAU_SEEDS) * len(XTAU_SIGMAS_MS)} evals")
    xtau_rows: list[dict] = []
    xtau_fit_summary: dict | None = None
    try:
        f_gamma_map = _xtau_load_exp041_f_gamma()
    except SystemExit as e:
        print(f"  [skip xtau] {e}")
        f_gamma_map = None
    if f_gamma_map is not None:
        for tau in XTAU_TAU_GABAS_MS:
            for seed in XTAU_SEEDS:
                train_dir = _xtau_exp041_cell_dir(tau, seed)
                if not (train_dir / "weights.pth").exists():
                    print(f"    [skip] missing {train_dir.relative_to(REPO)}")
                    continue
                f_gamma = f_gamma_map.get((tau, seed))
                if f_gamma is None or f_gamma <= 0:
                    print(f"    [skip] no f_γ for τ={tau} seed={seed}")
                    continue
                cycle_ms = 1000.0 / f_gamma
                for sigma in XTAU_SIGMAS_MS:
                    t0 = time.monotonic()
                    res = _xtau_evaluate_cell(
                        train_dir, sigma, cycle_ms,
                        seed_offset=seed + int(sigma),
                    )
                    res["seed"] = seed
                    res["f_gamma_hz"] = f_gamma
                    xtau_rows.append(res)
                    print(
                        f"    τ={tau:>5.1f}ms seed={seed} σ={sigma:>5.1f}ms  "
                        f"acc={res['acc']:5.2f}%  E={res['e_rate_hz']:5.2f}Hz  "
                        f"({time.monotonic() - t0:.1f}s)"
                    )
        plot_xtau_raw_sweeps(
            xtau_rows, FIGURES / "xtau_raw_sweeps", notebook_run_id,
        )
        print(f"wrote {FIGURES / 'xtau_raw_sweeps'}")
        plot_xtau_dimensional_collapse(
            xtau_rows, FIGURES / "xtau_dimensional_collapse",
            notebook_run_id,
        )
        print(f"wrote {FIGURES / 'xtau_dimensional_collapse'}")
        xtau_fit_summary = plot_xtau_inflection_vs_period(
            xtau_rows, FIGURES / "xtau_inflection_vs_period",
            notebook_run_id,
        )
        print(
            f"wrote {FIGURES / 'xtau_inflection_vs_period'}  "
            f"(α = {xtau_fit_summary['alpha']:.3f}, "
            f"R² = {xtau_fit_summary['r2']:.3f})"
        )

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "config": {
            "seeds": list(SEEDS),
            "conditions": list(CONDITIONS),
            "jitter_sigmas_ms": list(JITTER_SIGMAS_MS),
            "f_gamma_reference_hz": F_GAMMA_REFERENCE_HZ,
            "mix_alpha_grid": list(MIX_ALPHA_GRID),
            "mix_k_grid": list(MIX_K_GRID),
            "xtau_tau_gabas_ms": list(XTAU_TAU_GABAS_MS),
            "xtau_seeds": list(XTAU_SEEDS),
            "xtau_sigmas_ms": list(XTAU_SIGMAS_MS),
            "exp025_source": "ping__off__seed{seed} (θ_u = off baseline)",
            "exp041_source": "ping__tg{N}__seed{S} (100-epoch baselines)",
            "raster_sample_idx": RASTER_SAMPLE_IDX,
        },
        "results": rows,
        "jitter_sweep": jitter_rows,
        "cell_jitter_sweep": cell_jitter_rows,
        "pareto_sweep": pareto_rows,
        "pareto_summary": pareto_summary,
        "cross_tau_jitter": {
            "fit": xtau_fit_summary,
            "results": xtau_rows,
        },
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")



if __name__ == "__main__":
    main()
