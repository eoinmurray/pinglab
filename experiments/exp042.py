"""Notebook runner for entry 042 — rhythm vs mean-inhibition control.

Pure inference on the exp022 TR-02 PING baseline. Three conditions
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

Writing: writings/exp042.typ · figures + numbers.json: .artifacts/exp042/
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:  # torch is imported lazily inside the functions at runtime
    import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp022 import training_run_cell, training_run_values  # noqa: E402
from helpers import (
    runpod,  # noqa: E402
    theme,  # noqa: E402
)
from helpers.checkpoints import (  # noqa: E402
    cache_tag,
    checkpoint_policy,
    checkpoint_provenance,
    resolve_checkpoint,
)
from helpers.cli import Meta, parse_meta  # noqa: E402
from helpers.datasets import (  # noqa: E402
    MNIST_REDUCED_EVAL_SAMPLES,
    load_mnist_split,
)
from helpers.figsave import save_figure  # noqa: E402
from helpers.fmt import format_duration  # noqa: E402
from helpers.operating_point import F_GAMMA_HZ  # noqa: E402
from helpers.paths import (  # noqa: E402
    artifacts_and_figures,
    log_runner_event,
    runner_paths,
)
from helpers.run_cli import run_cli  # noqa: E402
from helpers.run_dirs import (
    finalize_prepared_run,  # noqa: E402
    preserve_active_view,  # noqa: E402
)
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "exp042"
RUN_PATHS = runner_paths(SLUG)
_ARTIFACTS_DEFAULT, FIGURES = artifacts_and_figures(SLUG)
ARTIFACTS = (
    RUN_PATHS.state if RUN_PATHS.isolated else runpod.artifacts_scratch(SLUG)
)
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"
ANALYSIS_PURPOSE = "endpoint_dynamics"
CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]
EVAL_SEED = 20260415  # mirror cli.encoders.EVAL_SEED (kept in sync by hand)

TRAINING_ROOT = runpod.training_root()
EXP022_TRAINING_ROOT = TRAINING_ROOT
SEEDS: tuple[int, ...] = training_run_values("TR-02", "seed")

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
OBSOLETE_OUTPUT_STEMS: tuple[str, ...] = (
    "bar_chart",
    "raster_strip",
    "cell_jitter_raster_strip",
    "jitter_raster_strip",
    "pareto_raster_strip",
    "pareto_sweep",
    "xtau_raw_sweeps",
    "xtau_dimensional_collapse",
    "xtau_inflection_vs_period",
)
# Raster panel: one trial per condition, MNIST digit 0 sample 0 — same
# convention as exp025/exp037 so the panels read against existing figures.
RASTER_SAMPLE_IDX: int = 0
RASTER_N_E_PLOT: int = 200
RASTER_N_I_PLOT: int = 64

SMOKE = os.environ.get("PINGLAB_SMOKE") == "1"
SMOKE_MAX_SAMPLES = 100
EVAL_MAX_SAMPLES = (
    SMOKE_MAX_SAMPLES if SMOKE else MNIST_REDUCED_EVAL_SAMPLES
)
if SMOKE:
    # Keep every anchor interpolated by the entry/manuscript while dropping the
    # dense production-only points between them.
    JITTER_SIGMAS_MS = (0.0, 14.0, 100.0)
    CELL_JITTER_SIGMAS_MS = (0.0, 0.5, 1.0, 2.0, 5.0, 9.0, 14.0)

# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
# exp042 is inference-only against the exp022 TR-02 PING baseline, so the
# dataset / max_samples / t_ms / dt_ms are inherited from each cell's own
# config.json at run time. What this runner declares is the evaluation grid:
# how many seeds and cells it sweeps, and the perturbation grids.
SCALE = {
    "dataset": "mnist",
    "max_samples": EVAL_MAX_SAMPLES,
    "seeds": len(SEEDS),
    "cells": len(CONDITIONS),
    "grid": (
        f"jitter σ ×{len(JITTER_SIGMAS_MS)}, "
        f"cell-jitter σ ×{len(CELL_JITTER_SIGMAS_MS)}"
    ),
}


# ─── trained-network loading (mirrors exp037 helper) ─────────────────


# ─── CLI-backed baseline + override (net execution runs in the CLI) ──────


def _load_eval(train_dir: Path):
    """Config + held-out MNIST test split for a trained cell (no net)."""
    cfg = json.loads((train_dir / "config.json").read_text())
    _, X_te, _, y_te = load_mnist_split(max_samples=int(cfg["max_samples"]))
    return cfg, X_te, y_te


def checkpoint_path(train_dir: Path) -> Path:
    return resolve_checkpoint(train_dir, CHECKPOINT_ROLE)["path"]


_BASE_CACHE: dict = {}


def _baseline_complete(rasters_path: Path, metrics_path: Path) -> bool:
    """True iff a finished baseline (raster + metrics) is already on disk and
    loadable. Lets all-but-the-first sharer of a train_dir reuse it — see
    _run_baseline."""
    if not (rasters_path.exists() and metrics_path.exists()):
        return False
    try:
        json.loads(metrics_path.read_text())
        with np.load(rasters_path):
            pass
    except Exception:  # noqa: BLE001 — a torn/legacy file counts as incomplete
        return False
    return True


def _run_baseline(train_dir: Path, tau_gaba=None):
    """Baseline pass via `sim --infer --outputs rasters`; return (metrics, rasters).
    Cached per (cell, τ_GABA) — the baseline I-stream is condition-independent.

    The ~800 MB raster is also cached on the shared volume so a re-fire, or a
    sibling pod running another condition of the same cell, REUSES it instead of
    recomputing. Under the 1-job-per-pod fan-out the busiest cell is shared by
    ~46 conditions on ~46 separate pods; without this every one would recompute
    and concurrently clobber the same file. The compute writes to a private temp
    dir and is published with os.replace (atomic on one filesystem), so a
    concurrent reader never sees a half-written raster."""
    checkpoint = resolve_checkpoint(train_dir, CHECKPOINT_ROLE)
    key = f"{train_dir}|{tau_gaba}|{cache_tag(checkpoint)}"
    if key not in _BASE_CACHE:
        out_dir = (ARTIFACTS / "baseline" / train_dir.name / cache_tag(checkpoint)).resolve()
        rasters_path = out_dir / "rasters.npz"
        metrics_path = out_dir / "metrics.json"
        if not _baseline_complete(rasters_path, metrics_path):
            out_dir.mkdir(parents=True, exist_ok=True)
            # A unique temp dir per WRITER: two pods are separate containers and
            # can share a PID, so a pid-named dir on the shared volume would
            # collide. mkdtemp guarantees uniqueness across pods.
            tmp = Path(tempfile.mkdtemp(
                prefix=f".{train_dir.name}.tmp.", dir=out_dir.parent))
            try:
                cmd = [
                    "sim", "--infer",
                    "--load-config", str((train_dir / "config.json").resolve()),
                    "--load-weights", str(checkpoint_path(train_dir)),
                    "--outputs", "rasters", "--out-dir", str(tmp),
                ]
                if tau_gaba is not None:
                    cmd += ["--tau-gaba", str(tau_gaba)]
                cmd += ["--max-samples", str(EVAL_MAX_SAMPLES)]
                run_cli(cmd)
                # Publish atomically; metrics last so _baseline_complete only
                # passes once both files are live.
                os.replace(tmp / "rasters.npz", rasters_path)
                os.replace(tmp / "metrics.json", metrics_path)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        m = json.loads(metrics_path.read_text())
        R = dict(np.load(metrics_path.parent / "rasters.npz"))
        _BASE_CACHE[key] = (m, R)
    return _BASE_CACHE[key]


def _build_override_file(R, condition, gen, dt_ms, out_path):
    """Build a sparse I-override NPZ from baseline rasters R by applying the pure
    _build_override transform per trial (per-trial independent). The transform stays
    in the notebook; the CLI only injects the result."""
    import torch
    T, n_i, n_tr = int(R["T"]), int(R["n_i"]), int(R["n_trials"])
    tr = R["i_trial"]
    order = np.argsort(tr, kind="stable")
    tr, tt, tc = tr[order], R["i_t"][order], R["i_cell"][order]
    bounds = np.searchsorted(tr, np.arange(n_tr + 1))
    out_tr, out_t, out_c = [], [], []
    for b in range(n_tr):
        lo, hi = bounds[b], bounds[b + 1]
        s_i = np.zeros((T, 1, n_i), dtype=np.float32)
        s_i[tt[lo:hi], 0, tc[lo:hi]] = 1.0
        ov = _build_override(torch.from_numpy(s_i), condition, gen, dt_ms=dt_ms)
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
    checkpoint = resolve_checkpoint(train_dir, CHECKPOINT_ROLE)
    out_dir = (
        ARTIFACTS / "ovrun" / f"{train_dir.name}__{override_path.stem}"
        / cache_tag(checkpoint)
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sim", "--infer",
        "--load-config", str((train_dir / "config.json").resolve()),
        "--load-weights", str(checkpoint_path(train_dir)),
        "--i-override-file", str(override_path), "--out-dir", str(out_dir),
    ]
    if tau_gaba is not None:
        cmd += ["--tau-gaba", str(tau_gaba)]
    cmd += ["--max-samples", str(EVAL_MAX_SAMPLES)]
    run_cli(cmd)
    return json.loads((out_dir / "metrics.json").read_text())


def _cached_condition_metrics(train_dir: Path, condition: str, seed_offset: int):
    """Read a previously-computed metrics.json for one (cell, condition) off disk,
    WITHOUT re-running the sim — for --skip-training builds over collected data.
    Returns the metrics dict, or None if not present. Reads the FINAL output only
    (does NOT need the excluded baseline rasters / override npz).

    Paths mirror exactly what the compute path writes:
      - baseline → ARTIFACTS / "baseline" / <cell> / "metrics.json"  (_run_baseline)
      - otherwise → ARTIFACTS / "ovrun" / "<cell>__<cell>_<cond>_<off>" / "metrics.json"
        (evaluate_condition's ov_path.stem fed to _run_with_override; identical to
        _job_metrics_path's ov_stem).
    """
    tag = cache_tag(resolve_checkpoint(train_dir, CHECKPOINT_ROLE))
    if condition == "baseline":
        path = ARTIFACTS / "baseline" / train_dir.name / tag / "metrics.json"
    else:
        ov_stem = f"{train_dir.name}_{condition}_{seed_offset}"
        path = ARTIFACTS / "ovrun" / f"{train_dir.name}__{ov_stem}" / tag / "metrics.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


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
) -> "torch.Tensor":
    """Construct the I-spike override tensor for one batch.

    s_i_base: (T, B, N_I) baseline recorded I-spikes.
    Returns (T, B, N_I) override tensor preserving per-(trial, cell)
    spike counts in expectation.

    Conditions:
      - phase_shuffled_i: permute time axis per trial (all I cells share permutation)
      - poisson_matched_i: per-(trial, cell) Bernoulli at matched mean rate
      - jitter_sigma_{X}: cycle-coherent Gaussian jitter with σ = X ms.
        Uses F_GAMMA_REFERENCE_HZ as the cycle period.
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
        out = _jitter_i_stream(s_i_base, sigma_ms, dt_ms, generator)
    elif condition.startswith("cell_jitter_sigma_"):
        sigma_ms = float(condition.split("_")[-1])
        out = _cell_jitter_i_stream(s_i_base, sigma_ms, dt_ms, generator)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return out


def _jitter_i_stream(
    s_i_base: "torch.Tensor", sigma_ms: float, dt_ms: float, generator,
) -> "torch.Tensor":
    """Cycle-coherent jitter on the I-spike stream.

    Bins time into blocks of one gamma cycle (1 / F_GAMMA_REFERENCE_HZ
    ≈ 28 ms at the trained operating point), draws one Gaussian offset
    Δ ~ 𝒩(0, σ²) per
    (trial, cycle), and shifts every I-spike in that block by Δ.
    Within-burst cross-cell synchrony is preserved exactly; what's
    perturbed is the *placement* of each burst relative to where the
    baseline cycle put it.

    The diagnostic prediction: rate release should be small when
    σ ≪ 1/f_γ (bursts barely move from their phase-locked slots) and
    large when σ ≳ 1/f_γ (bursts can land anywhere within the cycle,
    losing phase relation to E).

    σ in milliseconds; the conversion to timesteps uses dt_ms.
    """
    import torch

    T, B, N_I = s_i_base.shape
    if sigma_ms <= 0.0:
        return s_i_base.clone()

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


# ─── per-condition evaluation ───────────────────────────────────────


def evaluate_condition(
    train_dir: Path, condition: str, seed_offset: int = 0, reuse: bool = False,
) -> dict:
    """Accuracy + E/I rate for one I-override condition, via the CLI.

    Two passes: baseline (`--outputs rasters`, cached) supplies the I-stream, the
    notebook builds the override with its pure transforms, then `--i-override-file`
    replays it. baseline condition just reads the baseline metrics.

    reuse=True (--skip-training over collected data): read the already-computed
    metrics.json off disk and skip the sim entirely; fall through to compute on a
    cache miss.
    """
    if reuse:
        cached = _cached_condition_metrics(train_dir, condition, seed_offset)
        if cached is not None:
            return _pack_metrics(cached, condition)
    import torch
    cfg, _, _ = _load_eval(train_dir)
    m0, R = _run_baseline(train_dir)
    if condition == "baseline":
        return _pack_metrics(m0, condition)
    gen = torch.Generator().manual_seed(EVAL_SEED + 17 + seed_offset)
    stem = f"{train_dir.name}_{condition}_{seed_offset}"
    temp_root = (ARTIFACTS / ".override-tmp").resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{stem}.", dir=temp_root))
    ov_path = temp_dir / f"{stem}.npz"
    try:
        _build_override_file(
            R, condition, gen, float(cfg["dt"]), ov_path,
        )
        return _pack_metrics(_run_with_override(train_dir, ov_path), condition)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _snapshot(train_dir: Path, sample_idx: int, name: str, i_override=None, reuse=False):
    """Single-trial snapshot via `sim --infer --sample-index N` (optional
    --i-override-file); return the loaded snapshot.npz dict.

    reuse=True: read the already-collected snapshot.npz from the same out_dir the
    compute path writes to, WITHOUT running the sim. Returns None on a cache miss
    so callers can fall through to compute."""
    checkpoint = resolve_checkpoint(train_dir, CHECKPOINT_ROLE)
    out_dir = (
        ARTIFACTS / "condraster" / f"{train_dir.name}_{name}"
        / cache_tag(checkpoint)
    ).resolve()
    if reuse:
        try:
            return np.load(out_dir / "snapshot.npz")
        except (OSError, ValueError):
            return None
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "python", str(SNN_TOOL), "sim", "--infer",
        "--load-config", str((train_dir / "config.json").resolve()),
        "--load-weights", str(checkpoint_path(train_dir)),
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
    seed_offset: int = 0, reuse: bool = False,
) -> dict:
    """Single-trial raster under one I-override condition, via the CLI snapshot.

    Baseline snapshot supplies the trial's I-stream; the notebook builds the
    override and a second snapshot replays it under --i-override-file.

    reuse=True (--skip-training over collected data): load the FINAL collected
    snapshot.npz directly and skip the baseline pass + override build (their
    intermediates were excluded from collect); fall through to compute on a miss.
    """
    cfg = json.loads((train_dir / "config.json").read_text())

    d = None
    if reuse:
        name = (
            f"base_s{sample_idx}" if condition == "baseline"
            else f"{condition}_s{sample_idx}"
        )
        d = _snapshot(train_dir, sample_idx, name, reuse=True)

    if d is None:
        d0 = _snapshot(train_dir, sample_idx, f"base_s{sample_idx}")
        if condition == "baseline":
            d = d0
        else:
            import torch
            s_i = d0["spk_i"]
            if s_i.ndim == 3:
                s_i = s_i[:, 0, :]
            gen = torch.Generator().manual_seed(EVAL_SEED + 17 + seed_offset)
            ov = _build_override(
                torch.from_numpy(s_i[:, None, :].astype(np.float32)),
                condition, gen, dt_ms=float(cfg["dt"]),
            ).detach().cpu().numpy()[:, 0, :]  # (T, n_i)
            ti, ci = ov.nonzero()
            temp_root = (ARTIFACTS / ".override-tmp").resolve()
            temp_root.mkdir(parents=True, exist_ok=True)
            stem = f"{train_dir.name}_{condition}_s{sample_idx}"
            temp_dir = Path(tempfile.mkdtemp(prefix=f".{stem}.", dir=temp_root))
            ov_path = temp_dir / f"{stem}.npz"
            try:
                np.savez(
                    ov_path, n_trials=np.int32(1), T=np.int32(ov.shape[0]),
                    n_i=np.int32(ov.shape[1]),
                    i_trial=np.zeros(ti.size, "int32"),
                    i_t=ti.astype("int32"), i_cell=ci.astype("int32"),
                )
                d = _snapshot(
                    train_dir, sample_idx, f"{condition}_s{sample_idx}",
                    i_override=ov_path,
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

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




# σ values to render in the jitter-raster strip — diagnostic subset that
# spans the predicted transition at 1/f_γ ≈ 28 ms.




def plot_cell_jitter_sweep(
    cell_rows: list[dict], out_path: Path, run_id: str,
) -> None:
    """Per-I-cell jitter sweep — E rate, accuracy, and realised I rate.

    Same twin-axis layout and grey realised-I trace as plot_jitter_sweep, for
    the per-spike jitter family.
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

    i_means = [
        float(np.mean([r["i_rate_hz"] for r in by_sigma[s]])) for s in sigmas_sorted
    ]

    fig, ax_rate = plt.subplots(figsize=(5.6, 3.11))
    ax_rate.errorbar(
        sigmas_sorted, e_means, yerr=e_sems,
        marker="D", markersize=6, lw=1.4, color=theme.INK_BLACK, capsize=3,
        label="E rate (Hz)",
    )
    # Realised mean I rate — the "held fixed" control, same grey full-trace styling
    # as the cycle-coherent sweep. Per-cell jitter only moves each spike by a small
    # independent offset, so it stays flat near baseline through the E collapse.
    ax_rate.plot(
        sigmas_sorted, i_means, marker="o", markersize=6, lw=1.4,
        color=theme.GREY_MID, label="realised I rate (Hz)",
    )
    # Symlog x-axis (linthresh matched to plot_jitter_sweep) so the per-cell
    # collapse — all of which happens below σ ≈ 9 ms — spreads across the plot
    # instead of piling into the left margin, and the two paired sweep figures
    # share one x-scale for direct comparison.
    ax_rate.set_xscale("symlog", linthresh=1.0)
    ax_rate.set_xlabel(
        "Per-I-cell jitter σ on the I-stream (ms, symlog)",
        fontsize=theme.SIZE_LABEL,
    )
    ax_rate.set_ylabel("Firing rate (Hz)",
                       fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK)
    ax_rate.tick_params(axis="y", labelcolor=theme.INK_BLACK)

    ax_acc = ax_rate.twinx()
    ax_acc.errorbar(
        sigmas_sorted, acc_means, yerr=acc_sems,
        marker="s", markersize=6, lw=1.4, color=theme.DEEP_RED, capsize=3,
        label="Test accuracy (%)",
    )
    ax_acc.set_ylabel("Test accuracy (%)",
                      fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED)
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)

    # Self-identify all three traces (twin-axis colours alone don't survive
    # greyscale print): a single legend combining both axes' handles, replacing
    # the earlier inline grey-only label and activating the previously-unused
    # label= kwargs.
    h_rate, l_rate = ax_rate.get_legend_handles_labels()
    h_acc, l_acc = ax_acc.get_legend_handles_labels()
    ax_rate.legend(
        h_rate + h_acc, l_rate + l_acc,
        loc="center right", frameon=False, fontsize=theme.SIZE_LEGEND,
    )

    # H17: caption carries the takeaway
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)




def plot_jitter_sweep(
    jitter_rows: list[dict], out_path: Path, run_id: str,
) -> None:
    """E rate, accuracy, and realised I rate vs cycle-coherent jitter σ.

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

    i_means = [
        float(np.mean([r["i_rate_hz"] for r in by_sigma[s]])) for s in sigmas_sorted
    ]

    fig, ax_rate = plt.subplots(figsize=(5.6, 3.11))
    # Use a symlog x-axis so both σ = 0 and σ = 100 are visible.
    ax_rate.errorbar(
        sigmas_sorted, e_means, yerr=e_sems,
        marker="D", markersize=6, lw=1.4, color=theme.INK_BLACK, capsize=3,
        label="E rate (Hz)",
    )
    # Realised mean I rate — the "held fixed" control, same grey full-trace styling
    # as the per-cell sweep. Flat near baseline over the rate-matched range; droops at
    # large σ where the Gaussian block offset displaces part of each burst past the
    # trial window (see Methods note).
    ax_rate.plot(
        sigmas_sorted, i_means, marker="o", markersize=6, lw=1.4,
        color=theme.GREY_MID, label="realised I rate (Hz)",
    )
    ax_rate.set_xscale("symlog", linthresh=1.0)
    ax_rate.set_xlabel(
        "Cycle-coherent jitter σ on the I-stream (ms, symlog)",
        fontsize=theme.SIZE_LABEL,
    )
    ax_rate.set_ylabel("Firing rate (Hz)",
                       fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK)
    ax_rate.tick_params(axis="y", labelcolor=theme.INK_BLACK)

    ax_acc = ax_rate.twinx()
    ax_acc.errorbar(
        sigmas_sorted, acc_means, yerr=acc_sems,
        marker="s", markersize=6, lw=1.4, color=theme.DEEP_RED, capsize=3,
        label="Test accuracy (%)",
    )
    ax_acc.set_ylabel("Test accuracy (%)",
                      fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED)
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)

    # Self-identify all three traces with one legend combining both axes'
    # handles (replaces the inline grey-only label; matches plot_cell_jitter_sweep).
    h_rate, l_rate = ax_rate.get_legend_handles_labels()
    h_acc, l_acc = ax_acc.get_legend_handles_labels()
    ax_rate.legend(
        h_rate + h_acc, l_rate + l_acc,
        loc="center left", frameon=False, fontsize=theme.SIZE_LEGEND,
    )

    # H17: caption carries the takeaway
    ax_rate.spines["top"].set_visible(False)
    ax_acc.spines["top"].set_visible(False)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)




# ─── rhythm-vs-mean compound (the manuscript figure) ────────────────


def checkpoint_source_dirs() -> dict[str, list[Path]]:
    """Hard checkpoint inputs, grouped by their owning experiment and TR."""
    return {
        "exp022_tr02": [
            EXP022_TRAINING_ROOT
            / training_run_cell(
                "TR-02", model="ping", rate_target_hz=None, seed=seed,
            )["name"]
            for seed in SEEDS
        ],
    }


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
        # Opaque backing so the annotation reads over the dense I-raster instead
        # of crowding into the red spikes.
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="none", alpha=0.92),
    )
    _despine(ax)


def _compound_sweep_panel(
    ax, rows: list[dict], *, xlabel: str, title: str,
    symlog: bool, legend_loc: str,
) -> None:
    """One sweep panel — E rate (black) and accuracy (red, twin axis) vs σ."""
    by_sigma: dict[float, list[dict]] = {}
    for r in rows:
        by_sigma.setdefault(r["sigma_ms"], []).append(r)
    sig = sorted(by_sigma)
    e_means = [float(np.mean([r["e_rate_hz"] for r in by_sigma[s]])) for s in sig]
    i_means = [float(np.mean([r["i_rate_hz"] for r in by_sigma[s]])) for s in sig]
    a_means = [float(np.mean([r["acc"] for r in by_sigma[s]])) for s in sig]

    ax.plot(sig, e_means, marker="D", ms=5, lw=1.4, color=theme.INK_BLACK,
            label="E rate")
    # Realised (measured) mean I rate on the same Hz axis — makes the "mean
    # inhibition held fixed" control visible directly. It sits flat near
    # baseline over the rate-matched range and only droops where the finite
    # trial window truncates the displaced-burst tail (cycle-coherent, large σ).
    ax.plot(sig, i_means, marker=".", ms=4, lw=1.0, color=theme.GREY_MID,
            ls="-", alpha=0.75, label="realised I")
    if symlog:
        ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel(xlabel, fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("firing rate (Hz)", color=theme.INK_BLACK,
                  fontsize=theme.SIZE_LABEL)
    ax.tick_params(axis="y", labelcolor=theme.INK_BLACK)
    ax.set_title(title, fontsize=theme.SIZE_LABEL)

    ax_acc = ax.twinx()
    ax_acc.plot(sig, a_means, marker="s", ms=5, lw=1.4, color=theme.DEEP_RED,
                label="accuracy")
    ax_acc.set_ylabel("accuracy (%)", color=theme.DEEP_RED, fontsize=theme.SIZE_LABEL)
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax_acc.spines["top"].set_visible(False)

    # Self-identify all three traces; combine both axes' handles into one legend.
    h_rate, l_rate = ax.get_legend_handles_labels()
    h_acc, l_acc = ax_acc.get_legend_handles_labels()
    ax.legend(h_rate + h_acc, l_rate + l_acc, loc=legend_loc,
              frameon=False, fontsize=theme.SIZE_ANNOTATION)


def fig_rhythm_compound(
    cyc_rows: list[dict], cell_rows: list[dict],
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
        xlabel="per-I-cell jitter σ (ms, symlog)",
        title="Smear bursts → E rate falls to zero",
        # Symlog to match the cycle-coherent panel (and the standalone sweeps):
        # the per-cell collapse all happens below σ ≈ 9 ms and would otherwise
        # pile into the left margin, breaking the side-by-side read.
        symlog=True, legend_loc="center right",
    )
    _compound_sweep_panel(
        axes[1, 1], cyc_rows,
        xlabel="cycle-coherent jitter σ (ms, symlog)",
        title="Displace bursts → E rate rises",
        symlog=True, legend_loc="center left",
    )
    # H17: caption carries the takeaway
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense rasters: PNG, not SVG
    plt.close(fig)
    plt.rcParams["savefig.bbox"] = prev_bbox


def build_rhythm_compound(run_id: str | None = None) -> None:
    """Rebuild the compound from cached numbers.json — no sweep re-runs.

    Sweep curves load from numbers.json; the two example rasters are cheap
    single-trial forward passes against the cached exp022 TR-02 PING weights.
    ``run_id`` defaults to the cached notebook_run_id so the corner stamp stays
    consistent with the committed run rather than reading "replot".
    """
    data = json.loads((FIGURES / "numbers.json").read_text())
    if run_id is None:
        run_id = data["notebook_run_id"]
    cyc_rows = data["jitter_sweep"]
    cell_rows = data["cell_jitter_sweep"]

    seed = int(data["config"]["seeds"][0])
    train_dir = EXP022_TRAINING_ROOT / f"ping__off__seed{seed}"
    # Same jitter magnitude on both arms — σ = 14 ms. See the note in main() on why
    # this σ (a shared measured grid point), not σ = 100 ms, is the headline raster.
    raster_cyc = capture_condition_raster(
        train_dir, "jitter_sigma_14", RASTER_SAMPLE_IDX,
        seed_offset=seed + 14,
    )
    raster_cyc["sigma_ms"] = 14.0
    raster_cell = capture_condition_raster(
        train_dir, "cell_jitter_sigma_14", RASTER_SAMPLE_IDX,
        seed_offset=seed + int(14 * 13),
    )
    raster_cell["sigma_ms"] = 14.0

    fig_rhythm_compound(
        cyc_rows, cell_rows, raster_cyc, raster_cell,
        FIGURES / "rhythm_compound", run_id,
    )
    print(f"wrote {FIGURES / 'rhythm_compound'}")


def replot_sweeps_from_cache(which: str | None = None) -> None:
    """Redraw the numbers.json-only sweep figures without re-running inference.

    ``which`` selects one figure by name (``jitter_sweep`` or
    ``cell_jitter_sweep``); ``None`` redraws both. Each is stamped with the
    cached ``notebook_run_id`` so the corner tag and provenance stay consistent
    with the committed run — this only re-renders the plot, the numbers are
    untouched.
    """
    data = json.loads((FIGURES / "numbers.json").read_text())
    run_id = data["notebook_run_id"]
    targets = {
        "jitter_sweep": lambda: plot_jitter_sweep(
            data["jitter_sweep"], FIGURES / "jitter_sweep", run_id),
        "cell_jitter_sweep": lambda: plot_cell_jitter_sweep(
            data["cell_jitter_sweep"], FIGURES / "cell_jitter_sweep", run_id),
    }
    if which is not None and which not in targets:
        raise SystemExit(
            f"--plot-only {which}: not a cache-redrawable figure; choose from "
            f"{sorted(targets)} or 'compound'"
        )
    for name, draw in targets.items():
        if which in (None, name):
            draw()
            print(f"wrote {FIGURES / name}")


# ── RunPod infer jobs ────────────────────────────────────────────────

_JOB_CATALOG: list[dict] | None = None


def _enc_job_token(text: str) -> str:
    return re.sub(r"(\d)\.(\d)", r"\1p\2", text)


def _job_id(kind: str, *parts: str) -> str:
    return kind + "__" + "__".join(_enc_job_token(p) for p in parts)


def _job_catalog() -> list[dict]:
    global _JOB_CATALOG
    if _JOB_CATALOG is not None:
        return _JOB_CATALOG
    catalog: list[dict] = []
    for seed in SEEDS:
        train_dir = TRAINING_ROOT / f"ping__off__seed{seed}"
        for cond in CONDITIONS:
            catalog.append({
                "id": _job_id("eval", train_dir.name, cond),
                "run": lambda td=train_dir, c=cond, s=seed: evaluate_condition(
                    td, c, seed_offset=s),
                "train_dir": train_dir,
                "condition": cond,
                "seed_offset": seed,
            })
        for sigma_ms in JITTER_SIGMAS_MS:
            cond = f"jitter_sigma_{sigma_ms:g}"
            off = seed + int(sigma_ms)
            catalog.append({
                "id": _job_id("eval", train_dir.name, cond),
                "run": lambda td=train_dir, c=cond, o=off: evaluate_condition(
                    td, c, seed_offset=o),
                "train_dir": train_dir, "condition": cond, "seed_offset": off,
            })
        for sigma_ms in CELL_JITTER_SIGMAS_MS:
            cond = f"cell_jitter_sigma_{sigma_ms:g}"
            off = seed + int(sigma_ms * 13)
            catalog.append({
                "id": _job_id("eval", train_dir.name, cond),
                "run": lambda td=train_dir, c=cond, o=off: evaluate_condition(
                    td, c, seed_offset=o),
                "train_dir": train_dir, "condition": cond, "seed_offset": off,
            })
    _JOB_CATALOG = catalog
    return catalog


def infer_jobs() -> list[str]:
    return [j["id"] for j in _job_catalog()]


PLUMBING_JOBS = [
    _job_id("eval", "ping__off__seed42", "baseline"),
    _job_id("eval", "ping__off__seed42", "jitter_sigma_7.0"),
]


def _job_spec(job_id: str) -> dict:
    for spec in _job_catalog():
        if spec["id"] == job_id:
            return spec
    raise KeyError(job_id)


def _job_metrics_path(spec: dict) -> Path:
    train_dir = spec["train_dir"]
    condition = spec["condition"]
    seed_offset = spec["seed_offset"]
    tag = cache_tag(resolve_checkpoint(train_dir, CHECKPOINT_ROLE))
    if condition == "baseline":
        return ARTIFACTS / "baseline" / train_dir.name / tag / "metrics.json"
    ov_stem = f"{train_dir.name}_{condition}_{seed_offset}"
    return (
        ARTIFACTS
        / "ovrun"
        / f"{train_dir.name}__{ov_stem}"
        / tag
        / "metrics.json"
    )


def job_is_done(job_id: str) -> bool:
    try:
        spec = _job_spec(job_id)
    except KeyError:
        return False
    path = _job_metrics_path(spec)
    return path.exists()


def run_infer_job(job_id: str) -> None:
    spec = _job_spec(job_id)
    spec["run"]()


def pod_run() -> None:
    runpod.pod_run_loop(job_ids=infer_jobs(), is_done=job_is_done, run_job=run_infer_job)


def run_via_runpod(meta: Meta) -> None:
    jobs = list(PLUMBING_JOBS if meta.plumbing else infer_jobs())
    if meta.only_cells:
        wanted = set(meta.only_cells)
        jobs = [j for j in jobs if j in wanted]
        missing = wanted - set(jobs)
        if missing:
            raise SystemExit(f"unknown job(s): {sorted(missing)}")
    runpod.dispatch(
        slug=SLUG, runner=SLUG,
        buckets=runpod.chunk_buckets(jobs, meta.cells_per_pod, prefix="infer42"),
        gpu=meta.gpu, live=meta.live, plumbing=meta.plumbing, collect=meta.collect,
        collect_subdir=f"{runpod.ARTIFACTS_SUBDIR}/{SLUG}",
        local_collect_dir=str(runpod.artifacts_scratch(SLUG)),
        extra_env={
            "PINGLAB_ARTIFACTS_ROOT": f"{runpod.VOLUME_MOUNT}/{runpod.ARTIFACTS_SUBDIR}/{SLUG}",
            "PINGLAB_NO_SYNC": "1",
        },
    )


# ─── success criteria ───────────────────────────────────────────────

def _cleanup_successful_intermediates() -> None:
    """Remove reproducible tensor scratch after final outputs are complete.

    Baseline metrics and per-condition metrics remain as restart-safe completion
    records. Baseline rasters, condition snapshots, and override files are only
    process-boundary inputs and are not retained scientific evidence.
    """
    _BASE_CACHE.clear()
    baseline_root = ARTIFACTS / "baseline"
    if baseline_root.exists():
        for raster_path in baseline_root.rglob("rasters.npz"):
            raster_path.unlink()
    for scratch_dir in (
        ARTIFACTS / "condraster",
        ARTIFACTS / "override",
        ARTIFACTS / ".override-tmp",
    ):
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)


def _remove_obsolete_outputs() -> None:
    """Prevent removed diagnostics surviving a skip-training re-render."""
    for stem in OBSOLETE_OUTPUT_STEMS:
        for suffix in (".pdf", ".png", ".svg"):
            (FIGURES / f"{stem}{suffix}").unlink(missing_ok=True)


@preserve_active_view(SLUG)
def main() -> None:
    theme.set_paper_mode(True)

    meta = parse_meta(sys.argv, allow_dispatch=True)
    if meta.plot_only:
        if meta.plot_fig == "compound":
            build_rhythm_compound()
        else:
            replot_sweeps_from_cache(meta.plot_fig)
        return
    if meta.pod_run:
        pod_run()
        return
    if meta.reap:
        runpod.reap_all_pods()
        return
    if meta.runpod:
        run_via_runpod(meta)
        return

    if RUN_PATHS.isolated and not os.environ.get("PINGLAB_TRAINING_ROOT"):
        raise RuntimeError("isolated exp042 requires explicit PINGLAB_TRAINING_ROOT")

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} seeds={SEEDS}")
    log_runner_event(SLUG, "started", run_id=notebook_run_id)

    prepare_run_dirs(
        SLUG, notebook_run_id, wipe=not meta.skip_training, make_artifacts=True,
        scale=SCALE, host="local",
    )

    rows: list[dict] = []
    for seed in SEEDS:
        train_dir = EXP022_TRAINING_ROOT / f"ping__off__seed{seed}"
        if not (train_dir / "weights_final.pth").exists():
            raise SystemExit(
                f"missing exp022 TR-02 PING checkpoint at {train_dir} — "
                "run exp022 TR-02 first"
            )
        print(f"[eval] seed={seed} from {train_dir}")
        for cond in CONDITIONS:
            t0 = time.monotonic()
            res = evaluate_condition(
                train_dir, cond, seed_offset=seed, reuse=meta.skip_training,
            )
            res["seed"] = seed
            rows.append(res)
            print(
                f"    {cond:<22}  acc={res['acc']:5.2f}%  "
                f"E={res['e_rate_hz']:6.2f} Hz  I={res['i_rate_hz']:6.2f} Hz  "
                f"({time.monotonic() - t0:.1f}s)"
            )


    # ── Jitter sweep ───────────────────────────────────────────────
    # Adds Gaussian timing jitter σ to each I-spike at inference.
    # Predicts the rate-release transition at σ ≈ 1/f_γ ≈ 28 ms.
    print(f"[jitter] sweep σ ∈ {list(JITTER_SIGMAS_MS)} ms")
    jitter_rows: list[dict] = []
    for seed in SEEDS:
        train_dir = EXP022_TRAINING_ROOT / f"ping__off__seed{seed}"
        for sigma_ms in JITTER_SIGMAS_MS:
            cond = f"jitter_sigma_{sigma_ms:g}"
            t0 = time.monotonic()
            # Reuse evaluate_condition — it dispatches on the condition string.
            res = evaluate_condition(train_dir, cond,
                                     seed_offset=seed + int(sigma_ms),
                                     reuse=meta.skip_training)
            res["seed"] = seed
            res["sigma_ms"] = float(sigma_ms)
            jitter_rows.append(res)
            print(
                f"    σ={sigma_ms:>5.1f}ms seed={seed}  "
                f"acc={res['acc']:5.2f}%  E={res['e_rate_hz']:6.2f} Hz  "
                f"I={res['i_rate_hz']:6.2f} Hz  ({time.monotonic() - t0:.1f}s)"
            )

    plot_jitter_sweep(jitter_rows, FIGURES / "jitter_sweep", notebook_run_id)
    print(f"wrote {FIGURES / 'jitter_sweep'}")

    # ── Per-cell jitter sweep ──────────────────────────────────────
    # Independent Gaussian offset per spike — destroys within-burst
    # synchrony while preserving burst placement on average. Predicts
    # the rate-release transition at σ ≈ τ_GABA (the smearing width
    # at which the integrated g_i profile starts looking continuous).
    print(f"[cell-jitter] sweep σ ∈ {list(CELL_JITTER_SIGMAS_MS)} ms")
    cell_jitter_rows: list[dict] = []
    for seed in SEEDS:
        train_dir = EXP022_TRAINING_ROOT / f"ping__off__seed{seed}"
        for sigma_ms in CELL_JITTER_SIGMAS_MS:
            cond = f"cell_jitter_sigma_{sigma_ms:g}"
            t0 = time.monotonic()
            res = evaluate_condition(train_dir, cond,
                                     seed_offset=seed + int(sigma_ms * 13),
                                     reuse=meta.skip_training)
            res["seed"] = seed
            res["sigma_ms"] = float(sigma_ms)
            cell_jitter_rows.append(res)
            print(
                f"    σ={sigma_ms:>5.1f}ms seed={seed}  "
                f"acc={res['acc']:5.2f}%  E={res['e_rate_hz']:6.2f} Hz  "
                f"I={res['i_rate_hz']:6.2f} Hz  ({time.monotonic() - t0:.1f}s)"
            )

    plot_cell_jitter_sweep(
        cell_jitter_rows, FIGURES / "cell_jitter_sweep", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'cell_jitter_sweep'}")

    # Manuscript compound: matched mean I, opposite E response. Use the SAME jitter
    # magnitude on both arms — σ = 14 ms — so the figure reads as one manipulation
    # strength with opposite outcomes; only the KIND of jitter differs. σ = 14 ms is
    # a measured grid point on both sweeps where the per-cell arm has fully silenced E
    # and the cycle-coherent arm has raised it well above baseline, while realised I
    # still holds within a few percent on both. (σ = 100 ms would push realised I down
    # ~24% via finite-window truncation — that σ stays in the sweep panels below, not
    # as the headline raster.) Only these two illustrative snapshots are retained
    # long enough to render the compound figure.
    compound_sigma = 14.0
    raster_seed = SEEDS[0]
    raster_train_dir = EXP022_TRAINING_ROOT / f"ping__off__seed{raster_seed}"
    raster_cyc = capture_condition_raster(
        raster_train_dir, f"jitter_sigma_{compound_sigma:g}", RASTER_SAMPLE_IDX,
        seed_offset=raster_seed + int(compound_sigma), reuse=meta.skip_training,
    )
    raster_cyc["sigma_ms"] = compound_sigma
    raster_cell = capture_condition_raster(
        raster_train_dir, f"cell_jitter_sigma_{compound_sigma:g}", RASTER_SAMPLE_IDX,
        seed_offset=raster_seed + int(compound_sigma * 13), reuse=meta.skip_training,
    )
    raster_cell["sigma_ms"] = compound_sigma
    fig_rhythm_compound(
        jitter_rows, cell_jitter_rows, raster_cyc, raster_cell,
        FIGURES / "rhythm_compound", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'rhythm_compound'}")

    duration_s = time.monotonic() - t_start
    source_dirs = checkpoint_source_dirs()
    exp022_dirs = source_dirs["exp022_tr02"]
    source_provenance = {
        source: checkpoint_provenance(paths, CHECKPOINT_ROLE)
        for source, paths in source_dirs.items()
    }
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "checkpoint_policy": CHECKPOINT_POLICY,
        "checkpoint_provenance": checkpoint_provenance(
            exp022_dirs, CHECKPOINT_ROLE,
        ),
        "checkpoint_sources": source_provenance,
        "config": {
            "evaluation_samples_per_condition": EVAL_MAX_SAMPLES,
            "seeds": list(SEEDS),
            "conditions": list(CONDITIONS),
            "jitter_sigmas_ms": list(JITTER_SIGMAS_MS),
            "f_gamma_reference_hz": F_GAMMA_REFERENCE_HZ,
            "exp022_tr02_source": "ping__off__seed{seed}",
            "raster_sample_idx": RASTER_SAMPLE_IDX,
        },
        "results": rows,
        "jitter_sweep": jitter_rows,
        "cell_jitter_sweep": cell_jitter_rows,
    }
    (FIGURES / "numbers.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote {FIGURES / 'numbers.json'}")
    _remove_obsolete_outputs()
    _cleanup_successful_intermediates()
    print("removed exp042 tensor scratch; retained metrics and final outputs")
    print(f"  total duration: {summary['duration']}")
    log_runner_event(SLUG, "completed", run_id=notebook_run_id)
    finalize_prepared_run(SLUG, notebook_run_id)



if __name__ == "__main__":
    main()
