"""Notebook runner for entry 042 — rhythm vs mean-inhibition control.

Pure inference on the trained nb025 PING baseline. Three conditions
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

If the E rate stays clamped without the rhythm, art008's thesis
collapses to "inhibition lowers rates." If it shoots up toward COBA's
operating point, gamma is specifically what is doing the forbidding.

Notebook entry: src/docs/src/pages/notebooks/nb042.mdx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb042"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# nb025 trained PING baseline lives here. Three seeds available; the
# nb042 inference experiment runs against all three for error bars.
NB035_ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / "nb025"
SEEDS: tuple[int, ...] = (42, 43, 44)

CONDITIONS: tuple[str, ...] = ("baseline", "phase_shuffled_i", "poisson_matched_i")

# Jitter sweep — Gaussian timing jitter on each I-spike. σ in ms.
# 0 = baseline; well above the trained network's gamma period
# (≈ 28 ms at τ_GABA = 9 ms) the rate should approach the phase-shuffle
# release level. Predicted transition is at σ ≈ 1 / f_γ.
JITTER_SIGMAS_MS: tuple[float, ...] = (
    0.0, 1.0, 3.0, 7.0, 14.0, 21.0, 28.0, 42.0, 60.0, 100.0,
)
F_GAMMA_REFERENCE_HZ: float = 36.0   # trained nb025 PING f_γ at τ_GABA = 9 ms
# Inverse — annotated as the predicted inflection point on the sweep plot.
# 1 / 36 Hz ≈ 27.8 ms
CONDITION_LABELS = {
    "baseline": "baseline PING",
    "phase_shuffled_i": "phase-shuffled I",
    "poisson_matched_i": "rate-matched Poisson I",
}

# Raster panel: one trial per condition, MNIST digit 0 sample 0 — same
# convention as nb025/nb037 so the panels read against existing figures.
RASTER_SAMPLE_IDX: int = 0
RASTER_N_E_PLOT: int = 200
RASTER_N_I_PLOT: int = 64

TIER_CONFIG = {
    "extra small": dict(),
    "small": dict(),
    "medium": dict(),
    "large": dict(),
    "extra large": dict(),
}
DEFAULT_TIER = "medium"


# ─── trained-network loading (mirrors nb037 helper) ─────────────────


def _load_trained_full(train_dir: Path, device):
    """Load nb025 trained PING checkpoint. Returns (net, cfg, X_te, y_te)."""
    import torch

    import models as M
    from config import build_net, patch_dt
    from cli import load_dataset, seed_everything

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", 42)))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))
    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)

    _, X_te, _, y_te = load_dataset(
        cfg["dataset"], max_samples=int(cfg["max_samples"]), split=True
    )
    M.N_IN = 784 if cfg["dataset"] == "mnist" else 64

    w_in_cfg = cfg.get("w_in")
    w_in_arg = (
        (float(w_in_cfg[0]), float(w_in_cfg[1]))
        if isinstance(w_in_cfg, list) and len(w_in_cfg) >= 2
        else None
    )
    net = build_net(
        cfg["model"],
        w_in=w_in_arg,
        w_in_sparsity=float(cfg.get("w_in_sparsity") or 0.0),
        ei_strength=float(cfg.get("ei_strength") or 1.0),
        ei_ratio=float(cfg.get("ei_ratio") or 2.0),
        sparsity=float(cfg.get("sparsity") or 0.0),
        device=device,
        randomize_init=not bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
        readout_mode=cfg.get("readout_mode", "mem-mean"),
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True
    return net, cfg, X_te, y_te


# ─── I-stream override perturbation ─────────────────────────────────


def _make_i_override_fn(override: "object"):
    """Return a per-step callback that substitutes s_i with override[t].

    `override` is a (T, B, N_I) tensor — the I-spike stream to inject
    on each timestep. Holds a closure-local step counter; caller must
    call `.reset()` before each new forward pass.
    """
    state = {"step": 0}

    def fn(s_e, s_i, _layer):
        t = state["step"]
        state["step"] = t + 1
        if s_i is None:
            return s_e, s_i
        new_s_i = override[t].to(s_i.device, s_i.dtype)
        return s_e, new_s_i

    fn.reset = lambda: state.update(step=0)  # type: ignore[attr-defined]
    return fn


def _build_override(
    s_i_base: "object", condition: str, generator, dt_ms: float = 0.1,
    cycle_period_ms: float | None = None,
) -> "object":
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
    else:
        raise ValueError(f"unknown condition {condition!r}")
    return out


def _jitter_i_stream(
    s_i_base: "object", sigma_ms: float, dt_ms: float, generator,
    cycle_period_ms: float | None = None,
) -> "object":
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


# ─── per-condition evaluation ───────────────────────────────────────


def evaluate_condition(
    train_dir: Path, condition: str, device, seed_offset: int = 0
) -> dict:
    """Run the full test set under one condition; return acc/E rate/I rate."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import models as M
    from cli import EVAL_SEED, encode_batch

    net, cfg, X_te, y_te = _load_trained_full(train_dir, device)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )
    override_gen = torch.Generator().manual_seed(EVAL_SEED + 17 + seed_offset)
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)

    correct = total = 0
    e_spike_sum = i_spike_sum = 0.0
    n_e = M.N_HID
    n_i = M.N_INH or 1
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)

            if condition == "baseline":
                logits = net(input_spikes=spk)
            else:
                # Pass A: capture baseline I-spike stream for this batch.
                net._hidden_perturb_fn = None
                _ = net(input_spikes=spk)
                s_i_base = net.spike_record["inh"].detach().clone()
                # Pass B: build override and replay.
                override = _build_override(s_i_base, condition, override_gen,
                                           dt_ms=float(M.dt))
                fn = _make_i_override_fn(override)
                fn.reset()
                net._hidden_perturb_fn = fn
                logits = net(input_spikes=spk)
                net._hidden_perturb_fn = None

            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            e_spike_sum += float(net.spike_record["hid"].sum().item())
            i_spike_sum += float(net.spike_record["inh"].sum().item())

    t_sec = float(cfg["t_ms"]) / 1000.0
    return {
        "condition": condition,
        "acc": 100.0 * correct / total,
        "e_rate_hz": e_spike_sum / (total * n_e * t_sec),
        "i_rate_hz": i_spike_sum / (total * n_i * t_sec),
        "n_total": total,
    }


def capture_condition_raster(
    train_dir: Path, condition: str, sample_idx: int, device,
    seed_offset: int = 0,
) -> dict:
    """Single-trial raster under one condition. Same structure as eval."""
    import torch

    import models as M
    from cli import EVAL_SEED, encode_batch

    net, cfg, X_te, y_te = _load_trained_full(train_dir, device)
    X_b = torch.from_numpy(X_te[sample_idx : sample_idx + 1]).to(device)
    y_b = int(y_te[sample_idx])
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    override_gen = torch.Generator().manual_seed(EVAL_SEED + 17 + seed_offset)

    with torch.no_grad():
        spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
        if condition == "baseline":
            _ = net(input_spikes=spk)
        else:
            net._hidden_perturb_fn = None
            _ = net(input_spikes=spk)
            s_i_base = net.spike_record["inh"].detach().clone()
            override = _build_override(s_i_base, condition, override_gen,
                                       dt_ms=float(M.dt))
            fn = _make_i_override_fn(override)
            fn.reset()
            net._hidden_perturb_fn = fn
            _ = net(input_spikes=spk)
            net._hidden_perturb_fn = None

    e_full = net.spike_record["hid"].cpu().numpy()
    i_full = net.spike_record["inh"].cpu().numpy()
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


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


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

    fig, ax_rate = plt.subplots(figsize=(8.0, 4.5), dpi=150)
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
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_raster_strip(samples: list[dict], out_path: Path, run_id: str) -> None:
    """Stacked single-trial rasters across the three conditions."""
    theme.apply()
    n = len(samples)
    n_e = RASTER_N_E_PLOT
    n_i = RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(10.0, 5.625),
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
                "Single-trial rasters — trained PING (nb025 seed 42) "
                "under each I-stream condition"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
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
        n, 1, figsize=(10.0, 1.0 * n + 1.5),
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
                "Single-trial rasters — trained PING (nb025 seed 42) "
                "under cycle-coherent I-jitter"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
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

    fig, ax_rate = plt.subplots(figsize=(9.0, 5.0), dpi=150)
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
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── success criteria ───────────────────────────────────────────────


def evaluate_success(rows: list[dict], figures: Path) -> list[dict]:
    figs_root = figures.parents[2]

    def artifact(name: str, label: str) -> dict:
        path = figures / name
        ok = path.exists() and path.stat().st_size > 0
        href = "/" + str(path.relative_to(figs_root)) if ok else None
        return {
            "label": label,
            "passed": bool(ok),
            "detail": (
                f"{path.name} ({path.stat().st_size} bytes)"
                if ok else f"missing {path.name}"
            ),
            "detail_href": href,
        }

    crits: list[dict] = [
        artifact("bar_chart.png", "bar chart rendered"),
        artifact("raster_strip.png", "raster strip rendered"),
    ]

    # Control sanity: mean I rate held constant across conditions
    # (otherwise we're not isolating the rhythm).
    by_cond = {c: [] for c in CONDITIONS}
    for r in rows:
        by_cond[r["condition"]].append(r["i_rate_hz"])
    base_i = float(np.mean(by_cond["baseline"]))
    drift = []
    for cond in ("phase_shuffled_i", "poisson_matched_i"):
        i_mu = float(np.mean(by_cond[cond]))
        rel = abs(i_mu - base_i) / max(base_i, 1e-6)
        drift.append((cond, rel))
    max_drift = max(d for _, d in drift)
    crits.append({
        "label": "mean I rate held within 5% across conditions",
        "passed": bool(max_drift <= 0.05),
        "detail": ", ".join(f"{c}: {100*d:.1f}% drift" for c, d in drift),
    })

    # Discrimination: phase-shuffled / Poisson E rate clearly different
    # from baseline (the rhythm is doing real work).
    e_base = float(np.mean([r["e_rate_hz"] for r in rows if r["condition"] == "baseline"]))
    diffs = []
    for cond in ("phase_shuffled_i", "poisson_matched_i"):
        e_mu = float(np.mean([r["e_rate_hz"] for r in rows if r["condition"] == cond]))
        rel = abs(e_mu - e_base) / max(e_base, 1e-6)
        diffs.append((cond, e_mu, rel))
    max_e_jump = max(rel for _, _, rel in diffs)
    crits.append({
        "label": "E rate under desync ≥ 25% different from baseline",
        "passed": bool(max_e_jump >= 0.25),
        "detail": ", ".join(
            f"{c}: {mu:.2f} Hz ({100*rel:+.1f}% vs baseline {e_base:.2f} Hz)"
            for c, mu, rel in diffs
        ),
    })
    return crits


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default=DEFAULT_TIER)
    parser.add_argument("--modal-gpu", default="none")
    parser.add_argument("--no-wipe-dir", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS),
                        help="subset of nb025 PING seeds to evaluate")
    args = parser.parse_args()

    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    if modal_gpu:
        raise SystemExit(
            "nb042 is inference-only on local nb025 weights; --modal-gpu unused"
        )

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={tier} seeds={args.seeds}")

    if not args.no_wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    from cli import _auto_device

    device = _auto_device()
    print(f"device = {device}")

    rows: list[dict] = []
    for seed in args.seeds:
        train_dir = NB035_ARTIFACTS / f"ping__off__seed{seed}"
        if not (train_dir / "weights.pth").exists():
            raise SystemExit(
                f"missing nb025 trained PING checkpoint at {train_dir} — "
                "train nb025 baselines first"
            )
        print(f"[eval] seed={seed} from {train_dir.relative_to(REPO)}")
        for cond in CONDITIONS:
            t0 = time.monotonic()
            res = evaluate_condition(train_dir, cond, device, seed_offset=seed)
            res["seed"] = seed
            rows.append(res)
            print(
                f"    {cond:<22}  acc={res['acc']:5.2f}%  "
                f"E={res['e_rate_hz']:6.2f} Hz  I={res['i_rate_hz']:6.2f} Hz  "
                f"({time.monotonic() - t0:.1f}s)"
            )

    # Raster strip — single trial per condition from seed 42.
    raster_train_dir = NB035_ARTIFACTS / f"ping__off__seed{args.seeds[0]}"
    print(
        f"[raster] single-trial panels from seed {args.seeds[0]}, "
        f"sample {RASTER_SAMPLE_IDX}"
    )
    samples = [
        capture_condition_raster(
            raster_train_dir, cond, RASTER_SAMPLE_IDX, device,
            seed_offset=args.seeds[0],
        )
        for cond in CONDITIONS
    ]
    plot_raster_strip(samples, FIGURES / "raster_strip.png", notebook_run_id)
    print(f"wrote {FIGURES / 'raster_strip.png'}")

    plot_bar_chart(rows, FIGURES / "bar_chart.png", notebook_run_id)
    print(f"wrote {FIGURES / 'bar_chart.png'}")

    # ── Jitter sweep ───────────────────────────────────────────────
    # Adds Gaussian timing jitter σ to each I-spike at inference.
    # Predicts the rate-release transition at σ ≈ 1/f_γ ≈ 28 ms.
    print(f"[jitter] sweep σ ∈ {list(JITTER_SIGMAS_MS)} ms")
    jitter_rows: list[dict] = []
    for seed in args.seeds:
        train_dir = NB035_ARTIFACTS / f"ping__off__seed{seed}"
        for sigma_ms in JITTER_SIGMAS_MS:
            cond = f"jitter_sigma_{sigma_ms:g}"
            t0 = time.monotonic()
            # Reuse evaluate_condition — it dispatches on the condition string.
            res = evaluate_condition(train_dir, cond, device,
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
        FIGURES / "jitter_sweep.png", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'jitter_sweep.png'}")

    # Jitter raster strip — one panel per σ from the diagnostic subset,
    # all from the first seed at sample 0 so panels read against the
    # baseline raster strip directly.
    raster_seed = args.seeds[0]
    raster_train_dir = NB035_ARTIFACTS / f"ping__off__seed{raster_seed}"
    print(
        f"[jitter-raster] panels from seed {raster_seed}, "
        f"σ ∈ {list(JITTER_RASTER_SIGMAS_MS)} ms"
    )
    jitter_raster_samples = []
    for sigma_ms in JITTER_RASTER_SIGMAS_MS:
        cond = f"jitter_sigma_{sigma_ms:g}"
        sample = capture_condition_raster(
            raster_train_dir, cond, RASTER_SAMPLE_IDX, device,
            seed_offset=raster_seed + int(sigma_ms),
        )
        sample["sigma_ms"] = float(sigma_ms)
        jitter_raster_samples.append(sample)
    plot_jitter_raster_strip(
        jitter_raster_samples,
        FIGURES / "jitter_raster_strip.png",
        notebook_run_id,
    )
    print(f"wrote {FIGURES / 'jitter_raster_strip.png'}")

    duration_s = time.monotonic() - t_start
    crits = evaluate_success(rows, FIGURES)
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "seeds": list(args.seeds),
            "conditions": list(CONDITIONS),
            "jitter_sigmas_ms": list(JITTER_SIGMAS_MS),
            "f_gamma_reference_hz": F_GAMMA_REFERENCE_HZ,
            "nb025_source": "ping__off__seed{seed} (θ_u = off baseline)",
            "raster_sample_idx": RASTER_SAMPLE_IDX,
        },
        "results": rows,
        "jitter_sweep": jitter_rows,
        "success_criteria": crits,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")

    for c in crits:
        mark = "pass" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['label']} — {c['detail']}")
    if any(not c["passed"] for c in crits):
        sys.exit(1)


if __name__ == "__main__":
    main()
