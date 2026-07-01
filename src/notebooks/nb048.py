"""Notebook runner for entry 048 — Streaming digit classification on
trained PING.

Loads the canonical nb025 PING baseline (medium tier, seed 42),
constructs a sequential digit stream by concatenating Poisson-encoded
MNIST samples at τ ms per digit, runs inference over the long stream,
and reads out the class with a sliding-window mem-mean over the same
output LIF that the network was trained on.

Headline figure: 4-panel column at τ = 50 ms showing 5 sequential
digits, hidden E + I rasters, and the 10-class readout probability
tracking the ground-truth label over time.

Sweep figure: per-segment accuracy vs τ ∈ {25, 50, 100, 200} ms,
two protocols (constant input rate / rate-compensated for τ).

Notebook entry: src/docs/src/pages/notebooks/nb048.mdx
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "notebooks"))

from helpers.figsave import save_figure  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from helpers import theme  # noqa: E402

SLUG = "nb048"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

# Trained-baseline locator — nb025 medium-tier PING at θ_u = off, three
# seeds. Heatmap and τ-sweep average over all three; headline trials pick
# the first.
SEEDS: list[int] = [42, 43, 44]


def baseline_dir(seed: int) -> Path:
    # θ_u = off PING baseline now lives in the shared training root (nb022
    # train-once / reuse-many), not the retired per-notebook nb025 dir.
    return (
        REPO / "src" / "artifacts" / "notebooks" / "training"
        / f"ping__off__seed{seed}"
    )


NB025_BASELINE: Path = baseline_dir(SEEDS[0])

# Architecture constants for inference (must match nb025's trained config).
N_E: int = 1024
N_I: int = 256
N_IN: int = 784
N_CLASSES: int = 10
DT: float = 0.1                  # ms
TRAINED_T_MS: float = 200.0      # trained trial duration
INPUT_RATE_HZ: float = 25.0      # canonical Poisson input rate

# Headline experiment parameters.
TAU_HEADLINE_MS: float = 50.0    # digit duration in headline figure
N_DIGITS_HEADLINE: int = 5       # number of digits in the headline stream

# Sweep tier-driven counts. Stream count = streams × digits × seeds.
TIER_CONFIG: dict[str, dict] = {
    "extra small": dict(n_streams=4, n_per_stream=4, n_grid_streams=2),
    "small":       dict(n_streams=20, n_per_stream=10, n_grid_streams=40),
    "medium":      dict(n_streams=80, n_per_stream=10, n_grid_streams=80),
    "large":       dict(n_streams=200, n_per_stream=10, n_grid_streams=200),
    "extra large": dict(n_streams=400, n_per_stream=10, n_grid_streams=400),
}
DEFAULT_TIER: str = "small"

# τ sweep — multiples of the gamma cycle (≈ 28 ms at τ_GABA = 9 ms).
TAU_SWEEP_MS: list[float] = [25.0, 50.0, 100.0, 200.0]

# 2D heatmap grid over (τ, input_rate). Extended τ down to 10 ms to
# resolve the sub-cycle failure regime.
TAU_GRID_MS: list[float] = [10.0, 15.0, 25.0, 40.0, 50.0, 75.0, 100.0, 200.0]
RATE_GRID_HZ: list[float] = [5.0, 10.0, 25.0, 50.0, 100.0, 200.0]

# Varying-stream headline — per-segment (τ_ms, input_rate_hz). Spans a
# wide drive × duration range to show the trained network is robust to
# both. Each entry generates one segment in the headline stream.
VARYING_HEADLINE: list[tuple[float, float]] = [
    (200.0, 10.0),   # long, weak
    (50.0, 100.0),   # short, strong
    (100.0, 25.0),   # medium, canonical
    (25.0, 200.0),   # very short, very strong
    (75.0, 15.0),    # intermediate, weak-ish
]

# Raster display constants (Figure 1).
RASTER_N_E_PLOT: int = 200
RASTER_N_I_PLOT: int = 64

SEED: int = 42


# ── Net build / load ────────────────────────────────────────────────
def _load_trained_full(device, seed: int = SEEDS[0]):
    """Mirror nb025's loader but read τ_GABA from the trained config so
    we replay under the same biophysics. Returns (net, cfg, X_te, y_te)."""
    import models as M
    from cli.config import build_net
    from cli import load_dataset, seed_everything

    train_dir = baseline_dir(seed)
    if not (train_dir / "weights.pth").exists():
        raise SystemExit(
            f"nb048 needs nb025's PING baseline at {train_dir}"
        )
    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", 42)))
    M.T_ms = float(cfg["t_ms"])
    M.dt = float(cfg["dt"])
    M.T_steps = int(M.T_ms / M.dt)
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


# ── Stream construction ─────────────────────────────────────────────
def encode_stream(
    digit_pixels: np.ndarray, tau_ms: float, input_rate_hz: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Concatenate Poisson-encoded digits into one (T_stream, 1, N_IN)
    spike tensor. Each digit is encoded for tau_ms at the given
    Poisson rate; rate scaling for τ-compensation is handled by the
    caller.
    """
    tau_steps = int(round(tau_ms / DT))
    p_step = input_rate_hz * DT / 1000.0
    n_digits = digit_pixels.shape[0]
    streams: list[torch.Tensor] = []
    for d in range(n_digits):
        pixels = torch.from_numpy(digit_pixels[d:d + 1]).clamp(0, 1)
        # Same scheme as encode_images_poisson: per-step Bernoulli at p_step
        # weighted by pixel intensity in [0,1].
        rand = torch.rand(tau_steps, 1, N_IN, generator=generator)
        spikes = (rand < (p_step * pixels.unsqueeze(0))).float()
        streams.append(spikes)
    return torch.cat(streams, dim=0)


# ── Sliding-window readout ──────────────────────────────────────────
def _v_out_series(
    spk_e: np.ndarray, W_out: np.ndarray, tau_out_ms: float,
) -> np.ndarray:
    """Replay the trained output LIF on a recorded hidden spike train.
    Returns per-timestep v_out of shape (T, N_CLASSES)."""
    T, _ = spk_e.shape
    beta_out = float(np.exp(-DT / tau_out_ms))
    one_minus_beta = 1.0 - beta_out
    spike_scale = one_minus_beta / DT
    v_out = np.zeros(N_CLASSES, dtype=np.float32)
    series = np.zeros((T, N_CLASSES), dtype=np.float32)
    for t in range(T):
        if t > 0:
            v_out = beta_out * v_out + spike_scale * (spk_e[t - 1] @ W_out)
        series[t] = v_out
    return series


def sliding_readout(
    spk_e: np.ndarray, W_out: np.ndarray, tau_out_ms: float,
    window_ms: float,
) -> np.ndarray:
    """Replay the trained mem-mean readout post-hoc, with a sliding
    window of width `window_ms` instead of integrating from t=0.

    Pipeline (mirrors models._step_body when readout_mode='mem-mean'):
      1. v_out[t] = beta_out · v_out[t-1] + spike_scale · spk_e[t-1] @ W_out
         (after subtracting threshold-reset; for inference we skip the
         output spike because the trained readout's argmax is what we want)
      2. logits[t] = average of v_out over the last window_ms ms

    Returns: (T, N_CLASSES) array of logits per timestep.
    """
    v_out_series = _v_out_series(spk_e, W_out, tau_out_ms)
    T = v_out_series.shape[0]
    window_steps = max(1, int(round(window_ms / DT)))
    # Cumulative sum trick for the rolling mean.
    csum = np.concatenate([
        np.zeros((1, N_CLASSES), dtype=np.float32),
        np.cumsum(v_out_series, axis=0),
    ])  # (T+1, C)
    logits = np.empty_like(v_out_series)
    for t in range(T):
        lo = max(0, t + 1 - window_steps)
        hi = t + 1
        logits[t] = (csum[hi] - csum[lo]) / max(1, hi - lo)
    return logits


def softmax_rowwise(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ── Headline experiment ────────────────────────────────────────────
def pick_diverse_digits(X_te, y_te, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick n digits of n different classes, deterministic by seed."""
    rng = np.random.default_rng(seed)
    classes = list(range(N_CLASSES))
    rng.shuffle(classes)
    classes = classes[:n]
    pixels: list[np.ndarray] = []
    labels: list[int] = []
    for c in classes:
        idx = np.where(y_te == c)[0]
        if idx.size == 0:
            continue
        i = int(rng.choice(idx))
        pixels.append(X_te[i])
        labels.append(c)
    return np.stack(pixels, axis=0), np.array(labels, dtype=np.int64)


def run_headline_stream(net, cfg, X_te, y_te, device) -> dict:
    """Run a single 5-digit stream at TAU_HEADLINE_MS and capture
    everything the headline figure needs."""
    import models as M
    pixels, labels = pick_diverse_digits(
        X_te, y_te, N_DIGITS_HEADLINE, seed=SEED,
    )
    tau_steps = int(round(TAU_HEADLINE_MS / DT))
    T_stream_steps = tau_steps * N_DIGITS_HEADLINE
    M.T_steps = T_stream_steps
    M.T_ms = T_stream_steps * DT

    gen = torch.Generator().manual_seed(SEED + 1)
    spk_in = encode_stream(pixels, TAU_HEADLINE_MS, INPUT_RATE_HZ, gen)
    spk_in = spk_in.to(device)
    with torch.no_grad():
        _ = net(input_spikes=spk_in)
    rec = net.spike_record
    spk_e = rec["hid"].cpu().numpy()  # (T, N_E) with B=1 squeezed
    spk_i = rec["inh"].cpu().numpy()  # (T, N_I)
    if spk_e.ndim == 3:
        spk_e = spk_e[:, 0, :]
        spk_i = spk_i[:, 0, :]

    # W_out is the last entry in W_ff.
    W_out = list(net.W_ff)[-1].detach().cpu().numpy()  # (N_E, N_CLASSES)
    tau_out_ms = float(cfg.get("tau_out_ms", 2.0))
    logits = sliding_readout(
        spk_e, W_out, tau_out_ms, window_ms=TAU_HEADLINE_MS,
    )
    probs = softmax_rowwise(logits)
    pred = probs.argmax(axis=-1)

    # Per-segment accuracy: at the end-of-segment timestep.
    seg_ends = np.arange(1, N_DIGITS_HEADLINE + 1) * tau_steps - 1
    seg_pred = pred[seg_ends]
    seg_correct = (seg_pred == labels).astype(int)

    return {
        "tau_ms": TAU_HEADLINE_MS,
        "n_digits": N_DIGITS_HEADLINE,
        "tau_steps": tau_steps,
        "T_stream_steps": T_stream_steps,
        "labels": labels.tolist(),
        "pixels": pixels,
        "spk_e": spk_e,
        "spk_i": spk_i,
        "probs": probs,
        "pred_per_t": pred,
        "seg_correct": seg_correct.tolist(),
        "input_rate_hz": INPUT_RATE_HZ,
    }


# ── Varying-condition stream ────────────────────────────────────────
def encode_varying_stream(
    digit_pixels: np.ndarray, segments: list[tuple[float, float]],
    generator: torch.Generator,
) -> torch.Tensor:
    """Concatenate digits with per-segment (τ_ms, rate_hz)."""
    streams: list[torch.Tensor] = []
    for d, (tau_ms, rate_hz) in enumerate(segments):
        tau_steps = int(round(tau_ms / DT))
        p_step = rate_hz * DT / 1000.0
        pixels = torch.from_numpy(digit_pixels[d:d + 1]).clamp(0, 1)
        rand = torch.rand(tau_steps, 1, N_IN, generator=generator)
        spikes = (rand < (p_step * pixels.unsqueeze(0))).float()
        streams.append(spikes)
    return torch.cat(streams, dim=0)


def run_varying_headline(net, cfg, X_te, y_te, device) -> dict:
    """One stream where each segment has its own (τ, rate). Reads off
    accuracy at per-segment end with each segment's own τ as the
    sliding-window width."""
    import models as M
    pixels, labels = pick_diverse_digits(
        X_te, y_te, len(VARYING_HEADLINE), seed=SEED + 7,
    )
    segment_steps = [int(round(t / DT)) for (t, _) in VARYING_HEADLINE]
    T_stream_steps = sum(segment_steps)
    M.T_steps = T_stream_steps
    M.T_ms = T_stream_steps * DT

    gen = torch.Generator().manual_seed(SEED + 9)
    spk_in = encode_varying_stream(pixels, VARYING_HEADLINE, gen).to(device)
    with torch.no_grad():
        _ = net(input_spikes=spk_in)
    rec = net.spike_record
    spk_e = rec["hid"].cpu().numpy()
    spk_i = rec["inh"].cpu().numpy()
    if spk_e.ndim == 3:
        spk_e = spk_e[:, 0, :]
        spk_i = spk_i[:, 0, :]
    W_out = list(net.W_ff)[-1].detach().cpu().numpy()
    tau_out_ms = float(cfg.get("tau_out_ms", 2.0))

    # v_out is reusable; per-segment readout uses its own window.
    v_series = _v_out_series(spk_e, W_out, tau_out_ms)
    csum = np.concatenate([
        np.zeros((1, N_CLASSES), dtype=np.float32),
        np.cumsum(v_series, axis=0),
    ])
    seg_correct: list[int] = []
    seg_preds: list[int] = []
    seg_ends: list[int] = []
    cur = 0
    for d, ((tau_ms, _), s_steps) in enumerate(
        zip(VARYING_HEADLINE, segment_steps)
    ):
        end = cur + s_steps - 1
        lo = max(0, end + 1 - s_steps)
        logits = (csum[end + 1] - csum[lo]) / max(1, end + 1 - lo)
        pred = int(np.argmax(logits))
        seg_preds.append(pred)
        seg_correct.append(int(pred == labels[d]))
        seg_ends.append(end)
        cur += s_steps

    # For the continuous probability trace, use the per-segment window
    # that's currently active — gives an honest "what would I predict
    # right now" view.
    probs = np.zeros((T_stream_steps, N_CLASSES), dtype=np.float32)
    cur = 0
    for d, ((tau_ms, _), s_steps) in enumerate(
        zip(VARYING_HEADLINE, segment_steps)
    ):
        w = s_steps
        for t in range(cur, cur + s_steps):
            lo = max(0, t + 1 - w)
            logits_t = (csum[t + 1] - csum[lo]) / max(1, t + 1 - lo)
            probs[t] = logits_t
        cur += s_steps
    # Row-wise softmax to map logits → probabilities.
    probs = softmax_rowwise(probs)

    return {
        "segments": VARYING_HEADLINE,
        "segment_steps": segment_steps,
        "T_stream_steps": T_stream_steps,
        "labels": labels.tolist(),
        "pixels": pixels,
        "spk_e": spk_e,
        "spk_i": spk_i,
        "probs": probs,
        "seg_preds": seg_preds,
        "seg_correct": seg_correct,
        "seg_ends": seg_ends,
    }


# ── 2D grid sweep over (τ, input_rate) ──────────────────────────────
def run_grid_sweep(
    net, cfg, X_te, y_te, device, train_seed: int,
    n_streams: int, n_per_stream: int,
) -> list[dict]:
    """Per-(τ, rate) accuracy on uniform-segment streams for ONE trained
    seed. Each cell runs `n_streams` streams of `n_per_stream` random
    test digits. Caller loops over training seeds."""
    import models as M
    W_out = list(net.W_ff)[-1].detach().cpu().numpy()
    tau_out_ms = float(cfg.get("tau_out_ms", 2.0))
    rng = np.random.default_rng(SEED + 555 + train_seed)
    rows: list[dict] = []
    for tau_ms in TAU_GRID_MS:
        tau_steps = int(round(tau_ms / DT))
        T_stream_steps = tau_steps * n_per_stream
        M.T_steps = T_stream_steps
        M.T_ms = T_stream_steps * DT
        for rate_hz in RATE_GRID_HZ:
            n_correct = 0
            n_total = 0
            for s in range(n_streams):
                idx = rng.choice(len(y_te), n_per_stream, replace=False)
                pixels = X_te[idx]
                labels = y_te[idx]
                gen = torch.Generator().manual_seed(
                    SEED + 2000 + s + 100 * train_seed
                )
                spk_in = encode_stream(
                    pixels, tau_ms, rate_hz, gen,
                ).to(device)
                with torch.no_grad():
                    _ = net(input_spikes=spk_in)
                spk_e = net.spike_record["hid"].cpu().numpy()
                if spk_e.ndim == 3:
                    spk_e = spk_e[:, 0, :]
                logits = sliding_readout(
                    spk_e, W_out, tau_out_ms, window_ms=tau_ms,
                )
                pred = logits.argmax(axis=-1)
                seg_ends = np.arange(1, n_per_stream + 1) * tau_steps - 1
                n_correct += int((pred[seg_ends] == labels).sum())
                n_total += n_per_stream
            acc = 100.0 * n_correct / max(1, n_total)
            rows.append({
                "tau_ms": float(tau_ms),
                "input_rate_hz": float(rate_hz),
                "train_seed": int(train_seed),
                "n_correct": int(n_correct),
                "n_total": int(n_total),
                "acc": float(acc),
            })
            print(
                f"  seed={train_seed}  τ={tau_ms:>5.1f} ms  "
                f"rate={rate_hz:>6.1f} Hz  acc={acc:5.2f}%  "
                f"({n_correct}/{n_total})"
            )
    return rows


def aggregate_grid_rows(rows: list[dict]) -> list[dict]:
    """Collapse per-seed rows to mean ± SEM per (τ, rate) cell."""
    cells: dict[tuple[float, float], list[float]] = {}
    n_totals: dict[tuple[float, float], int] = {}
    for r in rows:
        key = (r["tau_ms"], r["input_rate_hz"])
        cells.setdefault(key, []).append(r["acc"])
        n_totals[key] = n_totals.get(key, 0) + int(r["n_total"])
    out: list[dict] = []
    for (tau_ms, rate_hz), accs in sorted(cells.items()):
        a = np.array(accs, dtype=np.float32)
        out.append({
            "tau_ms": float(tau_ms),
            "input_rate_hz": float(rate_hz),
            "acc": float(a.mean()),
            "acc_sem": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
            "n_seeds": int(len(a)),
            "n_total": int(n_totals[(tau_ms, rate_hz)]),
        })
    return out


def aggregate_tau_rows(rows: list[dict]) -> list[dict]:
    """Collapse per-seed rows to mean ± SEM per (τ, rate_protocol)."""
    cells: dict[tuple[float, bool], list[float]] = {}
    extras: dict[tuple[float, bool], dict] = {}
    for r in rows:
        key = (r["tau_ms"], bool(r["rate_compensate"]))
        cells.setdefault(key, []).append(r["acc"])
        extras[key] = {
            "input_rate_hz": r["input_rate_hz"],
        }
    out: list[dict] = []
    for (tau_ms, rate_compensate), accs in sorted(cells.items()):
        a = np.array(accs, dtype=np.float32)
        out.append({
            "tau_ms": float(tau_ms),
            "rate_compensate": bool(rate_compensate),
            "input_rate_hz": float(extras[(tau_ms, rate_compensate)]["input_rate_hz"]),
            "acc": float(a.mean()),
            "acc_sem": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
            "n_seeds": int(len(a)),
        })
    return out


# ── Sweep ───────────────────────────────────────────────────────────
def run_tau_sweep(
    net, cfg, X_te, y_te, device, train_seed: int,
    n_streams: int, n_per_stream: int,
    rate_compensate: bool,
) -> list[dict]:
    """For each τ in TAU_SWEEP_MS, run n_streams streams of n_per_stream
    digits each. Per-stream input_rate is either constant (25 Hz) or
    scaled to keep total spikes per digit constant (25 × 200/τ Hz).
    Returns one row per τ."""
    import models as M
    W_out = list(net.W_ff)[-1].detach().cpu().numpy()
    tau_out_ms = float(cfg.get("tau_out_ms", 2.0))
    rng = np.random.default_rng(SEED + 100 + train_seed)

    rows: list[dict] = []
    for tau_ms in TAU_SWEEP_MS:
        tau_steps = int(round(tau_ms / DT))
        input_rate = (
            INPUT_RATE_HZ * (TRAINED_T_MS / tau_ms) if rate_compensate
            else INPUT_RATE_HZ
        )
        T_stream_steps = tau_steps * n_per_stream
        M.T_steps = T_stream_steps
        M.T_ms = T_stream_steps * DT

        n_correct = 0
        n_total = 0
        for s in range(n_streams):
            idx = rng.choice(len(y_te), n_per_stream, replace=False)
            pixels = X_te[idx]
            labels = y_te[idx]
            gen = torch.Generator().manual_seed(
                SEED + 1000 + s + 100 * train_seed
            )
            spk_in = encode_stream(pixels, tau_ms, input_rate, gen).to(device)
            with torch.no_grad():
                _ = net(input_spikes=spk_in)
            spk_e = net.spike_record["hid"].cpu().numpy()
            if spk_e.ndim == 3:
                spk_e = spk_e[:, 0, :]
            logits = sliding_readout(
                spk_e, W_out, tau_out_ms, window_ms=tau_ms,
            )
            pred = logits.argmax(axis=-1)
            seg_ends = np.arange(1, n_per_stream + 1) * tau_steps - 1
            n_correct += int((pred[seg_ends] == labels).sum())
            n_total += n_per_stream
        acc = 100.0 * n_correct / max(1, n_total)
        rows.append({
            "tau_ms": float(tau_ms),
            "rate_compensate": bool(rate_compensate),
            "input_rate_hz": float(input_rate),
            "train_seed": int(train_seed),
            "n_streams": n_streams,
            "n_per_stream": n_per_stream,
            "n_correct": n_correct,
            "n_total": n_total,
            "acc": acc,
        })
        print(
            f"  seed={train_seed}  τ={tau_ms:>5.1f} ms  "
            f"rate={input_rate:>6.1f} Hz  acc={acc:5.2f}%  "
            f"({n_correct}/{n_total})"
        )
    return rows


# ── Plotting ────────────────────────────────────────────────────────


def plot_headline_stream(s: dict, out_path: Path, run_id: str) -> None:
    """4-panel headline figure for a 5-digit τ=50ms stream."""
    theme.apply()
    n_dig = s["n_digits"]
    tau_ms = s["tau_ms"]
    tau_steps = s["tau_steps"]
    T_stream_steps = s["T_stream_steps"]
    t_axis = np.arange(T_stream_steps) * DT  # ms
    seg_starts_ms = np.arange(n_dig) * tau_ms
    seg_ends_ms = (np.arange(n_dig) + 1) * tau_ms
    labels = s["labels"]
    seg_pred = s["pred_per_t"][np.arange(1, n_dig + 1) * tau_steps - 1]
    pixels = s["pixels"]
    spk_e = s["spk_e"]
    spk_i = s["spk_i"]
    probs = s["probs"]

    fig = plt.figure(figsize=(6.9, 5.33), dpi=150)
    gs = fig.add_gridspec(
        4, 1, height_ratios=[0.9, 2.2, 1.2, 2.0], hspace=0.18,
    )

    # ── Panel A: digit thumbnails + class labels
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, T_stream_steps * DT)
    ax_a.set_ylim(0, 1)
    ax_a.set_yticks([])
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.spines["left"].set_visible(False)
    for d in range(n_dig):
        # Thumbnail glued to the segment band, scaled to ~tau_ms wide.
        x_lo = seg_starts_ms[d]
        x_hi = seg_ends_ms[d]
        img = pixels[d].reshape(28, 28)
        # Build a tiny axes inset for the digit image to avoid axes-coord
        # warping with axhspan.
        sub_w = (x_hi - x_lo) / (T_stream_steps * DT) * 0.88
        sub_l = ax_a.get_position().x0 + (
            (x_lo / (T_stream_steps * DT)) * (
                ax_a.get_position().x1 - ax_a.get_position().x0
            )
        )
        sub = fig.add_axes([
            sub_l, ax_a.get_position().y0 + 0.005,
            sub_w, ax_a.get_position().height - 0.01,
        ])
        sub.imshow(img, cmap="Greys", interpolation="nearest", aspect="auto")
        sub.set_xticks([])
        sub.set_yticks([])
        sub.set_title(
            f"true {labels[d]} · pred {int(seg_pred[d])}",
            fontsize=theme.SIZE_LABEL,
            color=(theme.INK_BLACK if int(seg_pred[d]) == labels[d]
                   else theme.DEEP_RED),
            pad=2,
        )
    ax_a.set_xticks([])
    ax_a.tick_params(axis="x", labelbottom=False)

    # ── Panel B: hidden E raster
    ax_b = fig.add_subplot(gs[1])
    rng = np.random.default_rng(SEED)
    e_idx = np.sort(rng.choice(N_E, RASTER_N_E_PLOT, replace=False))
    e_t, e_n = np.where(spk_e[:, e_idx])
    ax_b.scatter(
        t_axis[e_t], e_n, s=2.0, c=theme.INK_BLACK,
        marker="|", linewidths=0.4,
    )
    for seg in seg_starts_ms[1:]:
        ax_b.axvline(seg, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)
    ax_b.set_xlim(0, T_stream_steps * DT)
    ax_b.set_ylim(0, RASTER_N_E_PLOT)
    ax_b.set_yticks([0, RASTER_N_E_PLOT])
    ax_b.set_yticklabels(["0", f"{N_E}"])
    ax_b.set_ylabel("E cell", fontsize=theme.SIZE_LABEL)
    ax_b.tick_params(axis="x", labelbottom=False)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # ── Panel C: hidden I raster
    ax_c = fig.add_subplot(gs[2])
    i_idx = np.sort(rng.choice(N_I, min(RASTER_N_I_PLOT, N_I), replace=False))
    i_t, i_n = np.where(spk_i[:, i_idx])
    ax_c.scatter(
        t_axis[i_t], i_n, s=2.0, c=theme.DEEP_RED,
        marker="|", linewidths=0.4,
    )
    for seg in seg_starts_ms[1:]:
        ax_c.axvline(seg, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)
    ax_c.set_xlim(0, T_stream_steps * DT)
    ax_c.set_ylim(0, len(i_idx))
    ax_c.set_yticks([0, len(i_idx)])
    ax_c.set_yticklabels(["0", f"{N_I}"])
    ax_c.set_ylabel("I cell", fontsize=theme.SIZE_LABEL)
    ax_c.tick_params(axis="x", labelbottom=False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ── Panel D: readout probabilities
    ax_d = fig.add_subplot(gs[3])
    cmap = plt.get_cmap("tab10")
    # Plot all 10 classes lightly, then highlight the true class per segment
    for c in range(N_CLASSES):
        ax_d.plot(
            t_axis, probs[:, c], color=cmap(c), lw=0.6, alpha=0.5,
        )
    # Heavy line: true-class trace per segment
    for d in range(n_dig):
        a = d * tau_steps
        b = (d + 1) * tau_steps
        c = labels[d]
        ax_d.plot(
            t_axis[a:b], probs[a:b, c], color=cmap(c), lw=2.2,
        )
    for seg in seg_starts_ms[1:]:
        ax_d.axvline(seg, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)
    ax_d.axhline(0.5, color=theme.GREY_MID, lw=0.5, ls="--", alpha=0.6)
    ax_d.set_xlim(0, T_stream_steps * DT)
    ax_d.set_ylim(0, 1)
    ax_d.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    ax_d.set_ylabel("readout p(class)", fontsize=theme.SIZE_LABEL)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    fig.suptitle(
        f"Streaming digit classification on trained PING — "
        f"{n_dig} digits at $\\tau$ = {tau_ms:g} ms per digit "
        f"(input {s['input_rate_hz']:g} Hz Poisson)",
        fontsize=theme.SIZE_TITLE,
    )
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def plot_acc_vs_tau(
    rows: list[dict], out_path: Path, run_id: str,
) -> None:
    theme.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.15), dpi=150)
    constant = sorted(
        [r for r in rows if not r["rate_compensate"]],
        key=lambda r: r["tau_ms"],
    )
    compensated = sorted(
        [r for r in rows if r["rate_compensate"]],
        key=lambda r: r["tau_ms"],
    )
    if constant:
        ax.errorbar(
            [r["tau_ms"] for r in constant], [r["acc"] for r in constant],
            yerr=[r.get("acc_sem", 0.0) for r in constant],
            marker="o", color=theme.INK_BLACK, lw=1.5, capsize=4,
            label=f"constant input ({INPUT_RATE_HZ:g} Hz)",
        )
    if compensated:
        ax.errorbar(
            [r["tau_ms"] for r in compensated], [r["acc"] for r in compensated],
            yerr=[r.get("acc_sem", 0.0) for r in compensated],
            marker="s", color=theme.DEEP_RED, lw=1.5, capsize=4,
            label=r"rate-compensated ($25 \cdot 200/\tau$ Hz)",
        )
    ax.set_xlabel(r"Segment duration $\tau$ (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Per-segment accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax.set_ylim(0, 100)
    ax.axhline(10.0, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.6)
    # Annotate the gamma cycle ≈ 28 ms.
    ax.axvline(28.0, color=theme.AMBER, lw=0.7, ls="--", alpha=0.8)
    ax.text(
        28.0, 92, " ≈ 1 gamma cycle",
        fontsize=theme.SIZE_ANNOTATION, color=theme.AMBER, va="top",
    )
    ax.set_xticks(TAU_SWEEP_MS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    fig.suptitle(
        "Streaming accuracy vs digit duration on trained PING",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def plot_varying_headline_stream(s: dict, out_path: Path, run_id: str) -> None:
    """4-panel headline with per-segment (τ, rate) varying."""
    theme.apply()
    segments = s["segments"]
    n_dig = len(segments)
    segment_steps = s["segment_steps"]
    T_stream_steps = s["T_stream_steps"]
    t_axis = np.arange(T_stream_steps) * DT
    seg_starts_steps = np.concatenate([[0], np.cumsum(segment_steps)[:-1]])
    seg_starts_ms = seg_starts_steps * DT
    seg_ends_ms = (seg_starts_steps + np.array(segment_steps)) * DT
    labels = s["labels"]
    seg_pred = s["seg_preds"]
    pixels = s["pixels"]
    spk_e = s["spk_e"]
    spk_i = s["spk_i"]
    probs = s["probs"]
    T_ms = T_stream_steps * DT

    fig = plt.figure(figsize=(6.9, 5.33), dpi=150)
    gs = fig.add_gridspec(
        4, 1, height_ratios=[1.1, 2.2, 1.2, 2.0], hspace=0.18,
    )

    # Panel A: digit thumbnails with per-segment (τ, rate) labels.
    ax_a = fig.add_subplot(gs[0])
    ax_a.set_xlim(0, T_ms)
    ax_a.set_ylim(0, 1)
    ax_a.set_yticks([])
    ax_a.set_xticks([])
    for sp in ("top", "right", "left", "bottom"):
        ax_a.spines[sp].set_visible(False)

    rates_all = [seg[1] for seg in segments]
    log_rmin = np.log(min(rates_all))
    log_rmax = np.log(max(rates_all))

    for d in range(n_dig):
        tau_ms, rate_hz = segments[d]
        x_lo = seg_starts_ms[d]
        x_hi = seg_ends_ms[d]
        img = pixels[d].reshape(28, 28)
        sub_w = (x_hi - x_lo) / T_ms * 0.88
        sub_l = ax_a.get_position().x0 + (
            (x_lo / T_ms) * (ax_a.get_position().x1 - ax_a.get_position().x0)
        )
        sub = fig.add_axes([
            sub_l, ax_a.get_position().y0 + 0.005,
            sub_w, ax_a.get_position().height - 0.02,
        ])
        # Opacity ∈ [0.2, 1.0] (log-rate) so the weakest drive is faintly
        # visible and the strongest is bold — input rate becomes a visual cue.
        if log_rmax > log_rmin:
            alpha = 0.2 + 0.8 * (np.log(rate_hz) - log_rmin) / (log_rmax - log_rmin)
        else:
            alpha = 1.0
        sub.imshow(
            img, cmap="Greys", interpolation="nearest", aspect="auto",
            alpha=alpha,
        )
        sub.set_xticks([])
        sub.set_yticks([])
        ok_color = theme.INK_BLACK if seg_pred[d] == labels[d] else theme.DEEP_RED
        # Compact 2-line title; rotate slightly to fit on narrow segments.
        # τ value first since that's the structural knob; rate beneath.
        sub.set_title(
            f"{tau_ms:g} ms\n{rate_hz:g} Hz",
            fontsize=theme.SIZE_LABEL - 1,
            color=theme.MUTED,
            pad=2,
        )
        # Per-segment prediction badge inset into the thumbnail's top
        # — same colour scheme as the readout traces below.
        sub.text(
            0.05, 0.95,
            f"{labels[d]}→{seg_pred[d]}",
            transform=sub.transAxes,
            ha="left", va="top",
            fontsize=theme.SIZE_LABEL,
            color="white",
            weight="bold",
            bbox=dict(
                facecolor=ok_color, edgecolor="none",
                boxstyle="round,pad=0.2", alpha=0.95,
            ),
        )

    # Panel B: hidden E raster.
    ax_b = fig.add_subplot(gs[1])
    rng = np.random.default_rng(SEED)
    e_idx = np.sort(rng.choice(N_E, RASTER_N_E_PLOT, replace=False))
    e_t, e_n = np.where(spk_e[:, e_idx])
    ax_b.scatter(
        t_axis[e_t], e_n, s=2.0, c=theme.INK_BLACK,
        marker="|", linewidths=0.4,
    )
    for seg in seg_starts_ms[1:]:
        ax_b.axvline(seg, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)
    ax_b.set_xlim(0, T_ms)
    ax_b.set_ylim(0, RASTER_N_E_PLOT)
    ax_b.set_yticks([0, RASTER_N_E_PLOT])
    ax_b.set_yticklabels(["0", f"{N_E}"])
    ax_b.set_ylabel("E cell", fontsize=theme.SIZE_LABEL)
    ax_b.tick_params(axis="x", labelbottom=False)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # Panel C: hidden I raster.
    ax_c = fig.add_subplot(gs[2])
    i_idx = np.sort(rng.choice(N_I, min(RASTER_N_I_PLOT, N_I), replace=False))
    i_t, i_n = np.where(spk_i[:, i_idx])
    ax_c.scatter(
        t_axis[i_t], i_n, s=2.0, c=theme.DEEP_RED,
        marker="|", linewidths=0.4,
    )
    for seg in seg_starts_ms[1:]:
        ax_c.axvline(seg, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)
    ax_c.set_xlim(0, T_ms)
    ax_c.set_ylim(0, len(i_idx))
    ax_c.set_yticks([0, len(i_idx)])
    ax_c.set_yticklabels(["0", f"{N_I}"])
    ax_c.set_ylabel("I cell", fontsize=theme.SIZE_LABEL)
    ax_c.tick_params(axis="x", labelbottom=False)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # Panel D: readout probabilities.
    ax_d = fig.add_subplot(gs[3])
    cmap = plt.get_cmap("tab10")
    for c in range(N_CLASSES):
        ax_d.plot(
            t_axis, probs[:, c], color=cmap(c), lw=0.6, alpha=0.5,
        )
    for d in range(n_dig):
        a = seg_starts_steps[d]
        b = a + segment_steps[d]
        c = labels[d]
        ax_d.plot(
            t_axis[a:b], probs[a:b, c], color=cmap(c), lw=2.2,
        )
    for seg in seg_starts_ms[1:]:
        ax_d.axvline(seg, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)
    ax_d.axhline(0.5, color=theme.GREY_MID, lw=0.5, ls="--", alpha=0.6)
    ax_d.set_xlim(0, T_ms)
    ax_d.set_ylim(0, 1)
    ax_d.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    ax_d.set_ylabel("readout p(class)", fontsize=theme.SIZE_LABEL)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    fig.suptitle(
        f"Streaming digit classification — per-segment $(\\tau, $ rate$)$ varies, "
        f"trained PING (no retraining)",
        fontsize=theme.SIZE_TITLE,
    )
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def plot_grid_heatmap(rows: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    taus = sorted(set(r["tau_ms"] for r in rows))
    rates = sorted(set(r["input_rate_hz"] for r in rows))
    grid = np.zeros((len(rates), len(taus)), dtype=np.float32)
    for r in rows:
        i = rates.index(r["input_rate_hz"])
        j = taus.index(r["tau_ms"])
        grid[i, j] = r["acc"]

    fig, ax = plt.subplots(figsize=(6.9, 5.06))
    im = ax.imshow(
        grid, origin="lower", aspect="auto", cmap="magma",
        vmin=0, vmax=100,
    )
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([f"{t:g}" for t in taus])
    ax.set_yticks(range(len(rates)))
    ax.set_yticklabels([f"{r:g}" for r in rates])
    ax.set_xlabel(r"Segment duration $\tau$ (ms)",
                  fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Input Poisson rate (Hz / channel)",
                   fontsize=theme.SIZE_LABEL)
    for i in range(len(rates)):
        for j in range(len(taus)):
            ax.text(
                j, i, f"{grid[i, j]:.0f}",
                ha="center", va="center",
                fontsize=theme.SIZE_LABEL,
                color=("white" if grid[i, j] < 55 else theme.INK_BLACK),
            )
    # Reference lines at the sub-cycle accuracy floor (τ ≈ 0.4 T_γ ≈ 15 ms)
    # and at one gamma period (T_γ ≈ 40 ms at f_γ ≈ 25 Hz). The x-axis is
    # discrete index; interpolate the τ value into a fractional index.
    F_GAMMA_HZ = 25.0
    T_GAMMA_MS = 1000.0 / F_GAMMA_HZ  # ≈ 40 ms

    def _tau_to_index(target_ms: float) -> float:
        if target_ms <= taus[0]:
            return 0.0
        if target_ms >= taus[-1]:
            return float(len(taus) - 1)
        for k in range(len(taus) - 1):
            if taus[k] <= target_ms <= taus[k + 1]:
                frac = (target_ms - taus[k]) / (taus[k + 1] - taus[k])
                return k + frac
        return float(len(taus) - 1)

    idx_subcycle = _tau_to_index(0.4 * T_GAMMA_MS)  # ≈ 15 ms (paper's floor)
    idx_tgamma   = _tau_to_index(T_GAMMA_MS)        # ≈ 40 ms

    ax.axvline(idx_subcycle, color=theme.FAINT, lw=0.8, ls=":", alpha=0.85,
               zorder=10)
    ax.axvline(idx_tgamma,   color=theme.PAPER, lw=1.4, ls="--", alpha=0.95,
               zorder=10)
    # Annotate just above the heatmap top.
    y_top = len(rates) - 0.5
    ax.annotate(
        r"$0.4\,T_\gamma$",
        xy=(idx_subcycle, y_top), xytext=(idx_subcycle, y_top + 0.45),
        ha="center", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.MUTED,
    )
    ax.annotate(
        r"$T_\gamma \approx 40$ ms",
        xy=(idx_tgamma, y_top), xytext=(idx_tgamma, y_top + 0.45),
        ha="center", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.INK,
    )
    ax.set_ylim(-0.5, y_top + 1.4)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Per-segment accuracy (%)", fontsize=theme.SIZE_LABEL)
    fig.suptitle(
        "Streaming accuracy across $(\\tau,$ input rate$)$ on trained PING",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # heatmap: PNG, not SVG
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    # Publication profile: every figure this notebook writes is a print-sized
    # artifact, emitted in both web (SVG/PNG) and manuscript (PDF) formats by
    # save_figure.
    theme.set_paper_mode(True)

    if "--replot-grid" in sys.argv:
        cached = FIGURES / "numbers.json"
        if not cached.exists():
            raise SystemExit(
                f"--replot-grid: no cached data at {cached}; run the full notebook first."
            )
        data = json.loads(cached.read_text())
        grid_agg = data.get("grid_sweep_agg", [])
        if not grid_agg:
            raise SystemExit("--replot-grid: cached numbers.json has no grid_sweep_agg.")
        plot_grid_heatmap(
            grid_agg, FIGURES / "acc_grid_tau_rate", "nb048-replot",
        )
        print(f"wrote {FIGURES / 'acc_grid_tau_rate'}.png (replotted from cache)")
        return

    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    cfg_tier = TIER_CONFIG[tier]
    notebook_run_id = next_run_id(SLUG)
    prepare_run_dirs(SLUG, notebook_run_id, wipe=False, make_artifacts=True)

    t_start = time.monotonic()
    from cli import _auto_device
    device = _auto_device()
    print(f"[streaming] device={device}  tier={tier}  seeds={SEEDS}")

    # Headlines (Figs 1, 3) use the first seed only — they're single-trial
    # demos, not aggregate measurements.
    net, cfg, X_te, y_te = _load_trained_full(device, seed=SEEDS[0])

    print(f"[headline] τ = {TAU_HEADLINE_MS:g} ms × {N_DIGITS_HEADLINE} digits "
          f"(seed {SEEDS[0]})")
    s = run_headline_stream(net, cfg, X_te, y_te, device)
    correct = sum(s["seg_correct"])
    print(f"  labels={s['labels']}  correct={correct}/{N_DIGITS_HEADLINE}")
    plot_headline_stream(s, FIGURES / "headline_stream", notebook_run_id)
    print(f"wrote {FIGURES / 'headline_stream'}.png")

    print(f"[varying-headline] segments={VARYING_HEADLINE} (seed {SEEDS[0]})")
    v = run_varying_headline(net, cfg, X_te, y_te, device)
    print(f"  labels={v['labels']}  correct="
          f"{sum(v['seg_correct'])}/{len(VARYING_HEADLINE)}")
    plot_varying_headline_stream(
        v, FIGURES / "varying_headline_stream", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'varying_headline_stream'}.png")

    # Multi-seed sweeps: τ-sweep and grid both run per-seed and average.
    rows_constant: list[dict] = []
    rows_comp: list[dict] = []
    grid_rows: list[dict] = []
    for sd in SEEDS:
        if sd != SEEDS[0]:
            net, cfg, X_te, y_te = _load_trained_full(device, seed=sd)
        print(f"[tau-sweep] constant input ({INPUT_RATE_HZ:g} Hz)  seed {sd}")
        rows_constant += run_tau_sweep(
            net, cfg, X_te, y_te, device, train_seed=sd,
            n_streams=int(cfg_tier["n_streams"]),
            n_per_stream=int(cfg_tier["n_per_stream"]),
            rate_compensate=False,
        )
        print(f"[tau-sweep] rate-compensated  seed {sd}")
        rows_comp += run_tau_sweep(
            net, cfg, X_te, y_te, device, train_seed=sd,
            n_streams=int(cfg_tier["n_streams"]),
            n_per_stream=int(cfg_tier["n_per_stream"]),
            rate_compensate=True,
        )
        print(f"[grid-sweep] τ × rate "
              f"({len(TAU_GRID_MS)}×{len(RATE_GRID_HZ)} cells)  seed {sd}")
        grid_rows += run_grid_sweep(
            net, cfg, X_te, y_te, device, train_seed=sd,
            n_streams=int(cfg_tier["n_grid_streams"]),
            n_per_stream=int(cfg_tier["n_per_stream"]),
        )

    tau_agg = (
        aggregate_tau_rows(rows_constant) + aggregate_tau_rows(rows_comp)
    )
    plot_acc_vs_tau(tau_agg, FIGURES / "acc_vs_tau", notebook_run_id)
    print(f"wrote {FIGURES / 'acc_vs_tau'}.svg")

    grid_agg = aggregate_grid_rows(grid_rows)
    plot_grid_heatmap(
        grid_agg, FIGURES / "acc_grid_tau_rate", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'acc_grid_tau_rate'}.png")

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "config": {
            "tier": tier,
            "n_e": N_E, "n_i": N_I, "n_in": N_IN, "n_classes": N_CLASSES,
            "dt": DT, "trained_t_ms": TRAINED_T_MS,
            "tau_headline_ms": TAU_HEADLINE_MS,
            "n_digits_headline": N_DIGITS_HEADLINE,
            "tau_sweep_ms": TAU_SWEEP_MS,
            "tau_grid_ms": TAU_GRID_MS,
            "rate_grid_hz": RATE_GRID_HZ,
            "input_rate_hz": INPUT_RATE_HZ,
            "n_streams": cfg_tier["n_streams"],
            "n_grid_streams": cfg_tier["n_grid_streams"],
            "n_per_stream": cfg_tier["n_per_stream"],
            "train_seeds": SEEDS,
            "seed": SEED,
        },
        "headline": {
            "labels": s["labels"],
            "seg_correct": s["seg_correct"],
        },
        "tau_sweep_per_seed": rows_constant + rows_comp,
        "tau_sweep_agg": tau_agg,
        "varying_headline": {
            "segments": [list(seg) for seg in VARYING_HEADLINE],
            "labels": v["labels"],
            "seg_preds": v["seg_preds"],
            "seg_correct": v["seg_correct"],
        },
        "grid_sweep_per_seed": grid_rows,
        "grid_sweep_agg": grid_agg,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
