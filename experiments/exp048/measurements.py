"""Preserved decoder and across-seed estimators; no simulation or plotting."""

import numpy as np

from .recipe import DT, N_CLASSES, TRAINED_T_MS


def _v_out_series(
    spk_e: np.ndarray,
    W_out: np.ndarray,
    tau_out_ms: float,
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
    spk_e: np.ndarray,
    W_out: np.ndarray,
    tau_out_ms: float,
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
    csum = np.concatenate(
        [
            np.zeros((1, N_CLASSES), dtype=np.float32),
            np.cumsum(v_out_series, axis=0),
        ]
    )  # (T+1, C)
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
        out.append(
            {
                "tau_ms": float(tau_ms),
                "input_rate_hz": float(rate_hz),
                "acc": float(a.mean()),
                "acc_sem": float(a.std(ddof=1) / np.sqrt(len(a)))
                if len(a) > 1
                else 0.0,
                "n_seeds": int(len(a)),
                "n_total": int(n_totals[(tau_ms, rate_hz)]),
            }
        )
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
        out.append(
            {
                "tau_ms": float(tau_ms),
                "rate_compensate": bool(rate_compensate),
                "input_rate_hz": float(
                    extras[(tau_ms, rate_compensate)]["input_rate_hz"]
                ),
                "acc": float(a.mean()),
                "acc_sem": float(a.std(ddof=1) / np.sqrt(len(a)))
                if len(a) > 1
                else 0.0,
                "n_seeds": int(len(a)),
            }
        )
    return out


def rate_curve(grid_rows: list[dict], low_rate_rows: list[dict]) -> list[dict]:
    reused = [
        {
            **row,
            "accuracy": row["acc"] / 100.0,
            "accuracy_sem": row["acc_sem"] / 100.0,
            "source": "exp048 grid",
        }
        for row in grid_rows
        if row["tau_ms"] == TRAINED_T_MS
    ]
    return sorted([*low_rate_rows, *reused], key=lambda row: row["input_rate_hz"])
