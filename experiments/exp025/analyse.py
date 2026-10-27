"""Measure retained exp025 evidence and prepare plot data; never simulate.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
import json

import numpy as np
from experiments.exp025 import recipe
from experiments.exp025.custom import evidence, inputs
from experiments.exp025.recipe import F_GAMMA_BAND_HZ, MODELS, RATE_TARGET_GRID_HZ
from pingstore.library import PingstoreError, write_json_atomic

MEASUREMENT_V1 = {
    "schema": "exp025.measurement/v1",
    "frontier": "final-epoch official-test accuracy and mean E rate; three independent seeds, mean and SEM",
    "pfg": "representative seed 42; Welch E-population frequency; per-trial frequency anchors inhibitory peak cycles; active cell-cycle fraction",
    "penalty": "0.041 times mean squared positive overshoot of per-sample E rate above 1 Hz",
    "low_w_in": "retained validation histories; original fallback rules and seed aggregation",
    "crossing": "midpoint of first adjacent scale pair crossing I rate 0.05 Hz",
}
MEASUREMENT = {
    **MEASUREMENT_V1,
    "schema": "exp025.measurement/v2",
    "low_w_in": "retained validation histories; prefer explicit test-rate fields, including zero; original seed aggregation",
}


def history_value(row, preferred, fallback=None):
    value = row.get(preferred)
    if value is None and fallback is not None:
        value = row.get(fallback)
    return float(value) if value is not None else 0.0


def low_curve(histories):
    def series(key, fallback=None):
        return np.asarray(
            [
                [history_value(e, key, fallback) for e in history]
                for history in histories
            ],
            dtype=float,
        )

    result = {"epochs": list(range(1, len(histories[0]) + 1))}
    for name, key, fallback in (
        ("acc", "acc", None),
        ("rate_e", "test_rate_e", "rate_e"),
        ("rate_i", "test_rate_i", "rate_i"),
    ):
        values = series(key, fallback)
        result[name + "_mean"] = values.mean(axis=0).tolist()
        result[name + "_sem"] = (
            values.std(axis=0, ddof=1) / np.sqrt(len(histories))
            if len(histories) > 1
            else np.zeros(values.shape[1])
        ).tolist()
    result["rate_max"] = float(
        max(
            series("test_rate_e", "rate_e").max(), series("test_rate_i", "rate_i").max()
        )
    )
    return result


def analyse(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "compute")
    cfg, bank, contract = inputs.compute_evidence(REPO, source)
    history = evidence.histories(bank.export, contract)
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": source, "bank": bank},
        run_id=run_id,
        configuration=MEASUREMENT,
    ) as run:
        frontier = []
        pfg = []
        scales = []
        by_name = {c["cell_name"]: c for c in contract["cells"]}
        for job in recipe.jobs(cfg):
            name = job["cell_name"]
            cell = by_name[name]
            train = contract["configs"][name]
            if job["kind"] == "snapshot":
                artifact = source.file(job["path"], "recording.npz")
            elif job["kind"] == "frontier":
                artifact = source.file(job["path"], "metrics.json")
            else:
                artifact = source.unit(job["path"])
                if not artifact.is_dir():
                    artifact = source.file(job["path"], "metrics.json")
            evidence.recordings(artifact, train, job)
            if job["kind"] == "snapshot":
                with np.load(artifact, allow_pickle=False) as raw:
                    et, ec = np.where(raw["spk_e"])
                    it, ic = np.where(raw["spk_i"])
                    np.savez_compressed(
                        run.export / f"raster__{job['model']}.npz",
                        dt=raw["dt"],
                        T=raw["spk_e"].shape[0],
                        n_e=raw["spk_e"].shape[1],
                        n_i=raw["spk_i"].shape[1],
                        e_t=et,
                        e_cell=ec,
                        i_t=it,
                        i_cell=ic,
                    )
            elif job["kind"] == "frontier":
                m = evidence.metric(artifact, train, job)
                frontier.append(
                    {
                        "cell_name": name,
                        "model": cell["model"],
                        "rate_target_hz": cell["rate_target_hz"],
                        "rate_target_display": recipe.rate_target_display(
                            cell["rate_target_hz"]
                        ),
                        "seed": cell["seed"],
                        "best_acc": float(history[name]["best_acc"]),
                        "best_epoch": int(history[name]["best_epoch"]),
                        "final_acc": float(m["best_acc"]),
                        "rate_e": float(m["rates_hz"]["hid"]),
                        "evaluation_partition": "official_mnist_test",
                        "evaluation_samples": job["samples"],
                        "checkpoint_role": recipe.CHECKPOINT_ROLE,
                    }
                )
            elif job["kind"] == "pfg":
                m = measure_p_fgamma(artifact, train["dt"], job["is_ping"])
                pfg.append(
                    {
                        "model": cell["model"],
                        "rate_target_hz": cell["rate_target_hz"],
                        "seed": 42,
                        "selection": "representative_seed",
                        **m,
                        "p_times_f_gamma": m["p"] * m["f_gamma"]
                        if m["p"] is not None and m["f_gamma"] is not None
                        else None,
                    }
                )
            else:
                acc, loss, penalty, er, ir = scaled_metrics(
                    artifact, 1.0, recipe.FR_STRENGTH_UPPER
                )
                scales.append(
                    {
                        "cell": job["label"],
                        "scale": job["scale"],
                        "loss": loss,
                        "penalty": penalty,
                        "total_loss": loss + penalty,
                        "acc": acc,
                        "rate_e": er,
                        "rate_i": ir,
                    }
                )
        low = []
        low_curves = {}
        baseline_curves = {}
        for w in recipe.LOW_W_IN_VALUES:
            per_seed = []
            hist = []
            for s in cfg["low_w_in_seeds"]:
                m = history[recipe.low_w_in_cell_name(w, s)]
                last = m["epochs"][-1]
                hist.append(m["epochs"])
                per_seed.append(
                    {
                        "seed": s,
                        "best_acc": float(m["best_acc"]),
                        "best_epoch": int(m["best_epoch"]),
                        "final_acc": float(last["acc"]),
                        "rate_e": history_value(last, "test_rate_e", "rate_e"),
                        "rate_i": history_value(last, "test_rate_i", "rate_i"),
                    }
                )
            low.append(aggregate_low_w_in_seed_rows(w, per_seed))
            low_curves[f"{w:g}"] = low_curve(hist)
        for model in recipe.MODELS:
            histories = [
                history[recipe.cell_name(model, None, s)]["epochs"]
                for s in recipe.SEEDS
            ]
            baseline_curves[model] = {
                "epochs": [e["ep"] for e in histories[0]],
                "acc_mean": np.asarray([[e["acc"] for e in h] for h in histories])
                .mean(0)
                .tolist(),
            }
        ping = sorted(
            [r for r in scales if r["cell"] == "ping@rt1hz"], key=lambda r: r["scale"]
        )
        crossing = next(
            (
                0.5 * (a["scale"] + b["scale"])
                for a, b in pairwise(ping)
                if a["rate_i"] < 0.05 <= b["rate_i"]
            ),
            None,
        )
        result = {
            "schema": "exp025.analysis/v2",
            "recipe": cfg,
            "results": frontier,
            "frontier_statistics": aggregate_frontier(frontier),
            "rate_target_p_fgamma": pfg,
            "low_w_in_sweep": low,
            "w_in_scale_sweep": scales,
            "plot_data": {
                "low_w_in": low_curves,
                "baseline": baseline_curves,
                "scale_crossing": crossing,
            },
        }
        write_json_atomic(run.export / "results.json", result)
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sources = p.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source")
    sources.add_argument("--reuse-analysis")
    p.add_argument("--baseline-bank")
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        if a.reuse_analysis:
            if not a.baseline_bank:
                p.error("--reuse-analysis requires --baseline-bank")
            from experiments.exp025.custom.full_mnist_baseline import (
                analyse as baseline_analyse,
            )

            baseline_analyse(REPO, a.reuse_analysis, a.baseline_bank, run_id=a.run_id)
        else:
            if a.baseline_bank:
                p.error("--baseline-bank requires --reuse-analysis")
            analyse(a.source, run_id=a.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        p.exit(1, f"exp025 analyse: {exc}\n")


def aggregate_frontier(rows: list[dict]) -> list[dict]:
    """Summarise each model/budget point across independent seeds."""
    aggregates: list[dict] = []
    for model in MODELS:
        for rate_target_hz in RATE_TARGET_GRID_HZ:
            cell_rows = [
                row
                for row in rows
                if row["model"] == model and row["rate_target_hz"] == rate_target_hz
            ]
            if not cell_rows:
                continue
            accs = np.asarray([row["final_acc"] for row in cell_rows], dtype=float)
            rates = np.asarray([row["rate_e"] for row in cell_rows], dtype=float)
            n = len(cell_rows)
            aggregates.append(
                {
                    "model": model,
                    "rate_target_hz": rate_target_hz,
                    "rate_target_display": cell_rows[0]["rate_target_display"],
                    "statistic": "mean_across_independent_seeds",
                    "uncertainty": "sem_across_independent_seeds",
                    "n_seeds": n,
                    "seeds": [int(row["seed"]) for row in cell_rows],
                    "cell_names": [row["cell_name"] for row in cell_rows],
                    "acc_mean": float(accs.mean()),
                    "rate_mean": float(rates.mean()),
                    "acc_sem": float(accs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                    "rate_sem": float(rates.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                }
            )
    return aggregates


def aggregate_low_w_in_seed_rows(w_in: float, seed_rows: list[dict]) -> dict:
    """Aggregate one local control across independent training seeds."""
    if not seed_rows:
        raise ValueError("low-W_in aggregation requires at least one seed")

    def mean_sem(key: str) -> tuple[float, float]:
        values = np.asarray([row[key] for row in seed_rows], dtype=float)
        sem = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        )
        return float(values.mean()), sem

    final_acc, final_acc_sem = mean_sem("final_acc")
    rate_e, rate_e_sem = mean_sem("rate_e")
    rate_i, rate_i_sem = mean_sem("rate_i")
    return {
        "w_in": float(w_in),
        "seeds": [int(row["seed"]) for row in seed_rows],
        "n_seeds": len(seed_rows),
        "statistic": "mean_across_independent_seeds",
        "uncertainty": "sem_across_independent_seeds",
        "final_acc": final_acc,
        "final_acc_sem": final_acc_sem,
        "rate_e": rate_e,
        "rate_e_sem": rate_e_sem,
        "rate_i": rate_i,
        "rate_i_sem": rate_i_sem,
        "per_seed": seed_rows,
    }


def _f_gamma_from_population(
    pop_traces: list[np.ndarray],
    fs_hz: float,
) -> float:
    """Welch PSD per trial, averaged across trials, peak frequency in band
    via parabolic interpolation. Returns NaN if the spectrum is flat or
    the population is silent."""
    from scipy import signal as sp_signal

    if not pop_traces or pop_traces[0].size == 0:
        return float("nan")
    nperseg = pop_traces[0].size
    psds: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for tr in pop_traces:
        if tr.std() == 0:
            continue
        f, p = sp_signal.welch(
            tr - tr.mean(),
            fs=fs_hz,
            nperseg=nperseg,
            scaling="density",
            detrend=False,
        )
        psds.append(p)
        freqs = f
    if not psds or freqs is None:
        return float("nan")
    psd_mean = np.mean(np.stack(psds, axis=0), axis=0)
    band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
    if not band.any() or psd_mean[band].max() <= 0:
        return float("nan")
    in_band = np.where(band)[0]
    peak_local = int(psd_mean[in_band].argmax())
    peak_idx = int(in_band[peak_local])
    if not (0 < peak_idx < len(psd_mean) - 1):
        return float(freqs[peak_idx])
    y0, y1, y2 = (
        float(psd_mean[peak_idx - 1]),
        float(psd_mean[peak_idx]),
        float(psd_mean[peak_idx + 1]),
    )
    denom = y0 - 2.0 * y1 + y2
    offset = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    offset = max(-0.5, min(0.5, offset))
    df = float(freqs[1] - freqs[0])
    return float(freqs[peak_idx] + offset * df)


def _detect_i_burst_peaks(
    s_i_trial: np.ndarray,
    dt_ms: float,
    f_gamma_hz: float,
) -> np.ndarray:
    """Mirrors exp046.detect_i_burst_steps. Smooth I population rate with
    1-ms Gaussian, find peaks separated by at least 0.5 cycle."""
    from scipy.signal import find_peaks

    rate = s_i_trial.sum(axis=1).astype(np.float32)
    sigma_steps = max(1.0, 1.0 / dt_ms)
    L = int(np.ceil(4 * sigma_steps))
    k = np.arange(-L, L + 1)
    kernel = np.exp(-0.5 * (k / sigma_steps) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(rate, kernel, mode="same")
    cycle_steps = max(1.0, 1000.0 / max(f_gamma_hz, 1e-3) / dt_ms)
    height = 0.05 * float(smooth.max()) if smooth.max() > 0 else 0.0
    peaks, _ = find_peaks(
        smooth,
        distance=max(1, int(0.5 * cycle_steps)),
        height=height,
    )
    return peaks


def _count_e_spikes_per_cycle(
    s_e_trial: np.ndarray,
    peak_steps: np.ndarray,
) -> np.ndarray:
    """Mirrors exp046.count_e_spikes_per_cycle."""
    T, N_E = s_e_trial.shape
    K = len(peak_steps)
    if K == 0:
        return np.zeros((0, N_E), dtype=np.int32)
    edges = np.concatenate(
        [
            [0],
            ((peak_steps[:-1] + peak_steps[1:]) // 2).astype(int),
            [T],
        ]
    )
    counts = np.zeros((K, N_E), dtype=np.int32)
    for kk in range(K):
        a, b = edges[kk], edges[kk + 1]
        if b > a:
            counts[kk] = s_e_trial[a:b].sum(axis=0)
    return counts


def measure_p_fgamma(out_dir: Path, dt_ms: float, is_ping: bool) -> dict:
    fs_hz = 1000.0 / dt_ms
    metrics = out_dir if out_dir.is_file() else out_dir / "metrics.json"
    m = json.loads(metrics.read_text())
    rates = m.get("rates_hz", {})
    acc = float(m["best_acc"])
    e_rate = float(rates.get("hid", 0.0))
    i_rate = float(rates.get("inh", 0.0))
    if not is_ping:
        return {
            "acc": acc,
            "e_rate": e_rate,
            "i_rate": i_rate,
            "f_gamma": None,
            "p": None,
        }

    if not out_dir.is_dir():
        raise ValueError("PING frequency evidence requires a multi-file unit")

    pt = np.load(out_dir / "pop_traces.npz")
    pop_e_traces = list(pt["pop_e"])
    f_gamma_val = _f_gamma_from_population(pop_e_traces, fs_hz)
    f_gamma = float(f_gamma_val) if np.isfinite(f_gamma_val) else None

    # Reconstruct per-trial dense E/I rasters from the sparse indices and count
    # (cell, cycle) participation. Cycle boundaries come from the I-burst peaks.
    R = np.load(out_dir / "rasters.npz")
    T, n_e, n_i = int(R["T"]), int(R["n_e"]), int(R["n_i"])
    n_trials = min(int(R["n_trials"]), len(pop_e_traces))

    def _by_trial(prefix):
        tr = R[f"{prefix}_trial"]
        order = np.argsort(tr, kind="stable")
        return (
            R[f"{prefix}_t"][order],
            R[f"{prefix}_cell"][order],
            np.searchsorted(tr[order], np.arange(n_trials + 1)),
        )

    e_t, e_c, e_b = _by_trial("e")
    i_t, i_c, i_b = _by_trial("i")

    n_cycle_pairs = 0
    n_cycle_pairs_active = 0
    for b in range(n_trials):
        s_i_trial = np.zeros((T, n_i), dtype=np.int8)
        s_i_trial[i_t[i_b[b] : i_b[b + 1]], i_c[i_b[b] : i_b[b + 1]]] = 1
        f_gamma_batch = _f_gamma_from_population([pop_e_traces[b]], fs_hz)
        if not np.isfinite(f_gamma_batch) or f_gamma_batch <= 0:
            continue
        peaks = _detect_i_burst_peaks(s_i_trial, dt_ms, f_gamma_batch)
        if peaks.size == 0:
            continue
        s_e_trial = np.zeros((T, n_e), dtype=np.int8)
        s_e_trial[e_t[e_b[b] : e_b[b + 1]], e_c[e_b[b] : e_b[b + 1]]] = 1
        counts = _count_e_spikes_per_cycle(s_e_trial, peaks)  # (K, N_E)
        n_cycle_pairs += counts.size
        n_cycle_pairs_active += int((counts > 0).sum())

    p = (
        float(n_cycle_pairs_active) / float(n_cycle_pairs)
        if n_cycle_pairs > 0
        else None
    )
    return {"acc": acc, "e_rate": e_rate, "i_rate": i_rate, "f_gamma": f_gamma, "p": p}


def scaled_metrics(out_dir: Path, fr_upper_target_hz: float, fr_upper_strength: float):
    m = json.loads((out_dir / "metrics.json").read_text())
    rates = m.get("rates_hz", {})
    penalty = 0.0
    if fr_upper_strength > 0:
        pc = np.load(out_dir / "per_cell_rates.npz")
        sample_rates = pc["rate_e_per_sample"]
        penalty = float(
            fr_upper_strength
            * (np.maximum(sample_rates - fr_upper_target_hz, 0.0) ** 2).mean()
        )
    return (
        float(m["best_acc"]),
        float(m["ce_loss"]),
        penalty,
        float(rates.get("hid", 0.0)),
        float(rates.get("inh", 0.0)),
    )


if __name__ == "__main__":
    main()
