"""Measure explicit compute evidence; never simulate or render."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp099 import inputs, recipe
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run
from scipy.signal import welch
from tools.snnsim.metrics import rhythmicity_metrics  # noqa: TID251


def activity_diagnostics(spikes, dt):
    """Irregularity, sampled pair correlation, and population spectrum."""
    cvs = []
    for cell in range(spikes.shape[1]):
        isi = np.diff(np.flatnonzero(spikes[:, cell]))
        if len(isi) >= 4:
            cvs.append(float(isi.std() / isi.mean()))
    width = round(10 / dt)
    sample = spikes[: len(spikes) // width * width, :: max(1, spikes.shape[1] // 100)]
    binned = sample.reshape(-1, width, sample.shape[1]).sum(1)
    valid = binned.std(0) > 0
    corr = np.corrcoef(binned[:, valid].T) if valid.sum() > 1 else None
    width1 = round(1 / dt)
    population = (
        spikes[: len(spikes) // width1 * width1]
        .reshape(-1, width1, spikes.shape[1])
        .sum((1, 2))
    )
    freq, power = welch(
        population.astype(float), fs=1000, nperseg=min(500, len(population))
    )
    band = (freq >= 20) & (freq <= 100)
    total = power[(freq >= 1) & (freq <= 200)].sum()
    rhythm = rhythmicity_metrics(spikes, dt, max_lag_ms=100.0, bin_ms=1.0)
    return {
        "autocorrelation_contrast": rhythm["contrast"],
        "active_fraction": float(spikes.any(0).mean()),
        "median_isi_cv": float(np.median(cvs)) if cvs else None,
        "cells_with_cv": len(cvs),
        "mean_pair_correlation_10ms": float(corr[np.triu_indices(len(corr), 1)].mean())
        if corr is not None
        else None,
        "population_peak_20_100_hz": float(freq[band][np.argmax(power[band])])
        if total > 0
        else None,
        "population_power_fraction_20_100_hz": float(power[band].sum() / total)
        if total > 0
        else None,
    }


def measure(data, cfg):
    dt = cfg["dt_ms"]
    times = np.arange(len(data["spk_e"])) * dt
    arrays = {
        "time_ms": times,
        "mean_g_e": data["mean_E_to_E"] + data["mean_private_e_to_E"],
        "mean_g_i": data["mean_I_to_E"],
        "mean_v_e": data["mean_v_e"],
        "mean_v_i": data["mean_v_i"],
    }
    arrays["rate_private_e"], arrays["rate_private_i"] = recipe.source_rates(times, cfg)
    width = round(20 / dt)
    for pop in ("e", "i"):
        raw = data[f"spk_{pop}"].mean(1) * 1000 / dt
        # Causal display smoothing never leaks a later spike into an earlier frame.
        arrays[f"rate_{pop}"] = np.convolve(raw, np.ones(width) / width, mode="full")[
            : len(raw)
        ]
    epochs = {
        "visible_baseline": (cfg["view_start_ms"], cfg["onset_ms"]),
        "plateau": (cfg["peak_ms"], cfg["plateau_end_ms"]),
        "recovery": (cfg["offset_ms"], cfg["t_ms"]),
    }
    summary = {
        "total_e_spikes": int(data["spk_e"].sum()),
        "total_i_spikes": int(data["spk_i"].sum()),
        "epochs": {},
    }
    for name, (start, stop) in epochs.items():
        sl = slice(round(start / dt), round(stop / dt))
        summary["epochs"][name] = {
            f"{p}_hz": float(
                data[f"spk_{p}"][sl].sum() / (cfg[f"n_{p}"] * (stop - start) / 1000)
            )
            for p in ("e", "i")
        }
        for p in ("e", "i"):
            summary["epochs"][name][p + "_diagnostics"] = activity_diagnostics(
                data[f"spk_{p}"][sl], dt
            )
    summary["interpretation"] = (
        "No E or I spikes: neither AI nor PING was established at the specified parameters."
        if summary["total_e_spikes"] == summary["total_i_spikes"] == 0
        else "Spiking occurred; a PING claim requires further E/I timing and rhythmicity analysis."
    )
    return arrays, summary


def analyse(identity, *, run_id=None):
    compute = inputs.source(REPO, identity, "compute")
    cfg = inputs.configuration(compute)
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        arrays, summary = measure(inputs.recording(compute), cfg)
        np.savez_compressed(run.export / "measurements.npz", **arrays)
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp099.analysis/v2",
                "parameters": cfg,
                "measurements": {
                    "population_rate_window_ms": 20.0,
                    "population_rate_alignment": "causal trailing",
                    "isi_cv": "median across neurons with at least five spikes in epoch",
                    "rhythmicity": "SNNSIM rhythmicity_metrics contrast, 1 ms bins, maximum lag 100 ms",
                    "pair_correlation": "mean Pearson correlation of 10 ms counts across at most 100 evenly sampled neurons with nonzero variance",
                    "spectrum": "Welch population counts, 1 ms bins, 500 ms segments, default Hann window and 50 percent overlap; peak 20-100 Hz and band fraction relative to 1-200 Hz; not a PING classifier",
                },
                "results": summary,
            },
        )
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    analyse(a.source, run_id=a.run_id)


if __name__ == "__main__":
    main()
