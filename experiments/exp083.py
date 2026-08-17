"""EXP083: characterize the default SNNLANG PING response to Poisson drive.

One authored graph is held fixed while homogeneous input rate varies.  The
experiment is deliberately small, local, and graph-native: it demonstrates the
ordinary SNNLANG path while asking whether the default PING component contains
a reproducible gamma regime without parameter tuning.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution import ExecutionSpec, simulate  # noqa: E402
from tools import snnlang as snn  # noqa: E402, TID251

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.gamma_frequency import (  # noqa: E402
    DEFAULT_PING_GAMMA,
    GammaFrequencyEstimate,
    estimate_gamma_from_raster,
)
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp083"
DT_MS = 0.1
T_MS = 1_000.0
BURN_MS = 200.0
N_INPUT = 128
N_E = 80
N_I = 20
INPUT_RATES_HZ = (0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0)
TRIAL_SEEDS = (8300, 8301, 8302, 8303, 8304)
NETWORK_SEED = 83
REPRESENTATIVE_RATES_HZ = (25.0, 75.0, 150.0)
DISPLAY_TRIAL = 0

GAMMA_CONFIG = replace(DEFAULT_PING_GAMMA, burn_ms=BURN_MS)

SCALE = {
    "dt_ms": DT_MS,
    "t_ms": T_MS,
    "burn_ms": BURN_MS,
    "n_input": N_INPUT,
    "n_e": N_E,
    "n_i": N_I,
    "rates_hz": list(INPUT_RATES_HZ),
    "trials": len(TRIAL_SEEDS),
    "network_seed": NETWORK_SEED,
}


def author_network() -> snn.Bundle:
    net = snn.Network("default_ping_drive_response", dt=DT_MS * snn.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", N_INPUT),
        signal_type="spikes",
        unit="spike",
    )
    cell = snn.components.ping(
        net,
        name="ping",
        n_e=N_E,
        n_i=N_I,
        source=drive,
    )
    net.expose(cell.E.spikes, cell.I.spikes, name="population")
    return snn.compile(net, target="tools/snn")


def make_inputs(rate_hz: float) -> np.ndarray:
    """Paired deterministic trials: each seed is reused at every drive rate."""
    steps = round(T_MS / DT_MS)
    probability = rate_hz * DT_MS / 1_000.0
    trials = []
    for seed in TRIAL_SEEDS:
        rng = np.random.default_rng(seed)
        trials.append(rng.random((steps, N_INPUT), dtype=np.float32) < probability)
    return np.stack(trials, axis=1).astype(np.uint8)


def _phase_lag_ms(e_spikes: np.ndarray, i_spikes: np.ndarray) -> float | None:
    burn = round(BURN_MS / DT_MS)
    bin_steps = round(1.0 / DT_MS)
    e = e_spikes[burn:].mean(axis=1)
    i = i_spikes[burn:].mean(axis=1)
    usable = len(e) // bin_steps * bin_steps
    if usable == 0:
        return None
    e = e[:usable].reshape(-1, bin_steps).sum(axis=1)
    i = i[:usable].reshape(-1, bin_steps).sum(axis=1)
    if e.std() == 0 or i.std() == 0:
        return None
    max_lag = 20
    correlation = np.correlate(e - e.mean(), i - i.mean(), mode="full")
    lags = np.arange(-len(e) + 1, len(e))
    keep = np.abs(lags) <= max_lag
    return float(lags[keep][np.argmax(correlation[keep])])


def _trial_rows(
    rate_hz: float,
    e_spikes: np.ndarray,
    i_spikes: np.ndarray,
    gamma: GammaFrequencyEstimate,
) -> list[dict]:
    burn = round(BURN_MS / DT_MS)
    duration_s = (T_MS - BURN_MS) / 1_000.0
    rows = []
    for index, seed in enumerate(TRIAL_SEEDS):
        peak = gamma.trials[index]
        rows.append(
            {
                "trial": index,
                "seed": seed,
                "input_rate_hz": rate_hz,
                "e_rate_hz": float(e_spikes[burn:, index].sum() / N_E / duration_s),
                "i_rate_hz": float(i_spikes[burn:, index].sum() / N_I / duration_s),
                "gamma": peak.json(),
                "e_i_peak_lag_ms": _phase_lag_ms(
                    e_spikes[:, index], i_spikes[:, index]
                ),
            }
        )
    return rows


def summarize_condition(rate_hz: float, rows: list[dict]) -> dict:
    resolved = [
        row["gamma"]["frequency_hz"] for row in rows if row["gamma"]["resolved"]
    ]
    lags = [
        row["e_i_peak_lag_ms"] for row in rows if row["e_i_peak_lag_ms"] is not None
    ]
    rhythmicity = [
        0.0
        if row["gamma"]["prominence_ratio"] is None
        else row["gamma"]["prominence_ratio"]
        for row in rows
    ]
    return {
        "input_rate_hz": rate_hz,
        "e_rate_mean_hz": float(np.mean([row["e_rate_hz"] for row in rows])),
        "e_rate_std_hz": float(np.std([row["e_rate_hz"] for row in rows], ddof=1)),
        "i_rate_mean_hz": float(np.mean([row["i_rate_hz"] for row in rows])),
        "i_rate_std_hz": float(np.std([row["i_rate_hz"] for row in rows], ddof=1)),
        "gamma_resolved_fraction": len(resolved) / len(rows),
        "rhythmicity_score_median": float(np.median(rhythmicity)),
        "rhythmicity_score_iqr": float(
            np.percentile(rhythmicity, 75) - np.percentile(rhythmicity, 25)
        ),
        "gamma_frequency_median_hz": None
        if not resolved
        else float(np.median(resolved)),
        "e_i_peak_lag_median_ms": None if not lags else float(np.median(lags)),
        "trials": rows,
    }


def plot_representative_rasters(
    recordings: dict[float, dict[str, np.ndarray]],
    summaries: list[dict],
    out: Path,
) -> None:
    theme.apply()
    by_rate = {row["input_rate_hz"]: row for row in summaries}
    gap = 6
    fig, axes = plt.subplots(
        3, 1, figsize=(6.5, 3.5), sharex=True, gridspec_kw={"hspace": 0.18}
    )
    for row, rate in enumerate(REPRESENTATIVE_RATES_HZ):
        arrays = recordings[rate]
        axis = axes[row]
        e_t, e_cells = np.nonzero(arrays["e"][:, DISPLAY_TRIAL])
        i_t, i_cells = np.nonzero(arrays["i"][:, DISPLAY_TRIAL])
        axis.scatter(
            e_t * DT_MS,
            e_cells,
            s=2.0,
            marker="|",
            linewidths=0.4,
            color=theme.INK_BLACK,
            rasterized=True,
        )
        axis.scatter(
            i_t * DT_MS,
            i_cells + N_E + gap,
            s=2.0,
            marker="|",
            linewidths=0.4,
            color=theme.DEEP_RED,
            rasterized=True,
        )
        axis.axvline(BURN_MS, color=theme.GREY_MID, ls="--", lw=0.8)
        axis.set_ylim(-2, N_E + gap + N_I + 2)
        axis.set_yticks([N_E / 2, N_E + gap + N_I / 2])
        axis.set_yticklabels(["E", "I"])
        axis.tick_params(axis="y", length=0)
        condition = by_rate[rate]
        axis.text(
            1.012,
            0.5,
            (
                f"{rate:g} Hz/channel\n"
                f"E {condition['e_rate_mean_hz']:.1f} · "
                f"I {condition['i_rate_mean_hz']:.1f} Hz"
            ),
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=theme.SIZE_ANNOTATION,
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlim(0, T_MS)
    axes[-1].set_xlabel("time (ms)")
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.15, top=0.98, hspace=0.18)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_response(summaries: list[dict], out: Path) -> None:
    theme.apply()
    x = np.array([row["input_rate_hz"] for row in summaries])
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 3.66))
    for key, std, label, colour in (
        ("e_rate_mean_hz", "e_rate_std_hz", "E", theme.INK_BLACK),
        ("i_rate_mean_hz", "i_rate_std_hz", "I", theme.DEEP_RED),
    ):
        axes[0].errorbar(
            x,
            [row[key] for row in summaries],
            yerr=[row[std] for row in summaries],
            marker="o",
            lw=1.3,
            capsize=3,
            color=colour,
            label=label,
        )
    axes[0].set_ylabel("rate (Hz)")
    axes[0].legend(frameon=False)
    rhythmicity = np.array([row["rhythmicity_score_median"] for row in summaries])
    rhythmicity_iqr = np.array([row["rhythmicity_score_iqr"] for row in summaries])
    axes[1].errorbar(
        x,
        rhythmicity,
        yerr=rhythmicity_iqr / 2.0,
        marker="o",
        capsize=3,
        color=theme.INK_BLACK,
    )
    axes[1].axhline(
        GAMMA_CONFIG.min_prominence_ratio,
        color=theme.DEEP_RED,
        ls="--",
        lw=1.2,
    )
    axes[1].set_ylim(bottom=-0.2)
    axes[1].set_ylabel("rhythmicity score")
    frequencies = [row["gamma_frequency_median_hz"] for row in summaries]
    axes[2].plot(x, frequencies, marker="o", color=theme.DEEP_RED)
    axes[2].axhspan(30, 80, color=theme.GREY_LIGHT, alpha=0.5)
    axes[2].set_ylim(20, 90)
    axes[2].set_ylabel("frequency (Hz)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("input rate per channel (Hz)")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spectra(estimates: dict[float, GammaFrequencyEstimate], out: Path) -> None:
    theme.apply()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for rate in REPRESENTATIVE_RATES_HZ:
        estimate = estimates[rate]
        frequencies = estimate.frequencies_hz
        keep = (frequencies >= 20) & (frequencies <= 100)
        power = estimate.mean_psd[keep]
        scale = power.max() if power.size and power.max() > 0 else 1.0
        ax.plot(frequencies[keep], power / scale, label=f"{rate:g} Hz input")
    ax.axvspan(30, 80, color=theme.GREY_LIGHT, alpha=0.45)
    ax.set(xlabel="frequency (Hz)", ylabel="mean PSD (peak-normalized)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = parse_meta(sys.argv)
    if meta.runpod:
        raise SystemExit(
            "exp083 is a bounded local experiment; RunPod is not supported"
        )
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id}")
    with published_run(SLUG, run_id, scale=SCALE, plot_only=meta.plot_only) as (
        scratch,
        staging,
    ):
        bundle = author_network()
        bundle_dir = staging / "network.bundle"
        bundle.write(bundle_dir, visualise=True)
        shutil.copy2(bundle_dir / "reports/circuit.svg", staging / "network.svg")

        all_recordings: dict[float, dict[str, np.ndarray]] = {}
        estimates: dict[float, GammaFrequencyEstimate] = {}
        summaries = []
        conditions = scratch / "conditions"
        conditions.mkdir(exist_ok=True)
        for rate in INPUT_RATES_HZ:
            print(f"[simulate] input rate {rate:g} Hz")
            input_spikes = make_inputs(rate)
            result = simulate(
                ExecutionSpec(
                    kind="simulate",
                    executor="graph",
                    graph=bundle.graph,
                    inputs={"drive": torch.from_numpy(input_spikes).float()},
                    seed=NETWORK_SEED,
                )
            )
            e_spikes = result.recordings["population_0"].cpu().numpy().astype(np.uint8)
            i_spikes = result.recordings["population_1"].cpu().numpy().astype(np.uint8)
            estimate = estimate_gamma_from_raster(
                e_spikes,
                dt_ms=DT_MS,
                config=GAMMA_CONFIG,
            )
            rows = _trial_rows(rate, e_spikes, i_spikes, estimate)
            summaries.append(summarize_condition(rate, rows))
            all_recordings[rate] = {"e": e_spikes, "i": i_spikes}
            estimates[rate] = estimate
            np.savez_compressed(
                conditions / f"rate-{rate:g}.npz",
                input_spikes=input_spikes,
                e_spikes=e_spikes,
                i_spikes=i_spikes,
            )

        plot_representative_rasters(
            all_recordings,
            summaries,
            staging / "representative_rasters.png",
        )
        plot_response(summaries, staging / "response.png")
        plot_spectra(estimates, staging / "spectra.png")
        shutil.copytree(conditions, staging / "conditions")
        payload = {
            "question": "Does the default SNNLANG PING component contain a reproducible gamma regime as homogeneous Poisson drive increases?",
            "config": SCALE,
            "gamma_frequency": GAMMA_CONFIG.json(),
            "representative_rates_hz": list(REPRESENTATIVE_RATES_HZ),
            "graph": {
                "digest": bundle.manifest["graph_digest"],
                "name": bundle.graph["name"],
            },
            "conditions": summaries,
        }
        (staging / "protocol.json").write_text(
            json.dumps(payload["config"], indent=2) + "\n"
        )
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload=payload,
        )
    print(f"exp083 complete: {run_id}")


if __name__ == "__main__":
    main()
