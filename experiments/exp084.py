"""EXP084: tune the default SNNLANG PING rhythm with inhibitory decay.

The reliable 100 Hz/channel operating point from exp083 is held fixed while
the public ``tau_gaba`` component parameter varies.  Paired input trials and a
fixed network seed isolate the effect of inhibitory recovery time.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution import ExecutionSpec, simulate  # noqa: E402
from experiments import exp083  # noqa: E402
from tools import snnlang as snn  # noqa: E402, TID251

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.gamma_frequency import (  # noqa: E402
    GammaFrequencyEstimate,
    estimate_gamma_from_raster,
)
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp084"
INPUT_RATE_HZ = 100.0
TAU_GABA_MS = (2.0, 4.0, 6.0, 9.0, 12.0, 16.0)
REPRESENTATIVE_TAU_MS = (2.0, 9.0, 16.0)
DISPLAY_TRIAL = 0

SCALE = {
    "dt_ms": exp083.DT_MS,
    "t_ms": exp083.T_MS,
    "burn_ms": exp083.BURN_MS,
    "n_input": exp083.N_INPUT,
    "n_e": exp083.N_E,
    "n_i": exp083.N_I,
    "input_rate_hz": INPUT_RATE_HZ,
    "tau_gaba_ms": list(TAU_GABA_MS),
    "trials": len(exp083.TRIAL_SEEDS),
    "trial_seeds": list(exp083.TRIAL_SEEDS),
    "network_seed": exp083.NETWORK_SEED,
}


def author_network(tau_gaba_ms: float) -> snn.Bundle:
    net = snn.Network("ping_inhibitory_timescale", dt=exp083.DT_MS * snn.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", exp083.N_INPUT),
        signal_type="spikes",
        unit="spike",
    )
    cell = snn.components.ping(
        net,
        name="ping",
        n_e=exp083.N_E,
        n_i=exp083.N_I,
        source=drive,
        tau_gaba=tau_gaba_ms * snn.ms,
    )
    net.expose(cell.E.spikes, cell.I.spikes, name="population")
    return snn.compile(net, target="tools/snnsim")


def summarize_condition(
    tau_gaba_ms: float,
    e_spikes: np.ndarray,
    i_spikes: np.ndarray,
    estimate: GammaFrequencyEstimate,
) -> dict:
    rows = exp083._trial_rows(INPUT_RATE_HZ, e_spikes, i_spikes, estimate)
    summary = exp083.summarize_condition(INPUT_RATE_HZ, rows)
    summary["tau_gaba_ms"] = tau_gaba_ms
    return summary


def plot_response(summaries: list[dict], out: Path) -> None:
    theme.apply()
    x = np.array([row["tau_gaba_ms"] for row in summaries])
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.2), sharex=True)
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
    axes[1].errorbar(
        x,
        [row["rhythmicity_score_median"] for row in summaries],
        yerr=[row["rhythmicity_score_iqr"] / 2 for row in summaries],
        marker="o",
        capsize=3,
        color=theme.INK_BLACK,
    )
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].set_ylabel("rhythmicity score")
    axes[2].plot(
        x,
        [row["rhythm_frequency_median_hz"] for row in summaries],
        marker="o",
        color=theme.DEEP_RED,
    )
    axes[2].axhspan(30, 80, color=theme.GREY_LIGHT, alpha=0.45)
    axes[2].set_ylim(0, 85)
    axes[2].set_ylabel("frequency (Hz)")
    axes[2].set_xticks(x)
    for axis in axes:
        axis.axvline(9.0, color=theme.GREY_MID, ls="--", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("inhibitory decay, τGABA (ms)")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_representative_rasters(
    recordings: dict[float, dict[str, np.ndarray]],
    summaries: list[dict],
    out: Path,
) -> None:
    theme.apply()
    by_tau = {row["tau_gaba_ms"]: row for row in summaries}
    gap = 6
    fig, axes = plt.subplots(
        3, 1, figsize=(6.5, 3.5), sharex=True, gridspec_kw={"hspace": 0.18}
    )
    for row, tau in enumerate(REPRESENTATIVE_TAU_MS):
        arrays = recordings[tau]
        axis = axes[row]
        e_t, e_cells = np.nonzero(arrays["e"][:, DISPLAY_TRIAL])
        i_t, i_cells = np.nonzero(arrays["i"][:, DISPLAY_TRIAL])
        axis.scatter(
            e_t * exp083.DT_MS,
            e_cells,
            s=2,
            marker="|",
            linewidths=0.4,
            color=theme.INK_BLACK,
            rasterized=True,
        )
        axis.scatter(
            i_t * exp083.DT_MS,
            i_cells + exp083.N_E + gap,
            s=2,
            marker="|",
            linewidths=0.4,
            color=theme.DEEP_RED,
            rasterized=True,
        )
        axis.axvline(exp083.BURN_MS, color=theme.GREY_MID, ls="--", lw=0.8)
        axis.set_ylim(-2, exp083.N_E + gap + exp083.N_I + 2)
        axis.set_yticks([exp083.N_E / 2, exp083.N_E + gap + exp083.N_I / 2])
        axis.set_yticklabels(["E", "I"])
        axis.tick_params(axis="y", length=0)
        condition = by_tau[tau]
        frequency = condition["rhythm_frequency_median_hz"]
        frequency_label = "unresolved" if frequency is None else f"{frequency:.1f} Hz"
        axis.text(
            1.012,
            0.5,
            f"τGABA {tau:g} ms\n{frequency_label}",
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=theme.SIZE_ANNOTATION,
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlim(0, exp083.T_MS)
    axes[-1].set_xlabel("time (ms)")
    fig.subplots_adjust(left=0.08, right=0.79, bottom=0.15, top=0.98, hspace=0.18)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = parse_meta(sys.argv)
    if meta.runpod:
        raise SystemExit("exp084 is a bounded local experiment; RunPod is not supported")
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id}")
    with published_run(SLUG, run_id, scale=SCALE, plot_only=meta.plot_only) as (
        scratch,
        staging,
    ):
        input_spikes = exp083.make_inputs(INPUT_RATE_HZ)
        summaries: list[dict] = []
        recordings: dict[float, dict[str, np.ndarray]] = {}
        conditions = scratch / "conditions"
        conditions.mkdir(exist_ok=True)
        graph_rows = []
        for tau in TAU_GABA_MS:
            print(f"[simulate] tau_GABA {tau:g} ms")
            bundle = author_network(tau)
            if tau == 9.0:
                bundle_dir = staging / "network.bundle"
                bundle.write(bundle_dir, visualise=True)
                shutil.copy2(bundle_dir / "reports/circuit.svg", staging / "network.svg")
            result = simulate(
                ExecutionSpec(
                    kind="simulate",
                    executor="graph",
                    graph=bundle.graph,
                    inputs={"drive": torch.from_numpy(input_spikes).float()},
                    seed=exp083.NETWORK_SEED,
                )
            )
            e_spikes = result.recordings["population_0"].cpu().numpy().astype(np.uint8)
            i_spikes = result.recordings["population_1"].cpu().numpy().astype(np.uint8)
            estimate = estimate_gamma_from_raster(
                e_spikes,
                dt_ms=exp083.DT_MS,
                config=exp083.FREQUENCY_CONFIG,
            )
            summaries.append(summarize_condition(tau, e_spikes, i_spikes, estimate))
            recordings[tau] = {"e": e_spikes, "i": i_spikes}
            graph_rows.append({"tau_gaba_ms": tau, "digest": bundle.manifest["graph_digest"]})
            np.savez_compressed(
                conditions / f"tau-{tau:g}.npz",
                input_spikes=input_spikes,
                e_spikes=e_spikes,
                i_spikes=i_spikes,
            )

        plot_response(summaries, staging / "response.svg")
        plot_representative_rasters(recordings, summaries, staging / "representative_rasters.png")
        shutil.copytree(conditions, staging / "conditions")
        payload = {
            "question": "Does inhibitory recovery time tune the default SNNLANG PING rhythm into gamma?",
            "config": SCALE,
            "frequency_analysis": exp083.FREQUENCY_CONFIG.json(),
            "representative_tau_gaba_ms": list(REPRESENTATIVE_TAU_MS),
            "graphs": graph_rows,
            "conditions": summaries,
        }
        (staging / "protocol.json").write_text(json.dumps(SCALE, indent=2) + "\n")
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload=payload,
        )
    print(f"exp084 complete: {run_id}")


if __name__ == "__main__":
    main()
