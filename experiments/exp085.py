"""EXP085: switch coupling on between equilibrated SNNLANG PING circuits."""

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
from scipy import signal as sp_signal
from scipy.ndimage import gaussian_filter1d

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution import ExecutionSpec, GraphRuntimeState, simulate  # noqa: E402
from experiments import exp083  # noqa: E402
from tools import snnlang as snn  # noqa: E402, TID251

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.gamma_frequency import estimate_gamma_from_raster  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp085"
DT_MS = 0.1
EQUILIBRATION_MS = 1_000.0
CONTINUATION_MS = 2_500.0
N_INPUT = 128
N_E = 80
N_I = 20
INPUT_RATE_HZ = 100.0
TAU_A_MS = 4.0
TAU_B_MS = 5.0
COUPLINGS = tuple(round(value, 2) for value in np.linspace(0.0, 0.10, 11))
REPRESENTATIVE_K = (0.0, 0.03, 0.06, 0.10)
TRIAL_SEEDS = (8500, 8501, 8502, 8503, 8504)
NETWORK_SEED = 85
PHASE_BAND_HZ = (20.0, 60.0)
PHASE_SMOOTH_MS = 5.0
PHASE_EDGE_TRIM_MS = 100.0
TRACE_BIN_MS = 1.0
SETTLING_K = (0.06, 0.07, 0.08, 0.10)
TERMINAL_PHASE_WINDOW_MS = 500.0
PHASE_ERROR_SMOOTH_MS = 50.0
DISPLAY_TRIAL = 0
FREQUENCY_CONFIG = replace(
    exp083.FREQUENCY_CONFIG,
    name="exp085-dominant-rhythm-v1",
    burn_ms=0.0,
)

POPULATION_KEYS = {
    "a_e": "population_0",
    "a_i": "population_1",
    "b_e": "population_2",
    "b_i": "population_3",
}

SCALE = {
    "dt_ms": DT_MS,
    "equilibration_ms": EQUILIBRATION_MS,
    "continuation_ms": CONTINUATION_MS,
    "n_input_per_circuit": N_INPUT,
    "n_e_per_circuit": N_E,
    "n_i_per_circuit": N_I,
    "input_rate_hz": INPUT_RATE_HZ,
    "tau_a_ms": TAU_A_MS,
    "tau_b_ms": TAU_B_MS,
    "couplings": list(COUPLINGS),
    "representative_couplings": list(REPRESENTATIVE_K),
    "trials": len(TRIAL_SEEDS),
    "trial_seeds": list(TRIAL_SEEDS),
    "network_seed": NETWORK_SEED,
    "phase_band_hz": list(PHASE_BAND_HZ),
    "phase_smooth_ms": PHASE_SMOOTH_MS,
    "phase_edge_trim_ms": PHASE_EDGE_TRIM_MS,
    "terminal_phase_window_ms": TERMINAL_PHASE_WINDOW_MS,
    "phase_error_smooth_ms": PHASE_ERROR_SMOOTH_MS,
}


def author_network(coupling: float) -> snn.Bundle:
    net = snn.Network("coupling_onset", dt=DT_MS * snn.ms)
    drives = {
        name: net.input(
            f"drive_{name}",
            shape=("time", "batch", N_INPUT),
            signal_type="spikes",
            unit="spike",
        )
        for name in ("a", "b")
    }
    circuits = {
        "a": snn.components.ping(
            net,
            name="a",
            n_e=N_E,
            n_i=N_I,
            source=drives["a"],
            tau_gaba=TAU_A_MS * snn.ms,
        ),
        "b": snn.components.ping(
            net,
            name="b",
            n_e=N_E,
            n_i=N_I,
            source=drives["b"],
            tau_gaba=TAU_B_MS * snn.ms,
        ),
    }
    for source, target in (("a", "b"), ("b", "a")):
        for population in ("E", "I"):
            target_port = (
                circuits[target].E.excitatory
                if population == "E"
                else circuits[target].I.excitatory
            )
            net.connect(
                circuits[source].E.spikes,
                target_port,
                name=f"{source}_E_to_{target}_{population}",
                synapse=snn.AMPA(tau=2 * snn.ms),
                weight=snn.Constant(coupling),
                constraint=snn.NonNegative(),
                connection="feedback",
                delay=0.1 * snn.ms,
            )
    net.expose(
        circuits["a"].E.spikes,
        circuits["a"].I.spikes,
        circuits["b"].E.spikes,
        circuits["b"].I.spikes,
        name="population",
    )
    return snn.compile(net, target="tools/snn")


def make_inputs() -> dict[str, np.ndarray]:
    """Private streams, paired across K and continuous across the state branch."""
    steps = round((EQUILIBRATION_MS + CONTINUATION_MS) / DT_MS)
    probability = INPUT_RATE_HZ * DT_MS / 1_000.0
    values: dict[str, list[np.ndarray]] = {"drive_a": [], "drive_b": []}
    for seed in TRIAL_SEEDS:
        for offset, name in enumerate(("drive_a", "drive_b")):
            rng = np.random.default_rng(seed * 10 + offset)
            values[name].append(
                rng.random((steps, N_INPUT), dtype=np.float32) < probability
            )
    return {
        name: np.stack(trials, axis=1).astype(np.uint8)
        for name, trials in values.items()
    }


def _population_rate(spikes: np.ndarray) -> np.ndarray:
    rate = spikes.mean(axis=-1) * (1_000.0 / DT_MS)
    return gaussian_filter1d(rate.astype(np.float64), PHASE_SMOOTH_MS / DT_MS)


def relative_phase(
    a_e: np.ndarray,
    b_e: np.ndarray,
    *,
    onset_step: int,
) -> np.ndarray:
    """Return onset-relative unwrapped A-minus-B phase, excluding filter edge."""
    sample_hz = 1_000.0 / DT_MS
    sos = sp_signal.butter(4, PHASE_BAND_HZ, btype="bandpass", fs=sample_hz, output="sos")
    phases = []
    for trial in range(a_e.shape[1]):
        rate_a = _population_rate(a_e[:, trial])
        rate_b = _population_rate(b_e[:, trial])
        filtered_a = sp_signal.sosfiltfilt(sos, rate_a)
        filtered_b = sp_signal.sosfiltfilt(sos, rate_b)
        wrapped = np.angle(sp_signal.hilbert(filtered_a)) - np.angle(
            sp_signal.hilbert(filtered_b)
        )
        unwrapped = np.unwrap(np.angle(np.exp(1j * wrapped)))
        edge_steps = round(PHASE_EDGE_TRIM_MS / DT_MS)
        post = unwrapped[onset_step:-edge_steps].copy()
        post -= post[0]
        phases.append(post)
    return np.stack(phases)


def _endpoint_frequency(spikes: np.ndarray) -> tuple[list[float], float | None]:
    endpoint_steps = round(1_500.0 / DT_MS)
    estimate = estimate_gamma_from_raster(
        spikes[-endpoint_steps:],
        dt_ms=DT_MS,
        config=FREQUENCY_CONFIG,
    )
    values = [
        float(trial.frequency_hz)
        for trial in estimate.trials
        if trial.resolved and trial.frequency_hz is not None
    ]
    return values, None if not values else float(np.median(values))


def summarize_condition(
    coupling: float,
    continuation: dict[str, np.ndarray],
    phase: np.ndarray,
) -> dict:
    frequency_a, median_a = _endpoint_frequency(continuation["a_e"])
    frequency_b, median_b = _endpoint_frequency(continuation["b_e"])
    duration_s = CONTINUATION_MS / 1_000.0
    rates = {
        name: array.sum(axis=(0, 2)) / array.shape[2] / duration_s
        for name, array in continuation.items()
    }
    plv = np.abs(np.mean(np.exp(1j * phase), axis=1))
    seconds = np.arange(phase.shape[1]) * DT_MS / 1_000.0
    slopes = np.array([np.polyfit(seconds, row, 1)[0] for row in phase])
    return {
        "coupling": coupling,
        "frequency_a_median_hz": median_a,
        "frequency_b_median_hz": median_b,
        "absolute_frequency_difference_hz": None
        if median_a is None or median_b is None
        else abs(median_a - median_b),
        "frequency_a_resolved_fraction": len(frequency_a) / len(TRIAL_SEEDS),
        "frequency_b_resolved_fraction": len(frequency_b) / len(TRIAL_SEEDS),
        "phase_locking_value_median": float(np.median(plv)),
        "phase_slope_median_rad_s": float(np.median(slopes)),
        **{
            f"{name}_rate_mean_hz": float(np.mean(value))
            for name, value in rates.items()
        },
        "trials": [
            {
                "trial": index,
                "seed": TRIAL_SEEDS[index],
                "phase_locking_value": float(plv[index]),
                "phase_slope_rad_s": float(slopes[index]),
                "frequency_a_hz": frequency_a[index] if len(frequency_a) == len(TRIAL_SEEDS) else None,
                "frequency_b_hz": frequency_b[index] if len(frequency_b) == len(TRIAL_SEEDS) else None,
            }
            for index in range(len(TRIAL_SEEDS))
        ],
    }


def _copy_runtime_state(state: GraphRuntimeState) -> GraphRuntimeState:
    return state.detached(device="cpu")


def plot_phase_small_multiples(phases: dict[float, np.ndarray], out: Path) -> None:
    theme.apply()
    bin_steps = round(TRACE_BIN_MS / DT_MS)
    trace_steps = next(iter(phases.values())).shape[1]
    time_s = np.arange(0, trace_steps, bin_steps) * DT_MS / 1_000.0
    fig, axes = plt.subplots(3, 4, figsize=(6.5, 5.2), sharex=True, sharey=True)
    flat = axes.ravel()
    for index, coupling in enumerate(COUPLINGS):
        axis = flat[index]
        trace = phases[coupling][:, ::bin_steps] / (2.0 * np.pi)
        for trial in range(1, len(TRIAL_SEEDS)):
            axis.plot(time_s, trace[trial], color=theme.GREY_MID, alpha=0.42, lw=0.65)
        axis.plot(time_s, trace[DISPLAY_TRIAL], color=theme.INK_BLACK, lw=1.0)
        axis.set_title(f"K = {coupling:.2f}", fontsize=theme.SIZE_ANNOTATION)
        axis.axhline(0, color=theme.GREY_LIGHT, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    flat[-1].axis("off")
    fig.supxlabel("time after coupling onset (s)")
    fig.supylabel("relative cycles gained")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def terminal_phase_error(
    phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return smoothed circular error from each trial's terminal phase offset."""
    bin_steps = round(TRACE_BIN_MS / DT_MS)
    binned = phase[:, ::bin_steps]
    terminal_steps = round(TERMINAL_PHASE_WINDOW_MS / TRACE_BIN_MS)
    smooth_steps = PHASE_ERROR_SMOOTH_MS / TRACE_BIN_MS
    trial_errors = []
    for row in binned:
        terminal_phase = np.angle(np.mean(np.exp(1j * row[-terminal_steps:])))
        circular_error = np.abs(np.angle(np.exp(1j * (row - terminal_phase))))
        trial_errors.append(gaussian_filter1d(circular_error, smooth_steps))
    errors = np.stack(trial_errors)
    time_ms = np.arange(errors.shape[1]) * TRACE_BIN_MS
    return (
        time_ms,
        np.median(errors, axis=0),
        np.quantile(errors, 0.25, axis=0),
        np.quantile(errors, 0.75, axis=0),
    )


def plot_synchrony_over_time(phases: dict[float, np.ndarray], out: Path) -> None:
    theme.apply()
    fig, axis = plt.subplots(figsize=(6.5, 3.2))
    styles = (
        (theme.GREY_LIGHT, ":"),
        (theme.GREY_DARK, "--"),
        (theme.DEEP_RED, "-."),
        (theme.INK_BLACK, "-"),
    )
    for coupling, (colour, linestyle) in zip(SETTLING_K, styles, strict=True):
        time_ms, median_error, lower, upper = terminal_phase_error(phases[coupling])
        axis.fill_between(time_ms / 1_000.0, lower, upper, color=colour, alpha=0.10, linewidth=0)
        axis.plot(
            time_ms / 1_000.0,
            median_error,
            color=colour,
            linestyle=linestyle,
            linewidth=1.0,
            label=f"K = {coupling:.2f}",
        )
    axis.set_xlabel("time after coupling onset (s)")
    axis.set_ylabel("terminal phase error (rad)")
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False, ncol=2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_response(summaries: list[dict], out: Path) -> None:
    theme.apply()
    x = np.array([row["coupling"] for row in summaries])
    fig, axes = plt.subplots(4, 1, figsize=(6.5, 5.5), sharex=True)
    axes[0].plot(x, [row["frequency_a_median_hz"] for row in summaries], "o-", color=theme.INK_BLACK, label="A")
    axes[0].plot(x, [row["frequency_b_median_hz"] for row in summaries], "s--", color=theme.DEEP_RED, label="B")
    axes[0].set_ylabel("frequency (Hz)")
    axes[0].legend(frameon=False)
    axes[1].plot(x, [row["absolute_frequency_difference_hz"] for row in summaries], "o-", color=theme.INK_BLACK)
    axes[1].set_ylabel("|fA − fB| (Hz)")
    axes[2].plot(x, [row["phase_locking_value_median"] for row in summaries], "o-", color=theme.INK_BLACK)
    axes[2].set_ylabel("phase-locking value")
    axes[2].set_ylim(-0.04, 1.04)
    for circuit, colour in (("a", theme.INK_BLACK), ("b", theme.DEEP_RED)):
        axes[3].plot(
            x,
            [row[f"{circuit}_e_rate_mean_hz"] for row in summaries],
            "o-",
            color=colour,
            label=f"{circuit.upper()}:E",
        )
        axes[3].plot(
            x,
            [row[f"{circuit}_i_rate_mean_hz"] for row in summaries],
            "s--",
            color=colour,
            label=f"{circuit.upper()}:I",
        )
    axes[3].set_ylabel("rate (Hz)")
    axes[3].set_xlabel("reciprocal coupling K")
    axes[3].legend(frameon=False, ncol=4)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_representative_rates(recordings: dict[float, dict[str, np.ndarray]], out: Path) -> None:
    theme.apply()
    fig, axes = plt.subplots(4, 1, figsize=(6.5, 5.2), sharex=True, sharey=True)
    for axis, coupling in zip(axes, REPRESENTATIVE_K):
        time_ms = np.arange(recordings[coupling]["a_e"].shape[0]) * DT_MS
        axis.plot(
            time_ms,
            _population_rate(recordings[coupling]["a_e"][:, DISPLAY_TRIAL]),
            color=theme.INK_BLACK,
            linewidth=0.55,
            label="A:E",
        )
        axis.plot(
            time_ms,
            _population_rate(recordings[coupling]["b_e"][:, DISPLAY_TRIAL]),
            color=theme.DEEP_RED,
            linestyle="--",
            linewidth=0.55,
            label="B:E",
        )
        axis.text(1.01, 0.5, f"K = {coupling:.2f}", transform=axis.transAxes, va="center", fontsize=theme.SIZE_ANNOTATION)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2)
    fig.supylabel("E-population rate (Hz)")
    axes[-1].set_xlim(0, CONTINUATION_MS)
    axes[-1].set_xlabel("time after coupling onset (ms)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = parse_meta(sys.argv)
    if meta.runpod:
        raise SystemExit("exp085 is a bounded local experiment; RunPod is not supported")
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id}")
    with published_run(SLUG, run_id, scale=SCALE, plot_only=meta.plot_only) as (scratch, staging):
        inputs = make_inputs()
        onset_step = round(EQUILIBRATION_MS / DT_MS)
        zero_bundle = author_network(0.0)
        zero_dir = staging / "network.bundle"
        zero_bundle.write(zero_dir, visualise=True)
        shutil.copy2(zero_dir / "reports/circuit.svg", staging / "network.svg")
        equilibration = simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=zero_bundle.graph,
                inputs={name: torch.from_numpy(value[:onset_step]).float() for name, value in inputs.items()},
                seed=NETWORK_SEED,
            )
        )
        assert equilibration.runtime_state is not None
        equilibration_arrays = {
            name: equilibration.recordings[key].cpu().numpy().astype(np.uint8)
            for name, key in POPULATION_KEYS.items()
        }

        phases: dict[float, np.ndarray] = {}
        representative: dict[float, dict[str, np.ndarray]] = {}
        summaries = []
        graph_rows = []
        continuation_inputs = {
            name: torch.from_numpy(value[onset_step:]).float()
            for name, value in inputs.items()
        }
        for coupling in COUPLINGS:
            print(f"[branch] K = {coupling:.2f}")
            bundle = author_network(coupling)
            result = simulate(
                ExecutionSpec(
                    kind="simulate",
                    executor="graph",
                    graph=bundle.graph,
                    inputs=continuation_inputs,
                    seed=NETWORK_SEED,
                    runtime_state=_copy_runtime_state(equilibration.runtime_state),
                )
            )
            continuation = {
                name: result.recordings[key].cpu().numpy().astype(np.uint8)
                for name, key in POPULATION_KEYS.items()
            }
            combined_a = np.concatenate((equilibration_arrays["a_e"], continuation["a_e"]))
            combined_b = np.concatenate((equilibration_arrays["b_e"], continuation["b_e"]))
            phase = relative_phase(combined_a, combined_b, onset_step=onset_step)
            phases[coupling] = phase
            summaries.append(summarize_condition(coupling, continuation, phase))
            graph_rows.append({"coupling": coupling, "digest": bundle.manifest["graph_digest"]})
            if coupling in REPRESENTATIVE_K:
                representative[coupling] = continuation

        plot_phase_small_multiples(phases, staging / "relative_phase.svg")
        plot_response(summaries, staging / "response.svg")
        plot_representative_rates(representative, staging / "representative_rates.svg")
        plot_synchrony_over_time(phases, staging / "synchrony_over_time.svg")
        bin_steps = round(TRACE_BIN_MS / DT_MS)
        np.savez_compressed(
            staging / "phase_traces.npz",
            couplings=np.asarray(COUPLINGS),
            time_ms=np.arange(0, next(iter(phases.values())).shape[1], bin_steps) * DT_MS,
            phase_rad=np.stack([phases[value][:, ::bin_steps] for value in COUPLINGS]),
        )
        payload = {
            "question": "How does reciprocal excitation change relative phase after coupling is switched on between equilibrated PING circuits?",
            "config": SCALE,
            "frequency_analysis": FREQUENCY_CONFIG.json(),
            "phase_analysis": {
                "population": "E",
                "smoothing_sigma_ms": PHASE_SMOOTH_MS,
                "band_hz": list(PHASE_BAND_HZ),
                "terminal_edge_trim_ms": PHASE_EDGE_TRIM_MS,
                "terminal_phase_window_ms": TERMINAL_PHASE_WINDOW_MS,
                "phase_error_smooth_ms": PHASE_ERROR_SMOOTH_MS,
                "method": "fourth-order zero-phase Butterworth plus Hilbert phase",
                "sign": "onset-relative A-minus-B",
            },
            "runtime_state": {
                "schema": equilibration.metrics["runtime_state_schema"],
                "signature": equilibration.metrics["runtime_state_signature"],
                "completed_steps": equilibration.metrics["completed_steps"],
            },
            "graphs": graph_rows,
            "conditions": summaries,
        }
        (staging / "protocol.json").write_text(json.dumps(SCALE, indent=2) + "\n")
        write_numbers(staging, run_id=run_id, duration_s=time.monotonic() - started, payload=payload)
    print(f"exp085 complete: {run_id}")


if __name__ == "__main__":
    main()
