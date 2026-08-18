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
SCOUT_MS = 2_400.0
CONTINUATION_MS = 1_500.0
N_INPUT = 128
N_E = 80
N_I = 20
INPUT_RATE_HZ = 100.0
TAU_A_MS = 4.0
TAU_B_MS = 5.0
COUPLING = 0.08
PHASE_TARGETS_RAD = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi)
TRIAL_SEEDS = (8500, 8501, 8502, 8503, 8504)
NETWORK_SEED = 85
PHASE_BAND_HZ = (20.0, 60.0)
PHASE_SMOOTH_MS = 5.0
PHASE_EDGE_TRIM_MS = 100.0
TRACE_BIN_MS = 1.0
TERMINAL_PHASE_WINDOW_MS = 500.0
PHASE_ERROR_SMOOTH_MS = 50.0
PHASE_PREHISTORY_MS = 300.0
# Legacy scan helpers remain importable for historical tests but are not published.
COUPLINGS = tuple(round(value, 2) for value in np.linspace(0.0, 0.10, 11))
REPRESENTATIVE_K = (0.0, 0.03, 0.06, 0.10)
DISPLAY_TRIAL = 0
SYNC_EXAMPLE_MIN_K = 0.07
SYNC_EXAMPLE_TERMINAL_P95_MAX_RAD = 0.25
SYNC_EXAMPLE_POST_SETTLING_MAX_RAD = 0.5
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
    "coupling": COUPLING,
    "phase_targets_rad": list(PHASE_TARGETS_RAD),
    "scout_ms": SCOUT_MS,
    "trials": len(TRIAL_SEEDS),
    "trial_seeds": list(TRIAL_SEEDS),
    "network_seed": NETWORK_SEED,
    "phase_band_hz": list(PHASE_BAND_HZ),
    "phase_smooth_ms": PHASE_SMOOTH_MS,
    "phase_edge_trim_ms": PHASE_EDGE_TRIM_MS,
    "terminal_phase_window_ms": TERMINAL_PHASE_WINDOW_MS,
    "phase_error_smooth_ms": PHASE_ERROR_SMOOTH_MS,
    "phase_prehistory_ms": PHASE_PREHISTORY_MS,
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
    sos = sp_signal.butter(
        4, PHASE_BAND_HZ, btype="bandpass", fs=sample_hz, output="sos"
    )
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
                "frequency_a_hz": frequency_a[index]
                if len(frequency_a) == len(TRIAL_SEEDS)
                else None,
                "frequency_b_hz": frequency_b[index]
                if len(frequency_b) == len(TRIAL_SEEDS)
                else None,
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


def trial_terminal_phase_error(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one trial's smoothed circular error from its terminal phase offset."""
    terminal_steps = round(TERMINAL_PHASE_WINDOW_MS / TRACE_BIN_MS)
    smooth_steps = PHASE_ERROR_SMOOTH_MS / TRACE_BIN_MS
    terminal_phase = np.angle(np.mean(np.exp(1j * phase[-terminal_steps:])))
    circular_error = np.abs(np.angle(np.exp(1j * (phase - terminal_phase))))
    error = gaussian_filter1d(circular_error, smooth_steps)
    return np.arange(error.size) * TRACE_BIN_MS, error


def clean_convergence_trials(phase: np.ndarray) -> list[int]:
    """Return trial indices satisfying the illustrative clean-convergence rule."""
    passed = []
    bin_steps = round(TRACE_BIN_MS / DT_MS)
    for trial, native_phase in enumerate(phase):
        _, error = trial_terminal_phase_error(native_phase[::bin_steps])
        terminal_p95 = np.quantile(error[-500:], 0.95)
        post_settling_max = np.max(error[500:])
        if (
            terminal_p95 <= SYNC_EXAMPLE_TERMINAL_P95_MAX_RAD
            and post_settling_max <= SYNC_EXAMPLE_POST_SETTLING_MAX_RAD
        ):
            passed.append(trial)
    return passed


def select_sync_example(
    phases: dict[float, np.ndarray],
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Select the largest clean onset-to-settled terminal-error reduction."""
    candidates = []
    for coupling in COUPLINGS:
        if coupling < SYNC_EXAMPLE_MIN_K:
            continue
        for trial, native_phase in enumerate(phases[coupling]):
            binned_phase = native_phase[:: round(TRACE_BIN_MS / DT_MS)]
            time_ms, error = trial_terminal_phase_error(binned_phase)
            terminal_p95 = np.quantile(error[-500:], 0.95)
            post_settling_max = np.max(error[500:])
            if (
                terminal_p95 <= SYNC_EXAMPLE_TERMINAL_P95_MAX_RAD
                and post_settling_max <= SYNC_EXAMPLE_POST_SETTLING_MAX_RAD
            ):
                reduction = np.mean(error[:100]) - np.mean(error[250:500])
                candidates.append((reduction, coupling, trial, time_ms, error))
    if not candidates:
        raise RuntimeError(
            "no phase-settling trace satisfies the example selection rule"
        )
    _, coupling, trial, time_ms, error = max(candidates, key=lambda row: row[0])
    return coupling, trial, time_ms, error


def plot_synchrony_over_time(phases: dict[float, np.ndarray], out: Path) -> dict:
    theme.apply()
    fig, axis = plt.subplots(figsize=(6.5, 3.2))
    coupling, selected_trial, _, _ = select_sync_example(phases)
    bin_steps = round(TRACE_BIN_MS / DT_MS)
    for trial, native_phase in enumerate(phases[coupling]):
        time_ms, error = trial_terminal_phase_error(native_phase[::bin_steps])
        selected = trial == selected_trial
        axis.plot(
            time_ms / 1_000.0,
            error,
            color=theme.INK_BLACK if selected else theme.GREY_MID,
            alpha=1.0 if selected else 0.55,
            linewidth=1.0 if selected else 0.7,
            label=f"seed {TRIAL_SEEDS[trial]}" if selected else None,
            zorder=2 if selected else 1,
        )
    axis.set_xlabel("time after coupling onset (s)")
    axis.set_ylabel("terminal phase error (rad)")
    axis.set_ylim(bottom=0.0)
    axis.text(
        0.98,
        0.92,
        f"K = {coupling:.2f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {
        "coupling": coupling,
        "trial": selected_trial,
        "seed": TRIAL_SEEDS[selected_trial],
    }


def plot_response(summaries: list[dict], out: Path) -> None:
    theme.apply()
    x = np.array([row["coupling"] for row in summaries])
    fig, axes = plt.subplots(4, 1, figsize=(6.5, 5.5), sharex=True)
    axes[0].plot(
        x,
        [row["frequency_a_median_hz"] for row in summaries],
        "o-",
        color=theme.INK_BLACK,
        label="A",
    )
    axes[0].plot(
        x,
        [row["frequency_b_median_hz"] for row in summaries],
        "s--",
        color=theme.DEEP_RED,
        label="B",
    )
    axes[0].set_ylabel("frequency (Hz)")
    axes[0].legend(frameon=False)
    axes[1].plot(
        x,
        [row["absolute_frequency_difference_hz"] for row in summaries],
        "o-",
        color=theme.INK_BLACK,
    )
    axes[1].set_ylabel("|fA − fB| (Hz)")
    axes[2].plot(
        x,
        [row["phase_locking_value_median"] for row in summaries],
        "o-",
        color=theme.INK_BLACK,
    )
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


def plot_representative_rates(
    recordings: dict[float, dict[str, np.ndarray]], out: Path
) -> None:
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
        axis.text(
            1.01,
            0.5,
            f"K = {coupling:.2f}",
            transform=axis.transAxes,
            va="center",
            fontsize=theme.SIZE_ANNOTATION,
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2)
    fig.supylabel("E-population rate (Hz)")
    axes[-1].set_xlim(0, CONTINUATION_MS)
    axes[-1].set_xlabel("time after coupling onset (ms)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _phase_signal(a_e: np.ndarray, b_e: np.ndarray) -> np.ndarray:
    sample_hz = 1_000.0 / DT_MS
    sos = sp_signal.butter(
        4, PHASE_BAND_HZ, btype="bandpass", fs=sample_hz, output="sos"
    )
    a = sp_signal.sosfiltfilt(sos, _population_rate(a_e))
    b = sp_signal.sosfiltfilt(sos, _population_rate(b_e))
    return np.angle(
        np.exp(1j * (np.angle(sp_signal.hilbert(a)) - np.angle(sp_signal.hilbert(b))))
    )


def select_phase_onsets(phase: np.ndarray) -> dict[float, int]:
    start = round(EQUILIBRATION_MS / DT_MS)
    stop = round((SCOUT_MS - PHASE_EDGE_TRIM_MS) / DT_MS)
    selected = {}
    for target in PHASE_TARGETS_RAD:
        error = np.abs(np.angle(np.exp(1j * (phase[start:stop] - target))))
        selected[target] = start + int(np.argmin(error))
    return selected


def _private_inputs(
    seed: int, duration_ms: float, *, offset: int
) -> dict[str, np.ndarray]:
    steps = round(duration_ms / DT_MS)
    probability = INPUT_RATE_HZ * DT_MS / 1_000.0
    result = {}
    for circuit, name in enumerate(("drive_a", "drive_b")):
        rng = np.random.default_rng(seed * 100 + offset + circuit)
        result[name] = (
            rng.random((steps, 1, N_INPUT), dtype=np.float32) < probability
        ).astype(np.uint8)
    return result


def _run(
    bundle: snn.Bundle,
    inputs: dict[str, np.ndarray],
    state: GraphRuntimeState | None = None,
):
    return simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            inputs={
                name: torch.from_numpy(value).float() for name, value in inputs.items()
            },
            seed=NETWORK_SEED,
            runtime_state=None if state is None else _copy_runtime_state(state),
        )
    )


def _phase_error(
    pre_a: np.ndarray, pre_b: np.ndarray, post_a: np.ndarray, post_b: np.ndarray
) -> np.ndarray:
    phase = _phase_signal(
        np.concatenate((pre_a, post_a)), np.concatenate((pre_b, post_b))
    )
    start = len(pre_a)
    edge = round(PHASE_EDGE_TRIM_MS / DT_MS)
    usable = phase[start:-edge]
    terminal = np.angle(
        np.mean(np.exp(1j * usable[-round(TERMINAL_PHASE_WINDOW_MS / DT_MS) :]))
    )
    error = np.abs(np.angle(np.exp(1j * (usable - terminal))))
    return gaussian_filter1d(error, PHASE_ERROR_SMOOTH_MS / DT_MS)[
        :: round(TRACE_BIN_MS / DT_MS)
    ]


def plot_phase_control(rows: list[dict], out: Path) -> None:
    theme.apply()
    fig, axis = plt.subplots(figsize=(4.8, 3.4))
    axis.scatter(
        [row["target_phase_rad"] for row in rows],
        [row["achieved_phase_rad"] for row in rows],
        color=theme.INK_BLACK,
        s=14,
    )
    axis.plot([0, 2 * np.pi], [0, 2 * np.pi], color=theme.GREY_LIGHT, linewidth=0.8)
    axis.set(
        xlabel="prescribed phase (rad)",
        ylabel="achieved phase (rad)",
        xlim=(-0.1, 2 * np.pi + 0.1),
        ylim=(-0.1, 2 * np.pi + 0.1),
    )
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_controlled_convergence(
    coupled: np.ndarray, control: np.ndarray, out: Path
) -> None:
    theme.apply()
    time_s = np.arange(coupled.shape[-1]) * TRACE_BIN_MS / 1_000.0
    colours = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN, theme.AMBER)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True, sharey=True)
    for target_index, (target, colour) in enumerate(
        zip(PHASE_TARGETS_RAD, colours, strict=True)
    ):
        for trial in range(len(TRIAL_SEEDS)):
            axes[0].plot(
                time_s,
                coupled[target_index, trial],
                color=colour,
                alpha=0.42,
                linewidth=0.65,
            )
            axes[1].plot(
                time_s,
                control[target_index, trial],
                color=colour,
                alpha=0.42,
                linewidth=0.65,
            )
        axes[0].plot(
            time_s,
            np.median(coupled[target_index], axis=0),
            color=colour,
            linewidth=1.2,
            label=f"{target / np.pi:.1g}π",
        )
        axes[1].plot(
            time_s,
            np.median(control[target_index], axis=0),
            color=colour,
            linewidth=1.2,
        )
    axes[0].set_title("coupling on")
    axes[1].set_title("uncoupled control")
    axes[0].legend(frameon=False, ncol=4)
    axes[1].set_xlabel("time after onset (s)")
    fig.supylabel("terminal phase error (rad)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = parse_meta(sys.argv)
    if meta.runpod:
        raise SystemExit(
            "exp085 is a bounded local experiment; RunPod is not supported"
        )
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id}")
    with published_run(SLUG, run_id, scale=SCALE, plot_only=meta.plot_only) as (
        _scratch,
        staging,
    ):
        zero = author_network(0.0)
        coupled_bundle = author_network(COUPLING)
        bundle_dir = staging / "network.bundle"
        zero.write(bundle_dir, visualise=True)
        shutil.copy2(bundle_dir / "reports/circuit.svg", staging / "network.svg")
        rows = []
        coupled_errors = []
        control_errors = []
        pre_steps = round(PHASE_PREHISTORY_MS / DT_MS)
        for seed in TRIAL_SEEDS:
            print(f"[scout] seed = {seed}")
            scout_inputs = _private_inputs(seed, SCOUT_MS, offset=0)
            scout = _run(zero, scout_inputs)
            scout_arrays = {
                name: scout.recordings[key].cpu().numpy()[:, 0].astype(np.uint8)
                for name, key in POPULATION_KEYS.items()
            }
            scout_phase = _phase_signal(scout_arrays["a_e"], scout_arrays["b_e"])
            onsets = select_phase_onsets(scout_phase)
            future = _private_inputs(seed, CONTINUATION_MS, offset=10)
            states = {}
            cursor = 0
            state = None
            for target, step in sorted(onsets.items(), key=lambda item: item[1]):
                segment = {
                    name: value[cursor:step] for name, value in scout_inputs.items()
                }
                replay = _run(zero, segment, state)
                assert replay.runtime_state is not None
                for name, key in POPULATION_KEYS.items():
                    np.testing.assert_array_equal(
                        replay.recordings[key].cpu().numpy()[:, 0],
                        scout_arrays[name][cursor:step],
                    )
                state = replay.runtime_state
                states[target] = _copy_runtime_state(state)
                cursor = step
            seed_coupled = []
            seed_control = []
            for target in PHASE_TARGETS_RAD:
                step = onsets[target]
                pre_a = scout_arrays["a_e"][step - pre_steps : step]
                pre_b = scout_arrays["b_e"][step - pre_steps : step]
                achieved = float(scout_phase[step] % (2 * np.pi))
                on = _run(coupled_bundle, future, states[target])
                off = _run(zero, future, states[target])
                on_a = on.recordings[POPULATION_KEYS["a_e"]].cpu().numpy()[:, 0]
                on_b = on.recordings[POPULATION_KEYS["b_e"]].cpu().numpy()[:, 0]
                off_a = off.recordings[POPULATION_KEYS["a_e"]].cpu().numpy()[:, 0]
                off_b = off.recordings[POPULATION_KEYS["b_e"]].cpu().numpy()[:, 0]
                seed_coupled.append(_phase_error(pre_a, pre_b, on_a, on_b))
                seed_control.append(_phase_error(pre_a, pre_b, off_a, off_b))
                rows.append(
                    {
                        "seed": seed,
                        "target_phase_rad": float(target),
                        "achieved_phase_rad": achieved,
                        "onset_ms": step * DT_MS,
                        "target_error_rad": float(
                            abs(np.angle(np.exp(1j * (achieved - target))))
                        ),
                    }
                )
            coupled_errors.append(seed_coupled)
            control_errors.append(seed_control)
        coupled_array = np.transpose(np.asarray(coupled_errors), (1, 0, 2))
        control_array = np.transpose(np.asarray(control_errors), (1, 0, 2))
        plot_phase_control(rows, staging / "phase_control.svg")
        plot_controlled_convergence(
            coupled_array, control_array, staging / "synchrony_over_time.svg"
        )
        np.savez_compressed(
            staging / "phase_traces.npz",
            targets_rad=np.asarray(PHASE_TARGETS_RAD),
            time_ms=np.arange(coupled_array.shape[-1]) * TRACE_BIN_MS,
            coupled_error_rad=coupled_array,
            control_error_rad=control_array,
        )
        payload = {
            "question": "Do mature detuned PING circuits converge from controlled relative phases when coupling switches on?",
            "config": SCALE,
            "phase_analysis": {
                "smoothing_sigma_ms": PHASE_SMOOTH_MS,
                "band_hz": list(PHASE_BAND_HZ),
                "terminal_edge_trim_ms": PHASE_EDGE_TRIM_MS,
                "terminal_phase_window_ms": TERMINAL_PHASE_WINDOW_MS,
                "phase_error_smooth_ms": PHASE_ERROR_SMOOTH_MS,
            },
            "phase_control": rows,
            "max_target_error_rad": max(row["target_error_rad"] for row in rows),
            "coupled_terminal_error_median_rad": float(
                np.median(coupled_array[:, :, -500:])
            ),
            "control_terminal_error_median_rad": float(
                np.median(control_array[:, :, -500:])
            ),
        }
        (staging / "protocol.json").write_text(json.dumps(SCALE, indent=2) + "\n")
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload=payload,
        )
    print(f"exp085 complete: {run_id}")


if __name__ == "__main__":
    main()
