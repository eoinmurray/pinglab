"""EXP085 methods 1-4: define PING networks and compare coupling pathways."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution import ExecutionSpec, simulate  # noqa: E402
from tools import snnlang as snn  # noqa: E402, TID251

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp085"
STATUS = "draft"

DT_MS = 0.1
T_MS = 2_000.0
BURN_MS = 300.0
COUPLING_ONSET_MS = 500.0
DISPLAY_START_MS = 500.0
DISPLAY_END_MS = 750.0
PRC_T_MS = 900.0
PRC_REFERENCE_MS = 700.0
PRC_PHASE_FRACTIONS = np.asarray(
    [
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
        0.14,
        0.16,
        0.18,
        0.20,
        0.22,
        0.24,
        0.26,
        0.28,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]
)
N_INPUT = 128
N_E = 80
N_I = 20
TAU_GABA_MS = 9.0
E_REFRACTORY_MS = 3.0
I_REFRACTORY_MS = 1.5
E_TO_I_WEIGHT = 0.5
E_TO_I_TAU_MS = 1.0

# These rates define the intended detuning. Method 2 must verify the resulting
# uncoupled gamma frequencies; they are design inputs, not completed results.
INPUT_RATE_A_HZ = 300.0
INPUT_RATE_B_HZ = 260.0
INPUT_SEEDS = (8501, 8502)
NETWORK_SEED = 85

# Separate controls even though their initial nominal values match. The graph
# executor divides each nominal total strength across the realised fan-in.
K_EE = 0.08
K_EI = 0.08
COUPLING_DELAY_MS = 2.0
CROSS_FAN_IN = 8
CROSS_ZERO_FRACTION = 1.0 - CROSS_FAN_IN / N_E
PING_GROUPS = ("PING_A", "PING_B")

SCALE = {
    "status": STATUS,
    "completed_methods": [1, 2, 3, 4],
    "dt_ms": DT_MS,
    "t_ms": T_MS,
    "burn_ms": BURN_MS,
    "coupling_onset_ms": COUPLING_ONSET_MS,
    "n_input_per_network": N_INPUT,
    "n_e_per_network": N_E,
    "n_i_per_network": N_I,
    "tau_gaba_ms": TAU_GABA_MS,
    "e_refractory_ms": E_REFRACTORY_MS,
    "i_refractory_ms": I_REFRACTORY_MS,
    "e_to_i_weight": E_TO_I_WEIGHT,
    "e_to_i_tau_ms": E_TO_I_TAU_MS,
    "input_rate_a_hz": INPUT_RATE_A_HZ,
    "input_rate_b_hz": INPUT_RATE_B_HZ,
    "k_ee": K_EE,
    "k_ei": K_EI,
    "coupling_delay_ms": COUPLING_DELAY_MS,
    "cross_fan_in": CROSS_FAN_IN,
    "prc_t_ms": PRC_T_MS,
    "prc_reference_ms": PRC_REFERENCE_MS,
    "prc_phase_fractions": PRC_PHASE_FRACTIONS.tolist(),
}


@dataclass(frozen=True)
class PING:
    E: snn.Population
    I: snn.Population


def add_ping(
    net: snn.Network,
    *,
    name: str,
    source: snn.Signal,
    e_to_i_weight: float = E_TO_I_WEIGHT,
    e_to_i_tau_ms: float = E_TO_I_TAU_MS,
) -> PING:
    """Add one matched, minimal E-to-I-to-E PING circuit."""
    with net.group(name):
        e = net.population(
            f"{name}_E",
            size=N_E,
            neuron=snn.COBA_LIF(
                tau_mem=20 * snn.ms,
                capacitance_nf=1.0,
                leak_us=0.05,
                resting_mv=-65.0,
                threshold_mv=-50.0,
                reset_mv=-65.0,
                refractory_steps=round(E_REFRACTORY_MS / DT_MS),
                voltage_grad_dampen=80.0,
                initial_voltage_mv=-65.0,
            ),
        )
        i = net.population(
            f"{name}_I",
            size=N_I,
            neuron=snn.COBA_LIF(
                tau_mem=5 * snn.ms,
                capacitance_nf=0.5,
                leak_us=0.1,
                resting_mv=-65.0,
                threshold_mv=-50.0,
                reset_mv=-65.0,
                refractory_steps=round(I_REFRACTORY_MS / DT_MS),
                voltage_grad_dampen=80.0,
                initial_voltage_mv=-65.0,
            ),
        )
        net.connect(
            source,
            e.excitatory,
            name=f"{name}_input_to_E",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.Normal(0.2, 0.03),
            constraint=snn.NonNegative(),
        )
        net.connect(
            e.spikes,
            i.excitatory,
            name=f"{name}_E_to_I",
            synapse=snn.AMPA(tau=e_to_i_tau_ms * snn.ms),
            weight=snn.Normal(e_to_i_weight, 0.1 * e_to_i_weight),
            constraint=snn.NonNegative(),
            connection="recurrent",
            delay=DT_MS * snn.ms,
        )
        net.connect(
            i.spikes,
            e.inhibitory,
            name=f"{name}_I_to_E",
            synapse=snn.GABA(tau=TAU_GABA_MS * snn.ms),
            weight=snn.Normal(1.0, 0.1),
            constraint=snn.NonNegative(),
            connection="recurrent",
            delay=DT_MS * snn.ms,
        )
    return PING(E=e, I=i)


def sparse_coupling(total_strength: float):
    """Return an exact-fan-in initializer for a long-range E projection."""
    return snn.LowerClampedNormal(
        total_strength,
        0.0,
        initial_zero_fraction=CROSS_ZERO_FRACTION,
        zeroing="exact_k",
    )


def author_network(
    *,
    k_ee: float = K_EE,
    k_ei: float = K_EI,
    coupling_delay_ms: float = COUPLING_DELAY_MS,
    e_to_i_weight: float = E_TO_I_WEIGHT,
    e_to_i_tau_ms: float = E_TO_I_TAU_MS,
) -> snn.Bundle:
    """Author the canonical coupled-PING graph for the remaining methods."""
    net = snn.Network("canonical_coupled_ping", dt=DT_MS * snn.ms)
    drive_a = net.input(
        f"drive_A_{INPUT_RATE_A_HZ:g}_Hz",
        shape=("time", "batch", N_INPUT),
        signal_type="spikes",
        unit="spike",
    )
    drive_b = net.input(
        f"drive_B_{INPUT_RATE_B_HZ:g}_Hz",
        shape=("time", "batch", N_INPUT),
        signal_type="spikes",
        unit="spike",
    )
    network_a = add_ping(
        net,
        name="PING_A",
        source=drive_a,
        e_to_i_weight=e_to_i_weight,
        e_to_i_tau_ms=e_to_i_tau_ms,
    )
    network_b = add_ping(
        net,
        name="PING_B",
        source=drive_b,
        e_to_i_weight=e_to_i_weight,
        e_to_i_tau_ms=e_to_i_tau_ms,
    )

    for source_name, source, target_name, target in (
        ("PING_A", network_a, "PING_B", network_b),
        ("PING_B", network_b, "PING_A", network_a),
    ):
        net.connect(
            source.E.spikes,
            target.E.excitatory,
            name=f"{source_name}_E_to_{target_name}_E_K_EE",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=sparse_coupling(k_ee),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=coupling_delay_ms * snn.ms,
        )
        net.connect(
            source.E.spikes,
            target.I.excitatory,
            name=f"{source_name}_E_to_{target_name}_I_K_EI",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=sparse_coupling(k_ei),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=coupling_delay_ms * snn.ms,
        )

    net.expose(
        network_a.E.spikes,
        network_a.I.spikes,
        network_b.E.spikes,
        network_b.I.spikes,
        name="population",
    )
    return snn.compile(net, target="tools/snn")


def author_phase_response_network() -> snn.Bundle:
    """Author one PING circuit with coupling-matched E and I probe paths."""
    net = snn.Network("ping_phase_response", dt=DT_MS * snn.ms)
    drive = net.input(
        f"drive_A_{INPUT_RATE_A_HZ:g}_Hz",
        shape=("time", "batch", N_INPUT),
        signal_type="spikes",
        unit="spike",
    )
    pulse_e = net.input(
        "coupling_matched_pulse_to_E",
        shape=("time", "batch", N_E),
        signal_type="spikes",
        unit="spike",
    )
    pulse_i = net.input(
        "coupling_matched_pulse_to_I",
        shape=("time", "batch", N_E),
        signal_type="spikes",
        unit="spike",
    )
    network = add_ping(net, name="PING_A", source=drive)
    net.connect(
        pulse_e,
        network.E.excitatory,
        name="probe_E_to_PING_A_E_K_EE",
        synapse=snn.AMPA(tau=2 * snn.ms),
        weight=sparse_coupling(K_EE),
        constraint=snn.NonNegative(),
        delay=COUPLING_DELAY_MS * snn.ms,
    )
    net.connect(
        pulse_i,
        network.I.excitatory,
        name="probe_E_to_PING_A_I_K_EI",
        synapse=snn.AMPA(tau=2 * snn.ms),
        weight=sparse_coupling(K_EI),
        constraint=snn.NonNegative(),
        delay=COUPLING_DELAY_MS * snn.ms,
    )
    net.expose(network.E.spikes, network.I.spikes, name="population")
    return snn.compile(net, target="tools/snn")


def poisson_input(*, rate_hz: float, seed: int, steps: int) -> torch.Tensor:
    probability = rate_hz * DT_MS / 1_000.0
    rng = np.random.default_rng(seed)
    spikes = rng.random((steps, 1, N_INPUT), dtype=np.float32) < probability
    return torch.from_numpy(spikes.astype(np.float32))


def make_uncoupled_inputs() -> dict[str, torch.Tensor]:
    """Create independent deterministic Poisson drives at the design rates."""
    steps = round(T_MS / DT_MS)
    inputs = {}
    rows = (
        (f"drive_A_{INPUT_RATE_A_HZ:g}_Hz", INPUT_RATE_A_HZ, INPUT_SEEDS[0]),
        (f"drive_B_{INPUT_RATE_B_HZ:g}_Hz", INPUT_RATE_B_HZ, INPUT_SEEDS[1]),
    )
    for name, rate_hz, seed in rows:
        inputs[name] = poisson_input(rate_hz=rate_hz, seed=seed, steps=steps)
    return inputs


def make_phase_response_inputs(
    *,
    target: str | None = None,
    arrival_step: int | None = None,
) -> dict[str, torch.Tensor]:
    """Create one fixed drive with an optional coupling-matched probe volley."""
    steps = round(PRC_T_MS / DT_MS)
    pulse_e = torch.zeros((steps, 1, N_E), dtype=torch.float32)
    pulse_i = torch.zeros((steps, 1, N_E), dtype=torch.float32)
    if target is not None:
        if arrival_step is None:
            raise ValueError("arrival_step is required when target is set")
        delay_steps = round(COUPLING_DELAY_MS / DT_MS)
        source_step = arrival_step - delay_steps
        if not (0 <= source_step < steps):
            raise ValueError("pulse source time is outside the simulation")
        pulses = {"E": pulse_e, "I": pulse_i}
        pulses[target][source_step, 0, :] = 1.0
    return {
        f"drive_A_{INPUT_RATE_A_HZ:g}_Hz": poisson_input(
            rate_hz=INPUT_RATE_A_HZ,
            seed=INPUT_SEEDS[0],
            steps=steps,
        ),
        "coupling_matched_pulse_to_E": pulse_e,
        "coupling_matched_pulse_to_I": pulse_i,
    }


def population_rate(spikes: np.ndarray, population_size: int) -> np.ndarray:
    """Return a 1 ms Gaussian-smoothed per-neuron firing rate in hertz."""
    counts = spikes[:, 0].sum(axis=1).astype(float)
    rate_hz = counts / population_size / (DT_MS / 1_000.0)
    return gaussian_filter1d(rate_hz, sigma=1.0 / DT_MS)


def detect_volleys(
    rate_hz: np.ndarray,
    *,
    burn_ms: float = BURN_MS,
) -> np.ndarray:
    """Detect separated excitatory population volleys after burn-in."""
    burn = round(burn_ms / DT_MS)
    post = rate_hz[burn:]
    if post.size == 0 or post.max() <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(
        post,
        distance=round(15.0 / DT_MS),
        prominence=0.1 * float(post.max()),
    )
    return peaks + burn


def interpolated_phase(peaks: np.ndarray, steps: int) -> np.ndarray:
    """Interpolate phase from zero to 2π between detected volleys."""
    phase = np.full(steps, np.nan)
    for left, right in zip(peaks[:-1], peaks[1:], strict=True):
        phase[left:right] = 2.0 * np.pi * np.arange(right - left) / (right - left)
    return phase


def rhythm_summary(peaks: np.ndarray) -> dict[str, float | int | None]:
    intervals_ms = np.diff(peaks) * DT_MS
    if intervals_ms.size == 0:
        return {"volleys": int(peaks.size), "frequency_hz": None, "iei_cv": None}
    mean_interval_ms = float(intervals_ms.mean())
    return {
        "volleys": int(peaks.size),
        "frequency_hz": 1_000.0 / mean_interval_ms,
        "iei_cv": float(intervals_ms.std() / mean_interval_ms),
    }


def inhibitory_cycle_summary(
    spikes: np.ndarray,
    excitatory_peaks: np.ndarray,
) -> dict[str, float | int]:
    """Summarize inhibitory spikes per neuron between excitatory volleys."""
    cycle_counts = [
        spikes[left:right, 0].sum(axis=0)
        for left, right in zip(
            excitatory_peaks[:-1], excitatory_peaks[1:], strict=True
        )
    ]
    if not cycle_counts:
        return {
            "cycles": 0,
            "mean_spikes_per_neuron": 0.0,
            "minimum": 0,
            "maximum": 0,
        }
    counts = np.concatenate(cycle_counts)
    return {
        "cycles": len(cycle_counts),
        "mean_spikes_per_neuron": float(counts.mean()),
        "minimum": int(counts.min()),
        "maximum": int(counts.max()),
    }


def analyse_uncoupled(recordings: dict[str, np.ndarray]) -> dict[str, object]:
    e_a = population_rate(recordings["population_0"], N_E)
    i_a = population_rate(recordings["population_1"], N_I)
    e_b = population_rate(recordings["population_2"], N_E)
    i_b = population_rate(recordings["population_3"], N_I)
    peaks_a = detect_volleys(e_a)
    peaks_b = detect_volleys(e_b)
    summary_a = rhythm_summary(peaks_a)
    summary_b = rhythm_summary(peaks_b)
    inhibition_a = inhibitory_cycle_summary(recordings["population_1"], peaks_a)
    inhibition_b = inhibitory_cycle_summary(recordings["population_3"], peaks_b)
    phase_a = interpolated_phase(peaks_a, len(e_a))
    phase_b = interpolated_phase(peaks_b, len(e_b))
    valid = np.isfinite(phase_a) & np.isfinite(phase_b)
    phase_difference = np.full_like(phase_a, np.nan)
    phase_difference[valid] = np.angle(
        np.exp(1j * (phase_a[valid] - phase_b[valid]))
    )
    valid_phase = phase_difference[valid]
    drift_wraps = int(np.count_nonzero(np.abs(np.diff(valid_phase)) > np.pi))

    for name, summary in (("A", summary_a), ("B", summary_b)):
        if summary["volleys"] < 20 or summary["iei_cv"] is None:
            raise RuntimeError(f"PING {name} did not produce a sustained rhythm")
        if float(summary["iei_cv"]) > 0.2:
            raise RuntimeError(f"PING {name} rhythm was too irregular")
    frequency_a = float(summary_a["frequency_hz"])
    frequency_b = float(summary_b["frequency_hz"])
    if not (30.0 <= frequency_a <= 80.0 and 30.0 <= frequency_b <= 80.0):
        raise RuntimeError("the uncoupled rhythms were outside the gamma band")
    if abs(frequency_a - frequency_b) < 0.5:
        raise RuntimeError("the uncoupled PING rhythms were not frequency-detuned")
    if drift_wraps < 2:
        raise RuntimeError("the uncoupled relative phase did not repeatedly wrap")
    for name, inhibition in (("A", inhibition_a), ("B", inhibition_b)):
        if inhibition["minimum"] != 1 or inhibition["maximum"] != 1:
            raise RuntimeError(
                f"PING {name} did not produce exactly one inhibitory spike "
                "per neuron per cycle before coupling"
            )

    return {
        "rate_e_a": e_a,
        "rate_i_a": i_a,
        "rate_e_b": e_b,
        "rate_i_b": i_b,
        "peaks_a": peaks_a,
        "peaks_b": peaks_b,
        "phase_difference": phase_difference,
        "network_a": summary_a,
        "network_b": summary_b,
        "inhibition_a": inhibition_a,
        "inhibition_b": inhibition_b,
        "drift_wraps": drift_wraps,
    }


def _normalise_window(trace: np.ndarray, window: slice) -> np.ndarray:
    values = trace[window]
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0 else values


def plot_uncoupled(analysis: dict[str, object], out: Path) -> None:
    """Show readable E/I rhythm excerpts above the full phase-drift trace."""
    theme.apply()
    start = round(DISPLAY_START_MS / DT_MS)
    stop = round(DISPLAY_END_MS / DT_MS)
    window = slice(start, stop)
    local_time_ms = np.arange(stop - start) * DT_MS + DISPLAY_START_MS
    full_time_ms = np.arange(len(analysis["phase_difference"])) * DT_MS

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2))
    for ax, name, e_key, i_key in (
        (axes[0], "Network A", "rate_e_a", "rate_i_a"),
        (axes[1], "Network B", "rate_e_b", "rate_i_b"),
    ):
        ax.plot(
            local_time_ms,
            _normalise_window(np.asarray(analysis[e_key]), window),
            color=theme.INK_BLACK,
            lw=1.0,
            label="E",
        )
        ax.plot(
            local_time_ms,
            _normalise_window(np.asarray(analysis[i_key]), window),
            color=theme.DEEP_RED,
            lw=1.0,
            label="I",
        )
        ax.set(ylabel=f"{name}\nnormalized rate", ylim=(-0.05, 1.1))
        ax.legend(frameon=False, ncol=2, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("time (ms), rhythm excerpt")

    axes[2].plot(
        full_time_ms,
        analysis["phase_difference"],
        color=theme.INK_BLACK,
        lw=0.9,
    )
    axes[2].axvline(BURN_MS, color=theme.GREY_MID, ls="--", lw=0.8)
    axes[2].set(
        xlim=(BURN_MS, T_MS),
        ylim=(-np.pi, np.pi),
        xlabel="time (ms)",
        ylabel="wrapped phase\ndifference (rad)",
    )
    axes[2].set_yticks((-np.pi, 0, np.pi), labels=(r"$-\pi$", "0", r"$\pi$"))
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def population_volley_events(
    spikes: np.ndarray,
    *,
    start: int,
    stop: int,
) -> list[dict[str, float | int]]:
    """Group adjacent occupied timesteps into population spike volleys."""
    counts = spikes[start:stop, 0].sum(axis=1)
    occupied = np.flatnonzero(counts)
    if occupied.size == 0:
        return []
    groups = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1)
    events = []
    for group in groups:
        group_counts = counts[group]
        centre = float(np.average(group + start, weights=group_counts))
        events.append(
            {
                "time_ms": centre * DT_MS,
                "spikes": int(group_counts.sum()),
            }
        )
    return events


def run_phase_response() -> tuple[dict[str, object], dict[str, object]]:
    """Measure the next-volley shift caused by E- and I-targeted probe volleys."""
    bundle = author_phase_response_network()
    baseline = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            inputs=make_phase_response_inputs(),
            seed=NETWORK_SEED,
        )
    )
    baseline_e = baseline.recordings["population_0"].cpu().numpy()
    baseline_i = baseline.recordings["population_1"].cpu().numpy()
    baseline_peaks = detect_volleys(population_rate(baseline_e, N_E))
    reference_step = round(PRC_REFERENCE_MS / DT_MS)
    left_index = int(np.searchsorted(baseline_peaks, reference_step) - 1)
    if left_index < 0 or left_index + 1 >= len(baseline_peaks):
        raise RuntimeError("no complete baseline cycle near the PRC reference time")
    left = int(baseline_peaks[left_index])
    baseline_next = int(baseline_peaks[left_index + 1])
    period_steps = baseline_next - left

    responses: dict[str, list[dict[str, float]]] = {"E": [], "I": []}
    representative_specs = {
        ("E", 0.70): "e_late_advance",
        ("I", 0.08): "i_early_no_doublet",
        ("I", 0.12): "i_early_doublet",
    }
    representative_cases: dict[str, dict[str, object]] = {}
    strongest_i_delay_steps = 0
    early_i_pulse_example: dict[str, object] | None = None
    for target in ("E", "I"):
        for fraction in PRC_PHASE_FRACTIONS:
            arrival = left + round(float(fraction) * period_steps)
            perturbed = simulate(
                ExecutionSpec(
                    kind="simulate",
                    executor="graph",
                    graph=bundle.graph,
                    inputs=make_phase_response_inputs(
                        target=target,
                        arrival_step=arrival,
                    ),
                    seed=NETWORK_SEED,
                )
            )
            perturbed_e = perturbed.recordings["population_0"].cpu().numpy()
            perturbed_i = perturbed.recordings["population_1"].cpu().numpy()
            perturbed_peaks = detect_volleys(population_rate(perturbed_e, N_E))
            candidates = perturbed_peaks[perturbed_peaks > arrival]
            if candidates.size == 0:
                raise RuntimeError(f"no E volley followed the {target}-targeted pulse")
            perturbed_next = int(candidates[0])
            shift_steps = baseline_next - perturbed_next
            response = {
                "pulse_phase_fraction": (arrival - left) / period_steps,
                "pulse_phase_rad": 2.0
                * np.pi
                * (arrival - left)
                / period_steps,
                "next_volley_shift_ms": shift_steps * DT_MS,
                "next_volley_phase_shift_rad": 2.0
                * np.pi
                * shift_steps
                / period_steps,
            }
            if target == "I":
                i_events = population_volley_events(
                    perturbed_i,
                    start=left,
                    stop=perturbed_next,
                )
                response["i_volleys_before_next_e"] = len(i_events)
                response["second_i_volley_latency_ms"] = (
                    i_events[1]["time_ms"] - i_events[0]["time_ms"]
                    if len(i_events) >= 2
                    else None
                )
            responses[target].append(response)
            case_name = representative_specs.get((target, round(float(fraction), 2)))
            if case_name is not None:
                representative_cases[case_name] = {
                    "target": target,
                    "pulse_phase_fraction": response["pulse_phase_fraction"],
                    "arrival_step": arrival,
                    "next_e_step": perturbed_next,
                    "rate_e": population_rate(perturbed_e, N_E),
                    "rate_i": population_rate(perturbed_i, N_I),
                    "i_volleys_before_next_e": response.get(
                        "i_volleys_before_next_e"
                    ),
                }
                if case_name == "i_early_doublet":
                    representative_cases[case_name].update(
                        {
                            "i_voltage": perturbed.recordings[
                                "PING_A_I.voltage"
                            ].cpu().numpy(),
                            "local_e_to_i_conductance": perturbed.recordings[
                                "PING_A_E_to_I.conductance"
                            ].cpu().numpy(),
                            "probe_e_to_i_conductance": perturbed.recordings[
                                "probe_E_to_PING_A_I_K_EI.conductance"
                            ].cpu().numpy(),
                        }
                    )
            if target == "I" and shift_steps < strongest_i_delay_steps:
                strongest_i_delay_steps = shift_steps
                event_stop = perturbed_next + round(2.0 / DT_MS)
                early_i_pulse_example = {
                    "pulse_phase_fraction": response["pulse_phase_fraction"],
                    "pulse_arrival_ms": arrival * DT_MS,
                    "baseline_next_e_volley_ms": baseline_next * DT_MS,
                    "perturbed_next_e_volley_ms": perturbed_next * DT_MS,
                    "baseline_i_volleys": population_volley_events(
                        baseline_i,
                        start=left,
                        stop=event_stop,
                    ),
                    "perturbed_i_volleys": population_volley_events(
                        perturbed_i,
                        start=left,
                        stop=event_stop,
                    ),
                }

    if early_i_pulse_example is None:
        raise RuntimeError("no I-targeted pulse delayed the next excitatory volley")
    if len(representative_cases) != len(representative_specs):
        raise RuntimeError("not all representative PRC cases were recorded")

    record = {
        "network": "PING_A",
        "baseline_cycle_start_ms": left * DT_MS,
        "baseline_cycle_period_ms": period_steps * DT_MS,
        "pulse": {
            "source_channels": N_E,
            "exact_fan_in": CROSS_FAN_IN,
            "arrival_delay_ms": COUPLING_DELAY_MS,
            "e_target_strength": K_EE,
            "i_target_strength": K_EI,
        },
        "positive_response_means": "next excitatory volley advanced",
        "sampling": {
            "coarse_pilot_interval_fraction": 0.1,
            "refined_region_fraction": [0.02, 0.30],
            "refined_interval_fraction": 0.02,
            "reason": "coarse pilot located the inhibitory transition near phase 0.1",
        },
        "responses": responses,
        "early_i_pulse_example": early_i_pulse_example,
    }
    illustration = {
        "left_step": left,
        "baseline_next_step": baseline_next,
        "baseline_rate_e": population_rate(baseline_e, N_E),
        "cases": representative_cases,
    }
    return record, illustration


def plot_phase_response_examples(
    illustration: dict[str, object],
    out: Path,
) -> None:
    """Show how three representative probes change the next PING volley."""
    theme.apply()
    left = int(illustration["left_step"])
    baseline_next = int(illustration["baseline_next_step"])
    start = left - round(2.0 / DT_MS)
    stop = left + round(32.0 / DT_MS)
    window = slice(start, stop)
    time_ms = (np.arange(start, stop) - left) * DT_MS
    baseline_e = _normalise_window(
        np.asarray(illustration["baseline_rate_e"]),
        window,
    )
    panels = (
        ("e_late_advance", "E probe at phase 0.70: next volley advances"),
        ("i_early_no_doublet", "I probe at phase 0.08: no doublet"),
        ("i_early_doublet", "I probe at phase 0.12: doublet delays next volley"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True, sharey=True)
    for ax, (case_name, title) in zip(axes, panels, strict=True):
        case = illustration["cases"][case_name]
        rate_e = _normalise_window(np.asarray(case["rate_e"]), window)
        rate_i = _normalise_window(np.asarray(case["rate_i"]), window)
        arrival_ms = (int(case["arrival_step"]) - left) * DT_MS
        next_e_ms = (int(case["next_e_step"]) - left) * DT_MS
        baseline_next_ms = (baseline_next - left) * DT_MS
        ax.plot(time_ms, baseline_e, color=theme.GREY_LIGHT, lw=1.0, ls="--")
        ax.plot(time_ms, rate_e, color=theme.INK_BLACK, lw=1.2, label="E")
        ax.plot(time_ms, rate_i, color=theme.DEEP_RED, lw=1.2, label="I")
        ax.axvline(
            arrival_ms,
            color=theme.ELECTRIC_CYAN,
            lw=1.1,
            label="probe arrival",
        )
        ax.axvline(
            baseline_next_ms,
            color=theme.GREY_MID,
            lw=0.9,
            ls="--",
            label="baseline next E",
        )
        if next_e_ms != baseline_next_ms:
            ax.axvline(next_e_ms, color=theme.INK_BLACK, lw=0.8, ls=":")
        ax.set(title=title, ylim=(-0.05, 1.12))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=4, loc="upper center")
    axes[1].set_ylabel("normalized population rate")
    axes[2].set_xlabel("time from reference E volley (ms)")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_phase_response(
    phase_response: dict[str, object],
    illustration: dict[str, object],
    out: Path,
) -> None:
    """Summarize whole-cycle responses and resolve the early-I transition."""
    theme.apply()
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(6.8, 8.4),
        gridspec_kw={"height_ratios": [1.35, 1.0, 0.8, 1.0]},
    )
    i_rows = phase_response["responses"]["I"]
    i_phase = np.asarray([row["pulse_phase_fraction"] for row in i_rows])
    i_shift = np.asarray([row["next_volley_shift_ms"] for row in i_rows])
    doublet = np.asarray(
        [row["i_volleys_before_next_e"] == 2 for row in i_rows]
    )
    doublet_indices = np.flatnonzero(doublet)
    first = int(doublet_indices[0])
    last = int(doublet_indices[-1])
    window_left = 0.5 * (i_phase[first - 1] + i_phase[first])
    window_right = 0.5 * (i_phase[last] + i_phase[last + 1])

    whole = axes[0]
    for target, color in (("E", theme.INK_BLACK), ("I", theme.DEEP_RED)):
        rows = phase_response["responses"][target]
        phase = np.asarray([row["pulse_phase_fraction"] for row in rows])
        response = np.asarray([row["next_volley_shift_ms"] for row in rows])
        whole.scatter(
            phase,
            response,
            s=22,
            color=color,
            label=f"pulse to {target}",
            zorder=3,
        )
    whole.axvspan(
        window_left,
        window_right,
        color=theme.DEEP_RED,
        alpha=0.08,
        lw=0,
    )
    whole.axhline(0.0, color=theme.GREY_MID, lw=0.8, ls="--")
    whole.annotate(
        "late E input\nadvances next volley",
        xy=(0.70, 2.7),
        xytext=(0.50, 1.45),
        arrowprops={"arrowstyle": "-", "color": theme.INK_BLACK, "lw": 0.8},
        ha="right",
        va="center",
        fontsize=theme.SIZE_ANNOTATION,
    )
    whole.annotate(
        "I doublet delays next volley",
        xy=(i_phase[doublet][1], i_shift[doublet][1]),
        xytext=(0.24, -4.7),
        arrowprops={"arrowstyle": "-", "color": theme.DEEP_RED, "lw": 0.8},
        color=theme.DEEP_RED,
        fontsize=theme.SIZE_ANNOTATION,
    )
    whole.set(
        xlim=(0.0, 1.0),
        ylim=(-6.8, 3.3),
        title="Whole cycle: E advances late; I delays only in a narrow window",
        ylabel="next E-volley shift (ms)",
    )
    whole.legend(frameon=False, ncol=2, loc="lower right")
    whole.spines[["top", "right"]].set_visible(False)

    early = axes[1]
    early_mask = i_phase <= 0.30 + 1e-9
    single_early = early_mask & ~doublet
    doublet_early = early_mask & doublet
    early.axvspan(
        window_left,
        window_right,
        color=theme.DEEP_RED,
        alpha=0.08,
        lw=0,
    )
    early.scatter(
        i_phase[single_early],
        i_shift[single_early],
        s=26,
        facecolors="white",
        edgecolors=theme.GREY_MID,
        label="one I volley",
        zorder=3,
    )
    early.scatter(
        i_phase[doublet_early],
        i_shift[doublet_early],
        s=30,
        color=theme.DEEP_RED,
        label="two I volleys",
        zorder=3,
    )
    early.axhline(0.0, color=theme.GREY_MID, lw=0.8, ls="--")
    for row in np.asarray(i_rows, dtype=object)[doublet]:
        early.annotate(
            f'{row["second_i_volley_latency_ms"]:.2f} ms',
            xy=(row["pulse_phase_fraction"], row["next_volley_shift_ms"]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            color=theme.DEEP_RED,
            fontsize=theme.SIZE_ANNOTATION,
        )
    early.text(
        0.5 * (window_left + window_right),
        0.8,
        "doublet window",
        ha="center",
        color=theme.DEEP_RED,
        fontsize=theme.SIZE_ANNOTATION,
    )
    early.set(
        xlim=(0.0, 0.30),
        ylim=(-6.8, 1.4),
        title="Early I response: filled points are doublets; labels give latency",
        xlabel="probe arrival phase (fraction of cycle)",
        ylabel="next E-volley shift (ms)",
    )
    early.legend(frameon=False, ncol=2, loc="upper right")
    early.spines[["top", "right"]].set_visible(False)

    state_case = illustration["cases"]["i_early_doublet"]
    state_left = int(illustration["left_step"])
    state_start = state_left - round(0.5 / DT_MS)
    state_stop = state_left + round(7.0 / DT_MS)
    state_window = slice(state_start, state_stop)
    state_time_ms = (np.arange(state_start, state_stop) - state_left) * DT_MS
    arrival_ms = (int(state_case["arrival_step"]) - state_left) * DT_MS

    local_g = np.asarray(state_case["local_e_to_i_conductance"])[
        state_window, 0
    ].mean(axis=1)
    probe_g = np.asarray(state_case["probe_e_to_i_conductance"])[
        state_window, 0
    ].mean(axis=1)
    conductance = axes[2]
    conductance.plot(
        state_time_ms,
        local_g,
        color=theme.AMBER,
        lw=1.3,
        ls="--",
        label="local E→I",
        zorder=3,
    )
    conductance.plot(
        state_time_ms,
        probe_g,
        color=theme.ELECTRIC_CYAN,
        lw=1.3,
        ls="-.",
        label="probe→I",
        zorder=3,
    )
    conductance.plot(
        state_time_ms,
        local_g + probe_g,
        color=theme.INK_BLACK,
        lw=1.6,
        label="total excitation",
        zorder=2,
    )
    conductance.axvline(arrival_ms, color=theme.ELECTRIC_CYAN, lw=0.9, ls=":")
    conductance.set(
        xlim=(-0.5, 7.0),
        title="Doublet case: the probe adds to local excitation after recovery",
        ylabel="mean I excitatory\nconductance (µS)",
    )
    conductance.legend(frameon=False, ncol=3, loc="upper right")
    conductance.spines[["top", "right"]].set_visible(False)

    voltage_values = np.asarray(state_case["i_voltage"])[state_window, 0]
    voltage = axes[3]
    for neuron_voltage in voltage_values.T:
        voltage.plot(
            state_time_ms,
            neuron_voltage,
            color=theme.DEEP_RED_LIGHT,
            lw=0.5,
            alpha=0.35,
        )
    voltage.plot(
        state_time_ms,
        voltage_values.mean(axis=1),
        color=theme.INK_BLACK,
        lw=1.2,
        label="I-cell mean",
    )
    voltage.axhline(-50.0, color=theme.GREY_MID, lw=0.8, ls="--", label="threshold")
    voltage.axvline(
        arrival_ms,
        color=theme.ELECTRIC_CYAN,
        lw=0.9,
        ls=":",
        label="probe arrival",
    )
    voltage.set(
        xlim=(-0.5, 7.0),
        xlabel="time from reference E volley (ms)",
        ylabel="I membrane\nvoltage (mV)",
    )
    voltage.legend(frameon=False, ncol=3, loc="upper right")
    voltage.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyse_pathway_branch(
    recordings: dict[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Measure relative-phase drift after coupling onset."""
    rate_a = population_rate(recordings["population_0"], N_E)
    rate_b = population_rate(recordings["population_2"], N_E)
    peaks_a = detect_volleys(rate_a, burn_ms=0.0)
    peaks_b = detect_volleys(rate_b, burn_ms=0.0)
    phase_a = interpolated_phase(peaks_a, len(rate_a))
    phase_b = interpolated_phase(peaks_b, len(rate_b))
    valid = np.isfinite(phase_a) & np.isfinite(phase_b)
    valid_steps = np.flatnonzero(valid)
    wrapped = np.angle(np.exp(1j * (phase_a[valid] - phase_b[valid])))
    unwrapped_cycles = np.unwrap(wrapped) / (2.0 * np.pi)
    unwrapped_cycles -= unwrapped_cycles[0]
    time_ms = valid_steps * DT_MS
    final = time_ms >= time_ms[-1] - 500.0
    final_time_s = time_ms[final] / 1_000.0
    final_cycles = unwrapped_cycles[final]
    drift_rate = float(np.polyfit(final_time_s, final_cycles, 1)[0])
    concentration = float(abs(np.mean(np.exp(1j * wrapped[final]))))
    locked = abs(drift_rate) < 0.25 and concentration > 0.95
    locked_phase = (
        float(np.angle(np.mean(np.exp(1j * wrapped[final])))) if locked else None
    )
    summary_a = rhythm_summary(peaks_a)
    summary_b = rhythm_summary(peaks_b)
    record = {
        "PING_A": summary_a,
        "PING_B": summary_b,
        "final_drift_rate_cycles_per_s": drift_rate,
        "final_phase_concentration": concentration,
        "phase_locked": locked,
        "locked_phase_rad": locked_phase,
    }
    traces = {
        "time_ms": time_ms,
        "unwrapped_phase_change_cycles": unwrapped_cycles,
    }
    return record, traces


def run_pathway_comparison() -> tuple[dict[str, object], dict[str, object]]:
    """Branch four coupling conditions from one uncoupled runtime state."""
    inputs = make_uncoupled_inputs()
    onset = round(COUPLING_ONSET_MS / DT_MS)
    prefix_graph = author_network(
        k_ee=0.0,
        k_ei=0.0,
        coupling_delay_ms=COUPLING_DELAY_MS,
    ).graph
    prefix = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=prefix_graph,
            inputs={name: value[:onset] for name, value in inputs.items()},
            seed=NETWORK_SEED,
        )
    )
    if prefix.runtime_state is None:
        raise RuntimeError("the uncoupled prefix did not return runtime state")

    specifications = (
        ("none", "No coupling", 0.0, 0.0),
        ("e_to_e", "E→E only", K_EE, 0.0),
        ("e_to_i", "E→I only", 0.0, K_EI),
        ("both", "Both pathways", K_EE, K_EI),
    )
    condition_records = []
    condition_traces = {}
    suffix_inputs = {name: value[onset:] for name, value in inputs.items()}
    for condition_id, label, k_ee, k_ei in specifications:
        graph = author_network(
            k_ee=k_ee,
            k_ei=k_ei,
            coupling_delay_ms=COUPLING_DELAY_MS,
        ).graph
        result = simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=graph,
                inputs=suffix_inputs,
                seed=NETWORK_SEED,
            ),
            runtime_state=prefix.runtime_state.detached(),
        )
        recordings = {
            key: value.cpu().numpy()
            for key, value in result.recordings.items()
        }
        condition_record, traces = analyse_pathway_branch(recordings)
        condition_record.update(
            {
                "id": condition_id,
                "label": label,
                "K_EE": k_ee,
                "K_EI": k_ei,
            }
        )
        condition_records.append(condition_record)
        condition_traces[condition_id] = traces

    return (
        {
            "coupling_onset_ms": COUPLING_ONSET_MS,
            "shared_delay_ms": COUPLING_DELAY_MS,
            "classification": {
                "final_window_ms": 500.0,
                "maximum_absolute_drift_cycles_per_s": 0.25,
                "minimum_phase_concentration": 0.95,
            },
            "conditions": condition_records,
        },
        condition_traces,
    )


def plot_pathway_comparison(
    pathway_comparison: dict[str, object],
    traces: dict[str, object],
    out: Path,
) -> None:
    """Show which coupling pathways arrest relative-phase drift."""
    theme.apply()
    colors = {
        "none": theme.GREY_MID,
        "e_to_e": theme.INK_BLACK,
        "e_to_i": theme.DEEP_RED,
        "both": theme.ELECTRIC_CYAN,
    }
    all_phase = np.concatenate(
        [
            np.asarray(traces[row["id"]]["unwrapped_phase_change_cycles"])
            for row in pathway_comparison["conditions"]
        ]
    )
    lower = float(all_phase.min()) - 0.3
    upper = float(all_phase.max()) + 0.3
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 6.4), sharex=True, sharey=True)
    for ax, condition in zip(
        axes,
        pathway_comparison["conditions"],
        strict=True,
    ):
        condition_id = condition["id"]
        condition_trace = traces[condition_id]
        state = "phase locked" if condition["phase_locked"] else "phase drift"
        ax.plot(
            condition_trace["time_ms"],
            condition_trace["unwrapped_phase_change_cycles"],
            color=colors[condition_id],
            lw=1.2,
        )
        ax.axhline(0.0, color=theme.GREY_LIGHT, lw=0.7, ls="--")
        ax.set(title=f'{condition["label"]}: {state}', ylim=(lower, upper))
        ax.text(
            0.99,
            0.82,
            f'{condition["final_drift_rate_cycles_per_s"]:.2f} cycles/s',
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=theme.SIZE_ANNOTATION,
            color=colors[condition_id],
        )
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_ylabel("unwrapped relative phase change (cycles)")
    axes[-1].set_xlabel("time after coupling onset (ms)")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def experiment_record(
    analysis: dict[str, object],
    phase_response: dict[str, object],
    pathway_comparison: dict[str, object],
) -> dict[str, object]:
    return {
        "status": STATUS,
        "completed_methods": [1, 2, 3, 4],
        "simulation_run": True,
        "network": {
            "local_circuit": "matched E-to-I-to-E PING",
            "populations_per_network": {"E": N_E, "I": N_I},
            "detuning_input_rates_hz": {
                "PING_A": INPUT_RATE_A_HZ,
                "PING_B": INPUT_RATE_B_HZ,
            },
            "cross_network_projections": ["E-to-E", "E-to-I"],
            "reciprocal": True,
            "exact_fan_in_per_target": CROSS_FAN_IN,
            "weights": {"K_EE": K_EE, "K_EI": K_EI},
            "delay_ms": COUPLING_DELAY_MS,
            "local_e_to_i": {
                "weight": E_TO_I_WEIGHT,
                "ampa_tau_ms": E_TO_I_TAU_MS,
            },
        },
        "uncoupled": {
            "PING_A": analysis["network_a"],
            "PING_B": analysis["network_b"],
            "inhibitory_spikes_per_cycle": {
                "PING_A": analysis["inhibition_a"],
                "PING_B": analysis["inhibition_b"],
            },
            "phase_wraps": analysis["drift_wraps"],
        },
        "phase_response": phase_response,
        "pathway_comparison": pathway_comparison,
        "remaining_methods_unrun": [5],
    }


def main() -> None:
    meta = parse_meta(sys.argv)
    if meta.runpod:
        raise SystemExit("exp085 methods 1-4 are a bounded local run")
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    with published_run(SLUG, run_id, scale=SCALE) as (_scratch, staging):
        bundle = author_network()
        bundle.write(staging / "network.bundle", visualise=True)
        bundle.visualise(
            staging / "network.svg",
            view="circuit",
            expand_groups=PING_GROUPS,
        )
        uncoupled = author_network(k_ee=0.0, k_ei=0.0)
        result = simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=uncoupled.graph,
                inputs=make_uncoupled_inputs(),
                seed=NETWORK_SEED,
            )
        )
        recordings = {
            key: value.cpu().numpy().astype(np.uint8)
            for key, value in result.recordings.items()
        }
        analysis = analyse_uncoupled(recordings)
        plot_uncoupled(analysis, staging / "uncoupled.png")
        np.savez_compressed(
            staging / "uncoupled_trace.npz",
            rate_e_a=analysis["rate_e_a"],
            rate_i_a=analysis["rate_i_a"],
            rate_e_b=analysis["rate_e_b"],
            rate_i_b=analysis["rate_i_b"],
            peaks_a=analysis["peaks_a"],
            peaks_b=analysis["peaks_b"],
            phase_difference=analysis["phase_difference"],
        )
        phase_response, phase_response_examples = run_phase_response()
        plot_phase_response_examples(
            phase_response_examples,
            staging / "phase_response_examples.png",
        )
        plot_phase_response(
            phase_response,
            phase_response_examples,
            staging / "phase_response.png",
        )
        pathway_comparison, pathway_traces = run_pathway_comparison()
        plot_pathway_comparison(
            pathway_comparison,
            pathway_traces,
            staging / "pathway_comparison.png",
        )
        record = experiment_record(
            analysis,
            phase_response,
            pathway_comparison,
        )
        (staging / "protocol.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload=record,
        )


if __name__ == "__main__":
    main()
