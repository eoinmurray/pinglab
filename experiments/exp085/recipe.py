"""Committed exp085 recipe and acquisition timing; no execution on import."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tools import snnlang as snn  # noqa: TID251

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
    "completed_methods": [1, 2, 3, 4, 5],
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
    return snn.compile(net, target="tools/snnsim")


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
    return snn.compile(net, target="tools/snnsim")


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


PATHWAYS = (
    ("none", "No coupling", 0.0, 0.0),
    ("e_to_e", "E→E only", K_EE, 0.0),
    ("e_to_i", "E→I only", 0.0, K_EI),
    ("both", "Both pathways", K_EE, K_EI),
)
FIGURES = (
    "network.svg",
    "uncoupled.png",
    "phase_response_examples.png",
    "phase_response.png",
    "pathway_comparison.png",
    "event_aligned_mechanism.png",
)


def reference_cycle(recordings):
    """The acquisition decision needed to place probes; no response analysis."""
    peaks = detect_volleys(population_rate(recordings["population_0"], N_E))
    index = int(np.searchsorted(peaks, round(PRC_REFERENCE_MS / DT_MS)) - 1)
    if index < 0 or index + 1 >= len(peaks):
        raise RuntimeError("no complete baseline cycle near the PRC reference time")
    return int(peaks[index]), int(peaks[index + 1])


def probe_schedule(left, baseline_next):
    return [
        {
            "id": f"prc-{target}-{index:02d}",
            "target": target,
            "fraction": float(fraction),
            "arrival_step": left + round(float(fraction) * (baseline_next - left)),
        }
        for target in ("E", "I")
        for index, fraction in enumerate(PRC_PHASE_FRACTIONS)
    ]


def graphs():
    return {
        **{
            name: author_network(k_ee=ee, k_ei=ei).graph for name, _, ee, ei in PATHWAYS
        },
        "prc": author_phase_response_network().graph,
    }


def graph_digest(graph):
    return hashlib.sha256(
        json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def configuration():
    return {
        "schema": "exp085.recipe/v1",
        **SCALE,
        "network_seed": NETWORK_SEED,
        "input_seeds": list(INPUT_SEEDS),
        "graph_hashes": {name: graph_digest(graph) for name, graph in graphs().items()},
    }
