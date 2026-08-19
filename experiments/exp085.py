"""EXP085 method 1: define and render two coupled cortical PING networks."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import snnlang as snn  # noqa: E402, TID251

from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp085"
STATUS = "draft"

DT_MS = 0.1
N_INPUT = 128
N_E = 80
N_I = 20
TAU_GABA_MS = 9.0
E_REFRACTORY_MS = 3.0
I_REFRACTORY_MS = 1.5

# These rates define the intended detuning. Method 2 must verify the resulting
# uncoupled gamma frequencies; they are design inputs, not completed results.
INPUT_RATE_A_HZ = 110.0
INPUT_RATE_B_HZ = 90.0

# Separate controls even though their initial nominal values match. The graph
# executor divides each nominal total strength across the realised fan-in.
K_EE = 0.08
K_EI = 0.08
COUPLING_DELAY_MS = 0.5
CROSS_FAN_IN = 8
CROSS_ZERO_FRACTION = 1.0 - CROSS_FAN_IN / N_E
PING_GROUPS = ("PING_A", "PING_B")

SCALE = {
    "status": STATUS,
    "completed_methods": [1],
    "dt_ms": DT_MS,
    "n_input_per_network": N_INPUT,
    "n_e_per_network": N_E,
    "n_i_per_network": N_I,
    "tau_gaba_ms": TAU_GABA_MS,
    "e_refractory_ms": E_REFRACTORY_MS,
    "i_refractory_ms": I_REFRACTORY_MS,
    "input_rate_a_hz": INPUT_RATE_A_HZ,
    "input_rate_b_hz": INPUT_RATE_B_HZ,
    "k_ee": K_EE,
    "k_ei": K_EI,
    "coupling_delay_ms": COUPLING_DELAY_MS,
    "cross_fan_in": CROSS_FAN_IN,
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
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.Normal(0.5, 0.05),
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
    network_a = add_ping(net, name="PING_A", source=drive_a)
    network_b = add_ping(net, name="PING_B", source=drive_b)

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


def method_1_record() -> dict[str, object]:
    return {
        "status": STATUS,
        "completed_methods": [1],
        "simulation_run": False,
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
        },
        "remaining_methods_unrun": [2, 3, 4, 5],
    }


def main() -> None:
    meta = parse_meta(sys.argv)
    if meta.runpod:
        raise SystemExit("exp085 method 1 is a bounded local graph build")
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
        record = method_1_record()
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
