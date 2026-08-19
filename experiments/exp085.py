"""EXP085: author the proposed pair of coupled SNNLANG PING networks.

This proposal runner exports the graph and its parameters. It does not run the
synchronization experiment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import snnlang as snn  # noqa: E402, TID251

from helpers.cli import parse_meta  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp085"
STATUS = "proposal"

DT_MS = 0.1
N_INPUT = 128
N_E = 80
N_I = 20
TAU_GABA_A_MS = 4.0
TAU_GABA_B_MS = 5.0
COUPLING_K = 0.08
COUPLING_DELAY_MS = 0.1
PING_GROUPS = ("PING_A", "PING_B")

SCALE = {
    "status": STATUS,
    "dt_ms": DT_MS,
    "n_input_per_network": N_INPUT,
    "n_e_per_network": N_E,
    "n_i_per_network": N_I,
    "tau_gaba_a_ms": TAU_GABA_A_MS,
    "tau_gaba_b_ms": TAU_GABA_B_MS,
    "coupling_k": COUPLING_K,
    "coupling_delay_ms": COUPLING_DELAY_MS,
}


def author_network(coupling_k: float = COUPLING_K) -> snn.Bundle:
    """Define the active-coupling graph used after coupling onset."""
    net = snn.Network("ping_pair", dt=DT_MS * snn.ms)
    drive_a = net.input(
        "drive_a",
        shape=("time", "batch", N_INPUT),
        signal_type="spikes",
        unit="spike",
    )
    drive_b = net.input(
        "drive_b",
        shape=("time", "batch", N_INPUT),
        signal_type="spikes",
        unit="spike",
    )
    network_a = snn.components.ping(
        net,
        name="PING_A",
        n_e=N_E,
        n_i=N_I,
        source=drive_a,
        tau_gaba=TAU_GABA_A_MS * snn.ms,
    )
    network_b = snn.components.ping(
        net,
        name="PING_B",
        n_e=N_E,
        n_i=N_I,
        source=drive_b,
        tau_gaba=TAU_GABA_B_MS * snn.ms,
    )

    for source_name, source, target_name, target in (
        ("PING_A", network_a, "PING_B", network_b),
        ("PING_B", network_b, "PING_A", network_a),
    ):
        for population_name, target_port in (
            ("E", target.E.excitatory),
            ("I", target.I.excitatory),
        ):
            net.connect(
                source.E.spikes,
                target_port,
                name=f"{source_name}_E_to_{target_name}_{population_name}",
                synapse=snn.AMPA(tau=2 * snn.ms),
                weight=snn.Constant(coupling_k),
                constraint=snn.NonNegative(),
                connection="feedback",
                delay=COUPLING_DELAY_MS * snn.ms,
            )

    net.expose(
        network_a.E.spikes,
        network_a.I.spikes,
        network_b.E.spikes,
        network_b.I.spikes,
        name="population",
    )
    return snn.compile(net, target="tools/snn")


def main() -> None:
    meta = parse_meta(sys.argv)
    run_id = next_run_id(SLUG)
    with published_run(SLUG, run_id, scale=SCALE, plot_only=meta.plot_only) as (
        _scratch,
        staging,
    ):
        bundle = author_network()
        bundle_dir = staging / "network.bundle"
        bundle.write(bundle_dir, visualise=True)
        bundle.visualise(
            staging / "network.svg",
            view="circuit",
            expand_groups=PING_GROUPS,
        )
        protocol = {
            "status": STATUS,
            "simulation_run": False,
            "main_parameter": {
                "name": "K",
                "value": COUPLING_K,
                "meaning": "weight of each reciprocal cross-network AMPA projection",
                "before_onset": 0.0,
                "after_onset": COUPLING_K,
            },
            "network": SCALE,
        }
        (staging / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
