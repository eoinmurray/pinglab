"""EXP087 draft: author the planned Diesmann-style synfire chain."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import snnlang as snn  # noqa: E402, TID251

from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp087"
STATUS = "draft"

DT_MS = 0.1
LAYERS = 6
NEURONS_PER_LAYER = 100
PACKET_CHANNELS = 100
BACKGROUND_CHANNELS = 600
FEEDFORWARD_FAN_IN = 40
FEEDFORWARD_TOTAL_STRENGTH_US = 0.8
FEEDFORWARD_DELAY_MS = 1.0
BACKGROUND_FAN_IN = 100
BACKGROUND_TOTAL_STRENGTH_US = 0.25

SCALE = {
    "status": STATUS,
    "simulation_run": False,
    "compiled_methods": [1],
    "dt_ms": DT_MS,
    "layers": LAYERS,
    "neurons_per_layer": NEURONS_PER_LAYER,
    "packet_channels": PACKET_CHANNELS,
    "background_channels": BACKGROUND_CHANNELS,
    "feedforward_fan_in": FEEDFORWARD_FAN_IN,
    "feedforward_total_strength_us": FEEDFORWARD_TOTAL_STRENGTH_US,
    "feedforward_delay_ms": FEEDFORWARD_DELAY_MS,
    "background_fan_in": BACKGROUND_FAN_IN,
    "background_total_strength_us": BACKGROUND_TOTAL_STRENGTH_US,
}


def exact_fan_in(total_strength: float, fan_in: int, source_size: int):
    """Create a fan-in-normalized sparse initializer."""
    return snn.LowerClampedNormal(
        total_strength,
        0.0,
        initial_zero_fraction=1.0 - fan_in / source_size,
        zeroing="exact_k",
    )


def author_network() -> snn.Bundle:
    """Author six feedforward pools with explicit packet and background inputs."""
    net = snn.Network("diesmann_synfire_chain", dt=DT_MS * snn.ms)
    packet = net.input(
        "pulse_packet",
        shape=("time", "batch", PACKET_CHANNELS),
        signal_type="spikes",
        unit="spike",
    )
    background = net.input(
        "independent_background",
        shape=("time", "batch", BACKGROUND_CHANNELS),
        signal_type="spikes",
        unit="spike",
    )

    pools = []
    with net.group("synfire_chain"):
        for layer in range(1, LAYERS + 1):
            pool = net.population(
                f"pool_{layer}",
                size=NEURONS_PER_LAYER,
                neuron=snn.COBA_LIF(
                    tau_mem=20 * snn.ms,
                    capacitance_nf=1.0,
                    leak_us=0.05,
                    resting_mv=-65.0,
                    threshold_mv=-50.0,
                    reset_mv=-65.0,
                    refractory_steps=round(2.0 / DT_MS),
                    voltage_grad_dampen=80.0,
                    initial_voltage_mv=-65.0,
                ),
            )
            pools.append(pool)
            net.connect(
                background,
                pool.excitatory,
                name=f"background_to_pool_{layer}",
                synapse=snn.AMPA(tau=2 * snn.ms),
                weight=exact_fan_in(
                    BACKGROUND_TOTAL_STRENGTH_US,
                    BACKGROUND_FAN_IN,
                    BACKGROUND_CHANNELS,
                ),
                constraint=snn.NonNegative(),
            )

        net.connect(
            packet,
            pools[0].excitatory,
            name="packet_to_pool_1",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=exact_fan_in(
                FEEDFORWARD_TOTAL_STRENGTH_US,
                FEEDFORWARD_FAN_IN,
                PACKET_CHANNELS,
            ),
            constraint=snn.NonNegative(),
            delay=FEEDFORWARD_DELAY_MS * snn.ms,
        )
        for layer, (source, target) in enumerate(
            zip(pools[:-1], pools[1:], strict=True),
            start=1,
        ):
            net.connect(
                source.spikes,
                target.excitatory,
                name=f"pool_{layer}_to_pool_{layer + 1}",
                synapse=snn.AMPA(tau=2 * snn.ms),
                weight=exact_fan_in(
                    FEEDFORWARD_TOTAL_STRENGTH_US,
                    FEEDFORWARD_FAN_IN,
                    NEURONS_PER_LAYER,
                ),
                constraint=snn.NonNegative(),
                connection="feedforward",
                delay=FEEDFORWARD_DELAY_MS * snn.ms,
            )

    net.expose(*(pool.spikes for pool in pools), name="pool_spikes")
    return snn.compile(net, target="tools/snn")


def main() -> None:
    meta = parse_meta(sys.argv)
    if meta.runpod:
        raise SystemExit("exp087 currently compiles a local draft graph only")
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    with published_run(SLUG, run_id, scale=SCALE) as (_scratch, staging):
        bundle = author_network()
        bundle.write(staging / "network.bundle", visualise=True)
        bundle.visualise(
            staging / "network.svg",
            view="circuit",
            expand_groups=("synfire_chain",),
        )
        record = {
            **SCALE,
            "question": "Can pulse packets converge to a stable size and width?",
            "remaining_methods_unrun": [2, 3],
        }
        (staging / "protocol.json").write_text(json.dumps(record, indent=2) + "\n")
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload=record,
        )


if __name__ == "__main__":
    main()
