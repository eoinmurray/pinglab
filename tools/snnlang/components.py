"""Reusable authoring functions; components expand before serialisation."""

from __future__ import annotations

from dataclasses import dataclass

from .core import (
    AMPA,
    COBA_LIF,
    GABA,
    Network,
    NonNegative,
    Normal,
    Population,
    Signal,
    ms,
)


@dataclass(frozen=True)
class PING:
    E: Population
    I: Population


def ping(
    net: Network,
    *,
    name: str,
    n_e: int,
    n_i: int,
    source: Signal | None = None,
    tau_gaba=9 * ms,
) -> PING:
    with net.group(name):
        e = net.population(f"{name}_E", size=n_e, neuron=COBA_LIF(tau_mem=20 * ms))
        i = net.population(f"{name}_I", size=n_i, neuron=COBA_LIF(tau_mem=10 * ms))
        net.connect(
            e.spikes,
            i.excitatory,
            name=f"{name}_E_to_I",
            synapse=AMPA(tau=5 * ms),
            weight=Normal(0.5, 0.05),
            constraint=NonNegative(),
            connection="recurrent",
            delay=0.1 * ms,
        )
        net.connect(
            i.spikes,
            e.inhibitory,
            name=f"{name}_I_to_E",
            synapse=GABA(tau=tau_gaba),
            weight=Normal(1.0, 0.1),
            constraint=NonNegative(),
            connection="recurrent",
            delay=0.1 * ms,
        )
        if source is not None:
            net.connect(
                source,
                e.excitatory,
                name=f"{name}_input",
                synapse=AMPA(tau=5 * ms),
                weight=Normal(0.2, 0.03),
                constraint=NonNegative(),
            )
    return PING(e, i)
