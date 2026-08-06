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
    include_silent_recurrence: bool = False,
) -> PING:
    with net.group(name):
        # The explicit step counts preserve the legacy COBANet numerical
        # contract: its refractory constants were derived at the historical
        # 0.25 ms module default and are 12 E / 6 I steps for every run.
        e = net.population(
            f"{name}_E", size=n_e,
            neuron=COBA_LIF(
                tau_mem=20 * ms, capacitance_nf=1.0, leak_us=0.05,
                resting_mv=-65.0, threshold_mv=-50.0, reset_mv=-65.0,
                refractory_steps=12, voltage_grad_dampen=80.0,
                initial_voltage_mv=-65.0,
            ),
        )
        i = net.population(
            f"{name}_I", size=n_i,
            neuron=COBA_LIF(
                tau_mem=5 * ms, capacitance_nf=0.5, leak_us=0.1,
                resting_mv=-65.0, threshold_mv=-50.0, reset_mv=-65.0,
                refractory_steps=6, voltage_grad_dampen=80.0,
                initial_voltage_mv=-65.0,
            ),
        )
        if include_silent_recurrence:
            net.connect(
                e.spikes,
                e.excitatory,
                name=f"{name}_E_to_E",
                synapse=AMPA(tau=2 * ms),
                weight=Normal(0.0, 0.0),
                constraint=NonNegative(),
                connection="recurrent",
                delay=0.1 * ms,
            )
        net.connect(
            e.spikes,
            i.excitatory,
            name=f"{name}_E_to_I",
            synapse=AMPA(tau=2 * ms),
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
        if include_silent_recurrence:
            net.connect(
                i.spikes,
                i.inhibitory,
                name=f"{name}_I_to_I",
                synapse=GABA(tau=tau_gaba),
                weight=Normal(0.0, 0.0),
                constraint=NonNegative(),
                connection="recurrent",
                delay=0.1 * ms,
            )
        if source is not None:
            net.connect(
                source,
                e.excitatory,
                name=f"{name}_input",
                synapse=AMPA(tau=2 * ms),
                weight=Normal(0.2, 0.03),
                constraint=NonNegative(),
            )
    return PING(e, i)
