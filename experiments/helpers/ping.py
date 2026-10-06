"""Author a reciprocal E/I circuit; experiment recipes own its parameters."""

from __future__ import annotations

from dataclasses import dataclass

from snnlab import lang


@dataclass(frozen=True)
class PINGCircuit:
    """Population and projection handles for experiment-specific observables."""

    e: lang.Population
    i: lang.Population
    input_to_e: lang.Projection
    e_to_i: lang.Projection
    i_to_e: lang.Projection


def build_ping(
    net: lang.Network,
    *,
    source: lang.Signal,
    n_e: int,
    n_i: int,
    neuron_e,
    neuron_i,
    ampa,
    gaba,
    input_weight,
    ei_weight,
    ie_weight,
    recurrent_delay: lang.Quantity,
    initialization_scaling: str,
) -> PINGCircuit:
    """Add E/I populations, E-only input, and nonnegative reciprocal weights.

    Names and declaration order are stable: E, I, input_to_E, E_to_I, I_to_E.
    Weights may be initializer specs or parameters belonging to ``net``;
    reused parameters retain their existing constraints.
    The caller owns inputs, readouts, observables, compilation and execution.
    No same-population recurrence is added.
    """
    e = net.population("E", size=n_e, neuron=neuron_e)
    i = net.population("I", size=n_i, neuron=neuron_i)
    external = net.connect(
        source,
        e.excitatory,
        name="input_to_E",
        synapse=ampa,
        weight=input_weight,
        constraint=lang.NonNegative(),
        initialization_scaling=initialization_scaling,
    )
    ei = net.connect(
        e.spikes,
        i.excitatory,
        name="E_to_I",
        synapse=ampa,
        weight=ei_weight,
        constraint=lang.NonNegative(),
        connection="recurrent",
        delay=recurrent_delay,
        initialization_scaling=initialization_scaling,
    )
    ie = net.connect(
        i.spikes,
        e.inhibitory,
        name="I_to_E",
        synapse=gaba,
        weight=ie_weight,
        constraint=lang.NonNegative(),
        connection="recurrent",
        delay=recurrent_delay,
        initialization_scaling=initialization_scaling,
    )
    return PINGCircuit(e, i, external, ei, ie)
