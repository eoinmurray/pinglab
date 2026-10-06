"""Small native-graph operations for explicitly described single-layer banks.

Recipes supply biophysics, observables and requests. Historical tensors remain
read-only [source, target] data; this module never constructs a CLI simulator.
"""

from __future__ import annotations

import torch
from experiments.helpers.ping import build_ping
from pingstore.contracts import PingstoreError
from snnlab import lang
from snnlab.sim.execution import GraphExecutor, plan_graph, resolve_device
from snnlab.sim.timing import refractory_steps

PARAMETERS = {
    "input_to_E.weight": "W_ff.0",
    "readout_projection.weight": "W_ff.1",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
}


def author_network(name, training, biophysics, refractory, *, observables=()):
    dt = training["dt"]
    net = lang.Network(name, dt=dt * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", training["n_in"]),
        signal_type="spikes",
        unit="spike",
    )
    neurons = {}
    for label in ("e", "i"):
        capacitance, leak = (
            biophysics[f"capacitance_{label}_nf"],
            biophysics[f"leak_{label}_us"],
        )
        neurons[label] = lang.COBA_LIF(
            tau_mem=capacitance / leak * lang.ms,
            capacitance_nf=capacitance,
            leak_us=leak,
            resting_mv=biophysics["resting_mv"],
            threshold_mv=biophysics["threshold_mv"],
            reset_mv=biophysics["reset_mv"],
            initial_voltage_mv=biophysics["resting_mv"],
            refractory_steps=refractory_steps(
                refractory[f"refractory_{label}_ms"],
                dt,
                policy=refractory["refractory_policy"],
            ),
            voltage_grad_dampen=training["v_grad_dampen"],
        )
    circuit = build_ping(
        net,
        source=drive,
        n_e=training["n_hidden"],
        n_i=training["n_inh"],
        neuron_e=neurons["e"],
        neuron_i=neurons["i"],
        ampa=lang.AMPA(tau=training["tau_ampa_ms"] * lang.ms),
        gaba=lang.GABA(tau=training["tau_gaba_ms"] * lang.ms),
        input_weight=lang.Constant(0),
        ei_weight=lang.Constant(0),
        ie_weight=lang.Constant(0),
        recurrent_delay=dt * lang.ms,
        initialization_scaling="direct",
    )
    output = net.population(
        "readout",
        size=training["n_out"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=biophysics["readout_tau_ms"] * lang.ms,
            soft_reset_threshold=biophysics["readout_threshold"],
            initial_voltage=0,
            surrogate_slope=training["surrogate_slope"],
        ),
    )
    net.connect(
        circuit.e.spikes,
        output.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=biophysics["readout_tau_ms"] * lang.ms),
        weight=lang.Constant(0),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    net.output(
        "class_scores",
        lang.ops.reduce(
            output.pre_reset_voltage
            if training.get("readout_mode", "mem-mean") == "mem-mean"
            else output.spikes,
            operation="mean"
            if training.get("readout_mode", "mem-mean") == "mem-mean"
            else "sum",
            over="time",
            name="readout_reduction",
        ),
    )
    for label, pop in (("e", circuit.e), ("i", circuit.i)):
        net.output(
            f"spk_{label}_count",
            lang.ops.reduce(
                pop.spikes, operation="sum", over="time", name=f"{label}_counts"
            ),
        )
        if "spikes" in observables:
            net.expose(pop.spikes, name=f"spk_{label}")
    if "traces" in observables:
        for key, value in {
            "v_e": circuit.e.voltage,
            "v_i": circuit.i.voltage,
            "ge_e": circuit.input_to_e.conductance,
            "ge_i": circuit.e_to_i.conductance,
            "gi_e": circuit.i_to_e.conductance,
        }.items():
            net.expose(value, name=key)
    return lang.compile(net, target="tools/snnsim")


def checkpoint_tensors(path, training):
    n_in, n_e, n_i, n_out = (
        training[k] for k in ("n_in", "n_hidden", "n_inh", "n_out")
    )
    for key in (
        "adaptive_threshold",
        "train_leak",
        "signed_readout",
        "readout_bias",
        "state_clamp",
        "trainable_w_ee",
        "trainable_w_ii",
    ):
        if training.get(key, False) is not False:
            raise PingstoreError(f"unsupported checkpoint dynamics: {key}")
    if training.get("dales_law") is not True:
        raise PingstoreError("requires Dale-constrained bank weights")
    if training.get("hidden_sizes", [n_e]) != [n_e] or training["readout_mode"] not in (
        "mem-mean",
        "spike-count",
    ):
        raise PingstoreError(
            "requires one E/I layer with mean pre-reset voltage readout"
        )
    shapes = {
        "W_ff.0": (n_in, n_e),
        "W_ff.1": (n_e, n_out),
        "W_ee.1": (n_e, n_e),
        "W_ei.1": (n_e, n_i),
        "W_ie.1": (n_i, n_e),
        "W_ii.1": (n_i, n_i),
    }
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise PingstoreError("checkpoint must contain exactly six matrices")
    for key, shape in shapes.items():
        value = state[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(value.shape) != shape
            or not torch.isfinite(value).all()
        ):
            raise PingstoreError(f"invalid checkpoint tensor: {key}")
        if key in ("W_ee.1", "W_ii.1") and torch.count_nonzero(value):
            raise PingstoreError(f"unsupported same-population recurrence: {key}")
        if key in ("W_ei.1", "W_ie.1") and (value < 0).any():
            raise PingstoreError(f"negative recurrent weight: {key}")
    return state


def initial_draws(training, *, ei_strength=None):
    """Ordered CPU weight draws, including unused zero recurrence.

    Return the generator cursor needed for CPU snapshots and the fresh loop
    needed for transfer probes. No model is rebuilt to recover these draws.
    """
    generator = torch.Generator().manual_seed(training["seed"])
    n_in, n_e, n_i, n_out = (
        training[k] for k in ("n_in", "n_hidden", "n_inh", "n_out")
    )

    def normal(shape, mean, sd, zero=0, scale=True):
        value = (
            torch.randn(*shape, generator=generator).mul_(sd).add_(mean).clamp_(min=0)
        )
        if zero:
            value *= (torch.rand(*shape, generator=generator) > zero).float()
            value /= 1 - zero
        return value / shape[0] if scale else value

    mean, sd = training["w_in"]
    external = normal((n_in, n_e), mean, sd, training["w_in_initial_zero_fraction"])
    if training.get("readout_w_init_mean") is not None:
        readout = normal(
            (n_e, n_out),
            training["readout_w_init_mean"],
            training["readout_w_init_std"],
            scale=False,
        )
    else:
        readout = normal((n_e, n_out), 5.1, 3.8) * training.get(
            "readout_w_out_scale", 225.0
        )
    normal((n_e, n_e), 0, 0)
    strength = training["ei_strength"] if ei_strength is None else ei_strength
    ei = normal((n_e, n_i), strength, strength * 0.1)
    ie_strength = strength * training["ei_ratio"]
    ie = normal((n_i, n_e), ie_strength, ie_strength * 0.1)
    normal((n_i, n_i), 0, 0)
    return generator, {
        "input_to_E.weight": external,
        "readout_projection.weight": readout,
        "E_to_I.weight": ei,
        "I_to_E.weight": ie,
    }


def bind_model(bundle, training, state, *, scale=1.0, fresh_loop=None, device=None):
    model = GraphExecutor(
        plan_graph(bundle.graph),
        seed=training["seed"],
        surrogate_slope=training["surrogate_slope"],
    )
    parameters = model.parameter_map()
    if set(parameters) != set(PARAMETERS):
        raise PingstoreError("graph parameter roles differ from checkpoint contract")
    with torch.no_grad():
        for name, key in PARAMETERS.items():
            value = state[key].clamp(min=0) if key.startswith("W_ff") else state[key]
            if fresh_loop is not None and name in ("E_to_I.weight", "I_to_E.weight"):
                value = fresh_loop[name]
            if name == "input_to_E.weight":
                value = value * scale
            if parameters[name].shape != value.shape:
                raise PingstoreError(f"graph/checkpoint shape mismatch: {name}")
            parameters[name].copy_(value)
    return model.to(device or resolve_device("auto")).eval()
