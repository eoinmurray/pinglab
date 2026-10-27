"""SNNLab extensions for bounded continuous-rate E/I dynamics.

Rates are dimensionless fractions; synapses convert weighted activity to nA.
Callbacks are used by the authored native graph and its differentiable executor.
"""

import torch
from snnlab import extensions as ext


def rate_initial_state(ctx):
    zeros = torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
    return dict(
        voltage=zeros,
        rate=zeros.clone() + ctx.config["initial_rate"],
        bias=zeros.clone()
        + torch.as_tensor(ctx.config["bias_na"], dtype=ctx.dtype, device=ctx.device),
    )


def rate_step(ctx):
    previous = ctx.state["rate"]
    net_current = ctx.excitatory - ctx.inhibitory - ctx.state["bias"]
    target = torch.sigmoid(net_current / ctx.config["sigmoid_scale_na"])
    updated = previous + ctx.dt_ms / ctx.config["tau_ms"] * (target - previous)
    return {**ctx.state, "rate": updated}, torch.zeros_like(updated)


def current_step(ctx):
    return {"value": ctx.drive * ctx.config["gain_na"]}


def fixed_values(shape, config, **kwargs):
    values = torch.tensor(config["values"], **kwargs)
    if tuple(values.shape) != tuple(shape):
        raise ValueError(f"fixed initializer: {values.shape} != {shape}")
    return values


def bounded_weights(value, config):
    return value.clamp(0.0, config["upper"])


def value_readout(sources, parameters, config):
    rates = sources[0][-config["tail_steps"] :].mean(dim=0)
    return rates @ parameters["value_weight"]


def register_extensions():
    present = {definition.name for definition in ext.definitions()}
    definitions = [
        (
            "ei_lifecycle.rate/v1",
            ext.register_neuron,
            rate_step,
            dict(
                initialize=rate_initial_state,
                input_unit="nA",
                state_units={"rate": "1"},
            ),
        ),
        (
            "ei_lifecycle.current/v1",
            ext.register_synapse,
            current_step,
            dict(input_unit="1", output_unit="nA"),
        ),
        ("ei_lifecycle.fixed/v1", ext.register_initializer, fixed_values, {}),
        ("ei_lifecycle.bounds/v1", ext.register_constraint, bounded_weights, {}),
        ("ei_lifecycle.value/v1", ext.register_operation, value_readout, {}),
    ]
    for name, register, function, kwargs in definitions:
        if name not in present:
            register(name, function, **kwargs)
