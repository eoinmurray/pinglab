"""Hodgkin–Huxley membrane dynamics for the joint controller.

Adapted from snnlab/examples/hodgkin-huxley/example.py. Retains its ionic
channels, steady-state gates, RK4 and upward zero-crossing spike detection.
AMPA/GABA conductances drive reversal-potential currents at every RK4 stage;
both populations use physical milliseconds without clock scaling.
The registered SNNLab callback uses PyTorch tensors for RK4 updates with
steps <= 0.025 ms, following the SNNLab custom-neuron example.
"""

import math
from functools import lru_cache

import torch
from snnlab import extensions

MAX_STEP_MS = 0.025
DEFINITION = "pinglab.exp152.hodgkin_huxley/v3"
PARAMETERS = (
    "capacitance_nf",
    "g_na_us",
    "g_k_us",
    "g_leak_us",
    "e_na_mv",
    "e_k_mv",
    "e_leak_mv",
    "excitatory_reversal_mv",
    "inhibitory_reversal_mv",
)
DEFAULT_PARAMETERS = (0.01, 1.2, 0.36, 0.003, 50.0, -77.0, -54.387, 0.0, -80.0)


def _vtrap(x):
    small = x.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 + x / 2 + x.square() / 12, safe / -torch.expm1(-safe))


def rates(voltage):
    return (
        (_vtrap((voltage + 40) / 10), 4 * torch.exp(-(voltage + 65) / 18)),
        (0.07 * torch.exp(-(voltage + 65) / 20), torch.sigmoid((voltage + 35) / 10)),
        (0.1 * _vtrap((voltage + 55) / 10), 0.125 * torch.exp(-(voltage + 65) / 80)),
    )


def initial_state(context):
    voltage = torch.full(
        context.shape,
        getattr(context, "config", {}).get("initial_voltage_mv", -65.0),
        device=context.device,
        dtype=context.dtype,
    )
    state = {"voltage": voltage}
    for gate, (alpha, beta) in zip(("m", "h", "n"), rates(voltage)):
        state[gate] = alpha / (alpha + beta)
    return state


def _derivative(values, excitatory, inhibitory, parameters):
    voltage, m, h, n = values
    capacitance, g_na, g_k, g_leak, e_na, e_k, e_leak, e_exc, e_inh = parameters
    ionic = (
        g_na * m**3 * h * (voltage - e_na)
        + g_k * n**4 * (voltage - e_k)
        + g_leak * (voltage - e_leak)
    )
    gates = tuple(
        alpha * (1 - gate) - beta * gate
        for gate, (alpha, beta) in zip((m, h, n), rates(voltage))
    )
    current = excitatory * (e_exc - voltage) + inhibitory * (e_inh - voltage)
    return ((current - ionic) / capacitance, *gates)


def step(context):
    """Advance membrane/gates and report any upward zero crossing, without reset."""
    keys = ("voltage", "m", "h", "n")
    values = tuple(context.state[key] for key in keys)
    excitatory = context.excitatory.to(values[0].dtype)
    inhibitory = context.inhibitory.to(values[0].dtype)
    parameters = tuple(
        context.config.get(key, value)
        for key, value in zip(PARAMETERS, DEFAULT_PARAMETERS)
    )
    substeps = max(1, math.ceil(context.dt_ms / MAX_STEP_MS))
    dt = context.dt_ms / substeps
    crossed = torch.zeros_like(values[0], dtype=torch.bool)
    for _ in range(substeps):
        k1 = _derivative(values, excitatory, inhibitory, parameters)
        k2 = _derivative(
            tuple(y + dt / 2 * k for y, k in zip(values, k1)),
            excitatory,
            inhibitory,
            parameters,
        )
        k3 = _derivative(
            tuple(y + dt / 2 * k for y, k in zip(values, k2)),
            excitatory,
            inhibitory,
            parameters,
        )
        k4 = _derivative(
            tuple(y + dt * k for y, k in zip(values, k3)),
            excitatory,
            inhibitory,
            parameters,
        )
        updated = tuple(
            y + dt / 6 * (a + 2 * b + 2 * c + d)
            for y, a, b, c, d in zip(values, k1, k2, k3, k4)
        )
        crossed = crossed | ((values[0] < 0) & (updated[0] >= 0))
        values = updated
    return {**context.state, **dict(zip(keys, values))}, crossed.to(values[0].dtype)


@lru_cache(maxsize=1)
def register():
    extensions.register_neuron(
        DEFINITION,
        step,
        initialize=initial_state,
        input_unit="uS",
        state_units={"m": "1", "h": "1", "n": "1"},
        capture_safe=True,
    )
