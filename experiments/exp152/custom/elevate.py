"""SNNLab tensor port of the supplied six-state Elevate neuron.

Filtered voltages precede sigmoid gating and frozen-conductance exponential
membrane updates, exactly as supplied. Numerical substeps resolve fast spikes;
only one spike flag is exported per external communication step. The KCa-labelled
current is driven by filtered voltage, without a calcium concentration state.
"""

import math
from functools import lru_cache

import torch
from snnlab import extensions

DEFINITION = "pinglab.exp152.elevate/v1"
KEYS = ("voltage", "vna", "vh", "vk", "vca", "vkca")
PARAMETERS = (
    "capacitance_nf",
    "g_na_us",
    "g_k_us",
    "g_ca_us",
    "g_kca_us",
    "g_leak_us",
    "e_na_mv",
    "e_k_mv",
    "e_ca_mv",
    "e_leak_mv",
    "excitatory_reversal_mv",
    "inhibitory_reversal_mv",
    "current_na",
)
TIME_CONSTANTS = (
    "tau_na_activation_ms",
    "tau_na_inactivation_ms",
    "tau_k_ms",
    "tau_ca_ms",
    "tau_kca_ms",
)


def initialize(context):
    return {
        key: torch.full(
            context.shape,
            context.config.get("initial_voltage_mv", -60.0),
            device=context.device,
            dtype=context.dtype,
        )
        for key in KEYS
    }


def integrate(
    state: torch.Tensor,
    exc: torch.Tensor,
    inh: torch.Tensor,
    duration: float,
    max_step: float,
    parameters: list[float],
    time_constants: list[float],
):
    count = int(math.ceil(duration / max_step))
    dt = duration / count
    (
        capacitance,
        g_na,
        g_k,
        g_ca,
        g_kca,
        g_leak,
        e_na,
        e_k,
        e_ca,
        e_leak,
        e_exc,
        e_inh,
        current,
    ) = parameters
    tau = torch.tensor(time_constants, dtype=state.dtype, device=state.device).reshape(
        5, 1, 1
    )
    offsets = torch.tensor(
        [25.0, 40.0, 15.0, 45.0, 30.0], dtype=state.dtype, device=state.device
    ).reshape(5, 1, 1)
    # Integer ratios avoid reduced-precision folding of decimal tensor constants.
    slopes = (
        torch.tensor(
            [2.0, -1.0, 1.0, 2.0, 1.0], dtype=state.dtype, device=state.device
        ).reshape(5, 1, 1)
        / 10.0
    )
    decay = torch.exp(-dt / tau)
    v = state[0]
    filtered = state[1:]
    crossed = torch.zeros_like(v, dtype=torch.bool)
    for _ in range(count):
        filtered = v + (filtered - v) * decay
        gates = torch.sigmoid((filtered + offsets) * slopes)
        gna = g_na * gates[0] * gates[1]
        gk = g_k * gates[2]
        gc = g_ca * gates[3]
        gkc = g_kca * gates[4]
        total = gna + gk + gc + gkc + g_leak + exc + inh
        vinf = (
            gna * e_na
            + (gk + gkc) * e_k
            + gc * e_ca
            + g_leak * e_leak
            + exc * e_exc
            + inh * e_inh
            + current
        ) / total
        updated = vinf + (v - vinf) * torch.exp(-dt * total / capacitance)
        crossed = crossed | ((updated > 0) & (v <= 0))
        v = updated
    return torch.cat((v.unsqueeze(0), filtered)), crossed.to(v.dtype)


# Eager execution avoids Python dispatch inside the intrinsic loop. During
# Dynamo tracing use the same Python function so its substeps can fuse together.
integrate_eager = torch.jit.script(integrate)


def step(context):
    cfg = context.config
    update = integrate if torch.compiler.is_compiling() else integrate_eager
    state, spikes = update(
        torch.stack([context.state[key] for key in KEYS]),
        context.excitatory,
        context.inhibitory,
        float(context.dt_ms),
        float(cfg["max_step_ms"]),
        [float(cfg[key]) for key in PARAMETERS],
        [float(cfg[key]) for key in TIME_CONSTANTS],
    )
    return {**context.state, **dict(zip(KEYS, state.unbind(0), strict=True))}, spikes


def direct(context):
    return {"value": context.drive.to(context.state["value"].dtype)}


@lru_cache(maxsize=1)
def register():
    extensions.register_neuron(
        DEFINITION,
        step,
        initialize=initialize,
        input_unit="uS",
        state_units=dict.fromkeys(KEYS[1:], "mV"),
        capture_safe=True,
    )
    extensions.register_synapse(
        "pinglab.exp152.direct_conductance/v1",
        direct,
        output_unit="uS",
        capture_safe=True,
    )
