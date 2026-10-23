"""Implement intrinsic ionic dynamics and separate graded inhibitory synapses.

Both callbacks read beginning-of-step voltage and advance gates before using
their conductances. A per-neuron integer clock implements the fixed current
protocol and persists through continuation; voltage is never reset on spikes.
"""

import torch


def neuron_initial_state(ctx):
    voltage = torch.full(
        ctx.shape, ctx.config["initial_voltage_mv"], device=ctx.device, dtype=ctx.dtype
    )
    return dict(
        voltage=voltage,
        z_na=voltage.clone(),
        z_k=voltage.clone(),
        current=torch.zeros_like(voltage),
        step_index=torch.zeros(ctx.shape, device=ctx.device, dtype=torch.int64),
    )


def neuron_step(ctx):
    cfg, state, dt = ctx.config, ctx.state, ctx.dt_ms
    voltage = state["voltage"]
    z_na = state["z_na"] + dt / cfg["tau_na_ms"] * (voltage - state["z_na"])
    z_k = state["z_k"] + dt / cfg["tau_k_ms"] * (voltage - state["z_k"])
    g_na = cfg["g_na_us"] * torch.sigmoid(
        cfg["gamma_na_per_mv"] * (z_na - cfg["bias_na_mv"])
    )
    g_k = cfg["g_k_us"] * torch.sigmoid(
        cfg["gamma_k_per_mv"] * (z_k - cfg["bias_k_mv"])
    )
    # Use integer intervals so current changes exactly at 5 and 10 ms.
    index = state["step_index"]
    current = torch.where(
        (index >= cfg["kick_start_step"]) & (index < cfg["kick_stop_step"]),
        torch.full_like(voltage, cfg["kick_na"]),
        torch.where(
            index >= cfg["drive_start_step"],
            torch.full_like(voltage, cfg["drive_na"]),
            torch.zeros_like(voltage),
        ),
    )
    outward = (
        cfg["leak_us"] * (voltage - cfg["leak_reversal_mv"])
        + g_na * (voltage - cfg["e_na_mv"])
        + g_k * (voltage - cfg["e_k_mv"])
        + ctx.inhibitory * (voltage - cfg["synapse_reversal_mv"])
    )
    updated = voltage + dt / cfg["capacitance_nf"] * (current - outward)
    spikes = (
        (voltage < cfg["spike_threshold_mv"]) & (updated >= cfg["spike_threshold_mv"])
    ).to(voltage.dtype)
    return {
        **state,
        "voltage": updated,
        "z_na": z_na,
        "z_k": z_k,
        "current": current,
        "step_index": index + 1,
    }, spikes


def synapse_initial_state(ctx):
    gate = torch.full(
        ctx.shape, ctx.config["initial_gate_mv"], device=ctx.device, dtype=ctx.dtype
    )
    value = ctx.config["g_max_us"] * torch.sigmoid(
        ctx.config["gamma_per_mv"] * (gate - ctx.config["bias_mv"])
    )
    return dict(gate=gate, value=value)


def synapse_step(ctx):
    gate = ctx.state["gate"] + ctx.dt_ms / ctx.config["tau_ms"] * (
        ctx.drive - ctx.state["gate"]
    )
    value = ctx.config["g_max_us"] * torch.sigmoid(
        ctx.config["gamma_per_mv"] * (gate - ctx.config["bias_mv"])
    )
    return dict(gate=gate, value=value)
