"""Evolve the proposal's intrinsic ionic conductances with native callbacks.

All gates read the previous membrane voltage, then the new conductances drive
the forward-Euler membrane update. Crossing events are observations only;
voltage is never reset and no refractory period is imposed.
"""

import torch


def conductances(voltage_na, voltage_k, cfg):
    return (
        cfg["g_na_us"]
        * torch.sigmoid(cfg["gamma_na_per_mv"] * (voltage_na - cfg["bias_na_mv"])),
        cfg["g_k_us"]
        * torch.sigmoid(cfg["gamma_k_per_mv"] * (voltage_k - cfg["bias_k_mv"])),
    )


def initial_state(ctx):
    voltage = torch.full(
        ctx.shape,
        ctx.config["initial_voltage_mv"],
        dtype=ctx.dtype,
        device=ctx.device,
    )
    g_na, g_k = conductances(voltage, voltage, ctx.config)
    return dict(
        voltage=voltage,
        z_na=voltage.clone(),
        z_k=voltage.clone(),
        g_na=g_na,
        g_k=g_k,
    )


def step(ctx):
    cfg, state, dt = ctx.config, ctx.state, ctx.dt_ms
    voltage = state["voltage"]
    z_na = state["z_na"] + dt / cfg["tau_na_ms"] * (voltage - state["z_na"])
    z_k = state["z_k"] + dt / cfg["tau_k_ms"] * (voltage - state["z_k"])
    g_na, g_k = conductances(z_na, z_k, cfg)
    ionic = (
        cfg["leak_us"] * (voltage - cfg["leak_reversal_mv"])
        + g_na * (voltage - cfg["e_na_mv"])
        + g_k * (voltage - cfg["e_k_mv"])
    )
    current = ctx.excitatory - ctx.inhibitory
    updated = voltage + dt * (current - ionic) / cfg["capacitance_nf"]
    threshold = cfg["spike_threshold_mv"]
    spikes = ((voltage < threshold) & (updated >= threshold)).to(voltage.dtype)
    return {
        **state,
        "voltage": updated,
        "z_na": z_na,
        "z_k": z_k,
        "g_na": g_na,
        "g_k": g_k,
    }, spikes


def inject_current(ctx):
    """Pass the projected signed current unchanged, without decay or clipping."""
    return {"value": ctx.drive.to(ctx.state["value"].dtype)}
