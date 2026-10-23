"""Author an exp145-equivalent circuit with a scheduled external-I drive reduction.

configuration() resolves the three measurement windows and the conductance-weight
schedule. author_network() compiles the six-projection COBA-LIF graph with spike
observations; neither function executes the protocol.
"""

import math
from dataclasses import asdict

import numpy as np
from snnlab import lang
from snnlab.sim import Performance


def configuration(
    *,
    seed=42,
    device="cpu",
    n_e=4000,
    n_i=1000,
    n_external=4000,
    baseline_ms=500.0,
    ramp_ms=500.0,
    hold_ms=500.0,
    final_i_gain=0.0,
):
    if any(type(n) is not int or n <= 0 for n in (n_e, n_i, n_external)):
        raise ValueError("population sizes must be positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name, value in (
        ("baseline_ms", baseline_ms),
        ("ramp_ms", ramp_ms),
        ("hold_ms", hold_ms),
    ):
        if (
            not math.isfinite(value)
            or value <= 0
            or not math.isclose(value, round(value))
        ):
            raise ValueError(
                f"{name} must be a positive integer number of milliseconds"
            )
    if not math.isfinite(final_i_gain) or not 0 <= final_i_gain <= 1:
        raise ValueError("final_i_gain must be finite and between zero and one")
    cfg = dict(
        n_e=n_e,
        n_i=n_i,
        n_external=n_external,
        seed=seed,
        device=device,
        torch_threads=1,
        dt_ms=0.05,
        duration_ms=50.0 + baseline_ms + ramp_ms + hold_ms,
        baseline_ms=baseline_ms,
        ramp_ms=ramp_ms,
        hold_ms=hold_ms,
        execution_block_ms=50.0,
        control=dict(
            parameter="X_to_I.weight",
            baseline_gain=1.0,
            final_gain=final_i_gain,
            ramp_start_ms=50.0 + baseline_ms,
            ramp_end_ms=50.0 + baseline_ms + ramp_ms,
            update_ms=1.0,
        ),
        burn_in_ms=50.0,
        external_rate_hz=2.5,
        external_seed_offset=10000,
        connection_probability=0.2,
        weight_relative_sd=0.5,
        capacitance_nf=0.25,
        leak_us=0.0167,
        resting_mv=-70.0,
        threshold_mv=-50.0,
        reset_mv=-60.0,
        excitatory_reversal_mv=0.0,
        inhibitory_reversal_mv=-80.0,
        refractory_e_ms=2.0,
        refractory_i_ms=1.0,
        tau_synapse_ms=5.0,
        reference_area_time_ms=1.0,
        excitatory_delay_ms=1.0,
        inhibitory_delay_ms=0.5,
        reference_coefficients_ns=dict(
            E_to_E=2.4,
            E_to_I=4.8,
            I_to_E=40.0,
            I_to_I=40.0,
            X_to_E=5.4,
            X_to_I=5.4,
        ),
        recording_block_steps=1000,
        analysis=dict(
            population_bin_ms=0.5,
            peak_separation_ms=10.0,
            peak_prominence_fraction=0.2,
        ),
        performance=asdict(
            Performance(compile_mode=None, precision="highest", cuda_graphs="off")
        ),
    )
    # Reference coefficients multiply kernels of area 1 ms, not unit-peak jumps.
    cfg["jump_means_us"] = {
        name: value * 1e-3 * cfg["reference_area_time_ms"] / cfg["tau_synapse_ms"]
        for name, value in cfg["reference_coefficients_ns"].items()
    }
    # Native Bernoulli sparsification divides retained weights by p. Compensate
    # its parent distribution so physical jumps do not grow as connectivity falls.
    cfg["initializer_parent_means_us"] = {
        name: jump * cfg["connection_probability"]
        for name, jump in cfg["jump_means_us"].items()
    }
    return cfg


def author_network(cfg):
    net = lang.Network("exp148", dt=cfg["dt_ms"] * lang.ms)
    external = net.input(
        "X",
        shape=("time", "batch", cfg["n_external"]),
        signal_type="spikes",
        unit="spike",
    )
    populations = {}
    for name in ("E", "I"):
        populations[name] = net.population(
            name,
            size=cfg[f"n_{name.lower()}"],
            neuron=lang.COBA_LIF(
                tau_mem=cfg["capacitance_nf"] / cfg["leak_us"] * lang.ms,
                capacitance_nf=cfg["capacitance_nf"],
                leak_us=cfg["leak_us"],
                resting_mv=cfg["resting_mv"],
                threshold_mv=cfg["threshold_mv"],
                reset_mv=cfg["reset_mv"],
                initial_voltage_mv=cfg["resting_mv"],
                excitatory_reversal_mv=cfg["excitatory_reversal_mv"],
                inhibitory_reversal_mv=cfg["inhibitory_reversal_mv"],
                refractory=cfg[f"refractory_{name.lower()}_ms"] * lang.ms,
            ),
        )
    for name, mean in cfg["initializer_parent_means_us"].items():
        source, target = name.split("_to_")
        inhibitory = source == "I"
        synapse = lang.GABA if inhibitory else lang.AMPA
        net.connect(
            external if source == "X" else populations[source].spikes,
            populations[target].inhibitory
            if inhibitory
            else populations[target].excitatory,
            name=name,
            synapse=synapse(tau=cfg["tau_synapse_ms"] * lang.ms),
            weight=lang.LowerClampedNormal(
                mean,
                mean * cfg["weight_relative_sd"],
                initial_zero_fraction=1.0 - cfg["connection_probability"],
            ),
            constraint=lang.NonNegative(),
            initialization_scaling="direct",
            connection="feedforward" if source == "X" else "recurrent",
            delay=cfg["inhibitory_delay_ms" if inhibitory else "excitatory_delay_ms"]
            * lang.ms,
            allow_autapses=False,
        )
    for name in ("E", "I"):
        net.output(name + "_spikes", populations[name].spikes)
    return lang.compile(net)


def drive_gain(time_ms, cfg):
    """Return the target X→I gain; execution holds it for each update interval."""
    schedule = cfg["control"]
    fraction = np.clip(
        (np.asarray(time_ms) - schedule["ramp_start_ms"])
        / (schedule["ramp_end_ms"] - schedule["ramp_start_ms"]),
        0,
        1,
    )
    return schedule["baseline_gain"] + fraction * (
        schedule["final_gain"] - schedule["baseline_gain"]
    )


def windows(cfg):
    """Return absolute simulation intervals, excluding the initial burn-in."""
    burn = cfg["burn_in_ms"]
    start = cfg["control"]["ramp_start_ms"]
    end = cfg["control"]["ramp_end_ms"]
    return (
        ("baseline", burn, start),
        ("ramp", start, end),
        ("hold", end, cfg["duration_ms"]),
    )


def execution_bounds(cfg):
    """Split constant windows into blocks and the ramp into one-ms updates."""
    dt = cfg["dt_ms"]
    ramp_start = round(cfg["control"]["ramp_start_ms"] / dt)
    ramp_end = round(cfg["control"]["ramp_end_ms"] / dt)
    steps = round(cfg["duration_ms"] / dt)
    block = round(cfg["execution_block_ms"] / dt)
    update = round(cfg["control"]["update_ms"] / dt)
    return [
        *range(0, ramp_start, block),
        *range(ramp_start, ramp_end, update),
        *range(ramp_end, steps, block),
        steps,
    ]
