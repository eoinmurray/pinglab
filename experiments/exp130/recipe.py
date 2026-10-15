"""Theta-modulated private-afferent protocol; no execution on import."""

from __future__ import annotations

import numpy as np
from snnlab import lang as snn  # noqa: TID251

SLUG = "exp130"
DT_MS, DURATION_MS, SEED = 0.1, 1000.0, 8
BURN_IN_MS = 0.0
N_E, N_I = 1600, 400
CAPACITANCE_NF, LEAK_US = 0.15, 0.01
BASELINE_HZ = 2.0
THETA_HZ, MODULATION_HZ = 5.0, 1.8
CONDITIONS = ("theta", "constant")
RECURRENT_SCALE, INHIBITORY_SCALE = 0.08, 5.0
VIEW_START_MS, VIEW_END_MS = BURN_IN_MS, DURATION_MS


def configuration() -> dict:
    return {
        "schema": "exp130.recipe/v1",
        "conditions": list(CONDITIONS),
        "theta_hz": THETA_HZ,
        "modulation_hz": MODULATION_HZ,
        "modulation_onset_ms": BURN_IN_MS,
        "analysis_start_ms": 200.0,
        "analysis_end_ms": 800.0,
        "phase_convention": "sine; phase zero is the rising mean crossing at modulation onset",
        "pairing": "same seed, initial state and graph; same private RNG streams; Poisson samples differ after onset",
        "dt_ms": DT_MS,
        "t_ms": DURATION_MS,
        "burn_in_ms": BURN_IN_MS,
        "seed": SEED,
        "n_e": N_E,
        "n_i": N_I,
        "connection_probability": 0.1,
        "external_afferents_per_neuron": 400,
        "baseline_e_hz": BASELINE_HZ,
        "baseline_i_hz": BASELINE_HZ,
        "capacitance_nf": CAPACITANCE_NF,
        "leak_us": LEAK_US,
        "external_weight_us": 0.004,
        "recurrent_e_weight_us": 0.0125 * RECURRENT_SCALE,
        "recurrent_i_weight_us": 0.00835 * RECURRENT_SCALE * INHIBITORY_SCALE,
        "tau_ampa_ms": 1.5,
        "tau_gaba_ms": 7.5,
        "delay_ms": 1.5,
        "refractory_e_ms": 3.0,
        "refractory_i_ms": 1.5,
        "view_start_ms": VIEW_START_MS,
        "view_end_ms": VIEW_END_MS,
        "initial_voltage_mv": -65.0,
        "external_representation": "independent Poisson counts per target; exact superposition of 400 sources; diagonal 4 nS projection",
        "recurrent_topology": "independent Bernoulli edges between distinct neurons; autapses disabled",
        "weight_initialization": "graph fan-in/sparsity normalization compensated in initializer; external matrices replaced by diagonal physical weights",
        "recording_policy": "all output spikes and afferent counts at dt; population mean voltages and projection conductances at dt",
        "reference": "https://doi.org/10.1371/journal.pcbi.1009416",
    }


def condition_configuration(cfg, condition):
    if condition not in CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    return {**cfg, "condition": condition}


def source_rates(times_ms, cfg):
    """Per-afferent rate; both targets share only the deterministic envelope."""
    times_ms = np.asarray(times_ms)
    condition = cfg["condition"]
    if condition not in CONDITIONS:
        raise ValueError(condition)
    phase = 2 * np.pi * cfg["theta_hz"] * (times_ms - cfg["modulation_onset_ms"]) / 1000
    oscillation = (
        cfg["modulation_hz"] * np.sin(phase)
        if condition == "theta"
        else np.zeros_like(times_ms)
    )
    oscillation = np.where(times_ms >= cfg["modulation_onset_ms"], oscillation, 0)
    return tuple(cfg[f"baseline_{pop}_hz"] + oscillation for pop in ("e", "i"))


def afferent_counts(cfg):
    """Independent across cells and populations; counts greater than one survive."""
    times = np.arange(round(cfg["t_ms"] / cfg["dt_ms"])) * cfg["dt_ms"]
    rngs = [
        np.random.default_rng(s) for s in np.random.SeedSequence(cfg["seed"]).spawn(2)
    ]
    result = {}
    for pop, rates, rng in zip(("e", "i"), source_rates(times, cfg), rngs):
        lam = (
            rates[:, None] * cfg["external_afferents_per_neuron"] * cfg["dt_ms"] / 1000
        )
        counts = rng.poisson(lam, size=(len(times), cfg[f"n_{pop}"]))
        if counts.max() > 255:
            raise ValueError("afferent counts exceed recording capacity")
        result[f"private_{pop}"] = counts.astype(np.uint8)
    return result


def author_network(cfg=None):
    cfg = configuration() if cfg is None else cfg
    net = snn.Network("exp130_private_afferents", dt=cfg["dt_ms"] * snn.ms)
    pops = {}
    for name in ("E", "I"):
        cap, leak = cfg["capacitance_nf"], cfg["leak_us"]
        tau = cap / leak
        pops[name] = net.population(
            name,
            size=cfg[f"n_{name.lower()}"],
            neuron=snn.COBA_LIF(
                tau_mem=tau * snn.ms,
                capacitance_nf=cap,
                leak_us=leak,
                resting_mv=-65.0,
                threshold_mv=-50.0,
                reset_mv=-65.0,
                refractory=cfg[f"refractory_{name.lower()}_ms"] * snn.ms,
                voltage_grad_dampen=80.0,
                initial_voltage_mv=cfg["initial_voltage_mv"],
            ),
        )
    for name, pop in pops.items():
        size = cfg[f"n_{name.lower()}"]
        source = net.input(
            f"private_{name.lower()}",
            shape=("time", "batch", size),
            signal_type="spikes",
            unit="spike",
        )
        projection = net.connect(
            source,
            pop.excitatory,
            name=f"private_{name.lower()}_to_{name}",
            synapse=snn.AMPA(tau=cfg["tau_ampa_ms"] * snn.ms),
            weight=snn.Constant(0.0),
            constraint=snn.NonNegative(),
            delay=cfg["delay_ms"] * snn.ms,
        )
        net.expose(projection.conductance, name=f"{projection.id}.conductance")
    for src in ("E", "I"):
        for dst in ("E", "I"):
            excitatory = src == "E"
            physical = cfg[f"recurrent_{src.lower()}_weight_us"]
            # Compensate normalization over eligible sources to retain physical weights.
            eligible_sources = cfg[f"n_{src.lower()}"] - int(src == dst)
            mean = physical * eligible_sources * cfg["connection_probability"]
            projection = net.connect(
                pops[src].spikes,
                pops[dst].excitatory if excitatory else pops[dst].inhibitory,
                name=f"{src}_to_{dst}",
                synapse=(
                    snn.AMPA(tau=cfg["tau_ampa_ms"] * snn.ms)
                    if excitatory
                    else snn.GABA(tau=cfg["tau_gaba_ms"] * snn.ms)
                ),
                weight=snn.LowerClampedNormal(
                    mean,
                    0.0,
                    initial_zero_fraction=1 - cfg["connection_probability"],
                    zeroing="bernoulli",
                ),
                constraint=snn.NonNegative(),
                connection="recurrent",
                allow_autapses=False,
                delay=cfg["delay_ms"] * snn.ms,
            )
            net.expose(projection.conductance, name=f"{projection.id}.conductance")
    for pop in pops.values():
        net.expose(pop.spikes, name=f"{pop.id}.spikes")
        net.expose(pop.voltage, name=f"{pop.id}.voltage")
    return snn.compile(net, target=None)
