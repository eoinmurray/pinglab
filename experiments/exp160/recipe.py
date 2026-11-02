"""Spiking one-way pair: conductance-LIF neurons and matched private Poisson input."""

import hashlib
import math

import numpy as np
from snnlab import lang

SLUG = "exp160"
SCHEMA = "exp160.recipe/v1"
REFERENCE_SNAPSHOT = "network-initialization/start/snapshot.json"


def alignment_configuration():
    return dict(
        min_fraction=0.8,
        min_output_lag_ms=0.2,
        max_output_lag_ms=10.0,
        description="illustrative timing selection rule, not an established CTC test",
    )


def configuration(*, network):
    """Resolve the single accepted Figure 1–2 condition from the exp159 reference."""
    import copy

    n = copy.deepcopy(network)
    for physical in n["populations"].values():
        physical["size"] *= 5
    n["input_rates_hz"] = dict(drive_A=26.0, drive_B=24.0)
    return dict(
        schema=SCHEMA,
        protocol="oneway-spiking-pair",
        network=n,
        weight_variant=dict(
            transformation_version=2,
            population_scale=5,
            sensory_fanin=128,
            ei_fanin=80,
            ii_gain=0.1,
            ei_gain=0.8,
            target_gain_cv=0.1,
            inhibitory_gain_cv=0.2,
            seed=303,
            description="fixed per-target lognormal excitatory gain; exact random sensory fan-in with expectation-preserving scaling",
        ),
        duration_ms=1500.0,
        difference_hz=2.0,
        coupling_us=0.3,
        cross_i_ratio=0.0,
        input_seeds=dict(drive_A=101, drive_B=202),
        input_encoding="independent per-channel Bernoulli(p=rate*dt/1000); matched complete trains across pair",
        phase=dict(
            window_ms=[800.0, 1500.0],
            min_peak_distance_ms=15.0,
            prominence_fraction=0.1,
            min_amplitude_hz=0.5,
            min_peaks=20,
            frequency_band_hz=[30.0, 80.0],
            max_interval_cv=0.2,
            max_abs_drift_hz=0.25,
            min_concentration=0.95,
            max_phase_span_cycles=1.0,
        ),
        rate_smoothing_ms=1.0,
        timing=dict(window_ms=[1000.0, 1450.0], low_inhibition_fraction=0.25),
        closeup_cycles=4,
        recording_signals=[
            "PING_A_E_to_PING_B_E_K_EE.conductance",
            "PING_B_I_to_E.conductance",
        ],
    )


def reference_parameters(original, cfg):
    """Deterministic physical weight intervention, shared by both paired trials.

    SNNLab matrices have source rows and target columns. Masking sensory rows
    reduces shared averaging; positive target gains break identical-cell drive.
    Neither intervention changes or jitters recorded spike times.
    """
    import torch

    variant = cfg["weight_variant"]
    rng = np.random.default_rng(variant["seed"])
    cv = variant["target_gain_cv"]
    gains = {}
    for c in ("A", "B"):
        for p in ("E", "I"):
            pop_cv = variant["inhibitory_gain_cv"] if p == "I" else cv
            sigma = math.sqrt(math.log1p(pop_cv * pop_cv))
            gains[c, p] = rng.lognormal(
                -sigma * sigma / 2, sigma, cfg["network"]["populations"][p]["size"]
            )
    result = {}
    for key, value in original.items():
        weights = value.clone()
        factor = variant.get("population_scale", 1)
        if factor != 1:
            weights = weights.repeat_interleave(factor, dim=1)
            if "input_to_E" not in key:
                weights = weights.repeat_interleave(factor, dim=0) / factor
        if "I_to_I" in key:
            weights *= variant.get("ii_gain", 1)
        if "E_to_I.weight" in key:
            weights *= variant.get("ei_gain", 1)
        key_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
        mask_rng = np.random.default_rng([variant["seed"], key_seed])
        if "input_to_E" in key or "E_to_I.weight" in key:
            fanin = variant.get("sensory_fanin" if "input_to_E" in key else "ei_fanin")
            if fanin is not None:
                mask = np.zeros(tuple(weights.shape), dtype=np.float32)
                for row in mask.T:
                    row[mask_rng.choice(len(row), fanin, replace=False)] = (
                        len(row) / fanin
                    )
                weights *= torch.from_numpy(mask).to(weights)
        if "input_to_E" in key or "E_to_" in key:
            target = excitatory_target(key)
            weights *= torch.as_tensor(gains[target], dtype=weights.dtype)[None, :]
        result[key] = weights
    return result


def excitatory_target(key):
    if "input_to_E" in key:
        return key.split("_")[1], "E"
    target = key.split("_to_")[1]
    if target.startswith("PING_"):
        return target.split("_")[1], target.split("_")[2]
    return key.split("_")[1], target[0]


def make_inputs(cfg):
    n = cfg["network"]
    shape = (round(cfg["duration_ms"] / n["dt_ms"]), 1, n["input_channels"])
    return {
        name: (
            np.random.default_rng(seed).random(shape, dtype=np.float32)
            < n["input_rates_hz"][name] * n["dt_ms"] / 1000
        ).astype(np.uint8)
        for name, seed in cfg["input_seeds"].items()
    }


def scales(cfg, k):
    return {
        **{
            f"PING_{c}_input_to_E": cfg["network"]["sensory"]["gain"]
            for c in ("A", "B")
        },
        **{
            f"PING_A_E_to_PING_B_{p}_K_E{p}": k
            / cfg["network"]["base_coupling_us"]
            * (cfg["cross_i_ratio"] if p == "I" else 1)
            for p in ("E", "I")
        },
    }


def add_circuit(net, circuit, drive, settings):
    """Author the complete local E/I reference and its private sensory projection."""
    cells = {}
    with net.group(f"PING_{circuit}"):
        for population, physical in settings["populations"].items():
            cells[population] = net.population(
                f"PING_{circuit}_{population}",
                size=physical["size"],
                neuron=lang.COBA_LIF(
                    tau_mem=physical["capacitance_nf"] / physical["leak_us"] * lang.ms,
                    capacitance_nf=physical["capacitance_nf"],
                    leak_us=physical["leak_us"],
                    resting_mv=settings["resting_mv"],
                    threshold_mv=settings["threshold_mv"],
                    reset_mv=settings["reset_mv"],
                    initial_voltage_mv=settings["resting_mv"],
                    refractory=physical["refractory_ms"] * lang.ms,
                    voltage_grad_dampen=80.0,
                ),
            )
        sensory = settings["sensory"]
        net.connect(
            drive,
            cells["E"].excitatory,
            name=f"PING_{circuit}_input_to_E",
            synapse=lang.AMPA(tau=sensory["tau_ms"] * lang.ms),
            weight=lang.LowerClampedNormal(
                sensory["mean_us"],
                sensory["std_us"],
            ),
            constraint=lang.NonNegative(),
        )
        for pathway, physical in settings["local"].items():
            source, target = pathway.split("_to_")
            inhibitory = source == "I"
            fixed = "total_us" in physical
            weight = (
                lang.Constant(
                    physical["total_us"] / settings["populations"][source]["size"]
                )
                if fixed
                else lang.LowerClampedNormal(physical["mean_us"], physical["std_us"])
            )
            net.connect(
                cells[source].spikes,
                getattr(cells[target], "inhibitory" if inhibitory else "excitatory"),
                name=f"PING_{circuit}_{pathway}",
                synapse=(lang.GABA if inhibitory else lang.AMPA)(
                    tau=physical["tau_ms"] * lang.ms
                ),
                weight=weight,
                initialization_scaling="direct" if fixed else "fan_in_normalized",
                constraint=lang.NonNegative(),
                connection="recurrent",
                delay=settings["dt_ms"] * lang.ms,
            )
    return cells


def author_network(cfg):
    n = cfg["network"]
    net = lang.Network("exp160_oneway_ctc", dt=n["dt_ms"] * lang.ms)
    drives = {
        c: net.input(
            f"drive_{c}",
            shape=("time", "batch", n["input_channels"]),
            signal_type="spikes",
            unit="spike",
        )
        for c in ("A", "B")
    }
    cells = {c: add_circuit(net, c, drives[c], n) for c in ("A", "B")}
    for p in ("E", "I"):
        net.connect(
            cells["A"]["E"].spikes,
            cells["B"][p].excitatory,
            name=f"PING_A_E_to_PING_B_{p}_K_E{p}",
            synapse=lang.AMPA(tau=n["cross"]["tau_ms"] * lang.ms),
            weight=lang.LowerClampedNormal(
                n["base_coupling_us"],
                0,
                initial_zero_fraction=n["cross"]["zero_fraction"],
                zeroing="exact_k",
            ),
            constraint=lang.NonNegative(),
            connection="feedback",
            delay=n["cross"]["delay_ms"] * lang.ms,
            modulation=lang.StepGain(at=n["onset_ms"] * lang.ms, before=0, after=1),
        )
    for c in ("A", "B"):
        for p in ("E", "I"):
            net.expose(cells[c][p].spikes, name=f"{c}_{p}")
    return lang.compile(net, target=None)
