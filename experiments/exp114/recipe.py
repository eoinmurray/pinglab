"""Scientific definition for the fixed coupling-by-detuning grid."""

from __future__ import annotations

import numpy as np
from tools import snnlang as snn  # noqa: TID251

SLUG = "exp114"
DT_MS = 0.2
DURATION_MS = 1200.0
BURN_IN_MS = 300.0
N_E, N_I = 100, 25
SEEDS = (17, 29, 43)
MEAN_DRIVE_HZ = 3.0
SIGNED_DRIVE_DIFFERENCES_HZ = (-0.80, -0.30, 0.0, 0.30, 0.80)
CROSS_WEIGHTS_US = (0.0, 0.0025, 0.00625, 0.0100)


def configuration() -> dict:
    return {
        "schema": "exp114.recipe/v1",
        "dt_ms": DT_MS,
        "duration_ms": DURATION_MS,
        "burn_in_ms": BURN_IN_MS,
        "n_e_per_module": N_E,
        "n_i_per_module": N_I,
        "seeds": list(SEEDS),
        "mean_afferent_rate_hz": MEAN_DRIVE_HZ,
        "signed_drive_differences_hz": list(SIGNED_DRIVE_DIFFERENCES_HZ),
        "cross_e_weight_us": list(CROSS_WEIGHTS_US),
        "afferents_per_e_cell": 200,
        "external_weight_us": 0.004,
        "connection_probability": 0.25,
        "cross_connection_probability": 0.05,
        "local_ee_weight_us": 0.001,
        "local_ei_weight_us": 0.010,
        "local_ie_weight_us": 0.010,
        "local_ii_weight_us": 0.00334,
        "capacitance_nf": 0.15,
        "leak_us": 0.01,
        "resting_mv": -65.0,
        "threshold_mv": -50.0,
        "reset_mv": -65.0,
        "refractory_e_ms": 1.2,
        "refractory_i_ms": 0.6,
        "tau_ampa_ms": 1.5,
        "tau_gaba_ms": 7.5,
        "delay_ms": 1.0,
        "gamma_band_hz": [20.0, 80.0],
        "locking_thresholds": {
            "phase_concentration": 0.70,
            "absolute_peak_difference_hz": 1.5,
            "minimum_gamma_power_fraction": 0.20,
        },
        "reference": "https://doi.org/10.1371/journal.pcbi.1004072",
    }


def condition_id(detuning_index: int, coupling_index: int, seed: int) -> str:
    return f"d{detuning_index}_c{coupling_index}_s{seed}"


def conditions(cfg=None):
    cfg = configuration() if cfg is None else cfg
    for di, difference in enumerate(cfg["signed_drive_differences_hz"]):
        for ci, coupling in enumerate(cfg["cross_e_weight_us"]):
            for seed in cfg["seeds"]:
                yield {
                    "condition_id": condition_id(di, ci, seed),
                    "detuning_index": di,
                    "coupling_index": ci,
                    "seed": seed,
                    "drive_difference_hz": difference,
                    "drive_1_hz": cfg["mean_afferent_rate_hz"] - difference / 2,
                    "drive_2_hz": cfg["mean_afferent_rate_hz"] + difference / 2,
                    "cross_weight_us": coupling,
                }


def afferent_counts(cfg, condition):
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    streams = np.random.SeedSequence(condition["seed"] + 1009 * condition["detuning_index"]).spawn(2)
    result = {}
    for module, rate, stream in zip((1, 2), (condition["drive_1_hz"], condition["drive_2_hz"]), streams):
        lam = rate * cfg["afferents_per_e_cell"] * cfg["dt_ms"] / 1000
        values = np.random.default_rng(stream).poisson(lam, size=(steps, cfg["n_e_per_module"]))
        result[f"private_{module}"] = values.astype(np.uint8)
    return result


def _projection(net, source, target, name, cfg, physical_weight, *, inhibitory=False, probability=None):
    n_source = cfg["n_i_per_module"] if inhibitory else cfg["n_e_per_module"]
    probability = cfg["connection_probability"] if probability is None else probability
    net.connect(
        source,
        target,
        name=name,
        synapse=snn.GABA(tau=cfg["tau_gaba_ms"] * snn.ms) if inhibitory else snn.AMPA(tau=cfg["tau_ampa_ms"] * snn.ms),
        weight=snn.LowerClampedNormal(
            physical_weight * n_source * probability,
            0.0,
            initial_zero_fraction=1 - probability,
            zeroing="bernoulli",
        ),
        constraint=snn.NonNegative(),
        connection="recurrent",
        delay=cfg["delay_ms"] * snn.ms,
    )


def author_network(cfg, condition):
    net = snn.Network("two_module_ping", dt=cfg["dt_ms"] * snn.ms)
    populations = {}
    for module in (1, 2):
        for kind in ("E", "I"):
            populations[(module, kind)] = net.population(
                f"{kind}{module}",
                size=cfg[f"n_{kind.lower()}_per_module"],
                neuron=snn.COBA_LIF(
                    tau_mem=(cfg["capacitance_nf"] / cfg["leak_us"]) * snn.ms,
                    capacitance_nf=cfg["capacitance_nf"],
                    leak_us=cfg["leak_us"],
                    resting_mv=cfg["resting_mv"],
                    threshold_mv=cfg["threshold_mv"],
                    reset_mv=cfg["reset_mv"],
                    refractory_steps=round(cfg[f"refractory_{kind.lower()}_ms"] / cfg["dt_ms"]),
                    voltage_grad_dampen=80.0,
                    initial_voltage_mv=cfg["resting_mv"],
                ),
            )
        external = net.input(f"private_{module}", shape=("time", "batch", cfg["n_e_per_module"]), signal_type="spikes", unit="spike")
        net.connect(
            external,
            populations[(module, "E")].excitatory,
            name=f"private_{module}_to_E{module}",
            synapse=snn.AMPA(tau=cfg["tau_ampa_ms"] * snn.ms),
            weight=snn.Constant(0.0),
            constraint=snn.NonNegative(),
            delay=cfg["delay_ms"] * snn.ms,
        )
        for source_kind in ("E", "I"):
            for target_kind in ("E", "I"):
                inhibitory = source_kind == "I"
                _projection(
                    net,
                    populations[(module, source_kind)].spikes,
                    populations[(module, target_kind)].inhibitory if inhibitory else populations[(module, target_kind)].excitatory,
                    f"{source_kind}{module}_to_{target_kind}{module}",
                    cfg,
                    cfg[f"local_{source_kind.lower()}{target_kind.lower()}_weight_us"],
                    inhibitory=inhibitory,
                )
    for source_module, target_module in ((1, 2), (2, 1)):
        for target_kind in ("E", "I"):
            _projection(
                net,
                populations[(source_module, "E")].spikes,
                populations[(target_module, target_kind)].excitatory,
                f"E{source_module}_to_{target_kind}{target_module}_cross",
                cfg,
                condition["cross_weight_us"],
                probability=cfg["cross_connection_probability"],
            )
    net.expose(*(populations[(m, k)].spikes for m in (1, 2) for k in ("E", "I")), name="populations")
    return snn.compile(net, target=None)
