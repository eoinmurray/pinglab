"""Resolved configuration, source pins and native EE/II-on authoring."""

import math

from snnlab import lang

SLUG = "exp168"
CONDITIONS = ("AI", "PING")
RECURRENT_CASES = ("on", "off")

SOURCES = {
    "off": dict(
        run_id="exp168-r002-compute",
        payload_digest="sha256:b9d00b62553a82b0d86241c9607c0b8a65e11e75b39ba9d64942f7c034588218",
        label="AI and PING with EE and II absent",
    ),
}


def _validate_positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def configuration(
    condition,
    *,
    recurrence="on",
    device="cpu",
    seed=43,
    external_rate_hz=None,
    input_connection_probability=None,
    input_i_gain=None,
):
    if condition not in CONDITIONS:
        raise ValueError("condition must be AI or PING")
    if recurrence not in RECURRENT_CASES:
        raise ValueError("recurrence must be on or off")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    _validate_positive_int("seed", seed)
    ai = condition == "AI"
    cfg = dict(
        experiment=SLUG,
        condition=condition,
        recurrence=recurrence,
        device=device,
        seed=seed,
        n_e=4000,
        n_i=1000,
        n_external=4000,
        torch_threads=4,
        dt_ms=0.05,
        duration_ms=2050.0,
        measurement_ms=[50.0, 2050.0],
        external_seed_offset=10000,
        external_rate_hz=1.25 if ai else 2.5,
        connection_probability=0.2,
        input_connection_probability=0.005 if ai else 0.05,
        input_i_gain=0.3 if ai else 0.0,
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
        excitatory_delay_ms=1.0,
        inhibitory_delay_ms=0.5,
        recording_block_steps=1000,
        performance=dict(compile_mode=None, precision="highest", cuda_graphs="off"),
    )
    # Exp167 reference coefficients, converted from 1-ms kernel area to jumps.
    reference_coefficients_ns = dict(
        E_to_E=2.4,
        E_to_I=4.8,
        I_to_E=40.0,
        I_to_I=40.0,
        X_to_E=5.4,
        X_to_I=5.4,
    )
    jump_means_us = {
        name: value * 1e-3 * 1.0 / cfg["tau_synapse_ms"]
        for name, value in reference_coefficients_ns.items()
    }
    # Exp168's absent-projection case scales input jumps to compensate sparsity.
    for name in ("X_to_E", "X_to_I"):
        jump_means_us[name] *= (
            cfg["connection_probability"] / cfg["input_connection_probability"]
        )
    # Exp168's off case uses weaker E→I and I→E recurrent jumps.
    if recurrence == "off":
        jump_means_us["E_to_I"] = 0.000012
        jump_means_us["I_to_E"] = 0.0002
    cfg["initializer_parent_means_us"] = {
        name: jump
        * (
            cfg["input_connection_probability"]
            if name.startswith("X_")
            else cfg["connection_probability"]
        )
        for name, jump in jump_means_us.items()
    }
    if recurrence == "off":
        cfg["initializer_parent_means_us"]["E_to_E"] = 0.0
        cfg["initializer_parent_means_us"]["I_to_I"] = 0.0
    for key, value in (
        ("external_rate_hz", external_rate_hz),
        ("input_connection_probability", input_connection_probability),
        ("input_i_gain", input_i_gain),
    ):
        if value is not None:
            cfg[key] = value
    if (
        not math.isfinite(cfg["external_rate_hz"])
        or not 0 < cfg["external_rate_hz"] < 20000
    ):
        raise ValueError("external_rate_hz must be between 0 and 20000 Hz")
    if (
        not math.isfinite(cfg["input_connection_probability"])
        or not 0 < cfg["input_connection_probability"] <= 1
    ):
        raise ValueError("input_connection_probability must be in (0, 1]")
    if not math.isfinite(cfg["input_i_gain"]) or cfg["input_i_gain"] < 0:
        raise ValueError("input_i_gain must be finite and nonnegative")
    return cfg


def author_network(cfg):
    net = lang.Network(SLUG, dt=cfg["dt_ms"] * lang.ms)
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
            size=cfg["n_" + name.lower()],
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
                refractory=cfg["refractory_" + name.lower() + "_ms"] * lang.ms,
            ),
        )
    for name, mean in cfg["initializer_parent_means_us"].items():
        if mean == 0:
            continue
        source, target = name.split("_to_")
        inhibitory = source == "I"
        probability = (
            cfg["input_connection_probability"]
            if source == "X"
            else cfg["connection_probability"]
        )
        net.connect(
            external if source == "X" else populations[source].spikes,
            populations[target].inhibitory
            if inhibitory
            else populations[target].excitatory,
            name=name,
            synapse=(lang.GABA if inhibitory else lang.AMPA)(
                tau=cfg["tau_synapse_ms"] * lang.ms
            ),
            weight=lang.LowerClampedNormal(
                mean,
                mean * cfg["weight_relative_sd"],
                initial_zero_fraction=1 - probability,
            ),
            constraint=lang.NonNegative(),
            initialization_scaling="direct",
            connection="feedforward" if source == "X" else "recurrent",
            delay=cfg["inhibitory_delay_ms" if inhibitory else "excitatory_delay_ms"]
            * lang.ms,
            allow_autapses=False,
            modulation=lang.RampGain(
                start=50.0 * lang.ms,
                end=50.05 * lang.ms,
                before=cfg["input_i_gain"],
                after=cfg["input_i_gain"],
            )
            if name == "X_to_I"
            else None,
        )
    for name in ("E", "I"):
        net.output(name + "_spikes", populations[name].spikes)
    return lang.compile(net)
