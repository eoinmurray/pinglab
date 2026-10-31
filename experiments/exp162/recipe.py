"""Spiking A/B → C competition with a redirected constant-rate cue."""

import copy
import math

import numpy as np
from snnlab import lang

SLUG = "exp162"
SCHEMA = "exp162.recipe/v1"
SNAPSHOT = "execution/start"


def configuration(
    *, reference_configuration, coupling_us=0.12, cue_hz=6.0, background_hz=28.0
):
    if not all(
        math.isfinite(x) and x > 0 for x in (coupling_us, cue_hz, background_hz)
    ):
        raise ValueError("coupling, cue and background must be finite and positive")
    return dict(
        schema=SCHEMA,
        protocol="one spiking trial; constant-rate cue redirected A to B",
        network=copy.deepcopy(reference_configuration["network"]),
        incoming_weights_us=copy.deepcopy(
            reference_configuration["incoming_weights_us"]
        ),
        reference_coupling_us=float(reference_configuration["coupling_us"]),
        coupling_us=float(coupling_us),
        cue_hz=float(cue_hz),
        background_hz=float(background_hz),
        sensory_hz=24.0,
        input_rates_hz=dict(
            X_A=24.0,
            X_B=24.0,
            X_C=float(background_hz),
            U_A=float(cue_hz),
            U_B=float(cue_hz),
        ),
        duration_ms=2400.0,
        switch_ms=1200.0,
        dt_ms=0.1,
        windows_ms=[[650.0, 1100.0], [1850.0, 2300.0]],
        seed=85,
        input_seeds=dict(X_A=101, X_B=202, X_C=303, U=404),
        input_encoding="independent per-channel Bernoulli(rate*dt/1000); cue ports share one constant-rate spike train with complementary delivery gates",
        rate_smoothing_ms=1.0,
        low_inhibition_fraction=0.25,
        phase=dict(
            min_peak_distance_ms=15.0,
            min_prominence_hz=0.05,
            prominence_fraction=0.1,
            min_plv=0.95,
            max_abs_drift_cycles_s=0.25,
            max_cued_phase_span_cycles=0.3,
            min_uncued_phase_span_cycles=0.5,
            min_receiver_amplitude_hz=0.5,
        ),
        timing=dict(
            reference_start_ms=500.0, closeup_before_ms=100.0, closeup_after_ms=250.0
        ),
        rhythm_criteria=dict(
            min_volleys=12, frequency_band_hz=[30.0, 80.0], max_interval_cv=0.2
        ),
        recording_signals=[
            "A_E_to_C_E.conductance",
            "B_E_to_C_E.conductance",
            "C_I_to_E.conductance",
            "cue_A.conductance",
            "cue_B.conductance",
        ],
    )


def scales(cfg):
    return {
        c + "_E_to_C_" + p: cfg["coupling_us"] / cfg["reference_coupling_us"]
        for c in "AB"
        for p in "EI"
    }


def make_inputs(cfg):
    shape = (
        round(cfg["duration_ms"] / cfg["dt_ms"]),
        1,
        cfg["network"]["input_channels"],
    )
    rates = {k: cfg["input_rates_hz"][k] for k in ["X_A", "X_B", "X_C"]}
    rates["U"] = cfg["cue_hz"]
    return {
        k: (
            np.random.default_rng(cfg["input_seeds"][k]).random(shape, dtype=np.float32)
            < rate * cfg["dt_ms"] / 1000
        ).astype(np.uint8)
        for k, rate in rates.items()
    }


def author_network(cfg):
    n = cfg["network"]
    net = lang.Network("exp162_spiking_competition", dt=n["dt_ms"] * lang.ms)
    source_weights = cfg["incoming_weights_us"]
    weights = {}
    cells = {}
    inputs = {}

    def connect(src, target, name, total, tau, delay=0.1, modulation=None):
        # Native direct matrices have the requested incoming sum at every cell.
        source_size = (
            n["input_channels"]
            if name.startswith(("sensory_", "cue_"))
            else n["populations"]["E" if "_E_to_" in name else "I"]["size"]
        )
        if name.endswith(("_E_to_E", "_I_to_I")):
            source_size -= 1
        net.connect(
            src,
            target,
            name=name,
            synapse=(lang.GABA if "_I_to_" in name else lang.AMPA)(tau=tau * lang.ms),
            weight=lang.Constant(total / source_size),
            initialization_scaling="direct",
            constraint=lang.NonNegative(),
            delay=delay * lang.ms,
            **(dict(modulation=modulation) if modulation is not None else {}),
        )
        weights[name] = total

    for c in "ABC":
        inputs[c] = net.input(
            "X_" + c,
            shape=("time", "batch", n["input_channels"]),
            signal_type="spikes",
            unit="spike",
        )
        cells[c] = {}
        for p, phys in n["populations"].items():
            cells[c][p] = net.population(
                c + "_" + p,
                size=phys["size"],
                neuron=lang.COBA_LIF(
                    tau_mem=phys["capacitance_nf"] / phys["leak_us"] * lang.ms,
                    capacitance_nf=phys["capacitance_nf"],
                    leak_us=phys["leak_us"],
                    resting_mv=n["resting_mv"],
                    threshold_mv=n["threshold_mv"],
                    reset_mv=n["reset_mv"],
                    initial_voltage_mv=n["resting_mv"],
                    refractory=phys["refractory_ms"] * lang.ms,
                    voltage_grad_dampen=80.0,
                ),
            )
        connect(
            inputs[c],
            cells[c]["E"].excitatory,
            "sensory_" + c,
            source_weights["PING_B_input_to_E"] * n["sensory"]["gain"],
            n["sensory"]["tau_ms"],
        )
        for path, phys in n["local"].items():
            s, t = path.split("_to_")
            connect(
                cells[c][s].spikes,
                getattr(cells[c][t], "inhibitory" if s == "I" else "excitatory"),
                c + "_" + path,
                source_weights["PING_B_" + path],
                phys["tau_ms"],
            )
        for p in "EI":
            net.expose(cells[c][p].spikes, name=c + "_" + p + "_spikes")
    for c in "AB":
        cue = net.input(
            "U_" + c,
            shape=("time", "batch", n["input_channels"]),
            signal_type="spikes",
            unit="spike",
        )
        connect(
            cue,
            cells[c]["E"].excitatory,
            "cue_" + c,
            source_weights["PING_B_input_to_E"] * n["sensory"]["gain"],
            n["sensory"]["tau_ms"],
            modulation=lang.StepGain(
                at=cfg["switch_ms"] * lang.ms,
                before=1 if c == "A" else 0,
                after=0 if c == "A" else 1,
            ),
        )
        for p in "EI":
            connect(
                cells[c]["E"].spikes,
                cells["C"][p].excitatory,
                c + "_E_to_C_" + p,
                cfg["reference_coupling_us"],
                n["cross"]["tau_ms"],
                delay=n["cross"]["delay_ms"],
            )
    return lang.compile(net, target=None), weights
