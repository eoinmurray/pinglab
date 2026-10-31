"""Spatial Gaussian drives and topographic excitation in a spiking A/B → C circuit."""

import copy
import math

import numpy as np
import torch
from snnlab import lang

SLUG = "exp163"
SCHEMA = "exp163.recipe/v1"
SNAPSHOT = "execution/start"


def configuration(
    *,
    reference_configuration,
    coupling_us=0.6,
    cue_hz=8.0,
    background_hz=22.0,
    contrast=0.4,
    width=0.1,
    kernel=0.06,
    swap=False,
):
    if (
        not all(
            math.isfinite(v) and v > 0
            for v in (coupling_us, cue_hz, background_hz, width, kernel)
        )
        or not math.isfinite(contrast)
        or not 0 < contrast <= 1
    ):
        raise ValueError("positive finite parameters and contrast in (0,1] required")
    return dict(
        schema=SCHEMA,
        protocol="one spiking cue-switch trial with spatial Gaussian sender drive",
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
        input_encoding="independent per-channel Bernoulli(rate*dt/1000); cue ports share one constant-rate spike train with complementary delivery gains",
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
        spatial=dict(
            centers=dict(A=0.75 if swap else 0.25, B=0.25 if swap else 0.75),
            width=float(width),
            contrast=float(contrast),
            kernel_width=float(kernel),
            min_receiver_contrast=0.15,
            min_sender_contrast=0.2,
            roi_half_width=0.1,
            max_peak_error=0.12,
            sensory_encoding="fixed target gains (1-contrast)+contrast*Gaussian/mean(Gaussian); unchanged constant-rate sensory spike sources",
            projection_encoding="Gaussian source-target distance kernel normalized per target; uniform inhibition; no C spatial background bias",
        ),
    )


def coordinates(cfg):
    n = cfg["network"]["populations"]["E"]["size"]
    return (np.arange(n) + 0.5) / n


def spatial_weights(original, cfg, reference_cfg):
    weights = {k: v.clone() for k, v in original.items()}
    x = coordinates(cfg)
    for c, center in cfg["spatial"]["centers"].items():
        g = np.exp(-0.5 * ((x - center) / cfg["spatial"]["width"]) ** 2)
        g /= g.mean()
        gain = (1 - cfg["spatial"]["contrast"]) + cfg["spatial"]["contrast"] * g
        weights["sensory_" + c + ".weight"] *= torch.as_tensor(
            gain, dtype=torch.float32
        )[None, :]
        key = c + "_E_to_C_E.weight"
        base = original[key].numpy()
        kernel = np.exp(
            -0.5 * ((x[:, None] - x[None, :]) / cfg["spatial"]["kernel_width"]) ** 2
        )
        kernel /= kernel.sum(axis=0, keepdims=True)
        weights[key] = torch.from_numpy(
            (kernel * base.sum(axis=0, keepdims=True)).astype(np.float32)
        )
    for c in "AB":
        for p in "EI":
            weights[c + "_E_to_C_" + p + ".weight"] *= (
                cfg["coupling_us"] / reference_cfg["coupling_us"]
            )
    # Both senders have equal total sensory drive and equal incoming sums at every C target.
    np.testing.assert_allclose(
        weights["sensory_A.weight"].numpy().sum(),
        weights["sensory_B.weight"].numpy().sum(),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        weights["A_E_to_C_E.weight"].numpy().sum(axis=0),
        weights["B_E_to_C_E.weight"].numpy().sum(axis=0),
        rtol=1e-6,
    )
    return weights


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
    net = lang.Network("exp163_spatial_competition", dt=n["dt_ms"] * lang.ms)
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
