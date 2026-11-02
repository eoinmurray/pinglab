"""Spatial Gaussian drives and topographic excitation in a spiking A/B → C circuit."""

import copy
import hashlib
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
    background_hz=28.0,
    contrast=0.08,
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
    network = copy.deepcopy(reference_configuration["network"])
    variant = dict(
        population_scale=5,
        sensory_fanin=128,
        ei_fanin=80,
        cross_fanin=40,
        target_gain_cv=0.1,
        receiver_gain_cv=0.0,
        reflect_with_centers=False,
        inhibitory_gain_cv=0.2,
        ei_gain=0.8,
        ii_gain=0.1,
        cross_i_ratio=0.65,
        seed=303,
    )
    for physical in network["populations"].values():
        physical["size"] *= variant["population_scale"]
    network["local"]["E_to_I"]["tau_ms"] = 1.0
    return dict(
        schema=SCHEMA,
        heterogeneity=variant,
        protocol="one spiking cue-switch trial with spatial Gaussian sender drive",
        network=network,
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


def heterogeneous_parameters(original, cfg):
    """Fixed input masks and excitatory target gains; saved native arrays are authoritative."""
    import torch

    if "heterogeneity" not in cfg:
        return {key: value.clone() for key, value in original.items()}
    variant = cfg["heterogeneity"]
    factor = variant["population_scale"]
    rng = np.random.default_rng(variant["seed"])
    gains = {}
    for c in "ABC":
        for p in "EI":
            cv = (
                variant["inhibitory_gain_cv"] if p == "I" else variant["target_gain_cv"]
            )
            if c == "C" and p == "E":
                cv = variant.get("receiver_gain_cv", cv)
            sigma = math.sqrt(math.log1p(cv * cv))
            gains[c, p] = rng.lognormal(
                -sigma * sigma / 2, sigma, cfg["network"]["populations"][p]["size"]
            )
    if "i_background_us" in variant:
        original = dict(original)
        for c in "ABC":
            original["background_I_" + c + ".weight"] = torch.full(
                (
                    cfg["network"]["input_channels"],
                    cfg["network"]["populations"]["I"]["size"] // factor,
                ),
                variant["i_background_us"] / cfg["network"]["input_channels"],
            )
    result = {}
    for key, value in original.items():
        weights = value.clone().repeat_interleave(factor, dim=1)
        external = key.startswith(("sensory_", "cue_", "background_I_"))
        if not external:
            weights = weights.repeat_interleave(factor, dim=0) / factor
        if key.endswith("_I_to_I.weight"):
            weights *= variant["ii_gain"]
        if key.endswith("_E_to_I.weight"):
            weights *= variant["ei_gain"]
        fanin = (
            variant["i_fanin"]
            if key.startswith("background_I_")
            else variant["sensory_fanin"]
            if external
            else (
                variant["ei_fanin"]
                if key.endswith("_E_to_I.weight")
                else variant["cross_fanin"]
                if "_to_C_" in key
                else None
            )
        )
        if fanin is not None:
            key_seed = int.from_bytes(
                hashlib.sha256(key.encode()).digest()[:4], "little"
            )
            mask_rng = np.random.default_rng([variant["seed"], key_seed])
            mask = np.zeros(tuple(weights.shape), dtype=np.float32)
            for column in mask.T:
                column[mask_rng.choice(len(column), fanin, replace=False)] = (
                    len(column) / fanin
                )
            weights *= torch.from_numpy(mask)
        if external or "_E_to_" in key:
            target = excitatory_target(key)
            weights *= torch.as_tensor(gains[target], dtype=weights.dtype)[None, :]
        result[key] = weights
    if variant.get("bilateral_symmetry", False):
        make_bilateral(result)
    return result


def make_bilateral(parameters):
    """Keep exact fan-ins while tying quenched weights across the midline."""
    for key, weights in parameters.items():
        half = weights.shape[1] // 2
        axes = (1,) if key.startswith(("sensory_", "cue_", "background_I_")) else (0, 1)
        weights[:, half:] = weights[:, :half].flip(axes)


def excitatory_target(key):
    if key.startswith("background_I_"):
        return key.removesuffix(".weight")[-1], "I"
    if key.startswith(("sensory_", "cue_")):
        return key.split("_")[1][0], "E"
    return ("C" if "_to_C_" in key else key[0]), key.removesuffix(".weight")[-1]


def coordinates(cfg):
    n = cfg["network"]["populations"]["E"]["size"]
    return (np.arange(n) + 0.5) / n


def spatial_weights(original, cfg, reference_cfg):
    weights = heterogeneous_parameters(original, cfg)
    x = coordinates(cfg)
    for c, center in cfg["spatial"]["centers"].items():
        g = np.exp(-0.5 * ((x - center) / cfg["spatial"]["width"]) ** 2)
        g /= g.mean()
        gain = (1 - cfg["spatial"]["contrast"]) + cfg["spatial"]["contrast"] * g
        weights["sensory_" + c + ".weight"] *= torch.as_tensor(
            gain, dtype=torch.float32
        )[None, :]
        if "heterogeneity" in cfg:
            target_total = (
                cfg["incoming_weights_us"]["PING_B_input_to_E"]
                * cfg["network"]["sensory"]["gain"]
                * len(x)
            )
            weights["sensory_" + c + ".weight"] *= (
                target_total / weights["sensory_" + c + ".weight"].sum()
            )
        key = c + "_E_to_C_E.weight"
        base = weights[key].numpy()
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
    if "heterogeneity" in cfg:
        for c in "AB":
            weights[c + "_E_to_C_I.weight"] *= cfg["heterogeneity"]["cross_i_ratio"]
        if (
            cfg["heterogeneity"].get("reflect_with_centers", True)
            and cfg["spatial"]["centers"]["A"] > cfg["spatial"]["centers"]["B"]
        ):
            # Reflect the complete heterogeneous circuit, not only its Gaussian field.
            unswapped = copy.deepcopy(cfg)
            unswapped["spatial"]["centers"] = {
                c: 1 - v for c, v in cfg["spatial"]["centers"].items()
            }
            baseline = spatial_weights(original, unswapped, reference_cfg)
            return {
                key: value.flip((1,))
                if key.startswith(("sensory_", "cue_", "background_I_"))
                else value.flip((0, 1))
                for key, value in baseline.items()
            }
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
            if name.startswith(("sensory_", "cue_", "background_I_"))
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
        if "i_background_us" in cfg.get("heterogeneity", {}):
            connect(
                inputs[c],
                cells[c]["I"].excitatory,
                "background_I_" + c,
                cfg["heterogeneity"]["i_background_us"],
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
