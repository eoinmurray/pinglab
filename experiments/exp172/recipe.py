"""Fixed A/B → C spatial relay: configuration and native network authoring.

Spatial widths use a circular unit interval; weights are conductances in µS.
Defaults use the selected weak-recurrence stability setting; this is single-seed tuning, not held-out validation.
"""

import copy

import numpy as np
import torch
from snnlab import lang


def configuration(**overrides):
    """Return fresh recorded defaults; reject invalid explicit overrides."""
    cfg = dict(
        network={
            "cross": {"delay_ms": 2.0, "tau_ms": 0.75},
            "dt_ms": 0.1,
            "input_channels": 1024,
            "local": {
                "E_to_E": {"tau_ms": 2.0},
                "E_to_I": {"tau_ms": 1.0},
                "I_to_E": {"tau_ms": 9.0},
                "I_to_I": {"tau_ms": 6.0},
            },
            "populations": {
                "E": {
                    "capacitance_nf": 1.0,
                    "leak_us": 0.05,
                    "refractory_ms": 3.0,
                    "size": 400,
                },
                "I": {
                    "capacitance_nf": 0.5,
                    "leak_us": 0.1,
                    "refractory_ms": 1.5,
                    "size": 100,
                },
            },
            "reset_mv": -65.0,
            "resting_mv": -65.0,
            "sensory": {"delay_ms": 0.1, "gain": 9.2, "tau_ms": 2.0},
            "threshold_mv": -50.0,
        },
        incoming_weights_us={
            "PING_A_E_to_E": 0.394999991171062,
            "PING_A_E_to_I": 0.7509690917562694,
            "PING_A_E_to_PING_B_E_K_EE": 0.11999998986721039,
            "PING_A_E_to_PING_B_I_K_EI": 0.11999998986721039,
            "PING_A_I_to_E": 1.4914241866208613,
            "PING_A_I_to_I": 0.7599999830126762,
            "PING_A_input_to_E": 0.19989175639675522,
            "PING_B_E_to_E": 0.394999991171062,
            "PING_B_E_to_I": 0.7493030397221446,
            "PING_B_I_to_E": 1.5004359916318208,
            "PING_B_I_to_I": 0.7599999830126762,
            "PING_B_input_to_E": 0.19995858191441584,
        },
        reference_coupling_us=0.04,
        background_hz=28.0,
        duration_ms=2400.0,
        switch_ms=1200.0,
        dt_ms=0.1,
        schema="exp172.recipe/v1",
        seeds=[185],
        cue_strengths_hz=[0.0, 2.5, 2.8, 3.0, 3.2, 4.5, 6.0],
        spatial_sigma=0.08,
        sensory_kernel_sigma=0.025,
        cross_kernel_sigma=0.005,
        spatial_baseline_hz=22.0,
        spatial_mean_hz=26.0,
        centres=[0.25, 0.75],
        minimum_total_amplitude_hz=1.0,
        sensory_topographic_fraction=0.5,
        cross_mapping="gaussian",
        spatial_assignments=[False],
        c_gain_scale=0.25,
        c_ie_scale=1.31,
        c_ee_scale=0.05,
        cross_total_us=0.7333333333333334,
        c_ee_kernel_sigma=0.025,
        cross_tau_ms=0.75,
        fidelity_bin_neurons=8,
        representative_cue_hz=3.0,
        stability_search=dict(
            windows_ms=[[250.0, 1100.0], [1450.0, 2300.0]],
            fit_window_ms=40.0,
            fit_hop_ms=5.0,
            sustained_dominance_ms=100.0,
        ),
        input_encoding=(
            "Independent per-channel Bernoulli(rate*dt/1000); spatial sensory rates; "
            "complementary StepGain gates deliver a shared uniform cue train"
        ),
        input_seed_rule="numpy.default_rng([trial_seed, offset]); offsets X_A=700, X_B=701, X_C=702, U=703",
        effective_weights="Pinned native source arrays with spatial mappings, local recurrence, inhibition scaling and recorded target-gain compression; executed arrays retained natively",
        windows_ms=[[650.0, 1100.0], [1850.0, 2300.0]],
        protocol="constant simultaneous spatial Gaussians; uniform cue A then B",
        backend="native SNNLab CPU, one thread, no compilation",
        experiment="exp172",
        source_run="exp162-r031-compute",
        source_digest="sha256:d81ac08e4c7aa2ec1ad923e36ec1f575714369d4a136b32abd284774b5a29035",
        source_snapshot_sha256="afcfcdd6b1773f06a8d429dd21600fa83afaa2085db32999cc0b3113bb8224fa",
        selection_reference=dict(
            run_id="exp172-r012-compute",
            payload_digest="sha256:cb3c9be5f6d8c62fc28c242a21bdb2d7057f01be602de12a87b54bf1f361126d",
        ),
    )
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    cfg.update(copy.deepcopy(overrides))
    validate_inputs(cfg)
    validate_weights(cfg)
    cfg["network"]["dt_ms"] = cfg["dt_ms"]
    cfg["network"]["cross"]["tau_ms"] = cfg["cross_tau_ms"]
    return cfg


def validate_inputs(cfg):
    if cfg["experiment"] != "exp172":
        raise ValueError("Wrong experiment")
    if (
        not cfg["seeds"]
        or not cfg["cue_strengths_hz"]
        or 0 not in cfg["cue_strengths_hz"]
    ):
        raise ValueError("A seed and matched no-cue condition are required")
    if any(not np.isfinite(v) or v < 0 for v in cfg["cue_strengths_hz"]):
        raise ValueError("Cue rates must be finite and nonnegative")
    if (
        cfg["dt_ms"] <= 0
        or cfg["duration_ms"] <= 0
        or not 0 < cfg["switch_ms"] < cfg["duration_ms"]
    ):
        raise ValueError("Invalid timebase or cue switch")
    if not 0 <= cfg["spatial_baseline_hz"] < cfg["spatial_mean_hz"]:
        raise ValueError("Invalid spatial rates")
    for key in ["sensory_topographic_fraction", "c_gain_scale"]:
        if not 0 <= cfg[key] <= 1:
            raise ValueError(key)


def validate_weights(cfg):
    if cfg["cross_mapping"] != "gaussian":
        raise ValueError("Unknown cross mapping")
    for key in ["sensory_kernel_sigma", "spatial_sigma", "cross_tau_ms", "c_ie_scale"]:
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(key)
    if cfg["c_ee_scale"] < 0:
        raise ValueError("Negative conductance scale")
    if cfg["cross_kernel_sigma"] <= 0:
        raise ValueError("Positive Gaussian width required")
    if cfg["cross_total_us"] is not None and cfg["cross_total_us"] <= 0:
        raise ValueError("Positive incoming sum required")
    if cfg["c_ee_kernel_sigma"] is not None and (
        not np.isfinite(cfg["c_ee_kernel_sigma"]) or cfg["c_ee_kernel_sigma"] <= 0
    ):
        raise ValueError("Positive local recurrent width required")


def distance(x, centre):
    return (x - centre + 0.5) % 1.0 - 0.5


def gaussian(x, centre, sigma):
    return np.exp(-0.5 * (distance(x, centre) / sigma) ** 2)


def topographic(original, cfg):
    params = {k: v.clone() for k, v in original.items()}
    for key in [
        "sensory_A.weight",
        "sensory_B.weight",
        "A_E_to_C_E.weight",
        "B_E_to_C_E.weight",
    ]:
        rows, cols = params[key].shape
        x = np.arange(rows) / rows
        y = np.arange(cols) / cols
        sigma = (
            cfg["sensory_kernel_sigma"]
            if key.startswith("sensory")
            else cfg["cross_kernel_sigma"]
        )
        kernel = gaussian(x[:, None], y[None, :], sigma)
        kernel /= kernel.sum(axis=0, keepdims=True)
        # Equal target-by-target sender coupling prevents anatomical preference.
        totals = (
            original["A_E_to_C_E.weight"].sum(0)
            if "to_C" in key
            else original[key].sum(0)
        )
        if "to_C" in key and cfg.get("cross_total_us") is not None:
            totals = torch.full_like(totals, cfg["cross_total_us"])
        params[key] = torch.tensor(kernel, dtype=params[key].dtype) * totals[None, :]
    apply_gain_interventions(params, original, cfg)
    return params


def apply_gain_interventions(params, original, cfg):
    fraction = cfg.get("sensory_topographic_fraction", 1.0)
    if fraction != 1:
        for c in "AB":
            key = "sensory_" + c + ".weight"
            params[key] = fraction * params[key] + (1 - fraction) * original[key]
    params["C_I_to_E.weight"] *= cfg.get("c_ie_scale", 1.0)
    if cfg.get("c_ee_scale", 1.0) != 1:
        params["C_E_to_E.weight"] *= cfg["c_ee_scale"]
    if cfg.get("c_ee_kernel_sigma") is not None and cfg.get("c_ee_scale", 1.0) > 0:
        key = "C_E_to_E.weight"
        n = params[key].shape[0]
        x = np.arange(n) / n
        kernel = gaussian(x[:, None], x[None, :], cfg["c_ee_kernel_sigma"])
        np.fill_diagonal(kernel, 0.0)
        kernel /= kernel.sum(0, keepdims=True)
        params[key] = (
            torch.tensor(kernel, dtype=params[key].dtype) * params[key].sum(0)[None, :]
        )
    compress_target_gains(params, cfg)


def compress_target_gains(params, cfg):
    scale = cfg.get("c_gain_scale", 1.0)
    if scale == 1:
        return
    for key in [
        "sensory_C.weight",
        "C_E_to_E.weight",
        "A_E_to_C_E.weight",
        "B_E_to_C_E.weight",
    ]:
        totals = params[key].sum(0)
        if not torch.any(totals):
            continue
        desired = totals.mean() + (totals - totals.mean()) * scale
        params[key] *= (desired / totals)[None, :]


def inputs(cfg, seed, cue, swap, active="AB"):
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    n = cfg["network"]["input_channels"]
    x = np.arange(n) / n
    centres = cfg["centres"][::-1] if swap else cfg["centres"]
    rates = {}
    for name, centre in zip("AB", centres, strict=False):
        g = gaussian(x, centre, cfg["spatial_sigma"])
        rates["X_" + name] = (
            cfg["spatial_baseline_hz"]
            + (cfg["spatial_mean_hz"] - cfg["spatial_baseline_hz"]) * g / g.mean()
        )
        if name not in active:
            rates["X_" + name] = np.zeros(n)
    rates["X_C"] = np.full(n, cfg["background_hz"])
    rates["U"] = np.full(n, cue)
    arrays = {}
    for i, (key, hz) in enumerate(rates.items()):
        rng = np.random.default_rng([seed, 700 + i])
        arrays[key] = (
            rng.random((steps, 1, n), dtype=np.float32)
            < hz[None, None, :] * cfg["dt_ms"] / 1000
        ).astype(np.uint8)
    return arrays, rates


def author_network(cfg):
    n = cfg["network"]
    net = lang.Network("spatial_gaussian_relay", dt=n["dt_ms"] * lang.ms)
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
