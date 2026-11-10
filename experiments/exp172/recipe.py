"""Fixed A/B → C spatial relay: configuration and native network authoring.

Spatial widths use a circular unit interval; weights are conductances in µS.
Defaults use the selected combined inhibition/coupling setting; this is single-seed tuning, not held-out validation.
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
        c_ie_scale=1.40,
        c_ee_scale=0.05,
        feedforward_multiplier=0.8,
        cross_total_us=0.7333333333333334 * 0.8,
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
        effective_weights="Pinned exp172 baseline arrays; C inhibition and recurrence scaled relative to recorded baseline; both sender-to-C E matrices scaled equally; all other arrays unchanged",
        windows_ms=[[650.0, 1100.0], [1850.0, 2300.0]],
        protocol="constant simultaneous spatial Gaussians; uniform cue A then B",
        mechanism=mechanism_configuration(),
        candidate=candidate_configuration(),
        backend="native SNNLab CPU, one thread, no compilation",
        experiment="exp172",
        source_run="exp172-r012-compute",
        source_digest="sha256:cb3c9be5f6d8c62fc28c242a21bdb2d7057f01be602de12a87b54bf1f361126d",
        source_snapshot="seed185-swap0-cue3/start/snapshot.json",
        source_snapshot_sha256="0171b5a83f89709035ffada15d7d6d62c33ba169e527a1a78418d7d51a351b76",
        parameter_selection="Cue 3 Hz/channel, sensory 1x, inhibition 1.40, recurrence 0.05 and feedforward multiplier 0.8 selected from single-seed scratch sweeps; no held-out validation",
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


def trial_plan(cfg):
    """Declare the report cue protocol and single-sender controls."""
    items = [
        dict(
            id=f"seed{seed}-swap{int(swap)}-cue{cue:g}",
            seed=seed,
            swap=swap,
            cue_hz=cue,
            active="AB",
        )
        for seed in cfg["seeds"]
        for swap in cfg["spatial_assignments"]
        for cue in cfg["cue_strengths_hz"]
    ]
    items += [
        dict(
            id=f"single{c}-swap{int(swap)}",
            seed=cfg["seeds"][0],
            swap=swap,
            cue_hz=0.0,
            active=c,
        )
        for swap in cfg["spatial_assignments"]
        for c in "AB"
    ]
    return items


def selected_parameters(original, cfg, baseline):
    """Apply the selected interventions to the exact saved exp172 arrays."""
    params = {key: value.clone() for key, value in original.items()}
    params["C_I_to_E.weight"] *= cfg["c_ie_scale"] / baseline["c_ie_scale"]
    params["C_E_to_E.weight"] *= cfg["c_ee_scale"] / baseline["c_ee_scale"]
    for key in ["A_E_to_C_E.weight", "B_E_to_C_E.weight"]:
        params[key] *= cfg["feedforward_multiplier"]
    expected = baseline["cross_total_us"] * cfg["feedforward_multiplier"]
    if not np.isclose(cfg["cross_total_us"], expected, rtol=1e-12, atol=0):
        raise ValueError(
            "Recorded incoming conductance disagrees with coupling multiplier"
        )
    return params


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


def cycle_analysis_configuration():
    """A switch must persist for three consecutive receiver cycles."""
    return dict(sustained_cycles=3)


def phase_plot_configuration():
    """Operational low-inhibition proxy, independent of decoded dominance."""
    return dict(seed=186, profile_samples=241, low_inhibition_fraction=0.5)


def entrainment_configuration():
    """Trailing local phase-locking value, one sample per receiver cycle."""
    return dict(window_cycles=5)


def probe_configuration():
    """A fixed weak AMPA volley; measure net extra C E spikes within 5 ms."""
    return dict(
        anchor_ms=[400, 600, 800, 1000, 1500, 1700, 1900, 2100],
        phases_cycles=np.linspace(-0.5, 0.5, 16, endpoint=False).tolist(),
        conductance_us=0.05,
        tau_ms=0.75,
        delay_ms=2.0,
        response_ms=5.0,
        window_fraction=0.5,
    )


def used_window_configuration():
    """Natural-volley deletion; require a spatial effect over one and two cycles."""
    return dict(
        phase_bins=16,
        minimum_events=3,
        window_fraction=0.5,
        effect_tolerance=1e-6,
        sensitivity_phase_bins=[12, 16, 24],
        sensitivity_window_fractions=[0.25, 0.5, 0.75],
    )


def contrast_rate_profiles(cfg, ratio, swap=False):
    """Two epochs of spatial rates: attended Gaussian excess is ratio times larger."""
    if not np.isfinite(ratio) or ratio < 1:
        raise ValueError("Contrast ratio must be finite and at least one")
    x = np.arange(cfg["network"]["input_channels"]) / cfg["network"]["input_channels"]
    centres = cfg["centres"][::-1] if swap else cfg["centres"]
    rates = {}
    for sender, centre in zip("AB", centres, strict=True):
        g = gaussian(x, centre, cfg["spatial_sigma"])
        bump = (cfg["spatial_mean_hz"] - cfg["spatial_baseline_hz"]) * g / g.mean()
        rates["X_" + sender] = np.stack(
            [
                cfg["spatial_baseline_hz"]
                + (ratio if sender == "AB"[epoch] else 1) * bump
                for epoch in range(2)
            ]
        )
    return rates


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
    if "contrast_ratio" in cfg:
        if active != "AB" or cue != 0:
            raise ValueError("Contrast cue requires both senders and no uniform cue")
        rates.update(contrast_rate_profiles(cfg, cfg["contrast_ratio"], swap))
    rates["X_C"] = np.full(n, cfg["background_hz"])
    rates["U"] = np.full(n, cue)
    arrays = {}
    for i, (key, hz) in enumerate(rates.items()):
        rng = np.random.default_rng([seed, 700 + i])
        uniform = rng.random((steps, 1, n), dtype=np.float32)
        if hz.ndim == 2:
            split = round(cfg["switch_ms"] / cfg["dt_ms"])
            arrays[key] = np.empty((steps, 1, n), dtype=np.uint8)
            for epoch, window in enumerate((slice(0, split), slice(split, steps))):
                arrays[key][window] = (
                    uniform[window] < hz[epoch][None, None, :] * cfg["dt_ms"] / 1000
                )
        else:
            arrays[key] = (uniform < hz[None, None, :] * cfg["dt_ms"] / 1000).astype(
                np.uint8
            )
    return arrays, rates


def arrival_distribution_configuration():
    """Cycle-normalised arrival histograms for the exact Fig1/Fig6 trials."""
    return dict(
        rate_smoothing_ms=1.0,
        minimum_peak_spacing_ms=15.0,
        peak_prominence_fraction=0.1,
        histogram_bins=24,
        smoothing_bins=1.0,
        profile_samples=240,
        reference="C inhibitory conductance minimum",
        gate_threshold=None,
        seed=185,
        cue_hz=3.0,
    )


def background_sweep_configuration():
    """Paired input-seed screen at fixed circuit weights and sensory contrast."""
    return dict(
        seeds=[185, 186, 187, 188, 189],
        rates_hz=[float(v) for v in range(0, 33, 4)],
        cue_hz=3.0,
        contrast_multiplier=1.0,
        weights="identical pinned baseline arrays",
        spread="sample standard deviation across input seeds",
    )


def background_sweep_plan(settings):
    return [
        dict(
            id=f"background-seed{seed}-rate{rate:g}",
            seed=seed,
            cue_hz=settings["cue_hz"],
            swap=False,
            active="AB",
            background_hz=rate,
        )
        for seed in settings["seeds"]
        for rate in settings["rates_hz"]
    ]


def cue_sweep_configuration():
    """Cue dose response at the best joint background-screen setting."""
    return dict(
        seeds=[185, 186, 187, 188, 189],
        rates_hz=[float(v) for v in range(7)],
        reference_cue_hz=3.0,
        selection="Highest mean S across attend A/B, then lowest mean F, then lowest mean observed latency",
        spread="sample standard deviation across input seeds",
    )


def best_background(summary):
    """Rank both attention conditions equally; disclose selection on these seeds."""
    rates = sorted({row["background_hz"] for row in summary})
    scores = []
    for rate in rates:
        rows = [row for row in summary if row["background_hz"] == rate]
        if sorted(row["attended"] for row in rows) != ["A", "B"]:
            raise ValueError("Expected attend-A and attend-B summaries")
        scores.append(
            (
                -np.mean([row["S"]["mean"] for row in rows]),
                np.mean([row["F"]["mean"] for row in rows]),
                np.mean([row["latency_cycles"]["mean"] for row in rows]),
                rate,
            )
        )
    return float(min(scores)[-1])


def cue_sweep_plan(settings):
    return [
        dict(
            id=f"cue-seed{seed}-rate{rate:g}",
            seed=seed,
            cue_hz=rate,
            swap=False,
            active="AB",
            background_hz=settings["background_hz"],
        )
        for seed in settings["seeds"]
        for rate in settings["rates_hz"]
    ]


def contrast_sweep_configuration():
    return dict(
        seeds=[185, 186, 187, 188, 189],
        rates_hz=[1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
        example_ratios=[1.25, 3.0],
        cue_hz=0.0,
        reference_ratio=1.0,
        definition="Attended/unattended Gaussian excess-amplitude ratio above a fixed uniform baseline",
        spread="sample standard deviation across input seeds",
    )


def contrast_sweep_plan(settings):
    return [
        dict(
            id=f"contrast-seed{seed}-ratio{ratio:g}",
            seed=seed,
            contrast_ratio=ratio,
            cue_hz=0.0,
            background_hz=settings["background_hz"],
            swap=False,
            active="AB",
        )
        for seed in settings["seeds"]
        for ratio in settings["rates_hz"]
    ]


def candidate_configuration():
    """Best joint-selection point in the separate one-seed screens."""
    return dict(
        background_hz=32.0,
        source_seed=185,
        cue_hz=3.0,
        contrast_multiplier=1.0,
        selection="Highest S in both attention periods among the tested contrast/background points; not a validated optimum",
    )


def mechanism_configuration():
    """Post hoc illustrative timing test; fixed before the native rerun."""
    return dict(
        sender="B",
        attended="A",
        packet_window_ms=[470.0, 490.0],
        delay_ms=6.0,
        display_window_ms=[470.0, 497.0],
        outcome_window_ms=[470.0, 497.0],
        source_seed=185,
        cue_hz=3.0,
    )


def shifted_packet(spikes, dt_ms, settings):
    """Delay a whole packet without losing, adding or wrapping any spikes."""
    lo, hi = [round(v / dt_ms) for v in settings["packet_window_ms"]]
    shift = round(settings["delay_ms"] / dt_ms)
    if not (0 <= lo < hi <= len(spikes) and 0 < shift < hi - lo):
        raise ValueError("Invalid packet timing window")
    if shift * dt_ms != settings["delay_ms"]:
        raise ValueError("Packet delay must be an integer number of timesteps")
    if spikes[hi - shift : hi].any():
        raise ValueError("Packet window would wrap nonzero spikes")
    result = spikes.copy()
    result[lo:hi] = np.roll(spikes[lo:hi], shift, axis=0)
    np.testing.assert_array_equal(result.sum(0), spikes.sum(0))
    return result


def author_network(cfg, *, replay_sender=None, probe=None, replay_both=False):
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
    if replay_sender is not None and replay_sender not in "AB":
        raise ValueError("Replay sender must be A or B")
    replay = (
        None
        if replay_sender is None
        else net.input(
            "replay_" + replay_sender,
            shape=("time", "batch", n["populations"]["E"]["size"]),
            signal_type="spikes",
            unit="spike",
        )
    )
    replay_sources = (
        {
            c: net.input(
                "replay_" + c,
                shape=("time", "batch", n["populations"]["E"]["size"]),
                signal_type="spikes",
                unit="spike",
            )
            for c in "AB"
        }
        if replay_both
        else {}
    )
    if replay_both and replay_sender is not None:
        raise ValueError("Choose single-sender or dual-sender replay")
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
                (replay_sources[c] if replay_both else replay)
                if p == "E" and (replay_both or replay_sender == c)
                else cells[c]["E"].spikes,
                cells["C"][p].excitatory,
                c + "_E_to_C_" + p,
                cfg["reference_coupling_us"],
                n["cross"]["tau_ms"],
                delay=n["cross"]["delay_ms"],
            )
    if probe is not None:
        source = net.input(
            "probe", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
        )
        net.connect(
            source,
            cells["C"]["E"].excitatory,
            name="probe_to_C_E",
            synapse=lang.AMPA(tau=probe["tau_ms"] * lang.ms),
            weight=lang.Constant(probe["conductance_us"]),
            initialization_scaling="direct",
            constraint=lang.NonNegative(),
            delay=probe["delay_ms"] * lang.ms,
        )
    return lang.compile(net, target=None), weights
