"""Author the three-window AI-target to PING-target protocol.

The full recurrent COBA-LIF circuit is active from initialization. After burn-in,
only direct external input to I changes; source rate and all other gains are fixed.
This recipe defines settings and the native graph without executing a simulation.
"""

import math
from copy import deepcopy
from dataclasses import asdict

import numpy as np
from snnlab import lang
from snnlab import meanfield as mf
from snnlab.meanfield import closures
from snnlab.sim import Performance


def configuration(
    *,
    seed=42,
    device="cpu",
    n_e=4000,
    n_i=1000,
    n_external=4000,
    baseline_ms=750.0,
    ramp_ms=750.0,
    hold_ms=750.0,
    final_i_gain=0.0,
    input_connection_probability=0.05,
    compile_mode="off",
    record_state=False,
):
    if any(type(n) is not int or n <= 0 for n in (n_e, n_i, n_external)):
        raise ValueError("population sizes must be positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if compile_mode not in {"off", "default"}:
        raise ValueError("unknown compile mode")
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
    if (
        not math.isfinite(input_connection_probability)
        or not 0 < input_connection_probability <= 1
    ):
        raise ValueError("input_connection_probability must be finite and in (0, 1]")
    cfg = dict(
        protocol="ai-to-ping",
        record_state=record_state,
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
        control=dict(
            parameter="X_to_I.weight",
            baseline_gain=1.0,
            final_gain=final_i_gain,
            ramp_start_ms=50.0 + baseline_ms,
            ramp_end_ms=50.0 + baseline_ms + ramp_ms,
            update_ms=0.05,
        ),
        burn_in_ms=50.0,
        external_rate_hz=2.5,
        external_seed_offset=10000,
        connection_probability=0.2,
        input_connection_probability=input_connection_probability,
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
            Performance(
                compile_mode=None if compile_mode == "off" else compile_mode,
                precision="highest",
                cuda_graphs="off",
            )
        ),
    )
    # Reference coefficients multiply kernels of area 1 ms, not unit-peak jumps.
    cfg["jump_means_us"] = {
        name: value * 1e-3 * cfg["reference_area_time_ms"] / cfg["tau_synapse_ms"]
        for name, value in cfg["reference_coefficients_ns"].items()
    }
    # Keep expected total input conductance fixed as input sparsity changes.
    for name in ("X_to_E", "X_to_I"):
        cfg["jump_means_us"][name] *= (
            cfg["connection_probability"] / cfg["input_connection_probability"]
        )
    # Native Bernoulli sparsification divides retained weights by p. Compensate
    # its parent distribution so physical jumps do not grow as connectivity falls.
    cfg["initializer_parent_means_us"] = {
        name: jump
        * (
            cfg["input_connection_probability"]
            if name.startswith("X_")
            else cfg["connection_probability"]
        )
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
                initial_zero_fraction=1.0
                - (
                    cfg["input_connection_probability"]
                    if source == "X"
                    else cfg["connection_probability"]
                ),
            ),
            constraint=lang.NonNegative(),
            initialization_scaling="direct",
            connection="feedforward" if source == "X" else "recurrent",
            delay=cfg["inhibitory_delay_ms" if inhibitory else "excitatory_delay_ms"]
            * lang.ms,
            allow_autapses=False,
            modulation=(
                lang.RampGain(
                    start=cfg["control"]["ramp_start_ms"] * lang.ms,
                    end=cfg["control"]["ramp_end_ms"] * lang.ms,
                    before=cfg["control"]["baseline_gain"],
                    after=cfg["control"]["final_gain"],
                )
                if name == "X_to_I"
                else None
            ),
        )
    for name in ("E", "I"):
        net.output(name + "_spikes", populations[name].spikes)
    return lang.compile(net)


def drive_gain(time_ms, cfg):
    """Scheduled multiplier on new external excitation of I; existing input decays."""
    control = cfg["control"]
    fraction = np.clip(
        (np.asarray(time_ms) - control["ramp_start_ms"])
        / (control["ramp_end_ms"] - control["ramp_start_ms"]),
        0,
        1,
    )
    return control["baseline_gain"] + fraction * (
        control["final_gain"] - control["baseline_gain"]
    )


def windows(cfg):
    """Absolute half-open measurement windows, excluding the 50 ms burn-in."""
    return (
        ("baseline", cfg["burn_in_ms"], cfg["control"]["ramp_start_ms"]),
        ("ramp", cfg["control"]["ramp_start_ms"], cfg["control"]["ramp_end_ms"]),
        ("hold", cfg["control"]["ramp_end_ms"], cfg["duration_ms"]),
    )


def hold_analysis_configuration():
    """Local pulse detection avoids a late large volley hiding early small ones."""
    return dict(
        interval_ms=100.0,
        bin_ms=0.5,
        minimum_separation_ms=6.0,
        prominence_fraction=0.12,
    )


def mean_field_configuration(spiking_configuration, summed_weights_us):
    """Native four-state rate–conductance closure with unfitted choices retained.

    Matrices are reduced to average incoming weight per target, including zeros.
    The shared pool supplies two mean drives; spike covariance is not retained.
    """
    scales = [
        spiking_configuration["tau_synapse_ms"] * summed_weights_us[path] / 1000
        for path in ("E_to_I", "I_to_E")
    ]
    if any(value <= 0 for value in scales):
        raise ValueError(
            "the four-state conductance basis requires positive E→I and I→E coupling"
        )
    return dict(
        model="snnlab-rate-conductance/v3",
        state_variables=["r_E_hz", "r_I_hz", "g_e_I_us", "g_i_E_us"],
        conductance_per_hz=scales,
        spiking_configuration=spiking_configuration,
        summed_weights_us=summed_weights_us,
        rate_relaxation_ms=[20.0, 5.0],
        voltage_noise_mv=4.0,
        closure_origin="exp117 fixed-resting-driving-force noisy-LIF closure; no fitted parameters",
        omitted_effects=[
            "shared-input covariance",
            "weight heterogeneity",
            "shunting",
            "axonal delays",
        ],
        alpha_points=201,
        gain_quadrature_nodes=48,
        equilibrium_scaled_derivative_tolerance=1e-9,
        equilibrium_guess=[0.0, 0.0, 0.0, 0.0],
        crossing_tolerance=1e-10,
        eigenvalue_tolerance_per_ms=1e-8,
        initial_e_perturbation_hz=0.001,
        solver=dict(
            method="LSODA",
            rtol=1e-8,
            atol=[1e-10, 1e-10, *[1e-10 * scale for scale in scales]],
            max_step_ms=0.5,
        ),
        sample_interval_ms=0.5,
    )


def mean_field_model(cfg, bundle):
    """Reduce recorded graph dynamics with explicit closure approximations.

    Only the Input→I ramp is translated to a numerical control. The source
    graph stays unchanged; unexpected modulation is rejected rather than lost.
    """
    net = cfg["spiking_configuration"]
    control = net["control"]
    expected = dict(
        kind="ramp_gain",
        before=control["baseline_gain"],
        after=control["final_gain"],
        start_ms=control["ramp_start_ms"],
        end_ms=control["ramp_end_ms"],
    )
    reduced_bundle = deepcopy(bundle)
    for projection in reduced_bundle.graph["projections"]:
        if projection["id"] == "X_to_I":
            if projection.get("modulation") != expected:
                raise ValueError(
                    "recorded Input→I modulation disagrees with its protocol"
                )
            projection["modulation"] = None
        elif projection.get("modulation"):
            raise ValueError("mean-field protocol supports only Input→I modulation")
    model = mf.reduce(
        reduced_bundle,
        closure=closures.RateConductance(
            gain=closures.NoisyLIF(
                sigma=cfg["voltage_noise_mv"],
                quadrature_nodes=cfg["gain_quadrature_nodes"],
            ),
            tau_rate=dict(zip(("E", "I"), cfg["rate_relaxation_ms"], strict=True)),
        ),
        weights=mf.MeanIncomingWeights(
            cfg["summed_weights_us"],
            provenance="effective sampled matrices from explicit source native start snapshot, sum(W)/N_target including zero edges",
        ),
        inputs={"X": mf.StationaryRate(net["external_rate_hz"])},
        external_filters="quasistatic",
        delays="omit",
    )
    return mf.reduce_states(model, retain=["E_to_I.conductance", "I_to_E.conductance"])


def mean_field_control(cfg):
    """Translate the recorded spiking ramp to SNNLab's numerical parameter gain."""
    control = cfg["spiking_configuration"]["control"]
    return mf.PiecewiseLinear(
        times=[control["ramp_start_ms"], control["ramp_end_ms"]],
        values=[control["baseline_gain"], control["final_gain"]],
    )
