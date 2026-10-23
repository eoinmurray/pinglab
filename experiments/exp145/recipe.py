"""Author a Renart-inspired conductance network with shared Poisson input.

The full-size circuit follows the supplement's population counts and neuron
parameters. Single-exponential synapses preserve conductance area, while fixed
mean delays and lower-clamped weights are explicit approximations.

configuration() resolves the measurement window (default 1 s) after 50 ms of burn-in.
author_network() compiles the circuit; neither function runs a simulation.
"""

from dataclasses import asdict

from snnlab import lang
from snnlab.sim import Performance


def configuration(
    *,
    seed=42,
    device="cpu",
    n_e=4000,
    n_i=1000,
    n_external=4000,
    measurement_ms=1000.0,
):
    if any(type(n) is not int or n <= 0 for n in (n_e, n_i, n_external)):
        raise ValueError("population sizes must be positive integers")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 < measurement_ms < float("inf"):
        raise ValueError("measurement_ms must be finite and positive")
    cfg = dict(
        n_e=n_e,
        n_i=n_i,
        n_external=n_external,
        seed=seed,
        device=device,
        torch_threads=1,
        dt_ms=0.05,
        duration_ms=50.0 + measurement_ms,
        burn_in_ms=50.0,
        external_rate_hz=2.5,
        external_seed_offset=10000,
        connection_probability=0.2,
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
        recording_cells=32,
        recording_block_steps=1000,
        analysis=dict(
            min_cv_spikes=20,
            correlation_cells=200,
            correlation_bins_ms=[5.0, 20.0, 50.0],
            correlation_step_ms=1.0,
            scaling_bin_ms=20.0,
            group_sizes=[4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
            subsample_repeats=50,
            subsample_seed_offset=30000,
            current_population="E",
            current_max_lag_ms=50.0,
            current_filter_ms=20.0,
        ),
        performance=asdict(
            Performance(compile_mode=None, precision="highest", cuda_graphs="off")
        ),
    )
    # Reference coefficients multiply kernels of area 1 ms, not unit-peak jumps.
    cfg["jump_means_us"] = {
        name: value * 1e-3 * cfg["reference_area_time_ms"] / cfg["tau_synapse_ms"]
        for name, value in cfg["reference_coefficients_ns"].items()
    }
    # Native Bernoulli sparsification divides retained weights by p. Compensate
    # its parent distribution so physical jumps do not grow as connectivity falls.
    cfg["initializer_parent_means_us"] = {
        name: jump * cfg["connection_probability"]
        for name, jump in cfg["jump_means_us"].items()
    }
    return cfg


def author_network(cfg):
    net = lang.Network("exp145", dt=cfg["dt_ms"] * lang.ms)
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
                initial_zero_fraction=1.0 - cfg["connection_probability"],
            ),
            constraint=lang.NonNegative(),
            initialization_scaling="direct",
            connection="feedforward" if source == "X" else "recurrent",
            delay=cfg["inhibitory_delay_ms" if inhibitory else "excitatory_delay_ms"]
            * lang.ms,
            allow_autapses=False,
        )
    return lang.compile(net)
