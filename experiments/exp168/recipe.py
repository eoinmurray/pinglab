"""Resolved two-state COBA-LIF recipe; EE and II projections are absent."""

from snnlab import lang

SLUG = "exp168"
CONDITIONS = ("AI", "PING")
IMPORTS = {
    "AI": dict(
        run_id="exp167-r008-compute",
        payload_digest="sha256:5f128a2be1c311fbfb75f2215a3f77e01289cf4264da57454be37e9eaf0fd6f8",
        prefix="confirm-00-EE_and_II_zero",
        numbers="search.json",
        trial="confirm-00-EE_and_II_zero",
        diagram_prefix="trial-00-EE_and_II_zero",
    ),
    "PING": dict(
        run_id="exp167-r015-compute",
        payload_digest="sha256:e88961f94c6d81cde1ef9b959bf2231811baee99980c9883ed8f29b0e103d958",
        prefix="smooth-drive",
        numbers="rhythmicity.json",
        diagram_prefix="smooth-drive",
    ),
}


def configuration(condition, *, seed=43, device="cpu"):
    if condition not in CONDITIONS:
        raise ValueError("condition must be AI or PING")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    ai = condition == "AI"
    cfg = dict(
        experiment=SLUG,
        condition=condition,
        seed=seed,
        device=device,
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
        analysis=dict(
            rhythmicity_bin_ms=1.0,
            max_lag_ms=100.0,
            cv_min_spikes=20,
            correlation_cells_per_population=200,
            correlation_bin_ms=20.0,
        ),
    )
    # Units are µS. Native Bernoulli zeroing divides retained weights by p.
    cfg["initializer_parent_means_us"] = dict(
        E_to_E=0.0,
        E_to_I=0.000012,
        I_to_E=0.0002,
        I_to_I=0.0,
        X_to_E=0.00021600000000000002,
        X_to_I=0.00021600000000000002,
    )
    return cfg


def author_network(cfg):
    net = lang.Network(SLUG, dt=cfg["dt_ms"] * lang.ms)
    external = net.input(
        "X",
        shape=("time", "batch", cfg["n_external"]),
        signal_type="spikes",
        unit="spike",
    )
    pops = {}
    for name in ("E", "I"):
        pops[name] = net.population(
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
        synapse = lang.GABA if inhibitory else lang.AMPA
        probability = (
            cfg["input_connection_probability"]
            if source == "X"
            else cfg["connection_probability"]
        )
        net.connect(
            external if source == "X" else pops[source].spikes,
            pops[target].inhibitory if inhibitory else pops[target].excitatory,
            name=name,
            synapse=synapse(tau=cfg["tau_synapse_ms"] * lang.ms),
            weight=lang.LowerClampedNormal(
                mean,
                mean * cfg["weight_relative_sd"],
                initial_zero_fraction=1 - probability,
            ),
            constraint=lang.NonNegative(),
            initialization_scaling="direct",
            connection="feedforward" if source == "X" else "recurrent",
            allow_autapses=False,
            delay=cfg["inhibitory_delay_ms" if inhibitory else "excitatory_delay_ms"]
            * lang.ms,
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
        net.output(name + "_spikes", pops[name].spikes)
    return lang.compile(net)
