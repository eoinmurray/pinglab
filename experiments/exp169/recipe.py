"""Frozen TR02 COBA with private, moment-matched GABA event streams."""

from snnlab import lang

SLUG = "exp169"


def configuration():
    return dict(
        source_run="exp022-r015-compute",
        source_digest="sha256:a542a714021fd7bb69c5d08ccf006763f19ce83b01d8ecdca86dba150118491d",
        coba_cell="coba__off__seed42",
        ping_cell="ping__off__seed42",
        dt_ms=0.1,
        duration_ms=200.0,
        tau_gaba_ms=6.0,
        tau_ampa_ms=2.0,
        artificial_seed=302211,
        seed=42,
        checkpoint_role="final_epoch",
        classification_images_per_digit=10,
        classification_selection_seed=42,
        classification_encoding_seed=402211,
        classification_burn_in_ms=0.0,
        classification_input_rate_hz=25.0,
        classification_retention="Expected <1 GB and <650 files: 302 native start snapshots, one bundle/diagram, compact predictions and calibration moments; full trial traces are transient",
        mean_relative_tolerance=0.001,
        std_relative_tolerance=0.03,
        calibration_probability_bounds=[0.001, 0.15],
        calibration_bisections=24,
        calibration_retries=20,
        calibration_sd_target=0.01,
        n_input=784,
        n_e=1024,
        n_i=256,
        n_output=10,
        capacitance_e_nf=1.0,
        capacitance_i_nf=0.5,
        leak_e_us=0.05,
        leak_i_us=0.1,
        refractory_e_ms=1.2,
        refractory_i_ms=0.6,
        resting_mv=-65.0,
        threshold_mv=-50.0,
        reset_mv=-65.0,
        initial_voltage_mv=-65.0,
        voltage_grad_dampen=1000.0,
        readout_tau_ms=2.0,
        readout_threshold=1.0,
        surrogate_slope=1.0,
        calibration="Independent Bernoulli-discretized Poisson streams; per-cell event probability and positive weight calibrated; independent stream draws resampled when needed to reach 1% temporal-SD error; fitted to full 200-ms PING temporal moments; zero initial conductance; one-step delivery delay",
        training="none; all original checkpoint parameters frozen",
        retention="native bundle, diagram, start snapshots, 200-ms recordings and calibration; expected <80 MB",
    )


def author_network(cfg, *, include_private=True, record_ping=False):
    net = lang.Network("TR02_artificial_inhibition", dt=cfg["dt_ms"] * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", cfg["n_input"]),
        signal_type="spikes",
        unit="spike",
    )
    if include_private:
        artificial = net.input(
            "artificial",
            shape=("time", "batch", cfg["n_e"]),
            signal_type="spikes",
            unit="spike",
        )
    pops = {}
    for name, n, c, g, ref in [
        (
            "E",
            cfg["n_e"],
            cfg["capacitance_e_nf"],
            cfg["leak_e_us"],
            cfg["refractory_e_ms"],
        ),
        (
            "I",
            cfg["n_i"],
            cfg["capacitance_i_nf"],
            cfg["leak_i_us"],
            cfg["refractory_i_ms"],
        ),
    ]:
        pops[name] = net.population(
            name,
            size=n,
            neuron=lang.COBA_LIF(
                tau_mem=c / g * lang.ms,
                capacitance_nf=c,
                leak_us=g,
                resting_mv=cfg["resting_mv"],
                threshold_mv=cfg["threshold_mv"],
                reset_mv=cfg["reset_mv"],
                initial_voltage_mv=cfg["initial_voltage_mv"],
                refractory=ref * lang.ms,
                voltage_grad_dampen=cfg["voltage_grad_dampen"],
            ),
        )
        net.output(name + "_spikes", pops[name].spikes)

    def connect(src, dst, name, tau, inhibitory=False, recurrent=False, delay=0.0):
        return net.connect(
            src,
            dst,
            name=name,
            synapse=(lang.GABA if inhibitory else lang.AMPA)(tau=tau * lang.ms),
            weight=lang.Constant(0.0),
            initialization_scaling="direct",
            constraint=lang.NonNegative(),
            delay=delay * lang.ms,
            **({"connection": "recurrent"} if recurrent else {}),
        )

    connect(drive, pops["E"].excitatory, "input_to_E", cfg["tau_ampa_ms"])
    connect(
        pops["E"].spikes,
        pops["I"].excitatory,
        "E_to_I",
        cfg["tau_ampa_ms"],
        recurrent=True,
        delay=cfg["dt_ms"],
    )
    inhibitory = connect(
        pops["I"].spikes,
        pops["E"].inhibitory,
        "I_to_E",
        cfg["tau_gaba_ms"],
        True,
        True,
        cfg["dt_ms"],
    )
    if record_ping:
        net.output("g_ping", inhibitory.conductance)
    if include_private:
        private = connect(
            artificial,
            pops["E"].inhibitory,
            "artificial_to_E",
            cfg["tau_gaba_ms"],
            True,
            delay=cfg["dt_ms"],
        )
        net.output("g_artificial", private.conductance)
    out = net.population(
        "readout",
        size=cfg["n_output"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=cfg["readout_tau_ms"] * lang.ms,
            soft_reset_threshold=cfg["readout_threshold"],
            initial_voltage=0.0,
            surrogate_slope=cfg["surrogate_slope"],
        ),
    )
    net.connect(
        pops["E"].spikes,
        out.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=cfg["readout_tau_ms"] * lang.ms),
        weight=lang.Constant(0.0),
        initialization_scaling="direct",
        constraint=lang.NonNegative(),
    )
    net.output(
        "scores",
        lang.reduce(
            out.pre_reset_voltage, operation="mean", over="time", name="mean_scores"
        ),
    )
    return lang.compile(net, target=None, refractory_timing="legacy_coba")
