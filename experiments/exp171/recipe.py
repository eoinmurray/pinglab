"""Small, source-pinned COBA adaptation pilot with private GABA inputs."""

import math

from snnlab import lang

SLUG = "exp171"
TRAINABLE = ("input_to_E.weight", "readout_projection.weight")
MAPPING = {
    "input_to_E.weight": "W_ff.0",
    "readout_projection.weight": "W_ff.1",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
}


def configuration(**overrides):
    cfg = dict(
        source_run="exp022-r015-compute",
        source_digest="sha256:a542a714021fd7bb69c5d08ccf006763f19ce83b01d8ecdca86dba150118491d",
        coba_cell="coba__off__seed42",
        ping_cell="ping__off__seed42",
        seed=42,
        selection_seed=42,
        sensory_seed=171001,
        calibration_seed=171002,
        shuffle_seed=171003,
        training_seed=171004,
        validation_seed=171005,
        train_per_digit=20,
        validation_per_digit=10,
        epochs=20,
        batch_size=20,
        validation_draws=3,
        dt_ms=0.1,
        duration_ms=200.0,
        inhibition_prehistory_ms=60.0,
        input_rate_hz=25.0,
        tau_gaba_ms=6.0,
        tau_ampa_ms=2.0,
        n_input=784,
        n_e=1024,
        n_i=256,
        n_output=10,
        lr=0.0004,
        weight_decay=0.0,
        grad_clip=1.0,
        surrogate_slope=1.0,
        voltage_grad_dampen=1000.0,
        probability_bounds=[0.001, 0.15],
        calibration_bisections=24,
        calibration_retries=20,
        calibration_sd_target=0.01,
        calibration_std_tolerance=0.03,
        rate_penalty=0.0,
        device="cpu",
        threads=1,
        compile_mode=None,
        precision="highest",
        calibration_batch_size=1,
        checkpoint_role="final_epoch",
        refractory_timing="legacy_coba",
        inhibition="Fixed per-image probabilities and amplitudes calibrated over scored trial after configured inhibition-only prehistory; fresh unconditional history and trial events each training epoch; three fixed validation histories and trials; carry only private conductance and pending events into fresh neuronal state",
        primary_comparison="final_epoch versus epoch_zero; paired validation inputs",
        retention_estimate="<250 MB; compact arrays, four native snapshots and two bundles; no full-trial traces",
        validation_limitation="Original validation partition already used for source checkpoint selection; no independent test-set claim",
    )
    if set(overrides) - set(cfg):
        raise ValueError("Unknown configuration override")
    cfg.update(overrides)
    for key in (
        "train_per_digit",
        "validation_per_digit",
        "epochs",
        "batch_size",
        "validation_draws",
        "calibration_batch_size",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"Invalid {key}")
    if cfg["device"] not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    if "calibration_batch_size" not in overrides:
        cfg["calibration_batch_size"] = 10 if cfg["device"] == "cuda" else 1
    if cfg["batch_size"] > 10 * cfg["train_per_digit"]:
        raise ValueError("Batch exceeds training pool")
    history = cfg["inhibition_prehistory_ms"] / cfg["dt_ms"]
    if (
        not math.isfinite(history)
        or history < 0
        or not math.isclose(history, round(history), abs_tol=1e-9)
    ):
        raise ValueError(
            "Inhibitory prehistory must be a nonnegative integer number of timesteps"
        )
    cfg["inhibition_prehistory_steps"] = round(history)
    cfg["steps"] = round(cfg["duration_ms"] / cfg["dt_ms"])
    return cfg


def author_network(cfg, *, teacher=False, record_spikes=False):
    net = lang.Network(
        "PING_teacher" if teacher else "COBA_adaptation", dt=cfg["dt_ms"] * lang.ms
    )
    drive = net.input(
        "drive",
        shape=("time", "batch", cfg["n_input"]),
        signal_type="spikes",
        unit="spike",
    )
    pops = {}
    for name, size, c, leak, ref in [
        ("E", cfg["n_e"], 1.0, 0.05, 1.2),
        ("I", cfg["n_i"], 0.5, 0.1, 0.6),
    ]:
        pops[name] = net.population(
            name,
            size=size,
            neuron=lang.COBA_LIF(
                tau_mem=c / leak * lang.ms,
                capacitance_nf=c,
                leak_us=leak,
                resting_mv=-65.0,
                threshold_mv=-50.0,
                reset_mv=-65.0,
                initial_voltage_mv=-65.0,
                refractory=ref * lang.ms,
                voltage_grad_dampen=cfg["voltage_grad_dampen"],
            ),
        )

    if record_spikes:
        for name, population in pops.items():
            net.output(name + "_spikes", population.spikes)

    def projection(src, dst, name, tau, inhibitory=False, recurrent=False, delay=0.0):
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

    projection(drive, pops["E"].excitatory, "input_to_E", cfg["tau_ampa_ms"])
    projection(
        pops["E"].spikes,
        pops["I"].excitatory,
        "E_to_I",
        cfg["tau_ampa_ms"],
        recurrent=True,
        delay=cfg["dt_ms"],
    )
    gaba = projection(
        pops["I"].spikes,
        pops["E"].inhibitory,
        "I_to_E",
        cfg["tau_gaba_ms"],
        True,
        True,
        cfg["dt_ms"],
    )
    if teacher:
        net.output("g_ping", gaba.conductance)
    else:
        # Events carry a fixed per-image amplitude; the frozen identity projection
        # keeps recipient j private without changing weights between batch samples.
        private = net.input(
            "artificial",
            shape=("time", "batch", cfg["n_e"]),
            signal_type="spikes",
            unit="spike",
        )
        gaba = projection(
            private,
            pops["E"].inhibitory,
            "artificial_to_E",
            cfg["tau_gaba_ms"],
            True,
            delay=cfg["dt_ms"],
        )
        net.output("g_artificial", gaba.conductance)
    out = net.population(
        "readout",
        size=cfg["n_output"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=2.0 * lang.ms,
            soft_reset_threshold=1.0,
            initial_voltage=0.0,
            surrogate_slope=cfg["surrogate_slope"],
        ),
    )
    net.connect(
        pops["E"].spikes,
        out.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=2.0 * lang.ms),
        weight=lang.Constant(0.0),
        initialization_scaling="direct",
        constraint=lang.NonNegative(),
    )
    scores = lang.reduce(
        out.pre_reset_voltage, operation="mean", over="time", name="mean_scores"
    )
    net.output("scores", scores)
    net.output(
        "e_count",
        lang.reduce(pops["E"].spikes, operation="sum", over="time", name="E_counts"),
    )
    training = (
        None
        if teacher
        else lang.TrainSpec(
            objectives=(lang.CrossEntropy(prediction=scores, target="digit"),),
            parameter_groups=(
                lang.ParameterGroup(list(TRAINABLE), name="learned", lr=cfg["lr"]),
                lang.ParameterGroup(
                    ["E_to_I.weight", "I_to_E.weight", "artificial_to_E.weight"],
                    name="frozen",
                    lr=0.0,
                    frozen=True,
                ),
            ),
            optimizer=lang.AdamW(lr=cfg["lr"], weight_decay=cfg["weight_decay"]),
            gradient_clip=cfg["grad_clip"],
            surrogate=lang.FastSigmoid(slope=cfg["surrogate_slope"]),
            epochs=cfg["epochs"],
            presentation_duration=cfg["duration_ms"] * lang.ms,
        )
    )
    return lang.compile(
        net, training=training, target=None, refractory_timing=cfg["refractory_timing"]
    )
