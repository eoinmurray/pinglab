"""Source-pinned native COBA fine-tuning with private GABA inputs."""

import math

from snnlab import lang

SLUG = "exp171"
TRAINABLE = ("input_to_E.weight", "readout_projection.weight")
RATE_TARGETS_HZ = (None, 25.0, 10.0, 5.0, 2.5, 1.0)
RATE_PENALTY_STRENGTH = 0.041
SOURCE_SEEDS = (42, 43, 44)
POISSON_METHOD = "poisson_moment_matched_v1"
TEACHER_COUNT_METHOD = "teacher_rate_count_timing_v3"
COUNT_METHODS = (
    "count_matched_timing_v1",
    "count_matched_timing_v2",
    TEACHER_COUNT_METHOD,
)
EXTENDED_METHODS = ("count_matched_timing_v2", TEACHER_COUNT_METHOD)
MAPPING = {
    "input_to_E.weight": "W_ff.0",
    "readout_projection.weight": "W_ff.1",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
}


def target_label(target):
    return "off" if target is None else "rt" + f"{target:g}".replace(".", "p") + "hz"


def sweep_configurations(*, seed=42, **overrides):
    """Matched TR02 sources; one independently retained training run per ceiling."""
    return [
        configuration(seed=seed, rate_target_hz=t, **overrides) for t in RATE_TARGETS_HZ
    ]


def resolve_condition(cfg, overrides):
    if type(cfg["seed"]) is not int or cfg["seed"] not in SOURCE_SEEDS:
        raise ValueError("Source seed must be 42, 43 or 44")
    target = cfg["rate_target_hz"]
    if isinstance(target, bool) or target not in RATE_TARGETS_HZ:
        raise ValueError("Unsupported TR02 activity ceiling")
    strength = 0.0 if target is None else RATE_PENALTY_STRENGTH
    if "rate_penalty" in overrides and cfg["rate_penalty"] != strength:
        raise ValueError("Rate penalty must match the TR02 ceiling protocol")
    cfg["rate_penalty"] = strength
    for kind in ("coba", "ping"):
        cell = f"{kind}__{target_label(target)}__seed{cfg['seed']}"
        if kind + "_cell" in overrides and cfg[kind + "_cell"] != cell:
            raise ValueError("Source cell must match the ceiling and seed")
        cfg[kind + "_cell"] = cell


def configuration(**overrides):
    cfg = dict(
        source_run="exp022-r015-compute",
        source_digest="sha256:a542a714021fd7bb69c5d08ccf006763f19ce83b01d8ecdca86dba150118491d",
        coba_cell=None,
        ping_cell=None,
        seed=42,
        execution_backend="snnlab.sim.train",
        train_images=6300,
        validation_images=700,
        microbatch_size=None,
        sensory_seed=171001,
        encoding_method="torch_device_v1",
        calibration_seed=171002,
        teacher_statistics="per_image_conductance_and_i_rate_v1",
        shuffle_seed=171003,
        training_seed=171004,
        validation_seed=171005,
        epochs=10,
        batch_size=256,
        validation_draws=3,
        dt_ms=0.1,
        duration_ms=200.0,
        inhibition_method=POISSON_METHOD,
        target_events_per_source=None,
        timing_bisections=24,
        bracket_attempts=1000,
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
        calibration_std_tolerance=0.01,
        rate_penalty=0.0,
        rate_target_hz=None,
        device="cpu",
        threads=1,
        profile_encoding=False,
        compile_mode="auto",
        precision="auto",
        evaluation_schedule="validation_each_epoch_train_final_only",
        final_training_draw=0,
        replay_batch_size=16,
        calibration_batch_size=1,
        calibration_inflight_batches=2,
        checkpoint_role="final_epoch",
        refractory_timing="legacy_coba",
        inhibition="Independent Poisson counts per timestep and private E channel; per-image rates and constant amplitudes analytically fitted to frozen PING conductance mean and variance using the exact discrete exponential filter; fitted parameters frozen, fresh training draws and three fixed validation draws; 60-ms inhibition-only prehistory; fresh neurons at t=0; one-step delay; finite-trial moments fluctuate",
        primary_comparison="final_epoch versus epoch_zero; paired validation inputs",
        retention_estimate="<350 MB; compact dataset/calibration arrays, native snapshots and final/selected checkpoints; one illustrative trial",
        validation_limitation="Original validation partition already used for source checkpoint selection; no independent test-set claim",
    )
    if set(overrides) - set(cfg):
        raise ValueError("Unknown configuration override")
    cfg.update(overrides)
    resolve_condition(cfg, overrides)
    for key in (
        "train_images",
        "validation_images",
        "replay_batch_size",
        "epochs",
        "batch_size",
        "validation_draws",
        "calibration_batch_size",
        "calibration_inflight_batches",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"Invalid {key}")
    if cfg["device"] not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    if "calibration_batch_size" not in overrides:
        cfg["calibration_batch_size"] = 64 if cfg["device"] == "cuda" else 1
    resolve_execution(cfg)
    if cfg["train_images"] > 6300 or cfg["validation_images"] > 700:
        raise ValueError("Requested images exceed the pinned source partitions")
    history = cfg["inhibition_prehistory_ms"] / cfg["dt_ms"]
    if (
        not math.isfinite(history)
        or history < 0
        or not math.isclose(history, round(history), abs_tol=1e-9)
    ):
        raise ValueError(
            "Inhibitory prehistory must be a nonnegative integer number of timesteps"
        )
    if cfg["inhibition_method"] not in (*COUNT_METHODS, POISSON_METHOD):
        raise ValueError("Unsupported inhibition method")
    resolve_event_count(cfg, overrides)
    for key in ("timing_bisections", "bracket_attempts"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"Invalid {key}")
    if history > round(cfg["duration_ms"] / cfg["dt_ms"]):
        raise ValueError("Inhibitory prehistory exceeds its preceding timing window")
    cfg["inhibition_prehistory_steps"] = round(history)
    cfg["steps"] = round(cfg["duration_ms"] / cfg["dt_ms"])
    return cfg


def resolve_event_count(cfg, overrides):
    if cfg["inhibition_method"] == POISSON_METHOD:
        if cfg["target_events_per_source"] is not None:
            raise ValueError(
                "Poisson event counts fluctuate; a fixed count is incompatible"
            )
        return
    if cfg["inhibition_method"] == TEACHER_COUNT_METHOD:
        if cfg["target_events_per_source"] is not None:
            raise ValueError(
                "Teacher-rate counts must come from each image, not a fixed override"
            )
        return
    if "target_events_per_source" not in overrides:
        cfg["target_events_per_source"] = 24
    count = cfg["target_events_per_source"]
    if type(count) is not int or not 1 <= count < round(
        cfg["duration_ms"] / cfg["dt_ms"]
    ):
        raise ValueError("Invalid target_events_per_source")


def resolve_execution(cfg):
    if cfg["encoding_method"] not in {"numpy_v1", "torch_device_v1"}:
        raise ValueError("Invalid encoding_method")
    if cfg["microbatch_size"] is not None and (
        type(cfg["microbatch_size"]) is not int
        or not 1 <= cfg["microbatch_size"] <= cfg["batch_size"]
    ):
        raise ValueError("Invalid microbatch_size")
    if cfg["compile_mode"] == "auto":
        cfg["compile_mode"] = "default" if cfg["device"] == "cuda" else None
    if cfg["compile_mode"] not in (None, "default"):
        raise ValueError("Invalid compile_mode")
    if cfg["precision"] not in ("auto", "highest"):
        raise ValueError("Invalid precision")


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

    def projection(
        src,
        dst,
        name,
        tau,
        inhibitory=False,
        recurrent=False,
        delay=0.0,
        *,
        private=False,
    ):
        return net.connect(
            src,
            dst,
            name=name,
            synapse=(lang.GABA if inhibitory else lang.AMPA)(tau=tau * lang.ms),
            weight=lang.Constant(1.0 if private else 0.0),
            connectivity=lang.OneToOne() if private else None,
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
        net.output(
            "i_count",
            lang.reduce(
                pops["I"].spikes, operation="sum", over="time", name="I_counts"
            ),
        )
    else:
        # Events carry a fixed per-image amplitude; the frozen identity projection
        # keeps recipient j private without changing weights between batch samples.
        private = net.input(
            "artificial",
            shape=("time", "batch", cfg["n_e"]),
            signal_type="continuous",
            unit="1",
        )
        gaba = projection(
            private,
            pops["E"].inhibitory,
            "artificial_to_E",
            cfg["tau_gaba_ms"],
            True,
            delay=cfg["dt_ms"],
            private=True,
        )
        if record_spikes:
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
            regularizers=(
                (
                    lang.SpikeBudgetPenalty(
                        signals=(pops["E"].spikes,),
                        ceiling_hz=cfg["rate_target_hz"],
                        strength=cfg["rate_penalty"],
                    ),
                )
                if cfg["rate_penalty"]
                else ()
            ),
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
