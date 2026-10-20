"""Define the calibrated two-layer SHD classifiers and resolved settings.

COBA learns a feedforward excitatory path; PING adds fixed reciprocal inhibitory
feedback in each hidden layer. Both learn the same three matrices and classify
20 spoken digits using a rate-normalized, non-spiking readout. One-second binary
inputs retain the historical insertion/deletion corruption protocol.

configuration() resolves full or local-check settings without loading data.
author_network() returns a compiled SNNLab bundle, without running it. The few
registered extensions preserve scientific operations absent from the native API;
execution, optimization, recording and snapshots use native SNNLab facilities.
"""

from dataclasses import asdict

from experiments.exp124.custom import extensions  # noqa: F401
from snnlab import lang
from snnlab.sim import CheckpointSelection, Performance, TrainingMetrics


def configuration(
    protocol="full",
    *,
    epochs=None,
    seed=42,
    batch_size=None,
    train_class_counts=None,
    validation_class_counts=None,
):
    if protocol not in ("full", "local-check"):
        raise ValueError(f"unknown protocol: {protocol}")
    local = protocol == "local-check"
    epochs = (1 if local else 50) if epochs is None else epochs
    batch_size = (20 if local else 256) if batch_size is None else batch_size
    if any(type(value) is not int or value < 1 for value in (epochs, batch_size)):
        raise ValueError("epochs and batch size must be positive integers")
    if type(seed) is not int or not 0 <= seed < 2**32 - 4:
        raise ValueError("seed must permit nonnegative 32-bit noise seeds")
    # All scientific defaults and requested execution policies live here.
    cfg = dict(
        experiment="exp124",
        schema="exp124.recipe/v5",
        protocol=protocol,
        models=["coba", "ping"],
        epochs=epochs,
        seed=seed,
        batch_size=batch_size,
        full_dataset=not local,
        validation_fraction=0.1,
        n_input=700,
        n_excitatory=128,
        n_inhibitory=32,
        n_classes=20,
        hidden_layers=2,
        dt_ms=1.0,
        duration_ms=1000.0,
        input_noise_hz=1.6,
        spike_deletion_probability=0.0008,
        tau_mem_ms=20.0,
        tau_synapse_ms=10.0,
        capacitance_e_nf=1.0,
        capacitance_i_nf=0.5,
        leak_e_us=0.05,
        leak_i_us=0.025,
        resting_mv=-65.0,
        reset_mv=-65.0,
        threshold_mv=-50.0,
        refractory_ms=0.0,
        voltage_grad_dampen=1.0,
        hidden_weight_mean_us=0.09,
        hidden_weight_std_us=0.009,
        initial_zero_fractions=[0.95, 0.90],
        feedback_sum_us=0.05,
        readout_weight_mean_us=0.0,
        readout_weight_std_us=0.1,
        score_gain=20.0,
        reference_rate_hz=20.0,
        minimum_rate_hz=1.0,
        learning_rates=dict(input=0.000005, interlayer=0.00002, readout=0.01),
        # With zero decay, native AdamW implements the existing Adam update.
        optimizer=dict(lr=0.001, betas=[0.9, 0.999], eps=1e-8, weight_decay=0.0),
        surrogate_slope_per_mv=100.0 / 15.0,
        surrogate_gradient_scale=0.01,
        performance=asdict(Performance()),
        training_metrics=TrainingMetrics(test="none").to_dict(),
        batching=dict(
            order="stratified",
            stratify_by="class",
            remainder="merge_small",
            min_batch_size=min(20, batch_size),
        ),
        checkpoint_selection="final_epoch",
        best_checkpoint_selection=CheckpointSelection().to_dict(),
        training_measurement="online_pre_update_minibatches",
        sample_selection="first validation utterance in split order",
        sample_device="cpu",
        recording=dict(
            coba=["e1", "e2"],
            ping=["e1", "e2", "i1", "i2"],
        ),
        autocorrelation_bin_ms=1.0,
        autocorrelation_max_lag_ms=200.0,
    )
    if (train_class_counts is None) != (validation_class_counts is None):
        raise ValueError("training and validation counts must be supplied together")
    if train_class_counts is not None:
        for split, counts in (
            ("train", train_class_counts),
            ("validation", validation_class_counts),
        ):
            if len(counts) != cfg["n_classes"] or any(
                type(count) is not int or count < 1 for count in counts
            ):
                raise ValueError("each class needs a positive integer example count")
            cfg[f"{split}_class_counts"] = list(counts)
            cfg[f"{split}_examples"] = sum(counts)
    return cfg


def author_network(cfg, model):
    """The calibrated scientific recipe, shared by training and inference."""
    if model not in cfg["models"]:
        raise ValueError(model)
    net = lang.Network(f"{cfg['experiment']}_{model}", dt=cfg["dt_ms"] * lang.ms)
    source = net.input(
        "drive",
        shape=("time", "batch", cfg["n_input"]),
        signal_type="spikes",
        unit="spike",
    )
    learned, fixed = [], []
    for layer in range(1, cfg["hidden_layers"] + 1):
        e = net.population(
            f"E{layer}",
            size=cfg["n_excitatory"],
            neuron=lang.COBA_LIF(
                tau_mem=cfg["tau_mem_ms"] * lang.ms,
                capacitance_nf=cfg["capacitance_e_nf"],
                leak_us=cfg["leak_e_us"],
                resting_mv=cfg["resting_mv"],
                reset_mv=cfg["reset_mv"],
                threshold_mv=cfg["threshold_mv"],
                refractory=cfg["refractory_ms"] * lang.ms,
                voltage_grad_dampen=cfg["voltage_grad_dampen"],
            ),
        )
        projection = net.connect(
            source,
            e.excitatory,
            name="input_to_E1" if layer == 1 else "E1_to_E2",
            synapse=lang.AMPA(tau=cfg["tau_synapse_ms"] * lang.ms),
            weight=lang.LowerClampedNormal(
                mean=cfg["hidden_weight_mean_us"],
                std=cfg["hidden_weight_std_us"],
                initial_zero_fraction=cfg["initial_zero_fractions"][layer - 1],
            ),
            constraint=lang.NonNegative(),
            initialization_scaling="fan_in_normalized",
        )
        learned.append(projection.weight)
        if model == "ping":
            i = net.population(
                f"I{layer}",
                size=cfg["n_inhibitory"],
                neuron=lang.COBA_LIF(
                    tau_mem=cfg["tau_mem_ms"] * lang.ms,
                    capacitance_nf=cfg["capacitance_i_nf"],
                    leak_us=cfg["leak_i_us"],
                    resting_mv=cfg["resting_mv"],
                    reset_mv=cfg["reset_mv"],
                    threshold_mv=cfg["threshold_mv"],
                    refractory=cfg["refractory_ms"] * lang.ms,
                    voltage_grad_dampen=cfg["voltage_grad_dampen"],
                ),
            )
            for pre, target, name, synapse, weight in (
                (
                    e.spikes,
                    i.excitatory,
                    f"E{layer}_to_I{layer}",
                    lang.AMPA(tau=cfg["tau_synapse_ms"] * lang.ms),
                    cfg["feedback_sum_us"] / cfg["n_excitatory"],
                ),
                (
                    i.spikes,
                    e.inhibitory,
                    f"I{layer}_to_E{layer}",
                    lang.GABA(tau=cfg["tau_synapse_ms"] * lang.ms),
                    cfg["feedback_sum_us"] / cfg["n_inhibitory"],
                ),
            ):
                loop = net.connect(
                    pre,
                    target,
                    name=name,
                    synapse=synapse,
                    weight=lang.Constant(weight),
                    constraint=lang.NonNegative(),
                    connection="recurrent",
                    delay=cfg["dt_ms"] * lang.ms,
                    initialization_scaling="direct",
                )
                fixed.append(loop.weight)
        source = e.spikes
    readout = net.population(
        "readout",
        size=cfg["n_classes"],
        spiking=False,
        neuron=lang.LeakyIntegrator(tau=cfg["tau_mem_ms"] * lang.ms),
    )
    projection = net.connect(
        source,
        readout.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=cfg["tau_synapse_ms"] * lang.ms),
        weight=lang.CustomInitializer(
            "exp124.centered_readout/v1",
            mean=cfg["readout_weight_mean_us"],
            std=cfg["readout_weight_std_us"],
            center=True,
        ),
        initialization_scaling="direct",
    )
    scores = lang.reduce(readout.voltage, operation="mean", over="time", name="scores")
    gain = net.parameter(
        "score_gain.weight",
        shape=(cfg["n_classes"], cfg["n_classes"]),
        initializer=lang.CustomInitializer(
            "exp124.identity_gain/v1", gain=cfg["score_gain"]
        ),
    )
    scores = net.operation(
        "linear",
        scores,
        name="scaled_scores",
        shape=scores.shape,
        unit=scores.unit,
        signal_type="continuous",
        parameters=(gain,),
        size=cfg["n_classes"],
        trainable=False,
    )
    counts = lang.reduce(
        source, operation="sum", over="time", name="normalization_counts"
    )
    scores = lang.custom_operation(
        "exp124.rate_normalize/v1",
        [scores, counts],
        name="normalized_scores",
        shape=scores.shape,
        unit="1",
        steps=round(cfg["duration_ms"] / cfg["dt_ms"]),
        reference_rate=cfg["reference_rate_hz"] * cfg["dt_ms"] / 1000,
        minimum_rate=cfg["minimum_rate_hz"] * cfg["dt_ms"] / 1000,
    )
    net.output("class_scores", scores)
    training = lang.TrainSpec(
        objectives=(
            lang.CustomObjective(
                "exp124.equal_class_cross_entropy/v1",
                prediction=scores,
                target="class",
                classes=cfg["n_classes"],
            ),
        ),
        parameter_groups=(
            lang.ParameterGroup(
                (learned[0],), name="input", lr=cfg["learning_rates"]["input"]
            ),
            lang.ParameterGroup(
                (learned[1],), name="interlayer", lr=cfg["learning_rates"]["interlayer"]
            ),
            lang.ParameterGroup(
                (projection.weight,),
                name="readout",
                lr=cfg["learning_rates"]["readout"],
            ),
            lang.ParameterGroup(
                tuple(fixed) + (gain,), name="fixed_loop", lr=0.0, frozen=True
            ),
        ),
        optimizer=lang.AdamW(**cfg["optimizer"]),
        surrogate=lang.FastSigmoid(
            slope=cfg["surrogate_slope_per_mv"],
            gradient_scale=cfg["surrogate_gradient_scale"],
            threshold_inclusive=False,
        ),
        epochs=cfg["epochs"],
        presentation_duration=cfg["duration_ms"] * lang.ms,
    )
    return lang.compile(net, training=training, target="tools/snnsim")
