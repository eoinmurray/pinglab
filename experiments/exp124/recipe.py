"""Directly declared two-layer SHD networks; no simulation on import."""

from snnlab import extensions, lang


def equal_class_loss(losses, target, classes):
    import torch

    counts = torch.zeros(classes, device=losses.device, dtype=losses.dtype)
    counts.scatter_add_(0, target, torch.ones_like(target, dtype=losses.dtype))
    return (losses / counts[target]).sum() / (counts > 0).sum()


def class_balanced_cross_entropy(prediction, target, config):
    import torch

    losses = torch.nn.functional.cross_entropy(
        prediction, target.long(), reduction="none"
    )
    return equal_class_loss(losses, target.long(), config["classes"])


extensions.register_objective(
    "exp124.equal_class_cross_entropy/v1",
    class_balanced_cross_entropy,
    classification=True,
)


def rate_normalized_scores(sources, parameters, config):
    """Rescale each presentation by its mean final-layer spike rate."""
    scores, counts = sources
    rate = counts.mean(dim=-1, keepdim=True) / config["steps"]
    return scores * config["reference_rate"] / rate.clamp_min(config["minimum_rate"])


extensions.register_operation("exp124.rate_normalize/v1", rate_normalized_scores)


def adam_optimizer(groups, config):
    import torch

    return torch.optim.Adam(groups, **config)


extensions.register_optimizer("exp124.adam/v1", adam_optimizer)


def centered_readout(shape, config, *, device, dtype):
    import torch

    value = (
        torch.randn(shape, device=device, dtype=dtype) * config["std"] + config["mean"]
    )
    return value - value.mean(dim=0, keepdim=True) if config["center"] else value


def identity_gain(shape, config, *, device, dtype):
    import torch

    return torch.eye(*shape, device=device, dtype=dtype) * config["gain"]


extensions.register_initializer("exp124.centered_readout/v1", centered_readout)
extensions.register_initializer("exp124.identity_gain/v1", identity_gain)


SLUG = "exp124"
MODELS = ("coba", "ping")
TRAINABLE = ("input_to_E1.weight", "E1_to_E2.weight", "readout_projection.weight")
POPULATIONS = {"coba": ("e1", "e2"), "ping": ("e1", "e2", "i1", "i2")}


def configuration(*, epochs=50, seed=42, batch_size=256, smoke=False):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    return {
        "schema": "exp124.recipe/v4",
        "models": list(MODELS),
        "epochs": epochs,
        "seed": seed,
        "batch_size": batch_size,
        "full_dataset": not smoke,
        "validation_fraction": 0.1,
        "dt_ms": 1.0,
        "duration_ms": 1000.0,
        "input_noise_hz": 1.6,
        "spike_deletion_probability": 0.0008,
        "training_measurement": "online_pre_update_minibatches",
    }


def author_network(cfg, model):
    """The calibrated scientific recipe, shared by training and inference."""
    if model not in MODELS:
        raise ValueError(model)
    net = lang.Network(f"{SLUG}_{model}", dt=cfg["dt_ms"] * lang.ms)
    source = net.input(
        "drive", shape=("time", "batch", 700), signal_type="spikes", unit="spike"
    )
    learned, fixed = [], []
    for layer in (1, 2):
        e = net.population(
            f"E{layer}",
            size=128,
            neuron=lang.COBA_LIF(
                tau_mem=20 * lang.ms,
                capacitance_nf=1.0,
                leak_us=0.05,
                resting_mv=-65.0,
                reset_mv=-65.0,
                threshold_mv=-50.0,
                refractory=0 * lang.ms,
                voltage_grad_dampen=1.0,
            ),
        )
        projection = net.connect(
            source,
            e.excitatory,
            name="input_to_E1" if layer == 1 else "E1_to_E2",
            synapse=lang.AMPA(tau=10 * lang.ms),
            weight=lang.LowerClampedNormal(
                mean=0.09, std=0.009, initial_zero_fraction=0.95 if layer == 1 else 0.90
            ),
            constraint=lang.NonNegative(),
            initialization_scaling="fan_in_normalized",
        )
        learned.append(projection.weight)
        if model == "ping":
            i = net.population(
                f"I{layer}",
                size=32,
                neuron=lang.COBA_LIF(
                    tau_mem=20 * lang.ms,
                    capacitance_nf=0.5,
                    leak_us=0.025,
                    resting_mv=-65.0,
                    reset_mv=-65.0,
                    threshold_mv=-50.0,
                    refractory=0 * lang.ms,
                    voltage_grad_dampen=1.0,
                ),
            )
            for pre, target, name, synapse, weight in (
                (
                    e.spikes,
                    i.excitatory,
                    f"E{layer}_to_I{layer}",
                    lang.AMPA(tau=10 * lang.ms),
                    0.05 / 128,
                ),
                (
                    i.spikes,
                    e.inhibitory,
                    f"I{layer}_to_E{layer}",
                    lang.GABA(tau=10 * lang.ms),
                    0.05 / 32,
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
        "readout", size=20, spiking=False, neuron=lang.LeakyIntegrator(tau=20 * lang.ms)
    )
    projection = net.connect(
        source,
        readout.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=10 * lang.ms),
        weight=lang.CustomInitializer(
            "exp124.centered_readout/v1", mean=0.0, std=0.1, center=True
        ),
        initialization_scaling="direct",
    )
    scores = lang.reduce(readout.voltage, operation="mean", over="time", name="scores")
    gain = net.parameter(
        "score_gain.weight",
        shape=(20, 20),
        initializer=lang.CustomInitializer("exp124.identity_gain/v1", gain=20.0),
    )
    scores = net.operation(
        "linear",
        scores,
        name="scaled_scores",
        shape=scores.shape,
        unit=scores.unit,
        signal_type="continuous",
        parameters=(gain,),
        size=20,
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
        reference_rate=0.02,
        minimum_rate=0.001,
    )
    net.output("class_scores", scores)
    training = lang.TrainSpec(
        objectives=(
            lang.CustomObjective(
                "exp124.equal_class_cross_entropy/v1",
                prediction=scores,
                target="class",
                classes=20,
            ),
        ),
        parameter_groups=(
            lang.ParameterGroup((learned[0],), name="input", lr=0.000005),
            lang.ParameterGroup((learned[1],), name="interlayer", lr=0.00002),
            lang.ParameterGroup((projection.weight,), name="readout", lr=0.01),
            lang.ParameterGroup(
                tuple(fixed) + (gain,), name="fixed_loop", lr=0.0, frozen=True
            ),
        ),
        optimizer=lang.CustomOptimizer(
            "exp124.adam/v1", lr=0.001, betas=[0.9, 0.999], weight_decay=0.0
        ),
        surrogate=lang.FastSigmoid(
            slope=100.0 / 15.0, gradient_scale=0.01, threshold_inclusive=False
        ),
        epochs=cfg["epochs"],
        presentation_duration=cfg["duration_ms"] * lang.ms,
    )
    return lang.compile(net, training=training, target="tools/snnsim")
