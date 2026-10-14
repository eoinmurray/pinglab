"""Directly declared two-layer SHD networks; no simulation on import."""

from snnlab import extensions, lang
from snnlab.lang import training


def rate_normalized_scores(sources, parameters, config):
    """Rescale each presentation by its mean final-layer spike rate."""
    scores, counts = sources
    rate = counts.mean(dim=-1, keepdim=True) / config["steps"]
    return scores * config["reference_rate"] / rate.clamp_min(config["minimum_rate"])


def maximum_scores(sources, parameters, config):
    """Reduce the complete time-major voltage history to class scores."""
    return sources[0].amax(dim=0)


extensions.register_operation("exp124.rate_normalize/v1", rate_normalized_scores)
extensions.register_operation("exp124.temporal_max/v1", maximum_scores)

SLUG = "exp124"
MODELS = ("coba", "ping")


TRAINABLE = ("input_to_E1.weight", "E1_to_E2.weight", "readout_projection.weight")


def configuration(
    train_examples=200, validation_examples=100, epochs=5, variant="tuned",
    seed=42, batch_size=None, learning_rate=None, readout_gain=None, damping=None,
    readout_learning_rate=None, balanced_batches=None, center_readout=None,
    interlayer_strength=None, normalize_readout=None, input_strength=None, models=None,
):
    models = tuple(MODELS if models is None else models)
    if not models or len(set(models)) != len(models) or any(model not in MODELS for model in models):
        raise ValueError("select unique supported models")
    tuned = variant == "tuned"
    batch_size = (20 if tuned else 16) if batch_size is None else batch_size
    balanced_batches = tuned if balanced_batches is None else balanced_batches
    center_readout = tuned if center_readout is None else center_readout
    normalize_readout = tuned if normalize_readout is None else normalize_readout
    if any(value < 1 for value in (train_examples, validation_examples, epochs)):
        raise ValueError("example counts and epochs must be positive")
    if train_examples % 20 or validation_examples % 20:
        raise ValueError("class-balanced example counts must be multiples of 20")
    if variant not in ("baseline", "diverse", "damping", "sum", "gain", "combined", "maximum", "tuned"):
        raise ValueError("unsupported training variant")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if balanced_batches and batch_size % 20:
        raise ValueError("balanced batches require batch size divisible by 20")
    train_per_class, validation_per_class = train_examples // 20, validation_examples // 20
    cfg = {
        "schema": "exp124.recipe/v2", "dataset": "shd", "seed": seed,
        "train_per_class": train_per_class, "validation_per_class": validation_per_class, "classes": 20,
        "n_input": 700, "n_e": 128, "n_i": 32, "epochs": epochs, "batch_size": batch_size,
        "dt_ms": 1.0, "duration_ms": 1000.0, "learning_rate": 0.0004,
        "weight_decay": 0.0, "gradient_clip": 1.0,
        "voltage_gradient_damping": 1000.0, "surrogate_slope": 1.0,
        "tau_ampa_ms": 2.0, "tau_gaba_ms": 6.0,
        "refractory_e_ms": 2.0, "refractory_i_ms": 1.0,
        "input_weight_mean": 0.9, "input_weight_sd": 0.09,
        "input_initial_zero_fraction": 0.95,
        "ei_summed_weight_us": 1.0, "ie_summed_weight_us": 2.0,
        "readout_weight_mean": 0.0, "readout_weight_sd": 0.1,
        "readout_tau_ms": 20.0,
        "readout": "non-spiking leaky integrator; mean voltage over presentation",
        "binning": "floor; binary OR; discard events at or after 1 s",
        "split": f"seeded {train_per_class + validation_per_class} per class from official train; first {train_per_class} train, last {validation_per_class} validation",
        "official_test_used": False, "hidden_state": "reset for every presentation",
        "recurrent_weights": "fixed; loop off for COBA, on for PING",
        "checkpoint": "final epoch; no test-based selection", "models": list(models),
    }
    cfg.update(hidden_layers=2,
               interlayer_weight_mean=1.8, interlayer_weight_sd=0.18,
               interlayer_initial_zero_fraction=0.0,
               rate_measurement="e/i aliases are final-layer rates; e1/i1/e2/i2 are per-layer rates")
    cfg.update(variant=variant, readout_reduction="mean", readout_gain=1.0,
               balanced_batches=balanced_batches,
               batch_order="equal examples per class, without replacement" if balanced_batches else "shuffled examples without replacement")
    if variant in ("diverse", "combined", "maximum", "tuned"):
        cfg["interlayer_initial_zero_fraction"] = 0.9
    if variant == "damping":
        cfg["voltage_gradient_damping"] = 1.0
    if variant in ("combined", "maximum", "tuned"):
        cfg["voltage_gradient_damping"] = 100.0
    if variant == "sum":
        cfg["readout_reduction"] = "sum"
    if variant in ("gain", "combined", "tuned"):
        cfg["readout_gain"] = 20.0
    if variant == "maximum":
        cfg.update(readout_reduction="max", readout_gain=20.0)
    if learning_rate is not None:
        cfg["learning_rate"] = learning_rate
    if readout_gain is not None:
        cfg["readout_gain"] = readout_gain
    if damping is not None:
        cfg["voltage_gradient_damping"] = damping
    if any(cfg[key] <= 0 for key in ("learning_rate", "readout_gain", "voltage_gradient_damping")):
        raise ValueError("learning rate, readout gain and damping must be positive")
    cfg.update(normalize_readout=normalize_readout, readout_reference_rate_per_step=0.02, readout_minimum_rate_per_step=0.001)
    cfg["center_readout_initialization"] = center_readout
    if interlayer_strength is not None:
        if interlayer_strength <= 0:
            raise ValueError("interlayer strength must be positive")
        cfg["interlayer_weight_mean"] = interlayer_strength
        cfg["interlayer_weight_sd"] = interlayer_strength / 10
    if input_strength is not None:
        if input_strength <= 0:
            raise ValueError("input strength must be positive")
        cfg["input_weight_mean"] = input_strength
        cfg["input_weight_sd"] = input_strength / 10
    cfg["readout_learning_rate"] = (0.01 if tuned else cfg["learning_rate"]) if readout_learning_rate is None else readout_learning_rate
    if cfg["readout_learning_rate"] <= 0:
        raise ValueError("readout learning rate must be positive")
    cfg["readout"] = f"non-spiking leaky integrator; {cfg['readout_reduction']} voltage over presentation"
    return cfg

def author_network(cfg, model):
    if cfg.get("schema") != "exp124.recipe/v2" or cfg.get("hidden_layers") != 2:
        raise ValueError("exp124 authors only the two-layer recipe")
    if model not in MODELS:
        raise ValueError(model)
    net = lang.Network(f"{SLUG}_{model}", dt=cfg["dt_ms"] * lang.ms)
    drive = net.input("drive", shape=("time", "batch", cfg["n_input"]),
                     signal_type="spikes", unit="spike")
    neurons = {}
    for label, tau, capacitance, leak in (("e", 20, 1.0, 0.05), ("i", 5, 0.5, 0.1)):
        neurons[label] = lang.COBA_LIF(
            tau_mem=tau * lang.ms, capacitance_nf=capacitance, leak_us=leak,
            resting_mv=-65.0, reset_mv=-65.0, threshold_mv=-50.0,
            refractory=cfg[f"refractory_{label}_ms"] * lang.ms,
            voltage_grad_dampen=cfg["voltage_gradient_damping"],
        )
    source = drive
    learned, fixed, populations = [], [], []
    for index in (1, 2):
        suffix = str(index)
        e = net.population(f"E{suffix}", size=cfg["n_e"], neuron=neurons["e"])
        i = net.population(f"I{suffix}", size=cfg["n_i"], neuron=neurons["i"])
        projection = net.connect(
            source, e.excitatory,
            name=f"input_to_E{suffix}" if index == 1 else "E1_to_E2",
            synapse=lang.AMPA(tau=cfg["tau_ampa_ms"] * lang.ms),
            weight=lang.Constant(0), constraint=lang.NonNegative(),
            initialization_scaling="direct")
        learned.append(projection.weight)
        for pre, target, name, synapse in (
            (e.spikes, i.excitatory, f"E{suffix}_to_I{suffix}", lang.AMPA(tau=cfg["tau_ampa_ms"] * lang.ms)),
            (i.spikes, e.inhibitory, f"I{suffix}_to_E{suffix}", lang.GABA(tau=cfg["tau_gaba_ms"] * lang.ms)),
        ):
            loop = net.connect(pre, target, name=name, synapse=synapse,
                               weight=lang.Constant(0), constraint=lang.NonNegative(),
                               connection="recurrent", delay=cfg["dt_ms"] * lang.ms,
                               initialization_scaling="direct")
            fixed.append(loop.weight)
        populations.append((e, i))
        source = e.spikes
    readout = net.population("readout", size=cfg["classes"], spiking=False,
                            neuron=lang.LeakyIntegrator(tau=cfg["readout_tau_ms"] * lang.ms))
    projection = net.connect(
        source, readout.excitatory, name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=cfg["readout_tau_ms"] * lang.ms),
        weight=lang.Constant(0), initialization_scaling="direct",
    )
    if cfg.get("readout_reduction") == "max":
        scores = lang.ops.custom("exp124.temporal_max/v1", readout.voltage, name="scores",
                                shape=("batch", cfg["classes"]), unit=readout.voltage.unit)
    else:
        scores = lang.ops.reduce(readout.voltage, operation=cfg.get("readout_reduction", "mean"), over="time", name="scores")
    if cfg.get("readout_gain", 1.0) != 1.0:
        gain = net.parameter("score_gain.weight", shape=(cfg["classes"], cfg["classes"]), initializer=lang.Constant(0))
        scores = net.operation("linear", scores, name="scaled_scores", shape=scores.shape,
                               unit=scores.unit, signal_type="continuous", parameters=(gain,),
                               size=cfg["classes"], trainable=False)
        fixed.append(gain)
    if cfg.get("normalize_readout", False):
        counts = lang.ops.reduce(populations[-1][0].spikes, operation="sum", over="time", name="normalization_counts")
        scores = lang.ops.custom("exp124.rate_normalize/v1", [scores, counts], name="normalized_scores",
                                shape=scores.shape, unit="1", steps=round(cfg["duration_ms"] / cfg["dt_ms"]),
                                reference_rate=cfg["readout_reference_rate_per_step"],
                                minimum_rate=cfg["readout_minimum_rate_per_step"])
    net.output("class_scores", scores)
    for index, (e, i) in enumerate(populations, 1):
        for label, population in (("e", e), ("i", i)):
            net.output(f"{label}{index}_counts", lang.ops.reduce(
                population.spikes, operation="sum", over="time", name=f"{label}{index}_count_reduction"))
            if index == 2:
                net.output(f"{label}_counts", lang.ops.reduce(
                    population.spikes, operation="sum", over="time", name=f"{label}_count_reduction"))
    spec = lang.TrainSpec(
        objectives=(training.CrossEntropy(prediction=scores, target="class"),),
        parameter_groups=(training.ParameterGroup(
            tuple(learned), name="hidden", lr=cfg["learning_rate"]),
            training.ParameterGroup((projection.weight,), name="readout", lr=cfg["readout_learning_rate"]),
            training.ParameterGroup(tuple(fixed),
                                    name="fixed_loop", lr=0.0, frozen=True)),
        optimizer=training.AdamW(weight_decay=cfg["weight_decay"]),
        surrogate=training.FastSigmoid(slope=cfg["surrogate_slope"]),
        gradient_clip=cfg["gradient_clip"],
        presentation_duration=cfg["duration_ms"] * lang.ms,
    )
    return lang.compile(net, training=spec, target="tools/snnsim")
