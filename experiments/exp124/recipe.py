"""Directly declared two-layer SHD networks; no simulation on import."""

from snnlab import extensions, lang


def equal_class_loss(losses, target, classes):
    import torch
    counts = torch.zeros(classes, device=losses.device, dtype=losses.dtype)
    counts.scatter_add_(0, target, torch.ones_like(target, dtype=losses.dtype))
    return (losses / counts[target]).sum() / (counts > 0).sum()


def class_balanced_cross_entropy(prediction, target, config):
    import torch
    losses = torch.nn.functional.cross_entropy(prediction, target.long(), reduction="none")
    return equal_class_loss(losses, target.long(), config["classes"])


extensions.register_objective("exp124.equal_class_cross_entropy/v1", class_balanced_cross_entropy,
                              classification=True)


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


def paper_surrogate(voltage, config):
    """Paper derivative in threshold-normalized voltage, converted to mV."""
    gap = config["voltage_scale_mv"]
    return 1.0 / gap / (1 + config["beta"] * (voltage / gap).abs()).square()


def adam_optimizer(groups, config):
    import torch
    return torch.optim.Adam(groups, **config)


extensions.register_surrogate("exp124.paper_fast_sigmoid/v1", paper_surrogate)
extensions.register_optimizer("exp124.adam/v1", adam_optimizer)

def centered_readout(shape, config, *, device, dtype):
    import torch
    value = torch.randn(shape, device=device, dtype=dtype) * config["std"] + config["mean"]
    return value - value.mean(dim=0, keepdim=True) if config["center"] else value


def identity_gain(shape, config, *, device, dtype):
    import torch
    return torch.eye(*shape, device=device, dtype=dtype) * config["gain"]


extensions.register_initializer("exp124.centered_readout/v1", centered_readout)
extensions.register_initializer("exp124.identity_gain/v1", identity_gain)


SLUG = "exp124"
MODELS = ("coba", "ping")


TRAINABLE = ("input_to_E1.weight", "E1_to_E2.weight", "readout_projection.weight")
CALIBRATION_CASES = (
    {"input": value, "interlayer": 0.09, "ei": 0.05, "ie": 0.05}
    for value in (0.03, 0.06, 0.09, 0.12)
)
CALIBRATION_CASES = tuple(CALIBRATION_CASES) + (
    {"input": 0.09, "interlayer": 0.03, "ei": 0.05, "ie": 0.05},
    {"input": 0.09, "interlayer": 0.3, "ei": 0.05, "ie": 0.05},
)
PHASES = {"operating-point": {
    "inputs": {"training_examples": "explicit completed full SHD compute run"},
    "required_exports": ["operating_points.json"], "optimizer_updates": 0,
    "examples": "one source training utterance per class; no validation utterances",
}, "sample-raster": {
    "inputs": {"coba_weights": "completed full-data COBA training", "ping_weights": "completed full-data PING training"},
    "required_exports": ["rasters.npz"], "optimizer_updates": 0,
    "examples": "first validation utterance in the paired training sources; one matched fresh corruption draw",
}, "gamma-probe": {
    "inputs": {"trained_network": "explicit completed SHD compute run with final weights"},
    "required_exports": ["rasters.npz"], "optimizer_updates": 0,
    "examples": "first two validation utterances per class, in source order",
    "conditions": "trained network; PING additionally with I-to-E feedback disabled at fixed learned weights",
    "analysis": "30-90 Hz population spectra, within-trial autocorrelation, 25 ms within-neuron spike-shuffle reference",
}}


def configuration(
    train_examples=200, validation_examples=100, epochs=5, variant="calibrated",
    seed=42, batch_size=None, learning_rate=None, readout_gain=None, damping=None,
    readout_learning_rate=None, balanced_batches=None, center_readout=None,
    interlayer_strength=None, normalize_readout=None, input_strength=None, models=None,
    full_dataset=False, validation_fraction=0.1,
    compile_mode=None,
):
    models = tuple(MODELS if models is None else models)
    if not models or len(set(models)) != len(models) or any(model not in MODELS for model in models):
        raise ValueError("select unique supported models")
    tuned = variant == "tuned"
    batch_size = (256 if variant in ("paper", "calibrated") else 20 if tuned else 16) if batch_size is None else batch_size
    balanced_batches = tuned if balanced_batches is None else balanced_batches
    center_readout = (tuned or variant == "calibrated") if center_readout is None else center_readout
    normalize_readout = (tuned or variant == "calibrated") if normalize_readout is None else normalize_readout
    if any(value < 1 for value in (train_examples, validation_examples, epochs)):
        raise ValueError("example counts and epochs must be positive")
    if train_examples % 20 or validation_examples % 20:
        raise ValueError("class-balanced example counts must be multiples of 20")
    if variant not in ("baseline", "diverse", "damping", "sum", "gain", "combined", "maximum", "tuned", "paper", "calibrated"):
        raise ValueError("unsupported training variant")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if full_dataset and balanced_batches:
        raise ValueError("full dataset requires shuffled batches to retain uneven class counts")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if balanced_batches and batch_size % 20:
        raise ValueError("balanced batches require batch size divisible by 20")
    train_per_class, validation_per_class = train_examples // 20, validation_examples // 20
    cfg = {
        "schema": "exp124.recipe/v2", "dataset": "shd", "seed": seed,
        "train_per_class": train_per_class, "validation_per_class": validation_per_class, "classes": 20,
        "n_input": 700, "n_e": 128, "n_i": 32, "epochs": epochs, "batch_size": batch_size,
        "train_examples": train_examples, "validation_examples": validation_examples,
        "full_dataset": full_dataset, "validation_fraction": validation_fraction,
        "optimizer": "adamw", "optimizer_betas": [0.9, 0.999],
        "tau_mem_e_ms": 20.0, "tau_mem_i_ms": 5.0,
        "input_noise_hz": 0.0, "spike_deletion_probability": 0.0,
        "firing_rate_regularization": False,
        "training_measurement": "full_split_evaluation",
        "class_stratified_batches": False, "class_balanced_loss": False,
        "merge_short_training_tail": False,
        "projection_mode": "fast", "extension_validation": "structural",
        "compile_mode": None,
        "forward_retention": {"retain_parameters": False, "retain_final_state": False,
                              "retain_runtime_state": False},
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
    if variant in ("paper", "calibrated"):
        cfg.update(schema="exp124.recipe/v3", dt_ms=1.0,
                   tau_ampa_ms=10.0, tau_gaba_ms=10.0,
                   tau_mem_e_ms=20.0, tau_mem_i_ms=20.0,
                   refractory_e_ms=0.0, refractory_i_ms=0.0,
                   learning_rate=0.001, voltage_gradient_damping=1.0,
                   surrogate_slope=100.0, surrogate_voltage_scale_mv=15.0,
                   surrogate_gradient_scale=0.01, surrogate_threshold_inclusive=False,
                   surrogate_implementation="builtin", compile_mode="default",
                   readout_reduction="max", readout_gain=1.0,
                   optimizer="adam", gradient_clip=None,
                   input_noise_hz=1.6, spike_deletion_probability=0.0008,
                   training_measurement="online_pre_update_minibatches",
                   gradient_measurement="none; no per-batch host diagnostic reads",
                   architecture="COBA: 128 neurons per layer; PING: 128 E + 32 I per layer",
                   population_sizes={"coba": {"e": 128, "i": 0}, "ping": {"e": 128, "i": 32}},
                   paper_reference="https://arxiv.org/pdf/1910.07407v1",
                   paper_adaptations="conductance neurons, nonnegative hidden weights, retained conductance initialization and fixed PING loops; batch size 256 and 1 ms steps match GPU timing benchmarks, not the original 2019 curve; stratified 10% validation is an explicit choice")
    cfg.update(input_learning_rate=cfg["learning_rate"], interlayer_learning_rate=cfg["learning_rate"])
    if variant == "calibrated":
        cfg.update(input_weight_mean=0.09, input_weight_sd=0.009,
                   interlayer_weight_mean=0.09, interlayer_weight_sd=0.009,
                   interlayer_initial_zero_fraction=0.9,
                   ei_summed_weight_us=0.05, ie_summed_weight_us=0.05,
                   input_learning_rate=0.000005, interlayer_learning_rate=0.00002,
                   readout_reduction="mean", readout_gain=20.0,
                   class_stratified_batches=True, class_balanced_loss=True,
                   merge_short_training_tail=True,
                   batch_order="class-stratified order without replacement; merge tails smaller than 20 examples",
                   optimization_objective="mean cross entropy per represented class; reported loss is example mean",
                   calibration_basis="operating-point phase: shared sparse conductances and weaker fixed PING loops",
                   paper_adaptations="calibrated conductance scales, sparse interlayer, centered rate-normalized mean readout, class-stratified batching/equal-class objective and separate physical-weight learning rates; not a paper reproduction")
    if learning_rate is not None:
        cfg["learning_rate"] = learning_rate
        cfg["input_learning_rate"] = cfg["interlayer_learning_rate"] = learning_rate
    if compile_mode is not None:
        if compile_mode not in ("default", "eager"):
            raise ValueError("compile_mode must be default or eager")
        cfg["compile_mode"] = None if compile_mode == "eager" else compile_mode
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
    cfg["readout_learning_rate"] = (0.01 if tuned or variant == "calibrated" else cfg["learning_rate"]) if readout_learning_rate is None else readout_learning_rate
    if cfg["readout_learning_rate"] <= 0:
        raise ValueError("readout learning rate must be positive")
    cfg["readout"] = f"non-spiking leaky integrator; {cfg['readout_reduction']} voltage over presentation"
    if full_dataset:
        cfg.update(train_per_class=None, validation_per_class=None,
                   train_examples=None, validation_examples=None,
                   split=f"seeded per-class {validation_fraction:.0%} holdout from official train; all remaining utterances train")
    cfg.update(training_engine="snnlab.sim.train", initialization_implementation="native",
               encoding_seed_policy="SNNLab indexed phase/epoch/draw hashing", training_timing=True,
               activity_measurement="separate inference; native online metrics do not support split population-rate observations")
    return cfg

def author_network(cfg, model):
    if cfg.get("schema") not in ("exp124.recipe/v2", "exp124.recipe/v3") or cfg.get("hidden_layers") != 2:
        raise ValueError("exp124 authors only the two-layer recipe")
    if model not in MODELS:
        raise ValueError(model)
    net = lang.Network(f"{SLUG}_{model}", dt=cfg["dt_ms"] * lang.ms)
    drive = net.input("drive", shape=("time", "batch", cfg["n_input"]),
                     signal_type="spikes", unit="spike")
    neurons = {}
    for label, capacitance in (("e", 1.0), ("i", 0.5)):
        tau = cfg[f"tau_mem_{label}_ms"]
        leak = capacitance / tau
        neurons[label] = lang.COBA_LIF(
            tau_mem=tau * lang.ms, capacitance_nf=capacitance, leak_us=leak,
            resting_mv=-65.0, reset_mv=-65.0, threshold_mv=-50.0,
            refractory=cfg[f"refractory_{label}_ms"] * lang.ms,
            voltage_grad_dampen=cfg["voltage_gradient_damping"],
        )
    native = cfg.get("initialization_implementation") == "native"
    source = drive
    learned, fixed, populations = [], [], []
    for index in (1, 2):
        suffix = str(index)
        e = net.population(f"E{suffix}", size=cfg["n_e"], neuron=neurons["e"])
        i = (net.population(f"I{suffix}", size=cfg["n_i"], neuron=neurons["i"])
             if model == "ping" or cfg["schema"] == "exp124.recipe/v2" else None)
        projection = net.connect(
            source, e.excitatory,
            name=f"input_to_E{suffix}" if index == 1 else "E1_to_E2",
            synapse=lang.AMPA(tau=cfg["tau_ampa_ms"] * lang.ms),
            weight=(lang.LowerClampedNormal(
                mean=cfg["input_weight_mean" if index == 1 else "interlayer_weight_mean"],
                std=cfg["input_weight_sd" if index == 1 else "interlayer_weight_sd"],
                initial_zero_fraction=cfg["input_initial_zero_fraction" if index == 1 else "interlayer_initial_zero_fraction"])
                if native else lang.Constant(0)), constraint=lang.NonNegative(),
            initialization_scaling="fan_in_normalized" if native else "direct")
        learned.append(projection.weight)
        for pre, target, name, synapse in (() if i is None else (
            (e.spikes, i.excitatory, f"E{suffix}_to_I{suffix}", lang.AMPA(tau=cfg["tau_ampa_ms"] * lang.ms)),
            (i.spikes, e.inhibitory, f"I{suffix}_to_E{suffix}", lang.GABA(tau=cfg["tau_gaba_ms"] * lang.ms)),
        )):
            loop = net.connect(pre, target, name=name, synapse=synapse,
                               weight=lang.Constant((cfg["ei_summed_weight_us"] / cfg["n_e"]
                                                     if name.startswith("E") else cfg["ie_summed_weight_us"] / cfg["n_i"])
                                                    if native and model == "ping" else 0), constraint=lang.NonNegative(),
                               connection="recurrent", delay=cfg["dt_ms"] * lang.ms,
                               initialization_scaling="direct")
            fixed.append(loop.weight)
        populations.append((e, i))
        source = e.spikes
    readout = net.population("readout", size=cfg["classes"], spiking=False,
                            neuron=lang.LeakyIntegrator(tau=cfg["readout_tau_ms"] * lang.ms))
    projection = net.connect(
        source, readout.excitatory, name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=(cfg["tau_ampa_ms"] if cfg["variant"] in ("paper", "calibrated")
                                        else cfg["readout_tau_ms"]) * lang.ms),
        weight=(lang.CustomInitializer("exp124.centered_readout/v1", mean=cfg["readout_weight_mean"],
                                      std=cfg["readout_weight_sd"], center=cfg["center_readout_initialization"])
                if native else lang.Constant(0)), initialization_scaling="direct",
    )
    if cfg.get("readout_reduction") == "max":
        scores = lang.custom_operation("exp124.temporal_max/v1", readout.voltage, name="scores",
                                shape=("batch", cfg["classes"]), unit=readout.voltage.unit)
    else:
        scores = lang.reduce(readout.voltage, operation=cfg.get("readout_reduction", "mean"), over="time", name="scores")
    if cfg.get("readout_gain", 1.0) != 1.0:
        gain = net.parameter("score_gain.weight", shape=(cfg["classes"], cfg["classes"]), initializer=(lang.CustomInitializer("exp124.identity_gain/v1", gain=cfg["readout_gain"]) if native else lang.Constant(0)))
        scores = net.operation("linear", scores, name="scaled_scores", shape=scores.shape,
                               unit=scores.unit, signal_type="continuous", parameters=(gain,),
                               size=cfg["classes"], trainable=False)
        fixed.append(gain)
    if cfg.get("normalize_readout", False):
        counts = lang.reduce(populations[-1][0].spikes, operation="sum", over="time", name="normalization_counts")
        scores = lang.custom_operation("exp124.rate_normalize/v1", [scores, counts], name="normalized_scores",
                                shape=scores.shape, unit="1", steps=round(cfg["duration_ms"] / cfg["dt_ms"]),
                                reference_rate=cfg["readout_reference_rate_per_step"],
                                minimum_rate=cfg["readout_minimum_rate_per_step"])
    net.output("class_scores", scores)
    for index, (e, i) in enumerate(populations, 1):
        for label, population in (("e", e), ("i", i)):
            if population is None:
                continue
            if cfg.get("record_spikes", False):
                net.output(f"{label}{index}_spikes", population.spikes)
            net.output(f"{label}{index}_counts", lang.reduce(
                population.spikes, operation="sum", over="time", name=f"{label}{index}_count_reduction"))
            if index == 2:
                net.output(f"{label}_counts", lang.reduce(
                    population.spikes, operation="sum", over="time", name=f"{label}_count_reduction"))
    spec = lang.TrainSpec(
        objectives=((lang.CustomObjective("exp124.equal_class_cross_entropy/v1", prediction=scores,
                     target="class", classes=cfg["classes"]) if cfg.get("class_balanced_loss", False)
                     else lang.CrossEntropy(prediction=scores, target="class")),),
        parameter_groups=(lang.ParameterGroup(
            (learned[0],), name="input", lr=cfg["input_learning_rate"]),
            lang.ParameterGroup((learned[1],), name="interlayer", lr=cfg["interlayer_learning_rate"]),
            lang.ParameterGroup((projection.weight,), name="readout", lr=cfg["readout_learning_rate"]),
            *((lang.ParameterGroup(tuple(fixed), name="fixed_loop", lr=0.0, frozen=True),)
              if fixed else ())),
        optimizer=(lang.CustomOptimizer("exp124.adam/v1", lr=cfg["learning_rate"],
                   betas=cfg["optimizer_betas"], weight_decay=cfg["weight_decay"])
                   if cfg["optimizer"] == "adam" else lang.AdamW(weight_decay=cfg["weight_decay"])),
        surrogate=(lang.FastSigmoid(slope=cfg["surrogate_slope"] / cfg["surrogate_voltage_scale_mv"],
                   gradient_scale=cfg["surrogate_gradient_scale"],
                   threshold_inclusive=cfg["surrogate_threshold_inclusive"])
                   if cfg["variant"] in ("paper", "calibrated") else lang.FastSigmoid(slope=cfg["surrogate_slope"])),
        gradient_clip=cfg["gradient_clip"],
        epochs=cfg["epochs"],
        presentation_duration=cfg["duration_ms"] * lang.ms,
    )
    return lang.compile(net, training=spec, target="tools/snnsim")
