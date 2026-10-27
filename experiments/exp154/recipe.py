"""Author a small E/I value controller and resolve both study protocols.

The capacity protocol supplies all targets. The lifetime protocol reveals targets
only after consumption; the simulator maintains energy and renews dead agents.
Only bounded E-to-I and I-to-E magnitudes can train in either protocol.
"""

import copy

import numpy as np
import torch
from experiments.exp154.custom.dynamics import register_extensions
from snnlab import lang
from snnlab.sim import GraphExecutor, plan_graph

SLUG = "exp154"


def inherited_genome(overrides=None):
    """Resolve one abstract inherited programme; no selection or mutation runs."""
    genome = dict(
        identity="development-g0/v1",
        recurrent_ranges=dict(
            EE=[0.003, 0.009], EI=[0.01, 0.05], IE=[0.10, 0.30], II=[0.01, 0.03]
        ),
        weight_upper=1.0,
        prenatal_windows=12,
        prenatal_learning_rate=0.005,
        homeostatic_target_rate=0.25,
        internal_drive_na=1.2,
        juvenile_encounters=20,
        juvenile_learning_rate=0.03,
        adult_learning_rate=0.008,
    )
    if overrides:
        if set(overrides) - set(genome):
            raise ValueError("unknown inherited genome parameter")
        genome.update(copy.deepcopy(overrides))
    for key in ("prenatal_windows", "juvenile_encounters"):
        if not isinstance(genome[key], int) or genome[key] < 0:
            raise ValueError("phase durations must be nonnegative integers")
    for key in (
        "weight_upper",
        "prenatal_learning_rate",
        "juvenile_learning_rate",
        "adult_learning_rate",
        "internal_drive_na",
    ):
        if not np.isfinite(genome[key]) or genome[key] <= 0:
            raise ValueError("genome scales must be finite and positive")
    if not 0 < genome["homeostatic_target_rate"] < 1:
        raise ValueError("homeostatic target must lie between zero and one")
    ranges = genome["recurrent_ranges"]
    if set(ranges) != {"EE", "EI", "IE", "II"} or any(
        not 0 <= low < high <= genome["weight_upper"] for low, high in ranges.values()
    ):
        raise ValueError("invalid inherited weight ranges")
    return genome


def configuration(profile="pilot", protocol="lifetime", *, genome=None):
    if profile not in {"pilot", "smoke"}:
        raise ValueError("profile must be pilot or smoke")
    if protocol not in {"capacity", "lifetime"}:
        raise ValueError("protocol must be capacity or lifetime")
    genome = inherited_genome(genome)
    return dict(
        genome=genome,
        protocol=protocol,
        initial_energy=2.5,
        metabolic_cost=0.236,
        exploration_probability=0.15,
        lifetime_presentations=300 if profile == "pilot" else 12,
        lifetime_learning_rate=genome["juvenile_learning_rate"],
        birth_policy="restore genome at conception; prenatal development; reset energy and food optimizer at birth; no evolution",
        lifetime_feedback="only consumed outcomes; including the terminal encounter",
        schema="exp154/v6",
        lifetime_record_signals=[
            "E.rate",
            "I.rate",
            "sensory_E.current",
            "sensory_I.current",
            "internal_E.current",
            "internal_I.current",
            "EE.current",
            "EI.current",
            "IE.current",
            "II.current",
        ],
        learning="always enabled after consumption",
        activity_validation_atol=1e-12,
        initialization="paired-e/v1",
        decision_threshold=1e-9,
        profile=profile,
        feedback=(
            "full outcome feedback; no consumption or energy budget"
            if protocol == "capacity"
            else "consumed outcomes only; metabolic energy budget"
        ),
        seeds=[7],
        mapping_signs=[1],
        e_cells=16,
        i_cells=4,
        dt_ms=1.0,
        presentation_ms=40.0,
        readout_tail_ms=10.0,
        probe_ms=200.0,
        probe_tail_ms=20.0,
        probe_cues=[0.1, 0.5, 0.9],
        tau_e_ms=10.0,
        tau_i_ms=5.0,
        sigmoid_scale_na=0.25,
        initial_rate=0.05,
        readout_gain=4.0,
        weight_upper=genome["weight_upper"],
        bias_mean_na=0.60,
        bias_std_na=0.04,
        sensory_noise_std=0.035,
        sensory_settings=dict(
            E=dict(base=0.35, amplitude=0.15, tonic=0.60),
            I=dict(base=0.50, amplitude=0.45, tonic=0.20),
        ),
        initial_recurrent_weights=copy.deepcopy(genome["recurrent_ranges"]),
        development_seed_offset=20000,
        training_seed_offset=10000,
        validation_seed=20261026,
        training_presentations=1000 if profile == "pilot" else 20,
        batch_size=10,
        evaluation_every=100 if profile == "pilot" else 10,
        evaluation_cues=201 if profile == "pilot" else 41,
        validation_cues=64,
        learning_rate=0.03,
        gradient_clip=5.0,
        dtype="float64",
        device="cpu",
        cpu_threads=1,
        pass_mse=0.02,
        required_seeds_per_sign=1,
        stability_step_rate_per_ms=1e-4,
    )


def _initializer(values):
    return lang.CustomInitializer("ei_lifecycle.fixed/v1", values=values.tolist())


def _sensory_weights(size, rng, cfg, name):
    preference = np.ones(size)
    preference[size // 2 :] = -1
    if name == "E":
        # Both opposing readout groups receive the same mix of cue tuning.
        preference = np.tile(np.where(np.arange(size // 2) % 2, -1, 1), 2)
    weights = np.zeros((3, size))
    settings = cfg["sensory_settings"][name]
    amplitude, base = settings["amplitude"], settings["base"]
    weights[0] = base + amplitude * preference
    weights[1] = base - amplitude * preference
    weights[2] = settings["tonic"]
    weights = np.maximum(
        weights + rng.normal(0, cfg["sensory_noise_std"], weights.shape), 0
    )
    return paired_columns(weights) if name == "E" else weights


def paired_columns(values):
    """Match E targets across the two opposing fixed-readout groups."""
    half = values.shape[-1] // 2
    values[..., half:] = values[..., :half]
    return values


def author_network(cfg, seed):
    register_extensions()
    rng = np.random.default_rng(seed)
    net = lang.Network("EI weight lifecycle", dt=cfg["dt_ms"] * lang.ms)
    sensory = net.input(
        "sensory", shape=("time", "batch", 3), signal_type="continuous", unit="1"
    )
    internal = net.input(
        "internal", shape=("time", "batch", 3), signal_type="continuous", unit="1"
    )
    populations = {}
    for name, size, tau in [
        ("E", cfg["e_cells"], cfg["tau_e_ms"]),
        ("I", cfg["i_cells"], cfg["tau_i_ms"]),
    ]:
        biases = rng.normal(cfg["bias_mean_na"], cfg["bias_std_na"], size)
        if name == "E":
            biases = paired_columns(biases)
        populations[name] = net.population(
            name,
            size=size,
            spiking=False,
            neuron=lang.CustomNeuron(
                "ei_lifecycle.rate/v1",
                tau_ms=tau,
                sigmoid_scale_na=cfg["sigmoid_scale_na"],
                initial_rate=cfg["initial_rate"],
                bias_na=biases.tolist(),
            ),
        )
        net.connect(
            sensory,
            populations[name].excitatory,
            name=f"sensory_{name}",
            synapse=lang.CustomSynapse("ei_lifecycle.current/v1", gain_na=1.0),
            weight=_initializer(_sensory_weights(size, rng, cfg, name)),
            constraint=lang.NonNegative(),
            initialization_scaling="direct",
        )
        net.connect(
            internal,
            populations[name].excitatory,
            name=f"internal_{name}",
            synapse=lang.CustomSynapse("ei_lifecycle.current/v1", gain_na=1.0),
            weight=_initializer(
                np.full((3, size), cfg["genome"]["internal_drive_na"] / 3)
            ),
            constraint=lang.NonNegative(),
            initialization_scaling="direct",
        )
        net.output(
            name + "_activity",
            lang.reduce(
                populations[name].state("rate"),
                operation="mean",
                over="time",
                name=name + "_mean_activity",
            ),
        )
    for name, (low, high) in cfg["initial_recurrent_weights"].items():
        source, target = name
        pre, post = populations[source], populations[target]
        weights = rng.uniform(low, high, (pre.size, post.size))
        if target == "E":
            weights = paired_columns(weights)
        if name == "EE":
            # Native removal of self-connections must preserve paired targets.
            indices = np.arange(pre.size)
            weights[indices, indices] = 0
            weights[indices, (indices + pre.size // 2) % pre.size] = 0
        net.connect(
            pre.state("rate"),
            post.excitatory if source == "E" else post.inhibitory,
            name=name,
            connection="recurrent",
            synapse=lang.CustomSynapse("ei_lifecycle.current/v1", gain_na=1.0),
            weight=_initializer(weights),
            constraint=lang.CustomConstraint(
                "ei_lifecycle.bounds/v1", upper=cfg["weight_upper"]
            ),
            initialization_scaling="direct",
        )
    readout = np.ones((cfg["e_cells"], 1))
    readout[cfg["e_cells"] // 2 :] = -1
    readout *= cfg["readout_gain"] / (cfg["e_cells"] // 2)
    weight = net.parameter(
        "value_weight", shape=readout.shape, initializer=_initializer(readout)
    )
    value = lang.custom_operation(
        "ei_lifecycle.value/v1",
        populations["E"].state("rate"),
        name="value_readout",
        shape=("batch", 1),
        unit="1",
        parameters=(weight,),
        tail_steps=round(cfg["readout_tail_ms"] / cfg["dt_ms"]),
    )
    net.output("prediction", value)
    return lang.compile(net)


def make_model(bundle, cfg, seed):
    return GraphExecutor(
        plan_graph(bundle.graph),
        seed=seed,
        trainable_parameters=("EI.weight", "IE.weight"),
        compile_mode=None,
        extension_validation="structural",
    ).to(device=cfg["device"], dtype=getattr(torch, cfg["dtype"]))


def encode(cues, cfg, *, duration_ms=None):
    cues = np.asarray(cues, dtype=float).reshape(-1)
    encoded = np.column_stack((cues, 1 - cues, np.ones_like(cues)))
    values = torch.as_tensor(
        encoded, dtype=getattr(torch, cfg["dtype"]), device=cfg["device"]
    )
    steps = round((duration_ms or cfg["presentation_ms"]) / cfg["dt_ms"])
    sensory = values.unsqueeze(0).repeat(steps, 1, 1)
    return {"sensory": sensory, "internal": torch.zeros_like(sensory)}


def encode_internal(drives, cfg):
    """Internally generated prenatal activity has no sensory cue or food target."""
    values = torch.as_tensor(
        drives, dtype=getattr(torch, cfg["dtype"]), device=cfg["device"]
    ).unsqueeze(1)
    return {"sensory": torch.zeros_like(values), "internal": values}


def developmental_phase(age, cfg):
    return "juvenile" if age <= cfg["genome"]["juvenile_encounters"] else "adult"


def targets(cues, sign):
    if sign not in {-1, 1}:
        raise ValueError("mapping sign must be -1 or +1")
    return sign * (2 * np.asarray(cues) - 1)
