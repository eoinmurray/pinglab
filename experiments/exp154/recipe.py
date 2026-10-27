"""Author the E/I controller, inherited genomes and resolved study protocols.

The capacity protocol supplies all targets. The lifetime protocol reveals targets
only after consumption; the simulator maintains energy and renews dead agents.
Only bounded E-to-I and I-to-E magnitudes can train in either protocol.
"""

import copy

import numpy as np
import torch
from experiments.exp154.custom.dynamics import register_extensions
from scipy.stats import truncnorm
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
        if set(overrides) - {*genome, "recurrent_normal"}:
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
    normal = genome.get("recurrent_normal")
    expected = {"EE", "II"} if normal is not None else {"EE", "EI", "IE", "II"}
    if set(ranges) != expected or any(
        not 0 <= low < high <= genome["weight_upper"] for low, high in ranges.values()
    ):
        raise ValueError("invalid inherited weight ranges")
    if normal is not None:
        validate_normal_weights(normal, genome["weight_upper"])
    return genome


def validate_normal_weights(distributions, upper):
    if set(distributions) != {"EI", "IE"}:
        raise ValueError("normal initial weights require EI and IE distributions")
    for distribution in distributions.values():
        if (
            set(distribution) != {"mean", "std"}
            or not np.isfinite(list(distribution.values())).all()
            or not 0 < distribution["mean"] <= upper
            or distribution["std"] <= 0
        ):
            raise ValueError("invalid normal initial weight mean or standard deviation")


def evolution_genome(overrides=None):
    """Evolve normal centres; fixed spreads match the former uniform variances."""
    genome = inherited_genome()
    genome.update(
        identity="development-g0/normal-v1",
        recurrent_normal={
            name: dict(mean=(low + high) / 2, std=float((high - low) / np.sqrt(12)))
            for name, (low, high) in genome["recurrent_ranges"].items()
            if name in {"EI", "IE"}
        },
        recurrent_ranges={
            name: genome["recurrent_ranges"][name] for name in ("EE", "II")
        },
    )
    if overrides:
        genome.update(copy.deepcopy(overrides))
    return inherited_genome(genome)


def evolution_settings(profile):
    """Keep the smoke search small while exercising selection and fresh seeds."""
    return dict(
        seed=154,
        population=16 if profile == "pilot" else 4,
        generations=10 if profile == "pilot" else 2,
        elites=4 if profile == "pilot" else 1,
        search_seeds=[7, 8, 9] if profile == "pilot" else [9],
        fresh_seeds=list(range(10, 15)) if profile == "pilot" else [10, 11],
        horizon=300,
        reproduction=dict(
            minimum_age_encounters=25,
            reserve_energy=0.5,
            cost_per_offspring=0.5,
            maximum_offspring=3,
            events_per_life=1,
            timing="after food energy and learning; adult and alive; total age since birth",
        ),
        population_policy="up to population genomes; retain up to elites unchanged; admit actual offspring of retained parents across evaluated lives, uniformly capped at free slots; stop if no offspring",
        log_mutation_std=0.1,
        target_mutation=0.02,
        duration_mutation=2,
        mean_bounds=[1e-5, 1.0],
        weight_sampling="normal conditioned on 0 < weight < weight_upper; shared uniform quantiles",
        standard_deviation_policy="fixed per pathway; describes the underlying normal before truncation",
        learning_rate_bounds=[1e-5, 0.1],
        target_bounds=[0.05, 0.95],
        prenatal_bounds=[0, 50],
        juvenile_bounds=[0, 100],
        fitness="mean offspring per life; then mean food encounters until death or horizon; then mean nonnegative remaining energy; then ascending candidate ID",
    )


def mutate_genome(parent, rng, settings):
    """Propose inherited settings only; acquired synaptic weights never enter."""
    child = copy.deepcopy(parent)
    mutate_initial_strengths(child, parent, rng, settings)
    for phase in ("prenatal", "juvenile", "adult"):
        name = phase + "_learning_rate"
        child[name] = float(
            np.clip(
                parent[name] * np.exp(rng.normal(0, settings["log_mutation_std"])),
                *settings["learning_rate_bounds"],
            )
        )
    child["adult_learning_rate"] = min(
        child["adult_learning_rate"], child["juvenile_learning_rate"]
    )
    child["homeostatic_target_rate"] = float(
        np.clip(
            parent["homeostatic_target_rate"]
            + rng.uniform(-settings["target_mutation"], settings["target_mutation"]),
            *settings["target_bounds"],
        )
    )
    for name, bounds in (
        ("prenatal_windows", "prenatal_bounds"),
        ("juvenile_encounters", "juvenile_bounds"),
    ):
        change = rng.integers(
            -settings["duration_mutation"], settings["duration_mutation"] + 1
        )
        child[name] = int(np.clip(parent[name] + change, *settings[bounds]))
    return inherited_genome(child)


def mutate_initial_strengths(child, parent, rng, settings):
    """Retain the recorded uniform law for reproducible historical run analysis."""
    for name in ("EI", "IE"):
        if "recurrent_normal" in parent:
            child["recurrent_normal"][name]["mean"] = float(
                np.clip(
                    parent["recurrent_normal"][name]["mean"]
                    * np.exp(rng.normal(0, settings["log_mutation_std"])),
                    *settings["mean_bounds"],
                )
            )
            continue
        while True:
            values = np.sort(
                np.asarray(parent["recurrent_ranges"][name])
                * np.exp(rng.normal(0, settings["log_mutation_std"], 2))
            )
            low, high = settings["range_bounds"]
            if low <= values[0] < values[1] <= high:
                child["recurrent_ranges"][name] = values.tolist()
                break


def validate_evolution_genome(genome, settings):
    """Reject unsupported search settings before a potentially unbounded proposal."""
    for name in ("EI", "IE"):
        if "recurrent_normal" in genome:
            mean = genome["recurrent_normal"][name]["mean"]
            if not settings["mean_bounds"][0] <= mean <= settings["mean_bounds"][1]:
                raise ValueError("evolution weight mean outside search bounds")
        else:
            low, high = genome["recurrent_ranges"][name]
            if (
                not settings["range_bounds"][0]
                <= low
                < high
                <= settings["range_bounds"][1]
            ):
                raise ValueError("evolution weight range outside search bounds")
    for phase in ("prenatal", "juvenile", "adult"):
        if (
            not settings["learning_rate_bounds"][0]
            <= genome[phase + "_learning_rate"]
            <= settings["learning_rate_bounds"][1]
        ):
            raise ValueError("evolution learning rate outside search bounds")
    if genome["adult_learning_rate"] > genome["juvenile_learning_rate"]:
        raise ValueError("adult evolution learning rate exceeds juvenile rate")
    for name, bounds in (
        ("homeostatic_target_rate", "target_bounds"),
        ("prenatal_windows", "prenatal_bounds"),
        ("juvenile_encounters", "juvenile_bounds"),
    ):
        if not settings[bounds][0] <= genome[name] <= settings[bounds][1]:
            raise ValueError("evolution developmental setting outside search bounds")


def rank_candidates(scores):
    """Rank reproductive output before survival; preserve historical survival runs."""
    return sorted(
        scores,
        key=lambda row: (
            -row.get("mean_offspring", 0.0),
            -row["mean_survival"],
            -row["mean_energy"],
            row["candidate"],
        ),
    )


def next_population(parents, settings, rng, generation):
    """Bootstrap a search, or replay historical fixed-quota generations."""
    if generation > 1 and "reproduction" in settings:
        raise ValueError("reproductive generations require actual offspring")
    population = copy.deepcopy(parents)
    for index in range(settings["population"] - len(parents)):
        parent = parents[index % len(parents)]
        identity = f"g{generation:03d}-c{len(population):03d}"
        genome = mutate_genome(parent["genome"], rng, settings)
        genome["identity"] = identity
        population.append(
            dict(
                candidate=identity,
                parent=parent["candidate"],
                created_generation=generation,
                genome=genome,
            )
        )
    return population


def reproductive_population(
    population, ranked, lives, settings, rng, generation, *, admit
):
    """Mutate each actual newborn once; capacity limits admission, never births."""
    individuals = {row["candidate"]: row for row in population}
    parents = [individuals[row["candidate"]] for row in ranked[: settings["elites"]]]
    retained = {row["candidate"] for row in parents}
    offspring = []
    for life in lives:
        parent = individuals[life["candidate"]]
        for sibling in range(life["offspring"]):
            identity = f"g{generation + 1:03d}-c{len(offspring):03d}"
            genome = mutate_genome(parent["genome"], rng, settings)
            genome["identity"] = identity
            offspring.append(
                dict(
                    candidate=identity,
                    parent=parent["candidate"],
                    created_generation=generation + 1,
                    parent_generation=generation,
                    parent_seed=life["seed"],
                    parent_trial=life["reproduction_age"],
                    sibling=sibling + 1,
                    genome=genome,
                    admitted=False,
                )
            )
    eligible = [
        index for index, child in enumerate(offspring) if child["parent"] in retained
    ]
    slots = settings["population"] - len(parents)
    selected = (
        sorted(
            rng.choice(eligible, size=min(slots, len(eligible)), replace=False).tolist()
        )
        if admit and eligible
        else []
    )
    for index in selected:
        offspring[index]["admitted"] = True
    return copy.deepcopy(parents + [offspring[index] for index in selected]), offspring


def configuration(profile="pilot", protocol="lifetime", *, genome=None):
    if profile not in {"pilot", "smoke"}:
        raise ValueError("profile must be pilot or smoke")
    if protocol not in {"capacity", "lifetime", "evolution"}:
        raise ValueError("protocol must be capacity, lifetime or evolution")
    genome = (
        evolution_genome(genome)
        if protocol == "evolution"
        else inherited_genome(genome)
    )
    cfg = dict(
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
    if protocol == "evolution":
        settings = evolution_settings(profile)
        validate_evolution_genome(genome, settings)
        cfg.update(
            schema="exp154/evolution-v3",
            evolution=settings,
            seeds=settings["search_seeds"],
            lifetime_presentations=settings["horizon"],
            stop_at_death=True,
            birth_policy="one independently conceived organism; stop at death; no acquired-weight inheritance",
        )
    return cfg


def individual_configuration(recorded, genome, seed):
    """Resolve a life from its recorded protocol, including historical weight laws."""
    cfg = copy.deepcopy(recorded)
    cfg.pop("evaluated_genomes", None)
    cfg.update(
        genome=inherited_genome(genome),
        initial_recurrent_weights=copy.deepcopy(genome["recurrent_ranges"]),
        lifetime_learning_rate=genome["juvenile_learning_rate"],
        seeds=[seed],
    )
    return cfg


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


def initial_weights(cfg, rng, name, shape):
    """Sample bounded normals with shared quantiles and paired native initializers.

    Native Normal/LowerClampedNormal do not implement this conditional law.
    Inverse-CDF sampling avoids candidate-dependent rejection-draw counts, which
    would otherwise alter fixed-pathway weights later in the same random stream.
    """
    normal = cfg["genome"].get("recurrent_normal", {})
    if name not in normal:
        return rng.uniform(*cfg["initial_recurrent_weights"][name], shape)
    mean, std = normal[name]["mean"], normal[name]["std"]
    return truncnorm.ppf(
        rng.uniform(size=shape),
        -mean / std,
        (cfg["weight_upper"] - mean) / std,
        loc=mean,
        scale=std,
    )


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
    for name in ("EE", "EI", "IE", "II"):
        source, target = name
        pre, post = populations[source], populations[target]
        weights = initial_weights(cfg, rng, name, (pre.size, post.size))
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
