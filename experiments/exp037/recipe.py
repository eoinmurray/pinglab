"""Explicit absolute-rate and baseline-relative perturbation recipes.

configuration() resolves fresh scientific settings before execution. Network
authoring and analysis definitions do not load data or launch a stage on import.
"""

import math
from copy import deepcopy

from experiments.exp037.custom.source_exp022_custom_checkpoints import checkpoint_policy

FR_STRENGTH_UPPER = 0.041
CELLS = [
    {
        "name": "coba__off__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__off__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__off__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt25hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt25hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt25hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt10hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt10hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt10hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt5hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt5hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt5hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt2p5hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt2p5hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt2p5hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt1hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt1hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt1hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__off__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__off__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__off__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt25hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt25hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt25hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt10hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt10hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt10hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt5hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt5hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt5hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt2p5hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt2p5hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt2p5hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt1hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt1hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt1hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
]


def training_run_cells(training_run_id):
    cells = [
        deepcopy(cell) for cell in CELLS if cell["training_run_id"] == training_run_id
    ]
    if not cells:
        raise ValueError(f"unsupported training family: {training_run_id}")
    return tuple(cells)


def training_run_values(training_run_id, field):
    values = []
    for cell in training_run_cells(training_run_id):
        value = cell[field]
        if value not in values:
            values.append(value)
    return tuple(values)


def training_run_cell(training_run_id, **identity):
    cells = [
        cell
        for cell in training_run_cells(training_run_id)
        if all(cell.get(key) == value for key, value in identity.items())
    ]
    if len(cells) != 1:
        raise ValueError(f"expected one {training_run_id} cell matching {identity}")
    return cells[0]


SLUG = "exp037"
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
SHARDS = 6
ANALYSIS_PURPOSE = "deployment_performance"
CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]
MODELS = list(training_run_values("TR-02", "model"))
SEEDS_BASELINE = list(training_run_values("TR-02", "seed"))
RATE_TARGET_GRID_HZ = list(training_run_values("TR-02", "rate_target_hz"))
# Evaluate 10% of the untouched 10,000-image official MNIST test partition.
EVAL_MAX_SAMPLES = 1_000
EI_RASTER_N_E_PLOT, EI_RASTER_N_I_PLOT = 200, 64
PERTURB_DROP_LEVELS = [round(0.1 * i, 1) for i in range(11)]
PERTURB_ADD_LEVELS = [float(2 * i) for i in range(21)]
PERTURB_RASTER_DROP_LEVELS = [0.0, 0.5, 1.0]
PERTURB_RASTER_ADD_LEVELS = [0.0, 20.0, 40.0]
SNAPSHOT_ARRAYS = ("dt", "n_e", "n_i", "label", "spk_e", "spk_i")
FIGURES = (
    "perturbation_rasters.svg",
    "perturbation_rasters.pdf",
    "perturbation_curves.svg",
    "perturbation_curves.pdf",
    *(
        f"perturb_rasters__{mode}__{model}.{ext}"
        for model in MODELS
        for mode in ("drop", "add")
        for ext in ("svg", "pdf")
    ),
)


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }


def cell_name(model: str, rate_target_hz: float | None, seed: int) -> str:
    return training_run_cell(
        "TR-02", model=model, rate_target_hz=rate_target_hz, seed=seed
    )["name"]


def rate_target_display(rate_target_hz: float | None) -> str:
    """Human label for plots / numbers.json."""
    if rate_target_hz is None:
        return "off"
    return f"{rate_target_hz:g}"


def seeds_for(rate_target_hz: float | None) -> list[int]:
    """Return the independent seeds used at every frontier point."""
    return list(SEEDS_BASELINE)


def _level_tag(level) -> str:
    if isinstance(level, (list, tuple)):
        return "_".join(_level_tag(x) for x in level)
    return f"{float(level):g}".replace(".", "p")


def _parse_job(job_id: str) -> tuple[str, str, int, str, str]:
    """Return (kind, model, seed, mode, level_tag)."""
    parts = job_id.split("__")
    if len(parts) != 5 or not parts[2].startswith("seed"):
        raise ValueError(f"bad job id {job_id!r}")
    return parts[0], parts[1], int(parts[2].removeprefix("seed")), parts[3], parts[4]


def bank_cells():
    return [
        {
            "cell_name": cell_name(m, t, s),
            "model": m,
            "rate_target_hz": t,
            "seed": s,
            "w_in": 0.9,
        }
        for m in MODELS
        for t in RATE_TARGET_GRID_HZ
        for s in SEEDS_BASELINE
    ]


def configuration(*, smoke=False, version=3):
    if type(smoke) is not bool:
        raise ValueError("smoke must be boolean")
    if version not in (1, 2, 3):
        raise ValueError("unsupported exp037 recipe version")
    cfg = dict(
        schema=f"exp037.recipe/v{version}",
        **(refractory_configuration() if version >= 3 else dict()),
        profile="smoke" if smoke else "production",
        checkpoint_policy=CHECKPOINT_POLICY,
        evaluation_samples=100 if smoke else EVAL_MAX_SAMPLES,
        seeds=SEEDS_BASELINE,
        illustrative_seed=SEEDS_BASELINE[0],
        sample_index=0,
        drop_levels=[0.0, 0.5, 0.8, 1.0] if smoke else PERTURB_DROP_LEVELS,
        add_levels=[0.0, 20.0, 40.0] if smoke else PERTURB_ADD_LEVELS,
        raster_drop_levels=PERTURB_RASTER_DROP_LEVELS,
        raster_add_levels=PERTURB_RASTER_ADD_LEVELS,
    )
    if version >= 2:
        cfg.update(
            add_levels=[0.0, 100.0, 200.0]
            if smoke
            else [float(i) for i in range(0, 201, 10)],
            raster_add_levels=[0.0, 100.0, 200.0],
            add_units="percent_of_per_seed_test_baseline_e_rate",
            calibration="unperturbed selected checkpoint, same evaluation images and encoding stream",
            shard_partition="one_model_seed",
        )
    return deepcopy(cfg)


def relative(cfg):
    return cfg["schema"] in ("exp037.recipe/v2", "exp037.recipe/v3")


def jobs(cfg):
    result = []
    for model in MODELS:
        if relative(cfg):
            for seed in cfg["seeds"]:
                identity = f"calibration__{model}__seed{seed}__drop__0"
                result.append(
                    {
                        "id": identity,
                        "path": f"jobs/{identity}",
                        "kind": "calibration",
                        "model": model,
                        "seed": seed,
                        "mode": "drop",
                        "level": 0.0,
                        "cell_name": cell_name(model, None, seed),
                        "samples": cfg["evaluation_samples"],
                    }
                )
        for kind, seeds in (
            ("sweep", cfg["seeds"]),
            ("raster", [cfg["illustrative_seed"]]),
        ):
            for seed in seeds:
                for mode in ("drop", "add"):
                    levels = cfg[
                        ("raster_" if kind == "raster" else "") + mode + "_levels"
                    ]
                    for level in levels:
                        identity = (
                            f"{kind}__{model}__seed{seed}__{mode}__{_level_tag(level)}"
                        )
                        result.append(
                            {
                                "id": identity,
                                "path": f"jobs/{identity}",
                                "kind": kind,
                                "model": model,
                                "seed": seed,
                                "mode": mode,
                                "level": level,
                                "cell_name": cell_name(model, None, seed),
                                "samples": cfg["evaluation_samples"],
                                **(
                                    {"sample_index": cfg["sample_index"]}
                                    if kind == "raster"
                                    else {}
                                ),
                            }
                        )
    if relative(cfg):
        for job in result:
            job["count_perturbations"] = True
            if job["mode"] == "add":
                job["level_units"] = cfg["add_units"]
    return result


def shard_jobs(cfg, index):
    if not 0 <= index < SHARDS:
        raise ValueError("invalid exp037 shard index")
    if not relative(cfg):
        return jobs(cfg)[index::SHARDS]
    model, seed = [(m, s) for m in MODELS for s in cfg["seeds"]][index]
    return [j for j in jobs(cfg) if (j["model"], j["seed"]) == (model, seed)]


def resolve_job(job, baseline_e_rate_hz, dt_ms):
    """Freeze an addition job's physical rate using its own calibration."""
    if job.get("level_units") is None:
        return job
    if not math.isfinite(baseline_e_rate_hz) or baseline_e_rate_hz <= 0:
        raise ValueError("relative insertion needs a finite positive baseline E rate")
    rate = job["level"] * baseline_e_rate_hz / 100.0
    if not 0 <= rate * dt_ms / 1000.0 <= 1:
        raise ValueError("insertion probability exceeds the binary timestep capacity")
    return {**job, "baseline_e_rate_hz": baseline_e_rate_hz, "applied_level": rate}


def infer_jobs():
    import os

    return [
        j["id"]
        for j in jobs(configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1"))
    ]


BIOPHYSICS = {
    "capacitance_e_nf": 1.0,
    "capacitance_i_nf": 0.5,
    "leak_e_us": 0.05,
    "leak_i_us": 0.10,
    "resting_mv": -65.0,
    "threshold_mv": -50.0,
    "reset_mv": -65.0,
    "readout_tau_ms": 2.0,
    "readout_threshold": 1.0,
}


def author_network(training, *, observables=()):
    """Compile this experiment's E/I classifier and requested observables."""
    from types import SimpleNamespace

    from snnlab import lang

    biophysics = BIOPHYSICS
    refractory = refractory_configuration()
    dt = training["dt"]
    net = lang.Network(SLUG, dt=dt * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", training["n_in"]),
        signal_type="spikes",
        unit="spike",
    )
    neurons = {}
    for label in ("e", "i"):
        capacitance, leak = (
            biophysics[f"capacitance_{label}_nf"],
            biophysics[f"leak_{label}_us"],
        )
        neurons[label] = lang.COBA_LIF(
            tau_mem=capacitance / leak * lang.ms,
            capacitance_nf=capacitance,
            leak_us=leak,
            resting_mv=biophysics["resting_mv"],
            threshold_mv=biophysics["threshold_mv"],
            reset_mv=biophysics["reset_mv"],
            initial_voltage_mv=biophysics["resting_mv"],
            refractory=refractory[f"refractory_{label}_ms"] * lang.ms,
            voltage_grad_dampen=training["v_grad_dampen"],
        )
    # Reciprocal E/I loop with excitatory drive to E only.
    ping_ampa = lang.AMPA(tau=training["tau_ampa_ms"] * lang.ms)
    ping_gaba = lang.GABA(tau=training["tau_gaba_ms"] * lang.ms)
    ping_recurrent_delay = dt * lang.ms
    ping_e = net.population("E", size=training["n_hidden"], neuron=neurons["e"])
    ping_i = net.population("I", size=training["n_inh"], neuron=neurons["i"])
    ping_input_to_e = net.connect(
        drive,
        ping_e.excitatory,
        name="input_to_E",
        synapse=ping_ampa,
        weight=lang.Constant(0),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    ping_e_to_i = net.connect(
        ping_e.spikes,
        ping_i.excitatory,
        name="E_to_I",
        synapse=ping_ampa,
        weight=lang.Constant(0),
        constraint=lang.NonNegative(),
        connection="recurrent",
        delay=ping_recurrent_delay,
        initialization_scaling="direct",
    )
    ping_i_to_e = net.connect(
        ping_i.spikes,
        ping_e.inhibitory,
        name="I_to_E",
        synapse=ping_gaba,
        weight=lang.Constant(0),
        constraint=lang.NonNegative(),
        connection="recurrent",
        delay=ping_recurrent_delay,
        initialization_scaling="direct",
    )
    circuit = SimpleNamespace(
        e=ping_e,
        i=ping_i,
        input_to_e=ping_input_to_e,
        e_to_i=ping_e_to_i,
        i_to_e=ping_i_to_e,
    )
    output = net.population(
        "readout",
        size=training["n_out"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=biophysics["readout_tau_ms"] * lang.ms,
            soft_reset_threshold=biophysics["readout_threshold"],
            initial_voltage=0,
            surrogate_slope=training["surrogate_slope"],
        ),
    )
    net.connect(
        circuit.e.spikes,
        output.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=biophysics["readout_tau_ms"] * lang.ms),
        weight=lang.Constant(0),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    net.output(
        "class_scores",
        lang.reduce(
            output.pre_reset_voltage
            if training.get("readout_mode", "mem-mean") == "mem-mean"
            else output.spikes,
            operation="mean"
            if training.get("readout_mode", "mem-mean") == "mem-mean"
            else "sum",
            over="time",
            name="readout_reduction",
        ),
    )
    for label, pop in (("e", circuit.e), ("i", circuit.i)):
        net.output(
            f"spk_{label}_count",
            lang.reduce(
                pop.spikes, operation="sum", over="time", name=f"{label}_counts"
            ),
        )
        if "spikes" in observables:
            net.expose(pop.spikes, name=f"spk_{label}")
    if "traces" in observables:
        for key, value in {
            "v_e": circuit.e.voltage,
            "v_i": circuit.i.voltage,
            "ge_e": circuit.input_to_e.conductance,
            "ge_i": circuit.e_to_i.conductance,
            "gi_e": circuit.i_to_e.conductance,
        }.items():
            net.expose(value, name=key)
    return lang.compile(
        net,
        target="tools/snnsim",
        refractory_timing="legacy_coba",
    )


def inference_request(training, job):
    if job.get("level_units") and "applied_level" not in job:
        raise ValueError("relative insertion requires explicit calibration")
    return {
        "checkpoint_role": CHECKPOINT_ROLE,
        "input": "snapshot" if job["kind"] == "raster" else "dataset",
        "t_ms": training["t_ms"],
        "input_rate_hz": training["input_rate"],
        "samples": job["samples"],
        "batch_size": 1 if job["kind"] == "raster" else 64,
        "subset_seed": 42,
        "encoder_seed": 20260415,
        "sample_index": job.get("sample_index"),
        "observables": ["spikes"],
        "products": [],
        "perturb_mode": job["mode"],
        "perturb_level": job.get("applied_level", job["level"]),
        "perturb_seed": 20260416,
        "perturb_stream": "sequential device E then I per timestep; continuous across batches",
    }
