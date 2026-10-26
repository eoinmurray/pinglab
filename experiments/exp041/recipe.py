"""The retained inhibitory-decay sweep recipe; training remains owned by exp022.

configuration() resolves fresh scientific settings before execution. Network
authoring and analysis definitions do not load data or launch a stage on import.
"""

from copy import deepcopy

from experiments.exp041.custom.source_exp022_custom_checkpoints import checkpoint_policy

FR_STRENGTH_UPPER = 0.041
CELLS = [
    {
        "name": "ping__tg4p5__seed42",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=4.5",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 4.5,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg4p5__seed43",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=4.5",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 4.5,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg4p5__seed44",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=4.5",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 4.5,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg6__seed42",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=6",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg6__seed43",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=6",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg6__seed44",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=6",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg9__seed42",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=9",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 9.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg9__seed43",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=9",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 9.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg9__seed44",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=9",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 9.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg12__seed42",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=12",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 12.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg12__seed43",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=12",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 12.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg12__seed44",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=12",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 12.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg18__seed42",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=18",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 18.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg18__seed43",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=18",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 18.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg18__seed44",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=18",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 18.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg27__seed42",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=27",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 27.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg27__seed43",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=27",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 27.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg27__seed44",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=27",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 27.0,
        "training_run_id": "TR-03",
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


SLUG = "exp041"
TRAINING_RUN = "TR-03"
ANALYSIS_PURPOSE = "endpoint_dynamics"
CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]
TAU_GABA_SWEEP = training_run_values(TRAINING_RUN, "tau_gaba")
SEEDS = training_run_values(TRAINING_RUN, "seed")
T_MS = 200.0
MAX_SAMPLES = 7000
EPOCHS = 50
# Evaluate 10% of the untouched 10,000-image official MNIST test partition.
EVAL_MAX_SAMPLES = 1_000
SMOKE_MAX_SAMPLES = 100
RASTER_SAMPLE_IDX = 0
RASTER_N_E_PLOT = 200
RASTER_N_I_PLOT = 64
RASTER_T_WINDOW_MS = 100.0
DT_TRAIN = 0.1
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
TAU_GABA_REFERENCE_MS = 6.0
F_GAMMA_BAND_HZ = (5.0, 150.0)
FIGURES = tuple(
    name + "." + ext
    for name, extensions in (
        ("rate_vs_fgamma", ("svg", "pdf")),
        ("training_curves", ("svg", "pdf")),
        ("psds", ("svg", "pdf")),
        ("per_trial_peaks", ("svg", "pdf")),
        ("raster_strip", ("svg", "pdf")),
    )
    for ext in extensions
)


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }


TRAINING_COMMON_FIELDS = (
    "model",
    "dataset",
    "max_samples",
    "epochs",
    "t_ms",
    "tau_ampa_ms",
    "dt",
    "input_rate",
    "input_rate_sampling",
    "hidden_sizes",
    "n_in",
    "n_hidden",
    "n_inh",
    "n_out",
    "ei_strength",
    "ei_ratio",
    "w_in",
    "w_in_initial_zero_fraction",
    "recurrent_initial_zero_fraction",
    "readout_mode",
    "readout_w_init_mean",
    "readout_w_init_std",
    "surrogate_slope",
    "lr",
    "batch_size",
    "weight_decay",
    "grad_clip",
    "v_grad_dampen",
    "dales_law",
    "trainable_w_ei",
    "trainable_w_ie",
    "dataset_split",
    "validation_encoder_draws",
    "fr_reg_upper_strength",
    "fr_reg_upper_target_hz",
    "adaptive_threshold",
    "train_leak",
    "signed_readout",
    "readout_bias",
    "trainable_w_ee",
    "trainable_w_ii",
    "state_clamp",
)


def cell_name(tau_ms: float, seed: int) -> str:
    return training_run_cell(TRAINING_RUN, tau_gaba=tau_ms, seed=seed)["name"]


def configuration(*, smoke: bool = False, version=2) -> dict:
    if type(smoke) is not bool:
        raise ValueError("smoke must be boolean")
    if version not in (1, 2):
        raise ValueError("unsupported exp041 recipe version")
    return deepcopy(
        dict(
            schema=f"exp041.recipe/v{version}",
            **(refractory_configuration() if version >= 2 else dict()),
            profile="smoke" if smoke else "production",
            tau_gaba_sweep_ms=list(TAU_GABA_SWEEP),
            seeds=list(SEEDS),
            evaluation_samples=SMOKE_MAX_SAMPLES if smoke else EVAL_MAX_SAMPLES,
            checkpoint_policy=CHECKPOINT_POLICY,
            f_gamma_band_hz=list(F_GAMMA_BAND_HZ),
            raster=dict(
                seed=SEEDS[0],
                sample_index=RASTER_SAMPLE_IDX,
                n_e_plot=RASTER_N_E_PLOT,
                n_i_plot=RASTER_N_I_PLOT,
                selection_seed=0,
                window_ms=RASTER_T_WINDOW_MS,
            ),
        )
    )


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


def inference_request(training, cfg, *, snapshot=False):
    return {
        "checkpoint_role": CHECKPOINT_ROLE,
        "input": "snapshot" if snapshot else "dataset",
        "t_ms": training["t_ms"],
        "input_rate_hz": training["input_rate"],
        "samples": cfg["evaluation_samples"],
        "batch_size": 1 if snapshot else 64,
        "subset_seed": 42,
        "encoder_seed": 20260415,
        "sample_index": cfg["raster"]["sample_index"],
        "observables": ["spikes"] if snapshot else [],
        "products": [] if snapshot else ["population"],
    }
