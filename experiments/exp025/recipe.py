"""The retained exp025 recipe and explicit inference jobs; no storage access."""

from experiments.exp022 import FR_STRENGTH_UPPER as FR_STRENGTH_UPPER
from experiments.exp022 import training_run_cell, training_run_values

SLUG = "exp025"
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
ANALYSIS_PURPOSE = "endpoint_dynamics"
CHECKPOINT_ROLE = "final_epoch"
CHECKPOINT_POLICY = {"purpose": ANALYSIS_PURPOSE, "role": CHECKPOINT_ROLE}
MODELS = list(training_run_values("TR-02", "model"))
SEEDS = list(training_run_values("TR-02", "seed"))
RATE_TARGET_GRID_HZ = list(training_run_values("TR-02", "rate_target_hz"))
LOW_W_IN_VALUES = training_run_values("TR-07", "w_in")
# Evaluate 10% of the untouched 10,000-image official MNIST test partition.
EVAL_MAX_SAMPLES = 1_000
F_GAMMA_BAND_HZ = (5.0, 150.0)
W_IN_SCALE_VALUES = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.80,
    0.90,
    1.00,
    1.15,
    1.30,
    1.50,
    1.75,
    2.00,
    2.50,
    3.00,
]
FIGURES = tuple(
    f"{name}.{ext}"
    for name in (
        "theta_p_fgamma",
        "low_w_in_sweep",
        "w_in_scale_sweep",
        "w_in_scale_sweep_vs_rate",
    )
    for ext in ("svg", "pdf")
) + tuple(f"results_compound.{ext}" for ext in ("svg", "pdf"))


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }



def cell_name(model, rate_target_hz, seed):
    return training_run_cell(
        "TR-02", model=model, rate_target_hz=rate_target_hz, seed=seed
    )["name"]


def low_w_in_cell_name(w_in, seed):
    return training_run_cell("TR-07", w_in=w_in, seed=seed)["name"]


def rate_target_display(value):
    return "off" if value is None else f"{value:g}"


def bank_cells():
    return [
        {
            "cell_name": cell_name(m, t, s),
            "group": "shared_tr02",
            "model": m,
            "rate_target_hz": t,
            "seed": s,
            "w_in": 0.9,
        }
        for m in MODELS
        for t in RATE_TARGET_GRID_HZ
        for s in SEEDS
    ] + [
        {
            "cell_name": low_w_in_cell_name(w, s),
            "group": "low_w_in_controls",
            "model": "ping",
            "rate_target_hz": 1.0,
            "seed": s,
            "w_in": w,
        }
        for w in LOW_W_IN_VALUES
        for s in SEEDS
    ]


def configuration(*, smoke=False, version=2):
    if version not in (1, 2):
        raise ValueError("unsupported exp025 recipe version")
    return {
        "schema": f"exp025.recipe/v{version}",
        **(refractory_configuration() if version >= 2 else {}),
        "profile": "smoke" if smoke else "production",
        "evaluation_samples": 100 if smoke else EVAL_MAX_SAMPLES,
        "pfg_samples": 100 if smoke else min(256 * 5, EVAL_MAX_SAMPLES),
        "low_w_in_seeds": [42] if smoke else list(SEEDS),
        "scales": [0.5, 1.0, 3.0] if smoke else list(W_IN_SCALE_VALUES),
        "checkpoint_policy": CHECKPOINT_POLICY,
    }


def jobs(cfg):
    rows = []
    for m in MODELS:
        for t in RATE_TARGET_GRID_HZ:
            for s in SEEDS:
                name = cell_name(m, t, s)
                rows.append(
                    {
                        "kind": "frontier",
                        "cell_name": name,
                        "path": f"frontier/{name}",
                        "samples": cfg["evaluation_samples"],
                    }
                )
            name = cell_name(m, t, 42)
            rows.append(
                {
                    "kind": "pfg",
                    "cell_name": name,
                    "path": f"pfg/{name}",
                    "samples": cfg["pfg_samples"],
                    "is_ping": m == "ping",
                }
            )
        name = cell_name(m, None, 42)
        rows.append(
            {"kind": "snapshot", "cell_name": name, "path": f"snapshot/{m}", "model": m}
        )
    for m in ("ping", "coba"):
        name = cell_name(m, 1.0, 42)
        for scale in cfg["scales"]:
            rows.append(
                {
                    "kind": "scale",
                    "cell_name": name,
                    "path": f"win_scale/s{scale:g}/{name}",
                    "samples": cfg["evaluation_samples"],
                    "scale": scale,
                    "label": f"{m}@rt1hz",
                }
            )
    return rows



BIOPHYSICS = {
    "capacitance_e_nf": 1.0, "capacitance_i_nf": 0.5,
    "leak_e_us": 0.05, "leak_i_us": 0.10,
    "resting_mv": -65.0, "threshold_mv": -50.0, "reset_mv": -65.0,
    "readout_tau_ms": 2.0, "readout_threshold": 1.0,
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
    snapshot = job["kind"] == "snapshot"
    request = {
        "checkpoint_role": CHECKPOINT_ROLE,
        "input": "snapshot" if snapshot else "dataset",
        "t_ms": 400.0 if snapshot else training["t_ms"],
        "input_rate_hz": training["input_rate"],
        "samples": job.get("samples"),
        "batch_size": 1 if snapshot else 64,
        "subset_seed": 42,
        "encoder_seed": 20260415,
        "digit": 0,
        "sample": 0,
        "observables": ["spikes"] if snapshot else [],
        "products": [],
        "scale": job.get("scale", 1),
    }
    if job["kind"] == "pfg" and job["is_ping"]:
        request["products"] = ["population", "rasters"]
    if job["kind"] == "scale":
        request["products"] = ["rates"]
    return request
