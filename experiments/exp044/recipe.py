"""The retained timestep audit recipe; training remains owned by exp022."""

from experiments.exp022.checkpoints import checkpoint_policy
from experiments.helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES
from snnlab.sim.timing import duration_metadata, duration_steps, refractory_metadata

SLUG = "exp044"
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
ANALYSIS_PURPOSE = "endpoint_dynamics"
CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]
DT_SWEEP_MS = (0.05, 0.1, 0.2, 0.3, 0.6)
SEEDS = (42, 43, 44)
T_MS = 200.0
MAX_SAMPLES = 7000
EPOCHS = 50
EVAL_MAX_SAMPLES = MNIST_REDUCED_EVAL_SAMPLES
SMOKE_MAX_SAMPLES = 100
RASTER_SAMPLE_IDX = 0
RASTER_N_E_PLOT = 200
RASTER_N_I_PLOT = 64
RASTER_T_WINDOW_MS = 100.0
FIGURES = (
    "dt_sweep.svg",
    "dt_sweep.pdf",
    "raster_strip.png",
    "raster_strip.pdf",
    "training_curves.svg",
    "training_curves.pdf",
)


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }


def duration_configuration(duration_ms: float, dt_ms: float) -> dict:
    return duration_metadata(duration_ms, dt_ms)


def refractory_execution_configuration(dt_ms: float) -> dict:
    return refractory_metadata(
        REFRACTORY_E_MS,
        REFRACTORY_I_MS,
        dt_ms,
        policy=REFRACTORY_POLICY,
    )


def dt_label(dt_ms: float) -> str:
    return "dt" + f"{dt_ms:g}".replace(".", "p")


def cell_name(dt_ms: float, seed: int) -> str:
    if dt_ms not in DT_SWEEP_MS or seed not in SEEDS:
        raise ValueError("unregistered exp044 timestep or seed")
    return f"ping__{dt_label(dt_ms)}__seed{seed}"


BIOPHYSICS = {
    "C_m_E_nF": 1.0,
    "C_m_I_nF": 0.5,
    "g_L_E_uS": 0.05,
    "g_L_I_uS": 0.10,
    "resting_mV": -65.0,
    "threshold_mV": -50.0,
    "reset_mV": -65.0,
    "readout_tau_ms": 2.0,
    "readout_threshold": 1.0,
}

# Graph declarations use [target, source]; checkpoint and runtime matrices
# both use [source, target], so binding needs no transpose or fan-in scaling.
CHECKPOINT_PARAMETERS = {
    "input_to_E.weight": "W_ff.0",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
    "readout_projection.weight": "W_ff.1",
}


def configuration(*, smoke: bool = False) -> dict:
    return {
        "schema": "exp044.recipe/v3",
        **refractory_configuration(),
        "profile": "smoke" if smoke else "production",
        "dt_sweep_ms": list(DT_SWEEP_MS),
        "trial_duration": {
            "nominal_ms": T_MS,
            "policy": "whole_steps_floor_with_integer_tolerance",
            "conditions": [
                {
                    "dt_ms": dt,
                    "steps": duration_steps(T_MS, dt),
                    "realized_ms": duration_steps(T_MS, dt) * dt,
                }
                for dt in DT_SWEEP_MS
            ],
        },
        "seeds": list(SEEDS),
        "evaluation_samples": SMOKE_MAX_SAMPLES if smoke else EVAL_MAX_SAMPLES,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "executor": "snnlab.sim.GraphExecutor",
        "evaluation_batch_size": 64,
        "evaluation_subset_seed": 42,
        "evaluation_encoder_seed": 20260415,
        "snapshot_encoder_stream": "native graph seed; distinct from historical CLI initialization draws",
        "biophysics": dict(BIOPHYSICS),
        "raster": {
            "seed": SEEDS[0],
            "sample_index": RASTER_SAMPLE_IDX,
            "n_e_plot": RASTER_N_E_PLOT,
            "n_i_plot": RASTER_N_I_PLOT,
            "selection_seed": 0,
            "window_ms": RASTER_T_WINDOW_MS,
        },
    }


def validate_network_settings(common: dict) -> None:
    """Validate the scientific quantities used by the native graph."""
    import math

    from pingstore.contracts import PingstoreError

    for name in ("n_in", "n_hidden", "n_inh", "n_out"):
        if type(common[name]) is not int or common[name] <= 0:
            raise PingstoreError(f"invalid exp044 population size: {name}")
    if common["n_in"] != 784 or common["n_out"] != 10:
        raise PingstoreError("exp044 requires 784 MNIST inputs and ten class outputs")
    for name in (
        "tau_ampa_ms",
        "tau_gaba_ms",
        "input_rate",
        "surrogate_slope",
        "v_grad_dampen",
    ):
        value = common[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise PingstoreError(f"invalid exp044 {name}")


def author_network(cfg: dict, cell: dict, common: dict, *, traces: bool):
    """Declare the checkpoint-backed circuit and its full-trial measurements."""
    from experiments.helpers.ping import build_ping
    from snnlab import lang
    from snnlab.sim.timing import refractory_steps

    validate_network_settings(common)
    dt, b = cell["dt_ms"], cfg["biophysics"]
    net = lang.Network("exp044_timestep_audit", dt=dt * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", common["n_in"]),
        signal_type="spikes",
        unit="spike",
    )
    neurons = {}
    for label in ("E", "I"):
        neurons[label] = lang.COBA_LIF(
            tau_mem=(b[f"C_m_{label}_nF"] / b[f"g_L_{label}_uS"]) * lang.ms,
            capacitance_nf=b[f"C_m_{label}_nF"],
            leak_us=b[f"g_L_{label}_uS"],
            resting_mv=b["resting_mV"],
            threshold_mv=b["threshold_mV"],
            reset_mv=b["reset_mV"],
            initial_voltage_mv=b["resting_mV"],
            refractory_steps=refractory_steps(
                cfg[f"refractory_{label.lower()}_ms"],
                dt,
                policy=cfg["refractory_policy"],
            ),
            voltage_grad_dampen=common["v_grad_dampen"],
        )
    circuit = build_ping(
        net,
        source=drive,
        n_e=common["n_hidden"],
        n_i=common["n_inh"],
        neuron_e=neurons["E"],
        neuron_i=neurons["I"],
        ampa=lang.AMPA(tau=common["tau_ampa_ms"] * lang.ms),
        gaba=lang.GABA(tau=common["tau_gaba_ms"] * lang.ms),
        input_weight=lang.Constant(0.0),
        ei_weight=lang.Constant(0.0),
        ie_weight=lang.Constant(0.0),
        recurrent_delay=dt * lang.ms,
        initialization_scaling="direct",
    )
    output = net.population(
        "readout",
        size=common["n_out"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=b["readout_tau_ms"] * lang.ms,
            soft_reset_threshold=b["readout_threshold"],
            surrogate_slope=common["surrogate_slope"],
            initial_voltage=0.0,
        ),
    )
    net.connect(
        circuit.e.spikes,
        output.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=b["readout_tau_ms"] * lang.ms),
        weight=lang.Constant(0.0),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    net.output(
        "class_scores",
        lang.ops.reduce(
            output.pre_reset_voltage, operation="mean", over="time", name="readout_mean"
        ),
    )
    for label, population in (("e", circuit.e), ("i", circuit.i)):
        net.output(
            f"spk_{label}_count",
            lang.ops.reduce(
                population.spikes, operation="sum", over="time", name=f"{label}_counts"
            ),
        )
        if traces:
            net.expose(population.spikes, name=f"spk_{label}")
    return lang.compile(net, target="tools/snnsim")
