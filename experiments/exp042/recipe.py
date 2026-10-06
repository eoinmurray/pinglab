"""The reduced timing intervention recipe; no execution or data selection on import."""

import re

from experiments.helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES

SLUG = "exp042"
TRAINING_RUN = "TR-02"
ANALYSIS_PURPOSE = "endpoint_dynamics"
CHECKPOINT_POLICY = {"purpose": ANALYSIS_PURPOSE, "role": "final_epoch"}
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]
EVAL_SEED = 20260415
SEEDS = (42, 43, 44)
JITTER_SIGMAS_MS = (0.0, 1.0, 3.0, 7.0, 14.0, 21.0, 28.0, 42.0, 60.0, 100.0)
CELL_JITTER_SIGMAS_MS = (0.0, 0.5, 1.0, 2.0, 5.0, 9.0, 14.0, 21.0, 50.0)
F_GAMMA_REFERENCE_HZ = 43.95
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
JITTER_BOUNDARY_POLICY = "reflect_in_range/v1"
JITTER_COLLISION_POLICY = "nearest_free_bounded_alternating/v1"
EVAL_MAX_SAMPLES = MNIST_REDUCED_EVAL_SAMPLES
SMOKE_MAX_SAMPLES = 100
RASTER_SAMPLE_IDX = 0
RASTER_N_E_PLOT = 200
RASTER_N_I_PLOT = 64
COMPOUND_SIGMA_MS = 14.0
SHARDS = 8
FIGURES = ("rhythm_compound.png",)

CHECKPOINT_FILENAME = "weights_final.pth"
BANK_REQUIREMENTS = {
    "training_run_id": TRAINING_RUN,
    "model": "ping",
    "dataset": "mnist",
    "ei_strength": 1.0,
    "fr_reg_upper_strength": 0.0,
    "readout_mode": "mem-mean",
    "input_rate_sampling": "fixed",
    "dales_law": True,
}
UNSUPPORTED_BANK_DYNAMICS = (
    "signed_readout",
    "readout_bias",
    "train_leak",
    "adaptive_threshold",
    "state_clamp",
)


def checkpoint_shapes(training):
    ne, ni = training["n_hidden"], training["n_inh"]
    return {
        "W_ff.0": (training["n_in"], ne),
        "W_ff.1": (ne, training["n_out"]),
        "W_ei.1": (ne, ni),
        "W_ie.1": (ni, ne),
        "W_ee.1": (ne, ne),
        "W_ii.1": (ni, ni),
    }


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }


def cell_name(seed):
    if seed not in SEEDS:
        raise ValueError("unregistered exp042 training seed")
    return f"ping__off__seed{seed}"


def configuration(*, smoke=False):
    return {
        "schema": "exp042.recipe/v7",
        **refractory_configuration(),
        "executor": "snnlab.sim.GraphExecutor",
        "evaluation_batch_size": 64,
        "evaluation_subset_seed": 42,
        "snapshot_encoder_stream": "explicit training seed; no obsolete model-initialization draws",
        "reset_boundary": "all neuronal, synaptic, delay and readout state per presentation",
        "biophysics": dict(BIOPHYSICS),
        "profile": "smoke" if smoke else "production",
        "seeds": list(SEEDS),
        "jitter_sigmas_ms": list((0.0, 14.0, 100.0) if smoke else JITTER_SIGMAS_MS),
        "cell_jitter_sigmas_ms": list(
            (0.0, 0.5, 1.0, 2.0, 5.0, 9.0, 14.0) if smoke else CELL_JITTER_SIGMAS_MS
        ),
        "evaluation_samples": SMOKE_MAX_SAMPLES if smoke else EVAL_MAX_SAMPLES,
        "evaluation_partition": "official_mnist_test",
        "evaluation_seed": EVAL_SEED,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "f_gamma_reference_hz": F_GAMMA_REFERENCE_HZ,
        "jitter_policy": {
            "boundary": JITTER_BOUNDARY_POLICY,
            "collision": JITTER_COLLISION_POLICY,
            "invariant": "exact_per_trial_per_cell_spike_count",
        },
        "raster": {
            "seed": SEEDS[0],
            "sample_index": RASTER_SAMPLE_IDX,
            "sigma_ms": COMPOUND_SIGMA_MS,
            "selection_seed": 0,
            "n_e_plot": RASTER_N_E_PLOT,
            "n_i_plot": RASTER_N_I_PLOT,
        },
    }


def jobs(cfg):
    result = []
    for seed in cfg["seeds"]:
        groups = [
            ("jitter_sweep", f"jitter_sigma_{s:g}", seed + int(s), s)
            for s in cfg["jitter_sigmas_ms"]
        ]
        groups += [
            ("cell_jitter_sweep", f"cell_jitter_sigma_{s:g}", seed + int(s * 13), s)
            for s in cfg["cell_jitter_sigmas_ms"]
        ]
        for group, condition, offset, sigma in groups:
            identity = re.sub(
                r"(\d)\.(\d)", r"\1p\2", f"eval__{cell_name(seed)}__{condition}"
            )
            result.append(
                {
                    "id": identity,
                    "seed": seed,
                    "cell": cell_name(seed),
                    "group": group,
                    "condition": condition,
                    "seed_offset": offset,
                    "sigma_ms": sigma,
                }
            )
    return result


def replay_job(job):
    """Both zero-zero arms replay identical I spikes; retain separate logical rows."""
    if job["condition"] == "cell_jitter_sigma_0":
        return {
            **job,
            "id": job["id"].replace("__cell_jitter_sigma_0", "__jitter_sigma_0"),
            "condition": "jitter_sigma_0",
            "group": "jitter_sweep",
        }
    return job


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

# Checkpoints and runtime tensors both use [source, target] orientation and
# already contain fan-in scaling. Bind directly; clamp feedforward weights as
# the trained forward pass does. Same-population matrices must remain zero.
CHECKPOINT_PARAMETERS = {
    "input_to_E.weight": "W_ff.0",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
    "readout_projection.weight": "W_ff.1",
}


def author_network(cfg, training):
    """Own the frozen circuit, readout and observables for each replay mode."""
    from experiments.helpers.ping import build_ping
    from snnlab import lang
    from snnlab.sim.timing import refractory_steps

    dt, b = training["dt"], cfg["biophysics"]
    net = lang.Network("exp042_inhibitory_replay", dt=dt * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", training["n_in"]),
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
            voltage_grad_dampen=training["v_grad_dampen"],
        )
    circuit = build_ping(
        net,
        source=drive,
        n_e=training["n_hidden"],
        n_i=training["n_inh"],
        neuron_e=neurons["E"],
        neuron_i=neurons["I"],
        ampa=lang.AMPA(tau=training["tau_ampa_ms"] * lang.ms),
        gaba=lang.GABA(tau=training["tau_gaba_ms"] * lang.ms),
        input_weight=lang.Constant(0.0),
        ei_weight=lang.Constant(0.0),
        ie_weight=lang.Constant(0.0),
        recurrent_delay=dt * lang.ms,
        initialization_scaling="direct",
    )
    output = net.population(
        "readout",
        size=training["n_out"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=b["readout_tau_ms"] * lang.ms,
            soft_reset_threshold=b["readout_threshold"],
            surrogate_slope=training["surrogate_slope"],
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
            output.pre_reset_voltage,
            operation="mean",
            over="time",
            name="readout_mean",
        ),
    )
    for label, population in (("e", circuit.e), ("i", circuit.i)):
        net.output(
            f"spk_{label}_count",
            lang.ops.reduce(
                population.spikes,
                operation="sum",
                over="time",
                name=f"{label}_counts",
            ),
        )
        net.expose(population.spikes, name=f"spk_{label}")
    return lang.compile(net, target="tools/snnsim")


def baseline_recording():
    from snnlab.sim.streaming import RecordingSpec, SignalRecording

    return RecordingSpec((SignalRecording("I.spikes", kind="spike_events"),))


def recording_jobs(cfg):
    seed, sigma = cfg["raster"]["seed"], cfg["raster"]["sigma_ms"]
    return (
        ("cycle", f"jitter_sigma_{sigma:g}", seed + int(sigma)),
        ("cell", f"cell_jitter_sigma_{sigma:g}", seed + int(sigma * 13)),
    )
