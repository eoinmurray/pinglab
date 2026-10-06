"""Retained TR-05 endpoint recipe; all training belongs to exp022."""

from experiments.exp022.checkpoints import checkpoint_policy
from experiments.exp022.recipe import training_run_cell, training_run_values
from experiments.helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES

SLUG = "exp049"
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"

ANALYSIS_PURPOSE = "endpoint_dynamics"

CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)

CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]

MAX_SAMPLES = 7000

EPOCHS = 50

T_MS = 200.0

DT_TRAIN = 0.1
N_E, N_I = 1024, 256

SEEDS: list[int] = list(training_run_values("TR-05", "seed"))

CONDITIONS: dict[str, dict] = {
    "frozen_ping": {
        "label": "Frozen PING (control)",
    },
    "trainable_ping_init": {
        "label": "Trainable, PING init",
    },
    "trainable_zero_init": {
        "label": "Trainable, zero init",
    },
    "trainable_small_init": {
        "label": "Trainable, small seed init",
    },
}

COND_ORDER = list(training_run_values("TR-05", "tag"))
if set(COND_ORDER) != set(CONDITIONS):
    raise ValueError("TR-05 condition contract drift")

COMMON_RECIPE = {
    "voltage_grad_dampen": 1000.0,
    "input_weight_mean": 0.9,
    "input_initial_zero_fraction": 0.95,
    "readout": "mem-mean",
    "surrogate_slope": 1.0,
    "readout_weight_mean": 1.12060546875,
    "readout_weight_sd": 0.8349609375,
    "learning_rate": 0.0004,
    "training_batch_size": 256,
}
RUNTIME_REQUIREMENTS = {
    "input_rate": 25.0,
    "tau_ampa_ms": 2.0,
    "ei_ratio": 2.0,
    "dales_law": True,
    "input_rate_sampling": "fixed",
    "recurrent_initial_zero_fraction": 0.0,
    "signed_readout": False,
    "readout_bias": False,
    "adaptive_threshold": False,
    "train_leak": False,
    "state_clamp": False,
    "trainable_w_ee": False,
    "trainable_w_ii": False,
    "w_ee": [0.0, 0.0],
}

F_GAMMA_BAND_HZ: tuple[float, float] = (5.0, 150.0)
EVAL_MAX_SAMPLES = MNIST_REDUCED_EVAL_SAMPLES
WEIGHT_ARRAYS = tuple(
    f"W_{direction}_1_{state}"
    for direction in ("ei", "ie")
    for state in ("init", "trained")
)
SNAPSHOT_ARRAYS = ("dt", "n_e", "n_i", "label", "spk_e", "spk_i")
ARRAYS = {
    "pop_traces.npz": ("dt", "pop_e"),
    "weights_dump.npz": WEIGHT_ARRAYS,
    "recording.npz": SNAPSHOT_ARRAYS,
}


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }


FIGURES = tuple(
    name + "." + ext
    for name, exts in [("card__" + c, ("png", "pdf")) for c in COND_ORDER]
    + [("weights__" + c, ("svg", "pdf")) for c in COND_ORDER]
    + [
        (n, ("svg", "pdf"))
        for n in (
            "attractor_ei",
            "training_curves",
            "phase_portrait",
            "acc_rate_trajectory",
        )
    ]
    + [("training_summary", ("svg", "pdf", "png"))]
    for ext in exts
)


def cell_name(cond, seed):
    return training_run_cell("TR-05", tag=cond, seed=seed)["name"]


def bank_cells():
    return [
        {"cell_name": cell_name(c, s), "condition": c, "seed": s}
        for c in COND_ORDER
        for s in SEEDS
    ]


def configuration(*, smoke=False, version=3):
    if version not in (1, 2, 3):
        raise ValueError("unsupported exp049 recipe version")
    cfg = {
        "schema": f"exp049.recipe/v{version}",
        **(refractory_configuration() if version >= 2 else {}),
        "profile": "smoke" if smoke else "production",
        "checkpoint_policy": CHECKPOINT_POLICY,
        "evaluation_samples": 100 if smoke else EVAL_MAX_SAMPLES,
        "seeds": SEEDS,
        "conditions": COND_ORDER,
        "snapshot_seed": SEEDS[0],
        "sample_index": 0,
    }

    if version == 3:
        cfg.update(
            executor="snnlab.sim.GraphExecutor",
            evaluation_batch_size=64,
            evaluation_subset_seed=42,
            evaluation_encoder_seed=20260415,
            snapshot_encoder_stream="training seed after ordered CPU initialization draws/v1",
            reset_boundary="fresh neuronal, synaptic, refractory, delay and readout state per presentation",
            biophysics=dict(BIOPHYSICS),
            initialization=dict(INITIALIZATION),
            recording={
                "infer": "full-trial E spike events and E/I online counts",
                "snapshot": "full-trial E/I spike events",
            },
            interventions=[],
        )
    return cfg


def jobs(cfg):
    return [
        {
            **cell,
            "kind": kind,
            "path": f"{kind}/{cell['cell_name']}",
            **(
                {"samples": cfg["evaluation_samples"]}
                if kind == "infer"
                else {"sample_index": cfg["sample_index"]}
                if kind == "snapshot"
                else {}
            ),
        }
        for cell in bank_cells()
        for kind in ("infer", "weights_dump", "snapshot")
        if kind != "snapshot" or cell["seed"] == cfg["snapshot_seed"]
    ]


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
INITIALIZATION = {
    "discarded_readout_mean": 5.1,
    "discarded_readout_sd": 3.8,
    "relative_sd": 0.1,
    "input_zero_fraction": 0.95,
    "order": ["input", "input_mask", "readout", "EE", "EI", "IE", "II"],
}
CHECKPOINT_PARAMETERS = {
    "input_to_E.weight": "W_ff.0",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
    "readout_projection.weight": "W_ff.1",
}


def checkpoint_shapes(train):
    ne, ni = train["n_hidden"], train["n_inh"]
    return {
        "W_ff.0": (train["n_in"], ne),
        "W_ff.1": (ne, train["n_out"]),
        "W_ee.1": (ne, ne),
        "W_ei.1": (ne, ni),
        "W_ie.1": (ni, ne),
        "W_ii.1": (ni, ni),
    }


def initial_recurrence(train):
    """Retain the endpoint producer's ordered draws without rebuilding a simulator.

    Its discarded readout used 5.1/3.8 divided by fan-in, even though training
    used a separately configured readout initializer. Those draws precede EI/IE
    and the CPU reference-image encoder and must therefore remain here.
    """
    import torch

    generator = torch.Generator(device="cpu").manual_seed(train["seed"])
    ne, ni = train["n_hidden"], train["n_inh"]
    spec = INITIALIZATION
    torch.randn(train["n_in"], ne, generator=generator, dtype=torch.float32)
    torch.rand(train["n_in"], ne, generator=generator, dtype=torch.float32)
    torch.randn(ne, train["n_out"], generator=generator, dtype=torch.float32)
    torch.randn(ne, ne, generator=generator, dtype=torch.float32)
    strength = train["ei_strength"]
    ei = (
        torch.randn(ne, ni, generator=generator, dtype=torch.float32)
        .mul_(strength * spec["relative_sd"])
        .add_(strength)
        .clamp_(min=0)
        / ne
    )
    strength *= train["ei_ratio"]
    ie = (
        torch.randn(ni, ne, generator=generator, dtype=torch.float32)
        .mul_(strength * spec["relative_sd"])
        .add_(strength)
        .clamp_(min=0)
        / ni
    )
    torch.randn(ni, ni, generator=generator, dtype=torch.float32)
    return {"W_ei_1_init": ei.numpy(), "W_ie_1_init": ie.numpy()}, generator


def author_network(cfg, train):
    from experiments.helpers.ping import build_ping
    from snnlab import lang
    from snnlab.sim.timing import refractory_steps

    dt, b = train["dt"], cfg["biophysics"]
    net = lang.Network("exp049_recurrent_training", dt=dt * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", train["n_in"]),
        signal_type="spikes",
        unit="spike",
    )
    neurons = {}
    for label in ("E", "I"):
        neurons[label] = lang.COBA_LIF(
            tau_mem=b[f"C_m_{label}_nF"] / b[f"g_L_{label}_uS"] * lang.ms,
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
            voltage_grad_dampen=train["v_grad_dampen"],
        )
    circuit = build_ping(
        net,
        source=drive,
        n_e=train["n_hidden"],
        n_i=train["n_inh"],
        neuron_e=neurons["E"],
        neuron_i=neurons["I"],
        ampa=lang.AMPA(tau=train["tau_ampa_ms"] * lang.ms),
        gaba=lang.GABA(tau=train["tau_gaba_ms"] * lang.ms),
        input_weight=lang.Constant(0.0),
        ei_weight=lang.Constant(0.0),
        ie_weight=lang.Constant(0.0),
        recurrent_delay=dt * lang.ms,
        initialization_scaling="direct",
    )
    output = net.population(
        "readout",
        size=train["n_out"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=b["readout_tau_ms"] * lang.ms,
            soft_reset_threshold=b["readout_threshold"],
            surrogate_slope=train["surrogate_slope"],
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
    return lang.compile(net, target="tools/snnsim")


def recording(*, snapshot=False):
    from snnlab.sim.streaming import RecordingSpec, SignalRecording

    return RecordingSpec(
        signals=tuple(
            SignalRecording(f"{label}.spikes", kind="spike_events")
            for label in (("E", "I") if snapshot else ("E",))
        )
    )


def training_settings(cell):
    return {
        "hidden_sizes": [N_E],
        "n_hidden": N_E,
        "n_inh": N_I,
        "model": "ping",
        "dataset": "mnist",
        "dt": DT_TRAIN,
        "t_ms": T_MS,
        "epochs": EPOCHS,
        "max_samples": MAX_SAMPLES,
        "seed": cell["seed"],
        "tau_gaba_ms": 6.0,
        "ei_strength": {
            "frozen_ping": 1.0,
            "trainable_ping_init": 1.0,
            "trainable_zero_init": 0.0,
            "trainable_small_init": 0.1,
        }[cell["condition"]],
        "trainable_w_ei": cell["condition"] != "frozen_ping",
        "trainable_w_ie": cell["condition"] != "frozen_ping",
        "v_grad_dampen": 1000.0,
        "w_in": [0.9, 0.09],
        "w_in_initial_zero_fraction": 0.95,
        "readout_mode": "mem-mean",
        "surrogate_slope": 1.0,
        "readout_w_init_mean": 1.12060546875,
        "readout_w_init_std": 0.8349609375,
        "lr": 0.0004,
        "batch_size": 256,
        "fr_reg_upper_strength": 0.0,
        "fr_reg_upper_target_hz": 0.0,
    }
