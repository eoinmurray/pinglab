"""Frozen TR02 (no rate penalty) inhibitory-pool comparison, authored with native SNNLab."""

from snnlab import lang

SLUG = "exp173"
MAPPING = {
    "input_to_E.weight": "W_ff.0",
    "readout_projection.weight": "W_ff.1",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
}


def configuration(*, source_run, source_digest):
    if not source_run or not source_digest:
        raise ValueError("Supply the explicit source run and digest")
    return dict(
        source_run=source_run,
        source_digest=source_digest,
        source_cell="ping__off__seed42",
        checkpoint_role="final_epoch",
        seed=42,
        original_n_i=256,
        pool_sizes=[32, 64, 128, 256],
        batch_size=1,
        device="cpu",
        threads=1,
        training=False,
        accuracy=True,
        accuracy_images=1000,
        accuracy_batch_size=64,
        accuracy_sample_seed=42,
        accuracy_encoder_seed=20260415,
        accuracy_draws=1,
        accuracy_partition="official MNIST test partition; uniform without replacement, no stratification",
        accuracy_protocol="Manuscript Figure 3D / exp025: fixed 1000-image subset, seed 42; CPU encoder 20260415, batch 64",
        accuracy_readout="argmax of trial-mean pre-reset readout voltage",
        inhibitory_selection_seed=42,
        ie_scaling="original_n_i / n_i; E-to-I unchanged",
        evaluation_partition="source validation split",
        measurement_window="full realised presentation; fresh state per image",
        rate_definition="spike count / (population size * duration_s)",
        rate_aggregation="single-image population mean from the raster trial",
        encoding_policy="one CPU encoder seed; identical digit-zero input across sizes",
        retention_estimate="<150 MB and <120 files: four bundles/start snapshots, single-trial counts, four sparse rasters, four diagrams; no full-dataset spike recordings",
    )


def selected_indices(n_i, original_n_i=256, seed=42):
    import numpy as np

    if n_i not in (32, 64, 128, original_n_i):
        raise ValueError("Unsupported inhibitory-pool size")
    if n_i == original_n_i:
        return np.arange(original_n_i)
    return np.random.default_rng(seed).permutation(original_n_i)[:n_i]


def resize_weights(weights, n_i):
    """Preserve stored source-to-target convention and exact original control."""
    indices = selected_indices(n_i)
    result = {name: weights[key].clone() for name, key in MAPPING.items()}
    if n_i != 256:
        result["E_to_I.weight"] = result["E_to_I.weight"][:, indices].clone()
        result["I_to_E.weight"] = result["I_to_E.weight"][indices, :].clone() * (
            256 / n_i
        )
    return result, indices


def test_sample_indices(*, total_images=10000, samples=1000, seed=42):
    """Reproduce the manuscript's ordered, unstratified official-test subset."""
    import numpy as np

    if type(samples) is not int or not 1 <= samples <= total_images:
        raise ValueError("Invalid test subset size")
    return np.random.RandomState(seed).choice(total_images, samples, replace=False)


def author_network(cfg, n_i):
    net = lang.Network("TR02_off_pool_size", dt=cfg["dt_ms"] * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", cfg["n_input"]),
        signal_type="spikes",
        unit="spike",
    )
    pops = {}
    for name, size in (("E", cfg["n_e"]), ("I", n_i)):
        neuron = dict(cfg["neurons"][name])
        neuron["tau_mem"] = neuron.pop("tau_mem_ms") * lang.ms
        neuron["refractory"] = neuron.pop("refractory_ms") * lang.ms
        pops[name] = net.population(name, size=size, neuron=lang.COBA_LIF(**neuron))
    for name, src, dst, tau, inhibitory, recurrent in (
        ("input_to_E", drive, pops["E"].excitatory, cfg["tau_ampa_ms"], False, False),
        (
            "E_to_I",
            pops["E"].spikes,
            pops["I"].excitatory,
            cfg["tau_ampa_ms"],
            False,
            True,
        ),
        (
            "I_to_E",
            pops["I"].spikes,
            pops["E"].inhibitory,
            cfg["tau_gaba_ms"],
            True,
            True,
        ),
    ):
        net.connect(
            src,
            dst,
            name=name,
            synapse=(lang.GABA if inhibitory else lang.AMPA)(tau=tau * lang.ms),
            weight=lang.Constant(0),
            initialization_scaling="direct",
            constraint=lang.NonNegative(),
            delay=(cfg["dt_ms"] if recurrent else 0) * lang.ms,
            **({"connection": "recurrent"} if recurrent else {}),
        )
    readout = net.population(
        "readout",
        size=cfg["n_output"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=cfg["readout_tau_ms"] * lang.ms,
            soft_reset_threshold=cfg["readout_threshold"],
            initial_voltage=0,
            surrogate_slope=cfg["surrogate_slope"],
        ),
    )
    net.connect(
        pops["E"].spikes,
        readout.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=cfg["readout_tau_ms"] * lang.ms),
        weight=lang.Constant(0),
        initialization_scaling="direct",
        constraint=lang.NonNegative(),
    )
    net.output(
        "class_scores",
        lang.reduce(
            readout.pre_reset_voltage,
            operation="mean",
            over="time",
            name="mean_class_scores",
        ),
    )
    for name in ("E", "I"):
        net.output(
            name + "_count",
            lang.reduce(
                pops[name].spikes, operation="sum", over="time", name=name + "_counts"
            ),
        )
        net.expose(pops[name].spikes, name=name + "_spikes")
    return lang.compile(net)
