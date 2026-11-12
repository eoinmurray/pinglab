"""Frozen no-rate-penalty models: decision time and cumulative spike budget."""

from snnlab import lang

SLUG = "exp174"
KINDS = ("coba", "ping", "artificial")
LABELS = ("COBA", "PING", "Artificial I")
MAPPING = {
    "input_to_E.weight": "W_ff.0",
    "readout_projection.weight": "W_ff.1",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
}


def configuration(**overrides):
    cfg = dict(
        source_run="exp171-r061-compute",
        source_digest="sha256:ce9438b46834dce6c2302eb9ff28186bad7b2ae6d00ac7708bdbb59c4ed70be8",
        bank_run="exp022-r015-compute",
        bank_digest="sha256:a542a714021fd7bb69c5d08ccf006763f19ce83b01d8ecdca86dba150118491d",
        test_origin="exp171-r057-compute",
        samples=1000,
        batch_size=16,
        manuscript_batch_size=64,
        dt_ms=0.1,
        duration_ms=200.0,
        n_input=784,
        n_e=1024,
        n_i=256,
        n_output=10,
        input_rate_hz=25.0,
        tau_ampa_ms=2.0,
        tau_gaba_ms=6.0,
        seed=42,
        inhibition_seed=171006,
        inhibition_prehistory_steps=600,
        threshold_accuracy=0.8,
        device="cpu",
        threads=1,
        recording_block_steps=100,
        checkpoint_role="final_epoch",
        training=False,
        rate_penalty=0.0,
        subset_policy="first N of exp171 ordered 1000-image official MNIST test subset; no resampling",
        sensory_policy="restore original CPU Torch RNG state for each full 64-image batch, then slice",
        inhibition_policy="CPU Torch per-image Poisson draw; seed derived from [171006, 0, original image ID]; frozen exp171 fits",
        decision_policy="argmax running mean pre-reset readout voltage; ties choose lowest digit",
        latency_policy="earliest timestep with accuracy >= 80% at every remaining timestep through 200 ms",
        spike_policy="population E spike count since stimulus onset, averaged over all images at the common threshold time",
        cost_scope="stimulus window only; excludes sensory/readout spikes, teacher calibration and artificial-I prehistory; I spikes and artificial events reported separately",
        retention_estimate="<200 MB / <100 files for 1000 images: three bundles/start snapshots, dataset fits and RNG states, per-image prefix predictions/population counts; no neuron rasters",
    )
    if set(overrides) - set(cfg):
        raise ValueError("Unknown configuration override")
    cfg.update(overrides)
    if type(cfg["samples"]) is not int or not 1 <= cfg["samples"] <= 1000:
        raise ValueError("Samples must be in 1..1000 for this retained subset")
    if type(cfg["batch_size"]) is not int or cfg["batch_size"] < 1:
        raise ValueError("Batch size must be a positive integer")
    if cfg["device"] != "cpu" or cfg["training"] or cfg["rate_penalty"] != 0:
        raise ValueError("This prototype is frozen CPU inference without rate penalty")
    cfg["steps"] = round(cfg["duration_ms"] / cfg["dt_ms"])
    return cfg


def author_network(cfg, kind):
    """Author baseline architectures; artificial-I uses its authenticated native bundle."""
    if kind not in ("coba", "ping"):
        raise ValueError("Only baseline models are authored here")
    net = lang.Network("TR02_off_" + kind, dt=cfg["dt_ms"] * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", cfg["n_input"]),
        signal_type="spikes",
        unit="spike",
    )
    pops = {}
    for name, size, c, leak, ref in (
        ("E", cfg["n_e"], 1.0, 0.05, 1.2),
        ("I", cfg["n_i"], 0.5, 0.1, 0.6),
    ):
        pops[name] = net.population(
            name,
            size=size,
            neuron=lang.COBA_LIF(
                tau_mem=c / leak * lang.ms,
                capacitance_nf=c,
                leak_us=leak,
                resting_mv=-65.0,
                threshold_mv=-50.0,
                reset_mv=-65.0,
                initial_voltage_mv=-65.0,
                refractory=ref * lang.ms,
                voltage_grad_dampen=1000.0,
            ),
        )
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
    out = net.population(
        "readout",
        size=cfg["n_output"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=2.0 * lang.ms,
            soft_reset_threshold=1.0,
            initial_voltage=0.0,
            surrogate_slope=1.0,
        ),
    )
    net.connect(
        pops["E"].spikes,
        out.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=2.0 * lang.ms),
        weight=lang.Constant(0),
        initialization_scaling="direct",
        constraint=lang.NonNegative(),
    )
    net.output(
        "scores",
        lang.reduce(
            out.pre_reset_voltage, operation="mean", over="time", name="mean_scores"
        ),
    )
    for name in ("E", "I"):
        net.output(
            name.lower() + "_count",
            lang.reduce(
                pops[name].spikes, operation="sum", over="time", name=name + "_counts"
            ),
        )
    return lang.compile(net, refractory_timing="following_steps")
