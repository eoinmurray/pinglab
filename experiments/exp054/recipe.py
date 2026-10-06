"""Committed coupling-grid and null-control recipe; no execution on import."""

import numpy as np

SLUG = "exp054"
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
FIGURES = (
    "turnon_maps_compound.png",
    "turnon_compound.png",
    "grid_maps.png",
    "grid_rasters.png",
    "grid_autocorr.png",
    "rate_invariance.png",
    "null_autocorr.png",
)


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }


def configuration(*, smoke=False, version=8):
    if version not in (7, 8):
        raise ValueError("unsupported exp054 recipe version")
    cfg = {
        "schema": f"exp054.recipe/v{version}",
        **refractory_configuration(),
        "profile": "smoke" if smoke else "production",
        "dt_ms": 0.1,
        "tau_gaba_ms": 6.0,
        "sim_ms": 400.0 if smoke else 1000.0,
        "burn_ms": 100.0,
        "n_e": 1024,
        "n_i": 256,
        "seed": 42,
        "input_rate_hz": 100.0,
        "private_w_in": 0.5,
        "shared_n_in": 200,
        "shared_w_in": 0.2,
        "shared_zero_fraction": 0.95,
        "max_lag_ms": 100.0,
        "bin_ms": 1.0,
        "wei_mean": [round(v, 3) for v in np.linspace(0, 3, 6 if smoke else 11)],
        "wie_mean": [round(v, 3) for v in np.linspace(0, 6, 6 if smoke else 11)],
        "private_null_hz": [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 70.0, 100.0],
        "shared_null_hz": [8.0, 12.0, 16.0, 20.0, 28.0, 40.0, 60.0, 100.0],
        "display_window_ms": 200.0,
        "display_e": 160,
        "display_i": 48,
        "display_stride": 1 if smoke else 2,
    }
    if version == 8:
        cfg.update(
            executor="snnlab.sim.GraphExecutor",
            biophysics=dict(BIOPHYSICS),
            initialization_stream="exp054 ordered CPU draws/v1",
            encoder_seed=43,
            weight_initialization={
                "relative_sd": 0.1,
                "discarded_readout": {"width": 10, "mean": 5.1, "sd": 3.8},
                "discarded_ee": {"mean": 0.0, "sd": 0.0},
            },
            reset="fresh state per probe; resting voltage; zero conductances and refractory",
        )
    return cfg


def validate(cfg):
    if cfg not in tuple(
        configuration(smoke=smoke, version=version)
        for smoke in (False, True)
        for version in (7, 8)
    ):
        from pingstore.contracts import PingstoreError

        raise PingstoreError("inconsistent exp054 recipe")
    return cfg


def job(cfg, wei, wie, rate, private=True):
    return {
        "id": f"{'priv' if private else 'shared'}_wei{wei:g}_wie{wie:g}_r{rate:g}_T{cfg['sim_ms']:g}",
        "wei": wei,
        "wie": wie,
        "rate_hz": rate,
        "private": private,
    }


def jobs(cfg):
    candidates = [
        job(cfg, e, i, cfg["input_rate_hz"])
        for i in cfg["wie_mean"]
        for e in cfg["wei_mean"]
    ]
    candidates += [
        job(cfg, 0.0, 0.0, rate, private)
        for private, key in ((True, "private_null_hz"), (False, "shared_null_hz"))
        for rate in cfg[key]
    ]
    # The origin at 100 Hz is the same seeded probe in the grid and private scan.
    return list({item["id"]: item for item in candidates}.values())


def turnon_points(cfg):
    weak = 1 if cfg["profile"] == "smoke" else 2
    return [
        ("A", 0, 0),
        ("B", weak, weak),
        ("C", len(cfg["wei_mean"]) - 1, len(cfg["wie_mean"]) - 1),
    ]


BIOPHYSICS = {
    "capacitance_e_nf": 1.0,
    "capacitance_i_nf": 0.5,
    "leak_e_us": 0.05,
    "leak_i_us": 0.10,
    "resting_mv": -65.0,
    "threshold_mv": -50.0,
    "reset_mv": -65.0,
    "tau_ampa_ms": 2.0,
    "reversal_ampa_mv": 0.0,
    "reversal_gaba_mv": -80.0,
}


def author_network(cfg, item):
    from experiments.helpers.ping import build_ping
    from snnlab import lang
    from snnlab.sim.timing import refractory_steps

    b = cfg["biophysics"]
    net = lang.Network("exp054_rhythmicity", dt=cfg["dt_ms"] * lang.ms)
    channels = cfg["n_e"] if item["private"] else cfg["shared_n_in"]
    drive = net.input(
        "drive", shape=("time", "batch", channels), signal_type="spikes", unit="spike"
    )
    neurons = {}
    for label in ("e", "i"):
        neurons[label] = lang.COBA_LIF(
            tau_mem=b[f"capacitance_{label}_nf"] / b[f"leak_{label}_us"] * lang.ms,
            capacitance_nf=b[f"capacitance_{label}_nf"],
            leak_us=b[f"leak_{label}_us"],
            resting_mv=b["resting_mv"],
            threshold_mv=b["threshold_mv"],
            reset_mv=b["reset_mv"],
            refractory_steps=refractory_steps(
                cfg[f"refractory_{label}_ms"],
                cfg["dt_ms"],
                policy=cfg["refractory_policy"],
            ),
            voltage_grad_dampen=80.0,
        )
    build_ping(
        net,
        source=drive,
        n_e=cfg["n_e"],
        n_i=cfg["n_i"],
        neuron_e=neurons["e"],
        neuron_i=neurons["i"],
        ampa=lang.AMPA(tau=b["tau_ampa_ms"] * lang.ms),
        gaba=lang.GABA(tau=cfg["tau_gaba_ms"] * lang.ms),
        input_weight=lang.Constant(0.0),
        ei_weight=lang.Constant(0.0),
        ie_weight=lang.Constant(0.0),
        recurrent_delay=cfg["dt_ms"] * lang.ms,
        initialization_scaling="direct",
    )
    return lang.compile(net, target="tools/snnsim")


def initial_parameters(cfg, item):
    """Preserve the probe's ordered CPU stream without a simulator or adapter.

    The discarded output and zero E→E draws precede reciprocal weights in
    the original protocol. Private identity replacement occurs after drawing
    the input matrix; shared inputs also consume compensated Bernoulli zeroing.
    Runtime matrices use [source, target] orientation.
    """
    import torch

    generator = torch.Generator().manual_seed(cfg["seed"])
    n_e, n_i = cfg["n_e"], cfg["n_i"]
    channels = n_e if item["private"] else cfg["shared_n_in"]

    def normal(shape, mean, sd):
        return (
            torch.randn(*shape, generator=generator).mul_(sd).add_(mean).clamp_(min=0)
        )

    initialization = cfg["weight_initialization"]
    relative_sd = initialization["relative_sd"]
    mean = cfg["private_w_in"] if item["private"] else cfg["shared_w_in"]
    external = normal((channels, n_e), mean, mean * relative_sd)
    if not item["private"]:
        external = (
            external
            * (
                torch.rand(channels, n_e, generator=generator)
                > cfg["shared_zero_fraction"]
            ).float()
        )
        external = external / (1.0 - cfg["shared_zero_fraction"])
    external = external / channels
    readout = initialization["discarded_readout"]
    normal((n_e, readout["width"]), readout["mean"], readout["sd"])
    ee = initialization["discarded_ee"]
    normal((n_e, n_e), ee["mean"], ee["sd"])
    ei = normal((n_e, n_i), item["wei"], item["wei"] * relative_sd) / n_e
    ie = normal((n_i, n_e), item["wie"], item["wie"] * relative_sd) / n_i
    if item["private"]:
        external = torch.eye(n_e) * cfg["private_w_in"]
    return {"input_to_E.weight": external, "E_to_I.weight": ei, "I_to_E.weight": ie}


def recording(cfg):
    from snnlab.sim.streaming import MeasurementWindow, RecordingSpec, SignalRecording
    from snnlab.sim.timing import duration_steps

    return RecordingSpec(
        signals=tuple(
            SignalRecording(f"{label}.spikes", kind="spike_events")
            for label in ("E", "I")
        ),
        window=MeasurementWindow(
            duration_steps(cfg["burn_ms"], cfg["dt_ms"]),
            duration_steps(cfg["sim_ms"], cfg["dt_ms"]),
        ),
    )
