"""Settings for the two constant-cue digit-routing excerpts."""

import math

import numpy as np
import torch
from snnlab import lang

SLUG = "exp164"
SCHEMA = "exp164.analysis/v1"
PATHS = ("A_E_to_C_E", "B_E_to_C_E", "C_I_to_E")


def configuration(
    *,
    source_configuration=None,
    end_ms=1100.0,
    window_ms=(650.0, 1100.0),
    closeup_ms=(900.0, 1050.0),
):
    if source_configuration is None:
        source_configuration = simulation_configuration()
    cfg = dict(
        schema=SCHEMA,
        display_end_ms=float(end_ms),
        window_ms=list(map(float, window_ms)),
        closeup_ms=list(map(float, closeup_ms)),
        rate_smoothing_sd_ms=1.0,
        min_peak_distance_ms=15.0,
        min_prominence_hz=0.05,
        prominence_fraction=0.1,
        reference_start_ms=500.0,
        low_inhibition_fraction=0.25,
        conductance_method="analytical linear filter of saved spikes; native snapshot weights and planner delays/decay; zero initial state; float64 means, not native conductance recordings",
    )
    times = [cfg["display_end_ms"], *cfg["window_ms"], *cfg["closeup_ms"]]
    if (
        len(cfg["window_ms"]) != 2
        or len(cfg["closeup_ms"]) != 2
        or not all(math.isfinite(t) for t in times)
    ):
        raise ValueError("finite two-endpoint windows are required")
    a, b = cfg["window_ms"]
    c, d = cfg["closeup_ms"]
    if (
        not 0 < a < b <= cfg["display_end_ms"] < source_configuration["switch_ms"]
        or not a <= c < d <= b
    ):
        raise ValueError(
            "measurement/close-up windows must precede the switch and nest inside the display"
        )
    if (
        source_configuration["encoding"] != "rates"
        or source_configuration["network"]["populations"]["E"]["size"] != 64
    ):
        raise ValueError("expected the fixed 8×8 rate encoder")
    return cfg


# Fixed standalone design defaults; historical trial bytes are not execution inputs.
def simulation_configuration(*, seed=101):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return dict(
        schema="exp164.recipe/v3",
        encoding="rates",
        seed=seed,
        network_seed=85,
        specification_origin="Recovered digit-routing recipe and recorded settings; fixed scalar defaults, no saved run required",
        protocol="two matched digit trials with opposite cue orders",
        duration_ms=2400.0,
        switch_ms=1200.0,
        dt_ms=0.1,
        sensory_hz=24.0,
        background_hz=28.0,
        cue_hz=8.0,
        contrast=0.4,
        kernel_pixels=0.45,
        coupling_us=0.6,
        labels=[0, 1],
        image_selection="first sample of each label in sklearn load_digits",
        input_encoding="per-channel Bernoulli(rate*dt/1000); matched sensory and background draws across conditions; shared cue draw gated by cue order",
        input_seed_offsets=dict(X_A=0, X_B=100, X_C=200, U=300),
        conditions=[
            dict(
                name="AB-cue-" + c,
                present="AB",
                order=c + ("B" if c == "A" else "A"),
                probe=False,
            )
            for c in "AB"
        ],
        incoming_weights_us=dict(
            PING_B_input_to_E=0.19995858244010378,
            PING_B_E_to_E=0.394999991171062,
            PING_B_E_to_I=0.7493030408397316,
            PING_B_I_to_E=1.501218814915046,
            PING_B_I_to_I=0.7599999830126762,
        ),
        network=dict(
            dt_ms=0.1,
            input_channels=1024,
            resting_mv=-65.0,
            threshold_mv=-50.0,
            reset_mv=-65.0,
            populations=dict(
                E=dict(size=64, capacitance_nf=1.0, leak_us=0.05, refractory_ms=3.0),
                I=dict(size=16, capacitance_nf=0.5, leak_us=0.1, refractory_ms=1.5),
            ),
            sensory=dict(gain=9.2, tau_ms=2.0, delay_ms=0.1),
            local=dict(
                E_to_E=dict(tau_ms=2.0),
                E_to_I=dict(tau_ms=1.0),
                I_to_E=dict(tau_ms=9.0),
                I_to_I=dict(tau_ms=6.0),
            ),
            cross=dict(tau_ms=0.5, delay_ms=2.0),
        ),
    )


def digit_images(cfg):
    from sklearn.datasets import load_digits

    dataset = load_digits()
    indices = [
        int(np.flatnonzero(dataset.target == label)[0]) for label in cfg["labels"]
    ]
    return dict(
        A=dataset.images[indices[0]].astype(np.float64),
        B=dataset.images[indices[1]].astype(np.float64),
        indices=np.asarray(indices),
        labels=np.asarray(cfg["labels"]),
    )


def make_inputs(cfg, images):
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    channels = cfg["network"]["input_channels"]
    drives = {}
    for name in ["X_A", "X_B", "X_C", "U"]:
        rate = (
            cfg["cue_hz"]
            if name == "U"
            else cfg["background_hz"]
            if name == "X_C"
            else cfg["sensory_hz"]
        )
        if name in ["X_A", "X_B"]:
            pixels = images[name[-1]].reshape(-1).astype(np.float64)
            rate = rate * np.repeat(
                (1 - cfg["contrast"]) + cfg["contrast"] * pixels / pixels.mean(),
                channels // 64,
            )
        rng = np.random.default_rng(cfg["seed"] + cfg["input_seed_offsets"][name])
        threshold = np.asarray(rate * cfg["dt_ms"] / 1000, dtype=np.float32)
        spikes = (
            rng.random((steps, 1, channels), dtype=np.float32) < threshold
        ).astype(np.uint8)
        drives[name] = np.repeat(spikes, len(cfg["conditions"]), axis=1)
    cue = drives.pop("U")
    before = np.arange(steps) * cfg["dt_ms"] < cfg["switch_ms"]
    for c in "AB":
        active = np.stack(
            [
                np.where(before, case["order"][0] == c, case["order"][1] == c)
                for case in cfg["conditions"]
            ],
            axis=1,
        )
        drives["U_" + c] = cue * active[:, :, None]
    return drives


def spatial_parameters(parameters, cfg):
    # Pixel blocks preserve total incoming sensory weight; C excitation is topographic.
    for c in "AB":
        target = parameters["sensory_" + c + ".weight"]
        block = target.shape[0] // 64
        mask = (
            torch.arange(target.shape[0])[:, None] // block == torch.arange(64)[None, :]
        )
        target.mul_(mask.to(target.dtype) * 64)
        y, x = np.indices((8, 8))
        xy = np.stack([y.ravel(), x.ravel()], axis=1)
        distance = ((xy[:, None] - xy[None, :]) ** 2).sum(axis=2)
        kernel = np.exp(-0.5 * distance / cfg["kernel_pixels"] ** 2)
        kernel /= kernel.sum(axis=0, keepdims=True)
        parameters[c + "_E_to_C_E.weight"].copy_(
            torch.as_tensor(kernel * cfg["coupling_us"], dtype=torch.float32)
        )


def author_network(cfg):
    n = cfg["network"]
    net = lang.Network("exp164_digit_routing", dt=n["dt_ms"] * lang.ms)
    source_weights = cfg["incoming_weights_us"]
    weights = {}
    cells = {}
    inputs = {}

    def connect(src, target, name, total, tau, delay=0.1, modulation=None):
        # Native direct matrices have the requested incoming sum at every cell.
        source_size = (
            n["input_channels"]
            if name.startswith(("sensory_", "cue_"))
            else n["populations"]["E" if "_E_to_" in name else "I"]["size"]
        )
        recurrent = name.endswith(("_E_to_E", "_I_to_I"))
        if recurrent:
            source_size -= 1
        net.connect(
            src,
            target,
            name=name,
            synapse=(lang.GABA if "_I_to_" in name else lang.AMPA)(tau=tau * lang.ms),
            weight=lang.Constant(total / source_size),
            initialization_scaling="direct",
            constraint=lang.NonNegative(),
            delay=delay * lang.ms,
            **(dict(connection="recurrent") if recurrent else {}),
            **(dict(modulation=modulation) if modulation is not None else {}),
        )
        weights[name] = total

    for c in "ABC":
        inputs[c] = net.input(
            "X_" + c,
            shape=("time", "batch", n["input_channels"]),
            signal_type="spikes",
            unit="spike",
        )
        cells[c] = {}
        for p, phys in n["populations"].items():
            cells[c][p] = net.population(
                c + "_" + p,
                size=phys["size"],
                neuron=lang.COBA_LIF(
                    tau_mem=phys["capacitance_nf"] / phys["leak_us"] * lang.ms,
                    capacitance_nf=phys["capacitance_nf"],
                    leak_us=phys["leak_us"],
                    resting_mv=n["resting_mv"],
                    threshold_mv=n["threshold_mv"],
                    reset_mv=n["reset_mv"],
                    initial_voltage_mv=n["resting_mv"],
                    refractory=phys["refractory_ms"] * lang.ms,
                    voltage_grad_dampen=80.0,
                ),
            )
        connect(
            inputs[c],
            cells[c]["E"].excitatory,
            "sensory_" + c,
            source_weights["PING_B_input_to_E"] * n["sensory"]["gain"],
            n["sensory"]["tau_ms"],
        )
        for path, phys in n["local"].items():
            s, t = path.split("_to_")
            connect(
                cells[c][s].spikes,
                getattr(cells[c][t], "inhibitory" if s == "I" else "excitatory"),
                c + "_" + path,
                source_weights["PING_B_" + path],
                phys["tau_ms"],
            )
        for p in "EI":
            net.expose(cells[c][p].spikes, name=c + "_" + p + "_spikes")
    for c in "AB":
        cue = net.input(
            "U_" + c,
            shape=("time", "batch", n["input_channels"]),
            signal_type="spikes",
            unit="spike",
        )
        connect(
            cue,
            cells[c]["E"].excitatory,
            "cue_" + c,
            source_weights["PING_B_input_to_E"] * n["sensory"]["gain"],
            n["sensory"]["tau_ms"],
        )
        for p in "EI":
            connect(
                cells[c]["E"].spikes,
                cells["C"][p].excitatory,
                c + "_E_to_C_" + p,
                cfg["coupling_us"],
                n["cross"]["tau_ms"],
                delay=n["cross"]["delay_ms"],
            )
    return lang.compile(net, target=None), weights
