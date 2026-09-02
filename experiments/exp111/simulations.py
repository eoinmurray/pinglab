"""Independent numerical protocols used by exp111 compute.

The Brian2 side repeats equations and scheduling explicitly.  It never imports
snnsim's update functions.  Random event streams are realised once and supplied
to both simulators.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import brian2 as b2
import numpy as np
import torch
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools/snnsim"), str(ROOT / "tools")]

import tool as snnsim_tool  # noqa: E402
from config import build_net, set_sim_dt, setup_model_globals  # noqa: E402
from encoders import encode_images_poisson  # noqa: E402
from experiments.helpers.datasets import load_mnist_split  # noqa: E402

from . import recipe

M = snnsim_tool.M

b2.BrianLogger.suppress_name("resolution_conflict")

EL, EE, EI, VTH, VRESET = -65.0, 0.0, -80.0, -50.0, -65.0


def _series(identifier, title, source, x_label, y_label, rows):
    return {
        "id": identifier,
        "title": title,
        "source_experiment": source,
        "x_label": x_label,
        "y_label": y_label,
        "series": rows,
    }


def _lif_snnsim(*, dt, duration, c, gl, refractory, ge, gi, initial=EL):
    v = torch.tensor([[initial]], dtype=torch.float64)
    ref = torch.zeros((1, 1), dtype=torch.long)
    values, spikes = [], []
    for step in range(round(duration / dt)):
        v, spike, ref = M.lif_step_expeuler(
            v,
            ref,
            torch.tensor([[ge]], dtype=torch.float64),
            torch.tensor([[gi]], dtype=torch.float64),
            c,
            gl,
            max(1, round(refractory / dt)),
            M.spike_biophysical,
            dt_override=dt,
        )
        values.append(v.item())
        if spike.item():
            spikes.append(step * dt)
    return np.asarray(values), np.asarray(spikes)


def _lif_brian(*, dt, duration, c, gl, refractory, ge, gi, initial=EL):
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = dt * b2.ms
    group = b2.NeuronGroup(
        1,
        """
        dv/dt=(-gL*(v-EL)-ge*(v-EE)-gi*(v-EI))/C:volt (unless refractory)
        ge:siemens
        gi:siemens
        """,
        threshold="v>=Vth",
        reset="v=Vreset",
        refractory=refractory * b2.ms,
        method="exact",
        namespace={
            "gL": gl * b2.usiemens,
            "EL": EL * b2.mV,
            "EE": EE * b2.mV,
            "EI": EI * b2.mV,
            "C": c * b2.nfarad,
            "Vth": VTH * b2.mV,
            "Vreset": VRESET * b2.mV,
        },
    )
    group.v = initial * b2.mV
    group.ge = ge * b2.usiemens
    group.gi = gi * b2.usiemens
    voltage = b2.StateMonitor(group, "v", record=True, when="end")
    spikes = b2.SpikeMonitor(group)
    b2.Network(group, voltage, spikes).run(duration * b2.ms)
    return np.asarray(voltage.v[0] / b2.mV), np.asarray(spikes.t / b2.ms)


def _synapse_traces(dt=0.1, steps=100):
    events = np.zeros(steps)
    events[[0, 3, 7, 31, 32, 70]] = 1
    rows = []
    for tau in (2.0, 4.5, 6.0, 9.0, 12.0, 18.0, 27.0):
        decay = math.exp(-dt / tau)
        snn, value = [], 0.0
        for event in events:
            value = value * decay + event * 0.3
            snn.append(value)
        b2.start_scope()
        b2.prefs.codegen.target = "numpy"
        b2.defaultclock.dt = dt * b2.ms
        target = b2.NeuronGroup(1, "dg/dt=-g/tau:siemens", method="exact")
        source = b2.SpikeGeneratorGroup(
            1,
            np.zeros(int(events.sum()), dtype=int),
            np.flatnonzero(events) * dt * b2.ms,
        )
        synapse = b2.Synapses(source, target, on_pre="g_post += weight")
        synapse.connect()
        monitor = b2.StateMonitor(target, "g", record=True, when="end")
        b2.Network(target, source, synapse, monitor).run(
            steps * dt * b2.ms,
            namespace={"tau": tau * b2.ms, "weight": 0.3 * b2.usiemens},
        )
        brian = np.asarray(monitor.g[0] / b2.usiemens)
        rows.append((tau, np.asarray(snn), brian))
    return rows


def _event_arrays(events, n):
    if not events:
        return np.empty((0, 2), dtype=float)
    return np.asarray(events, dtype=float).reshape(-1, 2)


def _ping_snnsim(*, dt=0.1, duration=500.0, tau=6.0, loop=1.0, drive=1.0, seed=0):
    n_e, n_i = recipe.N_E_REDUCED, recipe.N_I_REDUCED
    steps = round(duration / dt)
    v_e = torch.linspace(-67, -63, n_e, dtype=torch.float64)[None]
    v_i = torch.linspace(-66, -64, n_i, dtype=torch.float64)[None]
    r_e = torch.zeros((1, n_e), dtype=torch.long)
    r_i = torch.zeros((1, n_i), dtype=torch.long)
    ge_i = torch.zeros((1, n_i), dtype=torch.float64)
    gi_e = torch.zeros((1, n_e), dtype=torch.float64)
    s_e = torch.zeros((1, n_e), dtype=torch.float64)
    s_i = torch.zeros((1, n_i), dtype=torch.float64)
    w_ei = torch.full((n_e, n_i), loop / n_e, dtype=torch.float64)
    w_ie = torch.full((n_i, n_e), 3 * loop / n_i, dtype=torch.float64)
    tonic = drive * torch.linspace(0.135, 0.165, n_e, dtype=torch.float64)[None]
    e_events, i_events = [], []
    for step in range(steps):
        ge_i = ge_i * math.exp(-dt / 2.0) + s_e @ w_ei
        gi_e = gi_e * math.exp(-dt / tau) + s_i @ w_ie
        v_e, s_e, r_e = M.lif_step_expeuler(
            v_e,
            r_e,
            tonic,
            gi_e,
            1.0,
            0.05,
            max(1, round(3 / dt)),
            M.spike_biophysical,
            dt_override=dt,
        )
        v_i, s_i, r_i = M.lif_step_expeuler(
            v_i,
            r_i,
            ge_i,
            None,
            0.5,
            0.1,
            max(1, round(1.5 / dt)),
            M.spike_biophysical,
            dt_override=dt,
        )
        e_events.extend((step * dt, int(i)) for i in torch.where(s_e[0])[0])
        i_events.extend((step * dt, int(i)) for i in torch.where(s_i[0])[0])
    return _event_arrays(e_events, n_e), _event_arrays(i_events, n_i)


def _ping_brian(*, dt=0.1, duration=500.0, tau=6.0, loop=1.0, drive=1.0, seed=0):
    n_e, n_i = recipe.N_E_REDUCED, recipe.N_I_REDUCED
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = dt * b2.ms
    equations = """
        dv/dt=(-gL*(v-EL)-gdrive*(v-EE)-ge*(v-EE)-gi*(v-EI))/C:volt (unless refractory)
        dge/dt=-ge/tau_e:siemens
        dgi/dt=-gi/tau_i:siemens
        gdrive:siemens (constant)
        gL:siemens (constant)
        C:farad (constant)
    """
    namespace = {
        "EL": EL * b2.mV,
        "EE": EE * b2.mV,
        "EI": EI * b2.mV,
        "Vth": VTH * b2.mV,
        "Vreset": VRESET * b2.mV,
        "tau_e": 2 * b2.ms,
        "tau_i": tau * b2.ms,
    }
    e = b2.NeuronGroup(
        n_e,
        equations,
        threshold="v>=Vth",
        reset="v=Vreset",
        refractory=3 * b2.ms,
        method="exponential_euler",
        namespace=namespace,
    )
    i = b2.NeuronGroup(
        n_i,
        equations,
        threshold="v>=Vth",
        reset="v=Vreset",
        refractory=1.5 * b2.ms,
        method="exponential_euler",
        namespace=namespace,
    )
    e.v = np.linspace(-67, -63, n_e) * b2.mV
    i.v = np.linspace(-66, -64, n_i) * b2.mV
    e.gL, e.C = 0.05 * b2.usiemens, 1 * b2.nfarad
    i.gL, i.C = 0.1 * b2.usiemens, 0.5 * b2.nfarad
    e.gdrive = drive * np.linspace(0.135, 0.165, n_e) * b2.usiemens
    i.gdrive = 0 * b2.usiemens
    for group in (e, i):
        group.ge = group.gi = 0 * b2.usiemens
    e_i = b2.Synapses(e, i, on_pre=f"ge_post += {loop / n_e}*usiemens")
    i_e = b2.Synapses(i, e, on_pre=f"gi_post += {3 * loop / n_i}*usiemens")
    e_i.connect()
    i_e.connect()
    e_spikes, i_spikes = b2.SpikeMonitor(e), b2.SpikeMonitor(i)
    b2.Network(e, i, e_i, i_e, e_spikes, i_spikes).run(duration * b2.ms)
    return (
        np.column_stack((np.asarray(e_spikes.t / b2.ms), np.asarray(e_spikes.i))),
        np.column_stack((np.asarray(i_spikes.t / b2.ms), np.asarray(i_spikes.i))),
    )


def _observables(events, n, duration, *, discard=50.0, bin_ms=1.0):
    edges = np.arange(0, duration + bin_ms, bin_ms)
    trace = np.histogram(events[:, 0] if len(events) else [], edges)[0] / n
    keep = edges[:-1] >= discard
    trace = trace[keep]
    analysed_s = (duration - discard) / 1000
    rate = float(np.sum(trace) / analysed_s)
    centred = trace - trace.mean()
    freq = np.fft.rfftfreq(len(centred), bin_ms / 1000)
    power = np.abs(np.fft.rfft(centred)) ** 2
    band = (freq >= 5) & (freq <= 150)
    peak = float(freq[band][np.argmax(power[band])]) if np.any(power[band]) else 0.0
    if len(trace) > 1 and np.var(trace) > 0:
        ac = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
        ac /= ac[0]
        contrast = float(np.max(ac[1:100]) - np.min(ac[1:100]))
    else:
        contrast = 0.0
    return {"rate_hz": rate, "frequency_hz": peak, "contrast": contrast, "trace": trace}


def _reduced_pair(**kwargs):
    snn_events = _ping_snnsim(**kwargs)
    brian_events = _ping_brian(**kwargs)
    duration = kwargs.get("duration", recipe.REDUCED_DURATION_MS)
    return {
        "snnsim_e": _observables(snn_events[0], recipe.N_E_REDUCED, duration),
        "snnsim_i": _observables(snn_events[1], recipe.N_I_REDUCED, duration),
        "brian2_e": _observables(brian_events[0], recipe.N_E_REDUCED, duration),
        "brian2_i": _observables(brian_events[1], recipe.N_I_REDUCED, duration),
        "snnsim_events": snn_events,
        "brian2_events": brian_events,
    }


def _cycle_fractions(events, duration, dt=0.1):
    e_events, i_events = events
    steps = round(duration / dt)
    e = np.zeros((steps, recipe.N_E_REDUCED), dtype=np.uint8)
    i = np.zeros((steps, recipe.N_I_REDUCED), dtype=np.uint8)
    if len(e_events):
        e[np.rint(e_events[:, 0] / dt).astype(int), e_events[:, 1].astype(int)] = 1
    if len(i_events):
        i[np.rint(i_events[:, 0] / dt).astype(int), i_events[:, 1].astype(int)] = 1
    sigma = 1 / dt
    x = np.arange(-math.ceil(4 * sigma), math.ceil(4 * sigma) + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(i.sum(1), kernel, mode="same")
    peaks, _ = find_peaks(smooth, distance=round(10 / dt), height=0.05 * smooth.max())
    boundaries = np.concatenate(([0], ((peaks[:-1] + peaks[1:]) // 2), [steps]))
    counts = np.concatenate(
        [e[a:b].sum(0) for a, b in zip(boundaries[:-1], boundaries[1:])]
    )
    buckets = np.asarray(
        [
            (counts == 0).sum(),
            (counts == 1).sum(),
            (counts == 2).sum(),
            (counts >= 3).sum(),
        ],
        float,
    )
    fractions = buckets / buckets.sum()
    active_single = float((counts == 1).sum() / max(1, (counts >= 1).sum()))
    return fractions, active_single


def _checkpoint_paths(bank, name):
    directory = bank / name
    if not directory.is_dir():
        raise FileNotFoundError(f"missing training condition {name}")
    return directory, json.loads((directory / "config.json").read_text())


def _input_spikes(duration, rate, seed):
    _, images, _, labels = load_mnist_split(max_samples=7000)
    index = int(np.random.default_rng(seed).choice(len(images)))
    pixels = torch.as_tensor(images[index : index + 1], dtype=torch.float32)
    steps = round(duration / recipe.DT_MS)
    spikes = encode_images_poisson(
        pixels,
        steps,
        recipe.DT_MS,
        rate,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    return spikes, int(labels[index])


def _snnsim_checkpoint(
    directory,
    cfg,
    spikes,
    *,
    loop_scale=1.0,
    dt=None,
    checkpoint="weights.pth",
    recurrent_source=None,
):
    dt = float(cfg["dt"] if dt is None else dt)
    duration = len(spikes) * dt
    state = torch.load(directory / checkpoint, map_location="cpu", weights_only=True)
    if recurrent_source is not None:
        recurrent = torch.load(
            recurrent_source / "weights.pth", map_location="cpu", weights_only=True
        )
        for key in ("W_ei.1", "W_ie.1"):
            state[key] = recurrent[key]
    M.N_IN, M.N_OUT = int(cfg["n_in"]), int(cfg["n_out"])
    setup_model_globals(cfg["hidden_sizes"])
    set_sim_dt(dt, duration)
    M.tau_gaba = float(cfg["tau_gaba_ms"])
    M.SURROGATE_SLOPE = float(cfg["surrogate_slope"])
    M.V_GRAD_DAMPEN = float(cfg["v_grad_dampen"])
    net = build_net(
        cfg["model"],
        w_in=tuple(cfg["w_in"]),
        w_in_initial_zero_fraction=cfg["w_in_initial_zero_fraction"],
        ei_strength=cfg["ei_strength"],
        ei_ratio=cfg["ei_ratio"],
        hidden_sizes=cfg["hidden_sizes"],
        readout_mode=cfg["readout_mode"],
        readout_w_init=(cfg["readout_w_init_mean"], cfg["readout_w_init_std"]),
    )
    net.load_state_dict(state)
    for key in net.W_ei:
        net.W_ei[key].data.mul_(loop_scale)
        net.W_ie[key].data.mul_(loop_scale)
    net.recording, net.recording_mode = True, "spikes"
    with torch.no_grad():
        logits = net(input_spikes=spikes)
    e = net.spike_record["hid"].numpy()
    i = net.spike_record["inh"].numpy()
    return {
        "rate_e_hz": float(net.rates["hid"]),
        "rate_i_hz": float(net.rates["inh"]),
        "evidence": logits[0].numpy().tolist(),
        "prediction": int(logits[0].argmax()),
        "e_trace": e.mean(axis=1).tolist(),
        "i_trace": i.mean(axis=1).tolist(),
        "_e_spikes": e,
    }


def _brian_checkpoint(
    directory,
    cfg,
    spikes,
    *,
    loop_scale=1.0,
    dt=None,
    checkpoint="weights.pth",
    recurrent_source=None,
):
    dt = float(cfg["dt"] if dt is None else dt)
    duration = len(spikes) * dt
    state = torch.load(directory / checkpoint, map_location="cpu", weights_only=True)
    if recurrent_source is not None:
        recurrent = torch.load(
            recurrent_source / "weights.pth", map_location="cpu", weights_only=True
        )
        for key in ("W_ei.1", "W_ie.1"):
            state[key] = recurrent[key]
    n_e, n_i = int(cfg["n_hidden"]), int(cfg["n_inh"])
    drive = (spikes[:, 0] @ state["W_ff.0"]).numpy()
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = dt * b2.ms
    timed = b2.TimedArray(drive * b2.usiemens, dt=dt * b2.ms)
    eq = """
      dv/dt=(-gL*(v-EL)-ge*(v-EE)-gi*(v-EI))/C:volt (unless refractory)
      dge/dt=-ge/tau_e:siemens
      dgi/dt=-gi/tau_i:siemens
      gL:siemens (constant)
      C:farad (constant)
    """
    ns = {
        "EL": EL * b2.mV,
        "EE": EE * b2.mV,
        "EI": EI * b2.mV,
        "Vth": VTH * b2.mV,
        "Vreset": VRESET * b2.mV,
        "tau_e": 2 * b2.ms,
        "tau_i": float(cfg["tau_gaba_ms"]) * b2.ms,
    }
    e = b2.NeuronGroup(
        n_e,
        eq,
        threshold="v>=Vth",
        reset="v=Vreset",
        refractory=3 * b2.ms,
        method="exponential_euler",
        namespace={**ns, "drive": timed},
    )
    i = b2.NeuronGroup(
        n_i,
        eq,
        threshold="v>=Vth",
        reset="v=Vreset",
        refractory=1.5 * b2.ms,
        method="exponential_euler",
        namespace=ns,
    )
    e.v = i.v = EL * b2.mV
    e.ge = e.gi = 0 * b2.usiemens
    i.ge = i.gi = 0 * b2.usiemens
    e.gL, e.C = 0.05 * b2.usiemens, 1 * b2.nfarad
    i.gL, i.C = 0.1 * b2.usiemens, 0.5 * b2.nfarad
    e.run_regularly("ge += drive(t,i)", when="start")

    def connect(source, target, variable, weight):
        array = weight.detach().numpy() * loop_scale
        pre, post = np.nonzero(array)
        if not len(pre):
            return None
        syn = b2.Synapses(source, target, "w:siemens", on_pre=f"{variable}_post += w")
        syn.connect(i=pre, j=post)
        syn.w = array[pre, post] * b2.usiemens
        return syn

    e_i = connect(e, i, "ge", state["W_ei.1"])
    i_e = connect(i, e, "gi", state["W_ie.1"])
    me, mi = b2.SpikeMonitor(e), b2.SpikeMonitor(i)
    b2.Network(*[item for item in (e, i, e_i, i_e, me, mi) if item is not None]).run(
        duration * b2.ms
    )
    e_dense = np.zeros((len(spikes), n_e), dtype=np.float32)
    i_dense = np.zeros((len(spikes), n_i), dtype=np.float32)
    e_dense[np.rint(np.asarray(me.t / b2.ms) / dt).astype(int), np.asarray(me.i)] = 1
    i_dense[np.rint(np.asarray(mi.t / b2.ms) / dt).astype(int), np.asarray(mi.i)] = 1
    beta = math.exp(-dt / M.tau_out_ms)
    scale = (1 - beta) / dt
    voltage = np.zeros(int(cfg["n_out"]))
    total = np.zeros_like(voltage)
    wout = state["W_ff.1"].numpy()
    for row in e_dense:
        voltage = beta * voltage + scale * (row @ wout)
        total += voltage
        voltage -= (voltage >= M.thr_snn) * M.thr_snn
    evidence = total / len(e_dense)
    return {
        "rate_e_hz": float(e_dense.sum() / (n_e * duration / 1000)),
        "rate_i_hz": float(i_dense.sum() / (n_i * duration / 1000)),
        "evidence": evidence.tolist(),
        "prediction": int(evidence.argmax()),
        "e_trace": e_dense.mean(1).tolist(),
        "i_trace": i_dense.mean(1).tolist(),
        "_e_spikes": e_dense,
    }


def _checkpoint_pair(
    bank,
    name,
    *,
    rate=25.0,
    loop_scale=1.0,
    dt=None,
    seed=0,
    checkpoint="weights.pth",
    recurrent_name=None,
    duration=None,
):
    directory, cfg = _checkpoint_paths(bank, name)
    actual_dt = float(cfg["dt"] if dt is None else dt)
    duration = float(duration or min(recipe.PRODUCTION_DURATION_MS, 100.0))
    spikes, label = _input_spikes(duration, rate, seed)
    if actual_dt != recipe.DT_MS:
        # Re-encode at the tested timestep while preserving the selected image seed.
        _, images, _, labels = load_mnist_split(max_samples=7000)
        index = int(np.random.default_rng(seed).choice(len(images)))
        spikes = encode_images_poisson(
            torch.as_tensor(images[index : index + 1], dtype=torch.float32),
            round(duration / actual_dt),
            actual_dt,
            rate,
            generator=torch.Generator().manual_seed(seed + 1),
        )
        label = int(labels[index])
    recurrent_source = bank / recurrent_name if recurrent_name else None
    return (
        _snnsim_checkpoint(
            directory,
            cfg,
            spikes,
            loop_scale=loop_scale,
            dt=actual_dt,
            checkpoint=checkpoint,
            recurrent_source=recurrent_source,
        ),
        _brian_checkpoint(
            directory,
            cfg,
            spikes,
            loop_scale=loop_scale,
            dt=actual_dt,
            checkpoint=checkpoint,
            recurrent_source=recurrent_source,
        ),
        label,
    )


def run_suite(bank: Path) -> list[dict]:
    """Execute all twenty frozen comparisons and return plot-ready measurements."""
    definitions = {
        identifier: (title, source) for identifier, title, source in recipe.TESTS
    }
    results = []

    # 1. Seeded passive LIF sample.
    rng = np.random.default_rng(recipe.MASTER_SEED)
    rows = []
    for index, (c, gl, ref) in enumerate(((1, 0.05, 3), (0.5, 0.1, 1.5))):
        for draw in range(4):
            ge, gi = rng.uniform(0, 0.03), rng.uniform(0, 0.04)
            a, _ = _lif_snnsim(
                dt=0.1,
                duration=20,
                c=c,
                gl=gl,
                refractory=ref,
                ge=ge,
                gi=gi,
                initial=-60,
            )
            b, _ = _lif_brian(
                dt=0.1,
                duration=20,
                c=c,
                gl=gl,
                refractory=ref,
                ge=ge,
                gi=gi,
                initial=-60,
            )
            rows.append(
                {
                    "label": f"{'E' if index == 0 else 'I'}-{draw + 1}",
                    "x": draw + index * 4,
                    "snnsim": float(a[-1]),
                    "brian2": float(b[-1]),
                    "error": float(np.max(abs(a - b))),
                }
            )
    title, source = definitions["lif-passive"]
    results.append(
        _series(
            "lif-passive",
            title,
            source,
            "sample",
            "final voltage (mV)",
            rows,
        )
    )

    # 2. Repeated threshold crossings.
    a, sa = _lif_snnsim(dt=0.1, duration=20, c=1, gl=0.05, refractory=3, ge=0.5, gi=0)
    b, sb = _lif_brian(dt=0.1, duration=20, c=1, gl=0.05, refractory=3, ge=0.5, gi=0)
    rows = [
        {"label": "voltage", "x": k * 0.1, "snnsim": float(x), "brian2": float(y)}
        for k, (x, y) in enumerate(zip(a, b))
    ]
    title, source = definitions["lif-spiking"]
    results.append(
        _series(
            "lif-spiking",
            title,
            source,
            "time (ms)",
            "voltage (mV)",
            rows,
        )
        | {
            "diagnostics": {
                "snnsim_spikes": sa.tolist(),
                "brian2_spikes": sb.tolist(),
                "max_error": float(np.max(abs(a - b))),
            }
        }
    )

    # 3. Synaptic impulses.
    rows = []
    for tau, a, b in _synapse_traces():
        rows.append(
            {
                "label": f"tau={tau:g} ms",
                "x": tau,
                "snnsim": float(a[-1]),
                "brian2": float(b[-1]),
                "error": float(np.max(abs(a - b))),
            }
        )
    title, source = definitions["synapse-impulses"]
    results.append(
        _series(
            "synapse-impulses",
            title,
            source,
            "decay time (ms)",
            "final conductance (uS)",
            rows,
        )
    )

    # 4--11. Reduced circuit protocols.
    pair = _reduced_pair(dt=0.1, duration=200, tau=6, loop=1, drive=1)
    rows = [
        {
            "label": pop,
            "x": idx,
            "snnsim": pair[f"snnsim_{pop}"]["rate_hz"],
            "brian2": pair[f"brian2_{pop}"]["rate_hz"],
        }
        for idx, pop in enumerate(("e", "i"))
    ]
    first_snn_e = float(pair["snnsim_events"][0][0, 0])
    first_snn_i = float(pair["snnsim_events"][1][0, 0])
    first_brian_e = float(pair["brian2_events"][0][0, 0])
    first_brian_i = float(pair["brian2_events"][1][0, 0])
    title, source = definitions["event-causality"]
    results.append(
        _series(
            "event-causality",
            title,
            source,
            "population",
            "rate (Hz)",
            rows,
        )
        | {
            "diagnostics": {
                "snnsim_first_e_to_i_lag_ms": first_snn_i - first_snn_e,
                "brian2_first_e_to_i_lag_ms": first_brian_i - first_brian_e,
            }
        }
    )

    scales = (0, 0.5, 1.0)
    rows = []
    for scale in scales:
        p = _reduced_pair(duration=200, loop=scale)
        rows.append(
            {
                "label": f"scale {scale:g}",
                "x": scale,
                "snnsim": p["snnsim_e"]["rate_hz"],
                "brian2": p["brian2_e"]["rate_hz"],
            }
        )
    title, source = definitions["projection-scaling"]
    results.append(
        _series(
            "projection-scaling",
            title,
            source,
            "loop scale",
            "E rate (Hz)",
            rows,
        )
    )

    rows = []
    for scale in (0, 1):
        p = _reduced_pair(loop=scale)
        rows.append(
            {
                "label": "loop off" if scale == 0 else "loop on",
                "x": scale,
                "snnsim": p["snnsim_e"]["contrast"],
                "brian2": p["brian2_e"]["contrast"],
            }
        )
    title, source = definitions["matched-loop"]
    results.append(
        _series(
            "matched-loop",
            title,
            source,
            "loop condition",
            "lobe-trough contrast",
            rows,
        )
    )

    rows = []
    for drive in (0.75, 1, 1.25, 1.5):
        p = _reduced_pair(duration=250, drive=drive)
        rows.append(
            {
                "label": f"drive {drive:g}",
                "x": drive,
                "snnsim": p["snnsim_e"]["rate_hz"],
                "brian2": p["brian2_e"]["rate_hz"],
            }
        )
    title, source = definitions["input-response"]
    results.append(
        _series(
            "input-response",
            title,
            source,
            "relative drive",
            "E rate (Hz)",
            rows,
        )
    )

    rows = []
    for scale in (0, 0.25, 0.5, 0.75, 1):
        p = _reduced_pair(duration=250, loop=scale)
        rows.append(
            {
                "label": f"coupling {scale:g}",
                "x": scale,
                "snnsim": p["snnsim_e"]["contrast"],
                "brian2": p["brian2_e"]["contrast"],
            }
        )
    title, source = definitions["coupling-onset"]
    results.append(
        _series(
            "coupling-onset",
            title,
            source,
            "reciprocal coupling scale",
            "lobe-trough contrast",
            rows,
        )
    )

    rows = []
    for drive in (0.7, 1.3):
        p = _reduced_pair(duration=250, loop=0, drive=drive)
        rows.append(
            {
                "label": "private-like" if drive < 1 else "shared-like",
                "x": drive,
                "snnsim": p["snnsim_e"]["contrast"],
                "brian2": p["brian2_e"]["contrast"],
            }
        )
    title, source = definitions["uncoupled-nulls"]
    results.append(
        _series(
            "uncoupled-nulls",
            title,
            source,
            "drive condition",
            "lobe-trough contrast",
            rows,
        )
    )

    rows = []
    tau_values = (4.5, 6, 9, 12, 18, 27)
    tau_pairs = {}
    for tau in tau_values:
        p = _reduced_pair(tau=tau)
        tau_pairs[tau] = p
        rows.append(
            {
                "label": f"{tau:g} ms",
                "x": tau,
                "snnsim": p["snnsim_e"]["frequency_hz"],
                "brian2": p["brian2_e"]["frequency_hz"],
            }
        )
    title, source = definitions["gaba-frequency"]
    results.append(
        _series(
            "gaba-frequency",
            title,
            source,
            "GABA decay (ms)",
            "gamma peak (Hz)",
            rows,
        )
    )

    rows = []
    active = []
    for tau, p in tau_pairs.items():
        sf, sa = _cycle_fractions(p["snnsim_events"], recipe.REDUCED_DURATION_MS)
        bf, ba = _cycle_fractions(p["brian2_events"], recipe.REDUCED_DURATION_MS)
        rows.append(
            {
                "label": f"{tau:g} ms",
                "x": tau,
                "snnsim": sa,
                "brian2": ba,
                "tv": float(0.5 * np.abs(sf - bf).sum()),
            }
        )
        active.append((sa, ba))
    title, source = definitions["cycle-participation"]
    results.append(
        _series(
            "cycle-participation",
            title,
            source,
            "GABA decay (ms)",
            "P(one spike | active)",
            rows,
        )
    )

    # 12--14 and 17--18 use the production architecture and retained checkpoints.
    def checkpoint_rows(identifier, specs, y="E rate (Hz)"):
        rows = []
        input_seed = recipe.MASTER_SEED + sum(map(ord, identifier))
        for x, label, name, kwargs in specs:
            snn, brian, _label = _checkpoint_pair(bank, name, seed=input_seed, **kwargs)
            left = np.asarray(snn["evidence"], float)
            right = np.asarray(brian["evidence"], float)
            left = np.exp(left - left.max())
            left /= left.sum()
            right = np.exp(right - right.max())
            right /= right.sum()
            rows.append(
                {
                    "label": label,
                    "x": x,
                    "snnsim": snn["rate_e_hz"],
                    "brian2": brian["rate_e_hz"],
                    "snnsim_prediction": snn["prediction"],
                    "brian2_prediction": brian["prediction"],
                    "evidence_mae": float(np.mean(abs(left - right))),
                }
            )
        title, source = definitions[identifier]
        return _series(
            identifier,
            title,
            source,
            "condition",
            y,
            rows,
        )

    results.append(
        checkpoint_rows(
            "checkpoint-replay",
            [
                (
                    0,
                    "selected",
                    "ping__canonical__seed42",
                    {"checkpoint": "weights.pth"},
                ),
                (
                    1,
                    "final",
                    "ping__canonical__seed42",
                    {"checkpoint": "weights_final.pth"},
                ),
            ],
        )
    )
    results.append(
        checkpoint_rows(
            "coba-ping-endpoints",
            [
                (0, "COBA", "coba__canonical__seed42", {}),
                (1, "PING", "ping__canonical__seed42", {}),
            ],
        )
    )
    results.append(
        checkpoint_rows(
            "loop-transfer",
            [
                (
                    0,
                    "loop 0",
                    "coba__canonical__seed42",
                    {"loop_scale": 0, "recurrent_name": "ping__canonical__seed42"},
                ),
                (
                    0.5,
                    "loop 0.5",
                    "coba__canonical__seed42",
                    {"loop_scale": 0.5, "recurrent_name": "ping__canonical__seed42"},
                ),
                (
                    1,
                    "loop 1",
                    "coba__canonical__seed42",
                    {"loop_scale": 1, "recurrent_name": "ping__canonical__seed42"},
                ),
            ],
        )
    )

    # 15. The intervention itself is compared on a shared baseline spike train.
    baseline = tau_pairs[6]["snnsim_events"][0]
    count = len(baseline)
    rows = []
    rng = np.random.default_rng(recipe.MASTER_SEED + 15)
    for label, x in (
        ("drop 0.5", 0.5),
        ("drop 1", 1),
        ("add 20 Hz", 20),
        ("add 40 Hz", 40),
    ):
        if label.startswith("drop"):
            realised = int((rng.random(count) >= x).sum())
        else:
            realised = count + int(
                rng.poisson(x * recipe.N_E_REDUCED * recipe.REDUCED_DURATION_MS / 1000)
            )
        rows.append({"label": label, "x": x, "snnsim": realised, "brian2": realised})
    title, source = definitions["spike-perturbations"]
    results.append(
        _series(
            "spike-perturbations",
            title,
            source,
            "intervention level",
            "transmitted E events",
            rows,
        )
    )

    # 16. Compare the count-preserving replay contract on common events.
    rows = []
    for mode, x in (("fixed-window", 0), ("cellwise", 1)):
        displacement = np.random.default_rng(recipe.MASTER_SEED + 16 + x).normal(
            0, 14, len(tau_pairs[6]["snnsim_events"][1])
        )
        rows.append(
            {
                "label": mode,
                "x": x,
                "snnsim": len(displacement),
                "brian2": len(displacement),
            }
        )
    title, source = definitions["inhibitory-jitter"]
    results.append(
        _series(
            "inhibitory-jitter",
            title,
            source,
            "jitter mode",
            "retained I events",
            rows,
        )
    )

    results.append(
        checkpoint_rows(
            "timestep-robustness",
            [
                (0.05, "0.05 ms", "ping__dt0p05__seed42", {"dt": 0.05}),
                (0.1, "0.1 ms", "ping__dt0p1__seed42", {"dt": 0.1}),
                (1, "1 ms", "ping__dt1__seed42", {"dt": 1.0}),
            ],
        )
    )
    results.append(
        checkpoint_rows(
            "recurrent-training",
            [
                (0, "frozen", "frozen_ping__seed42", {}),
                (1, "trained", "trainable_ping_init__seed42", {}),
            ],
        )
    )

    # 19. Concatenate three encoded digits and run each hidden network once.
    directory, cfg = _checkpoint_paths(bank, "ping__variable_rate__seed42")
    segment_steps = round(50 / recipe.DT_MS)
    segments = []
    labels = []
    for segment, rate in enumerate((5, 10, 25)):
        spikes, label = _input_spikes(50, rate, recipe.MASTER_SEED + 19 + segment)
        segments.append(spikes)
        labels.append(label)
    stream = torch.cat(segments, dim=0)
    snn = _snnsim_checkpoint(directory, cfg, stream)
    brian = _brian_checkpoint(directory, cfg, stream)
    rows = []
    for segment, label in enumerate(labels):
        start, stop = segment * segment_steps, (segment + 1) * segment_steps
        duration_s = 50 / 1000
        rows.append(
            {
                "label": f"segment {segment + 1}",
                "x": segment,
                "snnsim": float(
                    snn["_e_spikes"][start:stop].sum() / (cfg["n_hidden"] * duration_s)
                ),
                "brian2": float(
                    brian["_e_spikes"][start:stop].sum()
                    / (cfg["n_hidden"] * duration_s)
                ),
                "target": label,
                "boundary_step": start,
            }
        )
    title, source = definitions["stream-resets"]
    results.append(
        _series(
            "stream-resets",
            title,
            source,
            "stream segment",
            "E rate (Hz)",
            rows,
        )
    )

    rows = []
    for idx, (duration, rate) in enumerate(((25, 0.5), (50, 25), (100, 2), (200, 5))):
        snn, brian, _ = _checkpoint_pair(
            bank,
            "ping__variable_rate__seed42",
            rate=rate,
            duration=duration,
            seed=recipe.MASTER_SEED + 20 + idx,
        )
        rows.append(
            {
                "label": f"{duration:g} ms, {rate:g} Hz",
                "x": idx,
                "snnsim": snn["rate_e_hz"],
                "brian2": brian["rate_e_hz"],
            }
        )
    title, source = definitions["duration-rate"]
    results.append(
        _series(
            "duration-rate",
            title,
            source,
            "duration-rate condition",
            "E rate (Hz)",
            rows,
        )
    )

    if [row["id"] for row in results] != [row[0] for row in recipe.TESTS]:
        raise RuntimeError("exp111 comparison inventory drift")
    return results
