"""Susin-inspired private-afferent protocol; no execution on import.

The configuration resolves scientific settings and the authoring functions compile
the network. Data loading, simulation and run storage belong to the execution stages.
"""

from __future__ import annotations

from snnlab import lang as snn  # noqa: TID251

SLUG = "exp099"
DT_MS, DURATION_MS, SEED = 0.1, 1100.0, 7
BURN_IN_MS = 500.0
N_E, N_I = 1600, 400
CAPACITANCE_NF, LEAK_US = 0.15, 0.01
BASELINE_HZ = 0.8
RECURRENT_SCALE, INHIBITORY_SCALE = 0.08, 5.0
VIEW_START_MS, VIEW_END_MS = BURN_IN_MS, DURATION_MS
ONSET_MS, PEAK_MS, PLATEAU_END_MS, OFFSET_MS = 700.0, 750.0, 850.0, 900.0


def configuration() -> dict:
    return dict(
        schema="exp099.recipe/v2",
        condition="private-ei-drive",
        dt_ms=DT_MS,
        t_ms=DURATION_MS,
        burn_in_ms=BURN_IN_MS,
        seed=SEED,
        n_e=N_E,
        n_i=N_I,
        connection_probability=0.1,
        external_afferents_per_neuron=400,
        baseline_e_hz=BASELINE_HZ,
        baseline_i_hz=BASELINE_HZ,
        stimulus_e_hz=BASELINE_HZ * 1.5,
        stimulus_i_hz=BASELINE_HZ * 1.5,
        capacitance_nf=CAPACITANCE_NF,
        leak_us=LEAK_US,
        external_weight_us=0.004,
        recurrent_e_weight_us=0.0125 * RECURRENT_SCALE,
        recurrent_i_weight_us=0.00835 * RECURRENT_SCALE * INHIBITORY_SCALE,
        tau_ampa_ms=1.5,
        tau_gaba_ms=7.5,
        delay_ms=1.5,
        refractory_e_ms=3.0,
        refractory_i_ms=1.5,
        onset_ms=ONSET_MS,
        peak_ms=PEAK_MS,
        plateau_end_ms=PLATEAU_END_MS,
        offset_ms=OFFSET_MS,
        view_start_ms=VIEW_START_MS,
        view_end_ms=VIEW_END_MS,
        initial_voltage_mv=-65.0,
        resting_mv=-65.0,
        reset_mv=-65.0,
        threshold_mv=-50.0,
        voltage_grad_dampen=80.0,
        refractory_timing="legacy_coba",
        device="cpu",
        torch_threads=1,
        performance=dict(compile_mode=None),
        external_representation="independent Poisson counts per target; exact superposition of 400 sources; diagonal 4 nS projection",
        recurrent_topology="independent Bernoulli edges, including possible self connections",
        weight_initialization="graph fan-in/sparsity normalization compensated in initializer; external matrices replaced by diagonal physical weights",
        recording_policy="all output spikes and afferent counts at dt; population mean voltages and projection conductances at dt",
        reference="https://doi.org/10.1371/journal.pcbi.1009416",
    )


def author_network(cfg=None):
    cfg = configuration() if cfg is None else cfg
    net = snn.Network("exp099_private_afferents", dt=cfg["dt_ms"] * snn.ms)
    pops = {}
    for name in ("E", "I"):
        cap, leak = cfg["capacitance_nf"], cfg["leak_us"]
        tau = cap / leak
        pops[name] = net.population(
            name,
            size=cfg[f"n_{name.lower()}"],
            neuron=snn.COBA_LIF(
                tau_mem=tau * snn.ms,
                capacitance_nf=cap,
                leak_us=leak,
                resting_mv=cfg["resting_mv"],
                threshold_mv=cfg["threshold_mv"],
                reset_mv=cfg["reset_mv"],
                refractory=cfg[f"refractory_{name.lower()}_ms"] * snn.ms,
                voltage_grad_dampen=cfg["voltage_grad_dampen"],
                initial_voltage_mv=cfg["initial_voltage_mv"],
            ),
        )
    for name, pop in pops.items():
        size = cfg[f"n_{name.lower()}"]
        source = net.input(
            f"private_{name.lower()}",
            shape=("time", "batch", size),
            signal_type="spikes",
            unit="spike",
        )
        projection = net.connect(
            source,
            pop.excitatory,
            name=f"private_{name.lower()}_to_{name}",
            synapse=snn.AMPA(tau=cfg["tau_ampa_ms"] * snn.ms),
            weight=snn.Constant(0.0),
            constraint=snn.NonNegative(),
            delay=cfg["delay_ms"] * snn.ms,
        )
        net.expose(projection.conductance, name=f"{projection.id}.conductance")
    for src in ("E", "I"):
        for dst in ("E", "I"):
            excitatory = src == "E"
            physical = cfg[f"recurrent_{src.lower()}_weight_us"]
            # SNNSIM divides by source count and renormalizes surviving edges.
            # Compensate both factors to retain the specified physical edge weight.
            mean = physical * cfg[f"n_{src.lower()}"] * cfg["connection_probability"]
            projection = net.connect(
                pops[src].spikes,
                pops[dst].excitatory if excitatory else pops[dst].inhibitory,
                name=f"{src}_to_{dst}",
                synapse=(
                    snn.AMPA(tau=cfg["tau_ampa_ms"] * snn.ms)
                    if excitatory
                    else snn.GABA(tau=cfg["tau_gaba_ms"] * snn.ms)
                ),
                weight=snn.LowerClampedNormal(
                    mean,
                    0.0,
                    initial_zero_fraction=1 - cfg["connection_probability"],
                    zeroing="bernoulli",
                ),
                constraint=snn.NonNegative(),
                connection="recurrent",
                allow_autapses=True,
                delay=cfg["delay_ms"] * snn.ms,
            )
            net.expose(projection.conductance, name=f"{projection.id}.conductance")
    for pop in pops.values():
        net.expose(pop.spikes, name=f"{pop.id}.spikes")
        net.expose(pop.voltage, name=f"{pop.id}.voltage")
    return snn.compile(net, target=None, refractory_timing=cfg["refractory_timing"])
