"""Scientific definitions for the paired input-rate decision study."""

from snnlab import lang as snn  # noqa: TID251

BANK = "exp022-r001-compute"
CHECKPOINT_SHA256 = "6af7b0c4e3e02d7ffcc12098bc84961c99a356c2495b90c34d7e9d914464e9bb"
RATES = (5.0, 2.5, 10.0, 20.0)
SCALES = (1.0, 0.9, 0.8, 0.5)
INTERNAL_STUDIES = ("gaba", "capacitance", "leak", "ampa", "inhibition", "threshold")
CONTROLS = {
    "gaba": SCALES,
    "capacitance": SCALES,
    "leak": SCALES,
    "ampa": SCALES,
    "inhibition": (1.0, 0.8, 1.2),
    "threshold": (-50.0, -52.0, -48.0),
}
CONFIG = dict(
    intervention="input_encoding_rate",
    weight_policy="fixed_peak_conductance",
    input_rates_hz=RATES,
    sample_seed=120082,
    images_per_class=10,
    encoding_seed_offset=820000,
    trained_seed=42,
    duration_ms=400,
    dt_ms=0.1,
    n_e=1024,
    n_i=256,
    e_leak_us=0.05,
    i_leak_us=0.1,
    e_capacitance_nf=1,
    i_capacitance_nf=0.5,
    tau_ampa_ms=2,
    tau_gaba_ms=6,
    refractory_e_ms=1.2,
    refractory_i_ms=0.6,
    bin_ms=1,
    histogram_bin_ms=20,
    minimum_peaks=3,
    cv_ddof=0,
    smoothing_sigma_ms=2,
    primary_threshold_hz=25,
    sensitivity_thresholds_hz=(10, 25, 50),
    minimum_peak_distance_ms=10,
    peak_window_ms=(0, 400),
    rate_window_ms=(0, 400),
)


def author_network(study="input", scale=1.0):
    """Compile the fixed network; restore weights from the pinned checkpoint."""
    cfg = dict(CONFIG)
    if study in ("gaba", "ampa"):
        cfg[f"tau_{study}_ms"] *= scale
    elif study in ("capacitance", "leak"):
        suffix = "capacitance_nf" if study == "capacitance" else "leak_us"
        for population in ("e", "i"):
            cfg[f"{population}_{suffix}"] *= scale
    elif study not in ("input", "inhibition", "threshold"):
        raise ValueError(f"Unknown study: {study}")
    net = snn.Network("exp121", dt=cfg["dt_ms"] * snn.ms)
    source = net.input(
        "input_spikes", shape=("time", "batch", 784), signal_type="spikes", unit="spike"
    )
    populations = {}
    for name in ("E", "I"):
        key = name.lower()
        cap, leak = cfg[f"{key}_capacitance_nf"], cfg[f"{key}_leak_us"]
        populations[name] = net.population(
            name,
            size=cfg[f"n_{key}"],
            neuron=snn.COBA_LIF(
                tau_mem=(cap / leak) * snn.ms,
                capacitance_nf=cap,
                leak_us=leak,
                resting_mv=-65.0,
                reset_mv=-65.0,
                threshold_mv=scale if study == "threshold" and name == "E" else -50.0,
                initial_voltage_mv=-65.0,
                voltage_grad_dampen=1000.0,
                refractory_steps=round(cfg[f"refractory_{key}_ms"] / cfg["dt_ms"]),
            ),
        )
    net.connect(
        source,
        populations["E"].excitatory,
        name="input_to_E",
        synapse=snn.AMPA(tau=cfg["tau_ampa_ms"] * snn.ms),
        weight=snn.Constant(0.0),
        constraint=snn.NonNegative(),
    )
    for src in ("E", "I"):
        for dst in ("E", "I"):
            excitatory = src == "E"
            net.connect(
                populations[src].spikes,
                populations[dst].excitatory
                if excitatory
                else populations[dst].inhibitory,
                name=f"{src}_to_{dst}",
                synapse=(
                    snn.AMPA(tau=cfg["tau_ampa_ms"] * snn.ms)
                    if excitatory
                    else snn.GABA(tau=cfg["tau_gaba_ms"] * snn.ms)
                ),
                weight=snn.Constant(0.0),
                constraint=snn.NonNegative(),
                connection="recurrent",
                delay=cfg["dt_ms"] * snn.ms,
            )
    readout = net.population(
        "readout",
        size=10,
        spiking=True,
        neuron=snn.LeakyIntegrator(
            tau=2 * snn.ms,
            soft_reset_threshold=1.0,
            surrogate_slope=1.0,
            initial_voltage=0.0,
        ),
    )
    net.connect(
        populations["E"].spikes,
        readout.excitatory,
        name="E_to_readout",
        synapse=snn.LeakyIntegrator(tau=2 * snn.ms),
        weight=snn.Constant(0.0),
        constraint=snn.NonNegative(),
    )
    count = snn.ops.reduce(readout.spikes, operation="sum", over="time", name="count")
    net.output("spike_count", count)
    for name, pop in {**populations, "out": readout}.items():
        net.expose(pop.spikes, name=f"{name.lower()}_spikes")
    return snn.compile(net, target=None)
