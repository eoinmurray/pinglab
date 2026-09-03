"""Committed EXP099 scientific definitions; no execution on import."""

from __future__ import annotations

from tools import snnlang as snn  # noqa: TID251

SLUG = "exp099"
DT_MS, DURATION_MS, SEED = 0.25, 2_000.0, 7
N_E, N_I = 400, 100
VIEW_START_MS, VIEW_END_MS = 300.0, 1_800.0
ONSET_MS, PEAK_MS, OFFSET_MS = 600.0, 850.0, 1_100.0
VIDEO = "richer-input-ai-to-intermittent-ping.mp4"
POSTER = "richer-input-ai-to-intermittent-ping.png"
SHARED_DRIVE_VIDEO = "shared-drive-ai-to-ping.mp4"
SHARED_DRIVE_POSTER = "shared-drive-ai-to-ping.png"
INPUT_MAP = "input-map-option-3.svg"
SCALE = {"dt_ms": DT_MS, "t_ms": DURATION_MS, "n_e": N_E, "n_i": N_I, "seed": SEED}


def recurrent(mean: float, std: float) -> snn.LowerClampedNormal:
    return snn.LowerClampedNormal(
        mean, std, initial_zero_fraction=0.975, zeroing="exact_k"
    )


def background(
    tau: float, group: int, private: float, shared: float, *, rate_scale: float = 1.0
):
    return snn.BackgroundChannel(
        private=snn.ShotNoise(500 * rate_scale, private, tau),
        shared=snn.GroupedShotNoise(80 * rate_scale, shared, tau, group),
        heterogeneity=snn.CellDistribution(
            rate=snn.LowerClampedNormal(1.0, 0.1),
            amplitude=snn.LowerClampedNormal(1.0, 0.1),
        ),
    )


def author_network(
    *,
    condition: str = "richer-input",
    shared_peak_scale: float = 6.5,
    private_afferent_scale: float = 1.0,
    background_rate_scale: float = 1.0,
    ampa_background_scale: float = 1.0,
    gaba_background_scale: float = 1.0,
    w_ee_scale: float = 1.0,
    w_ei_scale: float = 1.0,
    w_ie_scale: float = 1.0,
    w_in_e_scale: float = 1.0,
    w_in_i_scale: float = 1.0,
    tau_gaba_ms: float = 9.0,
    onset_ms: float = ONSET_MS,
    peak_ms: float = PEAK_MS,
    plateau_end_ms: float = PEAK_MS,
    offset_ms: float = OFFSET_MS,
) -> snn.Bundle:
    """Author the latest canvas condition without depending on canvas files."""
    if condition not in {"richer-input", "shared-drive-isolation"}:
        raise ValueError(f"unsupported exp099 condition: {condition}")
    if shared_peak_scale < 1 or not (
        0 < private_afferent_scale <= 1 and 0 < background_rate_scale <= 1
    ):
        raise ValueError("input scales require shared >= 1 and 0 < fixed scales <= 1")
    if (
        min(
            w_ee_scale,
            w_ei_scale,
            w_ie_scale,
            w_in_e_scale,
            w_in_i_scale,
            ampa_background_scale,
            gaba_background_scale,
            tau_gaba_ms,
        )
        <= 0
    ):
        raise ValueError("synaptic scales must be positive")
    if not 0 <= onset_ms < peak_ms <= plateau_end_ms < offset_ms:
        raise ValueError("input timing requires onset < peak <= plateau end < offset")
    isolated = condition == "shared-drive-isolation"
    private_scale = private_afferent_scale if isolated else 1.0
    background_scale = background_rate_scale if isolated else 1.0
    private_peak_scale = 1.0 if isolated else 1.2
    weather = (
        None if isolated else snn.StationaryRateWeather(tau_ms=250, std_fraction=0.12)
    )
    net = snn.Network(f"exp099_{condition.replace('-', '_')}", dt=DT_MS * snn.ms)
    source_e = net.input(
        "afferent_e", shape=("time", "batch", N_E), signal_type="spikes", unit="spike"
    )
    source_i = net.input(
        "afferent_i", shape=("time", "batch", N_E), signal_type="spikes", unit="spike"
    )
    cell = snn.components.ping(
        net,
        name="balanced_circuit",
        n_e=N_E,
        n_i=N_I,
        source_e=source_e,
        source_i=source_i,
        tau_gaba=tau_gaba_ms * snn.ms,
        w_in_e=snn.Normal(0.08 * w_in_e_scale, 0.008 * w_in_e_scale),
        w_in_i=snn.Normal(0.02 * w_in_i_scale, 0.002 * w_in_i_scale),
        w_ee=recurrent(0.85 * w_ee_scale, 0.255 * w_ee_scale),
        w_ei=recurrent(0.6 * w_ei_scale, 0.18 * w_ei_scale),
        w_ie=recurrent(3.0 * w_ie_scale, 0.9 * w_ie_scale),
        w_ii=recurrent(0.4, 0.12),
    )
    readout = snn.readouts.MeanVoltage(
        source=cell.E.spikes,
        classes=10,
        name="state_readout",
        tau=2 * snn.ms,
        weight=snn.Normal(5.1, 3.8),
    )
    net.output("state_logits", readout)
    net.expose(cell.E.spikes, cell.I.spikes, name="populations")
    simulation = snn.SimulationSpec(
        spike_sources=[
            snn.CorrelatedPoissonAfferents(
                source_e, source_i, 10, 15 * private_scale, 15 * private_scale
            )
        ],
        backgrounds=[
            snn.ConductanceBackground(
                cell.E,
                background(
                    2,
                    25,
                    0.06,
                    0.02,
                    rate_scale=background_scale * ampa_background_scale,
                ),
                background(
                    tau_gaba_ms,
                    25,
                    0.03,
                    0.01,
                    rate_scale=background_scale * gaba_background_scale,
                ),
            ),
            snn.ConductanceBackground(
                cell.I,
                background(
                    2,
                    10,
                    0.03,
                    0.01,
                    rate_scale=background_scale * ampa_background_scale,
                ),
                background(
                    tau_gaba_ms,
                    10,
                    0.03,
                    0.01,
                    rate_scale=background_scale * gaba_background_scale,
                ),
            ),
        ],
        weather=weather,
        afferent_wave=snn.TransientAfferentWave(
            onset_ms,
            peak_ms,
            offset_ms,
            private_peak_scale,
            shared_peak_scale,
            plateau_end_ms=plateau_end_ms,
        ),
    )
    return snn.compile(net, simulation=simulation, target="tools/snnsim")


def configuration(
    bundle=None,
    *,
    condition: str = "richer-input",
    shared_peak_scale: float = 6.5,
    private_afferent_scale: float = 1.0,
    background_rate_scale: float = 1.0,
    ampa_background_scale: float = 1.0,
    gaba_background_scale: float = 1.0,
    w_ee_scale: float = 1.0,
    w_ei_scale: float = 1.0,
    w_ie_scale: float = 1.0,
    w_in_e_scale: float = 1.0,
    w_in_i_scale: float = 1.0,
    tau_gaba_ms: float = 9.0,
    duration_ms: float = DURATION_MS,
    seed: int = SEED,
    onset_ms: float = ONSET_MS,
    peak_ms: float = PEAK_MS,
    plateau_end_ms: float = PEAK_MS,
    offset_ms: float = OFFSET_MS,
    view_start_ms: float = VIEW_START_MS,
    view_end_ms: float = VIEW_END_MS,
) -> dict:
    bundle = (
        bundle
        if bundle is not None
        else author_network(
            condition=condition,
            shared_peak_scale=shared_peak_scale,
            private_afferent_scale=private_afferent_scale,
            background_rate_scale=background_rate_scale,
            ampa_background_scale=ampa_background_scale,
            gaba_background_scale=gaba_background_scale,
            w_ee_scale=w_ee_scale,
            w_ei_scale=w_ei_scale,
            w_ie_scale=w_ie_scale,
            w_in_e_scale=w_in_e_scale,
            w_in_i_scale=w_in_i_scale,
            tau_gaba_ms=tau_gaba_ms,
            onset_ms=onset_ms,
            peak_ms=peak_ms,
            plateau_end_ms=plateau_end_ms,
            offset_ms=offset_ms,
        )
    )
    if not 0 <= view_start_ms < view_end_ms <= duration_ms:
        raise ValueError("view window must lie inside the simulation")
    return {
        "schema": "exp099.recipe/v1",
        "condition": condition,
        "controls": {
            "shared_peak_scale": float(shared_peak_scale),
            "private_afferent_scale": float(private_afferent_scale),
            "background_rate_scale": float(background_rate_scale),
            "ampa_background_scale": float(ampa_background_scale),
            "gaba_background_scale": float(gaba_background_scale),
            "w_ee_scale": float(w_ee_scale),
            "w_ei_scale": float(w_ei_scale),
            "w_ie_scale": float(w_ie_scale),
            "w_in_e_scale": float(w_in_e_scale),
            "w_in_i_scale": float(w_in_i_scale),
            "tau_gaba_ms": float(tau_gaba_ms),
            "onset_ms": float(onset_ms),
            "peak_ms": float(peak_ms),
            "plateau_end_ms": float(plateau_end_ms),
            "offset_ms": float(offset_ms),
            "view_start_ms": float(view_start_ms),
            "view_end_ms": float(view_end_ms),
        },
        **SCALE,
        "t_ms": float(duration_ms),
        "seed": int(seed),
        "graph": bundle.graph,
        "simulation": bundle.simulation,
    }


def media_names(condition: str) -> tuple[str, str]:
    if condition == "shared-drive-isolation":
        return SHARED_DRIVE_VIDEO, SHARED_DRIVE_POSTER
    return VIDEO, POSTER


def analysis_configuration(configuration: dict | None = None) -> dict:
    configuration = configuration or {}
    controls = configuration.get("controls", {})
    view_start_ms = float(controls.get("view_start_ms", VIEW_START_MS))
    view_end_ms = float(controls.get("view_end_ms", VIEW_END_MS))
    short_protocol = view_end_ms - view_start_ms <= 1_200
    return {
        "schema": "exp099.measurements/v1",
        "rhythm_window_ms": 160.0 if short_protocol else 400.0,
        "rhythm_stride_ms": 5.0 if short_protocol else 10.0,
        "rhythm_max_lag_ms": 60.0 if short_protocol else 100.0,
        "rhythm_bin_ms": 1.0,
        "view_start_ms": view_start_ms,
        "view_end_ms": view_end_ms,
        "loop_window_ms": 40.0,
        "loop_stride_ms": 5.0,
        "loop_smoothing_ms": 75.0,
        "loop_percentiles": [10, 95],
        "external_tau_ms": {"E AMPA": 2.0, "E GABA": 9.0, "I AMPA": 2.0, "I GABA": 9.0},
        "summary_endpoint": "exclusive",
        "plot_endpoint": "inclusive",
    }
