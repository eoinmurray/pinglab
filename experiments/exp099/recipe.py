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
INPUT_MAP = "input-map.svg"
SCALE = {"dt_ms": DT_MS, "t_ms": DURATION_MS, "n_e": N_E, "n_i": N_I, "seed": SEED}


def recurrent(mean: float, std: float) -> snn.LowerClampedNormal:
    return snn.LowerClampedNormal(
        mean, std, initial_zero_fraction=0.975, zeroing="exact_k"
    )


def background(tau: float, group: int, private: float, shared: float):
    return snn.BackgroundChannel(
        private=snn.ShotNoise(500, private, tau),
        shared=snn.GroupedShotNoise(80, shared, tau, group),
        heterogeneity=snn.CellDistribution(
            rate=snn.LowerClampedNormal(1.0, 0.1),
            amplitude=snn.LowerClampedNormal(1.0, 0.1),
        ),
    )


def author_network() -> snn.Bundle:
    """Author the latest canvas condition without depending on canvas files."""
    net = snn.Network("exp099_richer_input_ping", dt=DT_MS * snn.ms)
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
        tau_gaba=9 * snn.ms,
        w_in_e=snn.Normal(0.08, 0.008),
        w_in_i=snn.Normal(0.02, 0.002),
        w_ee=recurrent(0.85, 0.255),
        w_ei=recurrent(0.6, 0.18),
        w_ie=recurrent(3.0, 0.9),
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
        spike_sources=[snn.CorrelatedPoissonAfferents(source_e, source_i, 10, 15, 15)],
        backgrounds=[
            snn.ConductanceBackground(
                cell.E, background(2, 25, 0.06, 0.02), background(9, 25, 0.03, 0.01)
            ),
            snn.ConductanceBackground(
                cell.I, background(2, 10, 0.03, 0.01), background(9, 10, 0.03, 0.01)
            ),
        ],
        weather=snn.StationaryRateWeather(tau_ms=250, std_fraction=0.12),
        afferent_wave=snn.TransientAfferentWave(
            ONSET_MS, PEAK_MS, OFFSET_MS, 1.2, 6.5, plateau_end_ms=PEAK_MS
        ),
    )
    return snn.compile(net, simulation=simulation, target="tools/snnsim")


def configuration(bundle=None) -> dict:
    bundle = bundle if bundle is not None else author_network()
    return {
        "schema": "exp099.recipe/v1",
        **SCALE,
        "graph": bundle.graph,
        "simulation": bundle.simulation,
    }


def analysis_configuration() -> dict:
    return {
        "schema": "exp099.measurements/v1",
        "rhythm_window_ms": 400.0,
        "rhythm_stride_ms": 10.0,
        "rhythm_max_lag_ms": 100.0,
        "rhythm_bin_ms": 1.0,
        "view_start_ms": VIEW_START_MS,
        "view_end_ms": VIEW_END_MS,
        "loop_window_ms": 40.0,
        "loop_stride_ms": 5.0,
        "loop_smoothing_ms": 75.0,
        "loop_percentiles": [10, 95],
        "external_tau_ms": {"E AMPA": 2.0, "E GABA": 9.0, "I AMPA": 2.0, "I GABA": 9.0},
        "summary_endpoint": "exclusive",
        "plot_endpoint": "inclusive",
    }
