"""Author and compile the matched AI/PING network definitions with SNNLang."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))

from tools import snnlang as snn  # noqa: E402

BUNDLES = ROOT / "snnlang"


def build(
    name: str,
    w_ee_mean: float,
    w_ee_std: float,
    *,
    afferent_wave: snn.TransientAfferentWave | None = None,
):
    net = snn.Network(name, dt=0.25 * snn.ms)
    afferent_e = net.input(
        "afferent_e",
        shape=("time", "batch", 400),
        signal_type="spikes",
        unit="spike",
    )
    afferent_i = net.input(
        "afferent_i",
        shape=("time", "batch", 400),
        signal_type="spikes",
        unit="spike",
    )

    def recurrent(mean: float, std: float):
        return snn.LowerClampedNormal(
            mean,
            std,
            initial_zero_fraction=0.975,
            zeroing="exact_k",
        )

    cell = snn.components.ping(
        net,
        name="balanced_circuit",
        n_e=400,
        n_i=100,
        source_e=afferent_e,
        source_i=afferent_i,
        tau_gaba=9 * snn.ms,
        w_in_e=snn.Normal(0.08, 0.008),
        w_in_i=snn.Normal(0.02, 0.002),
        w_ee=recurrent(w_ee_mean, w_ee_std),
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

    def background(
        tau_ms: float,
        group_size: int,
        private_amplitude: float,
        shared_amplitude: float,
    ):
        return snn.BackgroundChannel(
            private=snn.ShotNoise(500, private_amplitude, tau_ms),
            shared=snn.GroupedShotNoise(80, shared_amplitude, tau_ms, group_size),
            heterogeneity=snn.CellDistribution(
                rate=snn.LowerClampedNormal(1.0, 0.1),
                amplitude=snn.LowerClampedNormal(1.0, 0.1),
            ),
        )

    simulation = snn.SimulationSpec(
        spike_sources=[
            snn.CorrelatedPoissonAfferents(
                afferent_e,
                afferent_i,
                shared_rate_hz=10,
                e_private_rate_hz=15,
                i_private_rate_hz=15,
            )
        ],
        backgrounds=[
            snn.ConductanceBackground(
                cell.E,
                background(2, 25, 0.06, 0.02),
                background(9, 25, 0.03, 0.01),
            ),
            snn.ConductanceBackground(
                cell.I,
                background(2, 10, 0.03, 0.01),
                background(9, 10, 0.03, 0.01),
            ),
        ],
        weather=snn.StationaryRateWeather(tau_ms=250, std_fraction=0.12),
        afferent_wave=afferent_wave,
    )
    return snn.compile(net, simulation=simulation, target="tools/snnsim")


def main():
    BUNDLES.mkdir(parents=True, exist_ok=True)
    definitions = {
        "ai": build("balanced_ai_state", 0.4, 0.12),
        "ping": build("balanced_ping_state", 4.34, 1.302),
    }
    for state, bundle in definitions.items():
        path = bundle.write(BUNDLES / f"{state}.bundle", visualise=True)
        print(path)


if __name__ == "__main__":
    main()
