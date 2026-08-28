"""Committed default PING drive sweep; no execution on import."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
from experiments.helpers.gamma_frequency import DEFAULT_PING_GAMMA

SLUG = "exp083"
DT_MS = 0.1
T_MS = 1_000.0
BURN_MS = 200.0
N_INPUT = 128
N_E = 80
N_I = 20
INPUT_RATES_HZ = (0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0)
TRIAL_SEEDS = (8300, 8301, 8302, 8303, 8304)
NETWORK_SEED = 83
REPRESENTATIVE_RATES_HZ = (25.0, 75.0, 150.0)
DISPLAY_TRIAL = 0

FREQUENCY_CONFIG = replace(
    DEFAULT_PING_GAMMA,
    name="default-ping-dominant-rhythm-v1",
    band_hz=(5.0, 80.0),
    burn_ms=BURN_MS,
    subharmonic_ratio=0.3,
)

SCALE = {
    "dt_ms": DT_MS,
    "t_ms": T_MS,
    "burn_ms": BURN_MS,
    "n_input": N_INPUT,
    "n_e": N_E,
    "n_i": N_I,
    "rates_hz": list(INPUT_RATES_HZ),
    "trials": len(TRIAL_SEEDS),
    "network_seed": NETWORK_SEED,
}


def author_network():
    from tools import snnlang as snn  # noqa: TID251

    net = snn.Network("default_ping_drive_response", dt=DT_MS * snn.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", N_INPUT),
        signal_type="spikes",
        unit="spike",
    )
    cell = snn.components.ping(
        net,
        name="ping",
        n_e=N_E,
        n_i=N_I,
        source=drive,
    )
    net.expose(cell.E.spikes, cell.I.spikes, name="population")
    return snn.compile(net, target="tools/snnsim")


def make_inputs(rate_hz: float) -> np.ndarray:
    """Paired deterministic trials: each seed is reused at every drive rate."""
    steps = round(T_MS / DT_MS)
    probability = rate_hz * DT_MS / 1_000.0
    trials = []
    for seed in TRIAL_SEEDS:
        rng = np.random.default_rng(seed)
        trials.append(rng.random((steps, N_INPUT), dtype=np.float32) < probability)
    return np.stack(trials, axis=1).astype(np.uint8)


def configuration():
    return {
        "schema": "exp083.recipe/v1",
        "config": deepcopy(SCALE),
        "trial_seeds": list(TRIAL_SEEDS),
        "frequency_analysis": FREQUENCY_CONFIG.json(),
        "representative_rates_hz": list(REPRESENTATIVE_RATES_HZ),
        "display_trial": DISPLAY_TRIAL,
        "input_generator": "numpy.default_rng.float32-bernoulli/v1",
        "rate_sd_ddof": 1,
        "rhythmicity": "autocorrelation-lobe-trough/v1",
        "lag_bin_ms": 1.0,
        "lag_max_ms": 20,
    }


def conditions():
    return [
        {"input_rate_hz": rate, "file": f"conditions/rate-{rate:g}.npz"}
        for rate in INPUT_RATES_HZ
    ]
