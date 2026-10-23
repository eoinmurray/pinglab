"""Committed coupling-grid and null-control recipe; no execution on import.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

import numpy as np

REFRACTORY_E_MS = 1.2

REFRACTORY_I_MS = 0.6

REFRACTORY_POLICY = "exact"

FIGURES = (
    "turnon_maps_compound.svg",
    "turnon_compound.svg",
    "grid_maps.svg",
    "grid_rasters.svg",
    "grid_autocorr.svg",
    "rate_invariance.svg",
    "null_autocorr.svg",
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
