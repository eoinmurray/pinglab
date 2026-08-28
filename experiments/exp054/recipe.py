"""Committed coupling-grid and null-control recipe; no execution on import."""

import numpy as np
from experiments.exp033 import recipe as mean_field

SLUG = "exp054"
FIGURES = (
    "turnon_maps_compound.png",
    "turnon_compound.png",
    "grid_maps.png",
    "grid_rasters.png",
    "grid_autocorr.png",
    "rate_invariance.png",
    "null_autocorr.png",
    "onset_super_compound.png",
    "onset_super_compound.pdf",
)


def configuration(*, smoke=False):
    mf = mean_field.configuration()
    return {
        "schema": "exp054.recipe/v1",
        "profile": "smoke" if smoke else "production",
        "dt_ms": 0.25,
        "sim_ms": 400.0 if smoke else 1000.0,
        "burn_ms": 100.0,
        "n_e": 256,
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
        "mean_field": {
            k: v
            for k, v in mf.items()
            if k
            not in {
                "schema",
                "profile",
                "sigma_grid_mV",
                "sensitivity_grid",
                "convergence_grid",
                "cycle",
                "comparison",
                "ladder",
                "comparison_and_ladder_solver",
            }
        },
    }


def validate(cfg):
    if cfg not in (configuration(), configuration(smoke=True)):
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


def simulation_args(cfg, item, output):
    args = [
        "sim",
        "--input",
        "synthetic-spikes",
        "--model",
        "ping",
        "--n-hidden",
        str(cfg["n_e"]),
        "--n-inh",
        str(cfg["n_i"]),
        "--n-in",
        str(cfg["n_e"] if item["private"] else cfg["shared_n_in"]),
        "--w-ei-mean",
        str(item["wei"]),
        "--w-ie-mean",
        str(item["wie"]),
        "--input-rate",
        str(item["rate_hz"]),
        "--n-batch",
        "1",
        "--t-ms",
        str(cfg["sim_ms"]),
        "--dt",
        str(cfg["dt_ms"]),
        "--seed",
        str(cfg["seed"]),
        "--outputs",
        "rasters",
        "--out-dir",
        str(output),
    ]
    if item["private"]:
        args += ["--private-w-in", "--w-in", str(cfg["private_w_in"])]
    else:
        args += [
            "--w-in",
            str(cfg["shared_w_in"]),
            "--w-in-initial-zero-fraction",
            str(cfg["shared_zero_fraction"]),
        ]
    return args + [
        "--recording-mode",
        "spikes",
        "--recording-start-step",
        str(int(cfg["burn_ms"] / cfg["dt_ms"])),
        "--output-fields",
        "e_trial",
        "e_t",
        "e_cell",
        "i_trial",
        "i_t",
        "i_cell",
    ]
