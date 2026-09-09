"""Committed coupling-grid and null-control recipe; no execution on import."""

import copy

import numpy as np
from experiments.exp033 import recipe as mean_field
from experiments.helpers.operating_point import (
    refractory_args,
    refractory_configuration,
)

SLUG = "exp054"
FIGURES = (
    "turnon_maps_compound.png",
    "turnon_compound.png",
    "grid_maps.png",
    "grid_rasters.png",
    "grid_autocorr.png",
    "rate_invariance.png",
    "null_autocorr.png",
)


def configuration(*, smoke=False, version=6):
    if version not in (1, 2, 3, 4, 5, 6):
        raise ValueError("unsupported exp054 recipe version")
    mf = mean_field.configuration(version=2 if version >= 6 else 1)
    return {
        "schema": f"exp054.recipe/v{version}",
        **(refractory_configuration() if version >= 5 else {}),
        "profile": "smoke" if smoke else "production",
        "dt_ms": 0.1 if version >= 3 else 0.25,
        **({"tau_gaba_ms": 6.0} if version >= 4 else {}),
        "sim_ms": 400.0 if smoke else 1000.0,
        "burn_ms": 100.0,
        "n_e": 256 if version == 1 else 1024,
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
    # Recipe versions describe scientific conditions within v4 storage runs.
    # Earlier scientific recipes stay readable; new compute uses v4.
    if cfg not in tuple(
        configuration(smoke=smoke, version=version)
        for version in (1, 2, 3, 4, 5, 6)
        for smoke in (False, True)
    ):
        from pingstore.contracts import PingstoreError

        raise PingstoreError("inconsistent exp054 recipe")
    return cfg


def refresh_configuration(spikes, theory):
    """An analysis recipe, keeping the two independent source recipes intact."""
    from pingstore.contracts import PingstoreError

    validate(spikes)
    if spikes["schema"] not in {"exp054.recipe/v4", "exp054.recipe/v5", "exp054.recipe/v6"}:
        raise PingstoreError("theory refresh requires 1024-E, 0.1-ms, 6-ms-GABA spikes")
    if theory != mean_field.configuration(version=2):
        raise PingstoreError("theory refresh requires the adopted exp033 recipe")
    return {
        "schema": "exp054.theory-refresh/v1",
        "spike_source_recipe": copy.deepcopy(spikes),
        "theory_recipe": copy.deepcopy(theory),
    }


def validate_analysis(cfg):
    from pingstore.contracts import PingstoreError

    if isinstance(cfg, dict) and cfg.get("schema") == "exp054.theory-refresh/v1":
        if set(cfg) != {
            "schema",
            "spike_source_recipe",
            "theory_recipe",
        } or cfg != refresh_configuration(
            cfg["spike_source_recipe"], cfg["theory_recipe"]
        ):
            raise PingstoreError("inconsistent exp054 theory refresh recipe")
        return cfg
    return validate(cfg)


def spike_configuration(cfg):
    validate_analysis(cfg)
    return cfg.get("spike_source_recipe", cfg)


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
        *refractory_args(),
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
    if "tau_gaba_ms" in cfg:
        args += ["--tau-gaba", str(cfg["tau_gaba_ms"])]
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
