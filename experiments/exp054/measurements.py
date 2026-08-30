"""Original rhythmicity and rate estimators applied to explicit saved evidence."""

import numpy as np
from experiments.exp033 import measurements as mf_measurements
from experiments.helpers.rhythmicity import (
    iei_histogram,
    population_event_times,
    rhythmicity_scalars,
    spike_autocorrelogram,
)
from pingstore.contracts import PingstoreError

from . import evidence, recipe


def dense(data, cfg):
    burn = int(cfg["burn_ms"] / cfg["dt_ms"])
    result = []
    for prefix, size in (("e", cfg["n_e"]), ("i", cfg["n_i"])):
        a = np.zeros((int(data["T"]) - burn, size), dtype=np.int8)
        times, cells = data[f"{prefix}_t"], data[f"{prefix}_cell"]
        keep = times >= burn
        a[times[keep] - burn, cells[keep]] = 1
        result.append(a)
    return result


def score(spikes, cfg):
    dt, lag, bin_ms = cfg["dt_ms"], cfg["max_lag_ms"], cfg["bin_ms"]
    rate = float(spikes.sum() / (spikes.shape[1] * spikes.shape[0] * dt / 1000.0))
    ac_lags, ac = spike_autocorrelogram(spikes, dt, lag, bin_ms)
    iei_lags, iei = iei_histogram(population_event_times(spikes, dt), lag, bin_ms)
    scalars = rhythmicity_scalars(ac_lags, ac, iei_lags, iei, bin_ms)
    return {
        "rate": rate,
        "ac_lags": ac_lags,
        "ac": ac,
        "contrast": scalars["contrast"] if scalars["contrast"] is not None else np.nan,
        "lobe_lag": scalars["lobe_lag"],
        "trough_lag": scalars["trough_lag"],
    }


def recordings(source, cfg):
    measured, grid_cells = {}, {}
    for item in recipe.jobs(cfg):
        data = evidence.raster(
            source.file("probe", item["id"], "rasters.npz"), cfg
        )
        e, i = dense(data, cfg)
        s = score(e, cfg)
        measured[item["id"]] = s
        window = int(cfg["display_window_ms"] / cfg["dt_ms"])
        grid_cells[item["id"]] = {
            "contrast": s["contrast"],
            "rate_hz": s["rate"],
            "rate_i_hz": float(
                i.sum() / (i.shape[1] * i.shape[0] * cfg["dt_ms"] / 1000.0)
            ),
            "ac_lags": s["ac_lags"],
            "ac": s["ac"],
            "lobe_lag": s["lobe_lag"],
            "trough_lag": s["trough_lag"],
            "e": e[:window, : cfg["display_e"]],
            "i": i[:window, : cfg["display_i"]],
        }
    grid = [
        [
            grid_cells[recipe.job(cfg, e, i, cfg["input_rate_hz"])["id"]]
            for e in cfg["wei_mean"]
        ]
        for i in cfg["wie_mean"]
    ]
    nulls = {}
    for private, key in ((True, "private_null_hz"), (False, "shared_null_hz")):
        rows = [
            measured[recipe.job(cfg, 0.0, 0.0, rate, private)["id"]]
            for rate in cfg[key]
        ]
        nulls["private" if private else "shared"] = sorted(
            rows, key=lambda r: r["rate"]
        )
    return {
        "grid": grid,
        "private_null": nulls["private"],
        "shared_null": nulls["shared"],
    }


def summary(coords, cfg):
    grid, private, shared = (
        coords["grid"],
        coords["private_null"],
        coords["shared_null"],
    )

    def matrix(key):
        return [[c[key] if np.isfinite(c[key]) else None for c in row] for row in grid]

    rates = [c["rate_hz"] for row in grid for c in row]
    contrasts = [c["contrast"] for row in grid for c in row]
    return {
        "config": {
            "source": "untrained PING networks, private per-cell Poisson input",
            **{
                k: cfg[k]
                for k in (
                    "dt_ms",
                    "sim_ms",
                    "burn_ms",
                    "n_e",
                    "input_rate_hz",
                    "private_w_in",
                    "max_lag_ms",
                    "bin_ms",
                )
            },
        },
        "grid": {
            "wei_mean": cfg["wei_mean"],
            "wie_mean": cfg["wie_mean"],
            "contrast": matrix("contrast"),
            "rate_e_hz": matrix("rate_hz"),
            "rate_i_hz": matrix("rate_i_hz"),
            "rate_e_min_hz": float(np.nanmin(rates)),
            "rate_e_max_hz": float(np.nanmax(rates)),
            "contrast_min": float(np.nanmin(contrasts)),
            "contrast_max": float(np.nanmax(contrasts)),
        },
        "rate_invariance": {
            "private_null_max": float(max(d["contrast"] for d in private)),
            "shared_null_max": float(max(d["contrast"] for d in shared)),
            "private_scan": {
                "rate_hz": [d["rate"] for d in private],
                "contrast": [d["contrast"] for d in private],
            },
            "shared_scan": {
                "rate_hz": [d["rate"] for d in shared],
                "contrast": [d["contrast"] for d in shared],
            },
        },
    }


def mean_field(raw, cfg):
    if raw.get("schema") != "exp054.mean-field/v1":
        raise PingstoreError("invalid exp054 mean-field evidence")
    grid = np.linspace(*cfg["mean_field"]["drive_grid"])
    reference = raw["reference"]
    mf_measurements.validate_continuation(reference, grid)
    if len(reference["sweep"]) != len(grid) or reference["hopf"] is None:
        raise PingstoreError("incomplete exp054 reference continuation")
    criticality = mf_measurements.hysteresis(reference["ramp"], reference["hopf"])
    if [r["tau_gaba_ms"] for r in raw["frequency"]] != cfg["mean_field"]["tau_grid_ms"]:
        raise PingstoreError("incomplete exp054 frequency sweep")
    frequencies = []
    for row in raw["frequency"]:
        mf_measurements.validate_continuation(row, grid)
        if len(row["sweep"]) != len(grid):
            raise PingstoreError("incomplete exp054 frequency continuation")
        h = row["hopf"]
        frequencies.append(
            {
                "tau_gaba_ms": row["tau_gaba_ms"],
                "f_star_Hz": h["freq_star_Hz"] if h else None,
                "I_ext_star": h["I_ext_star"] if h else None,
            }
        )
    return {
        "sweep": reference["sweep"],
        "hopf": reference["hopf"],
        "criticality": criticality,
        "frequency_vs_tau_gaba": frequencies,
    }
