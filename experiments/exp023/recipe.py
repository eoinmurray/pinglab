"""PING fundamentals: preserved scientific settings, without execution on import."""

from pathlib import Path

SLUG = "exp023"
DT_MS = 0.1
N_E, N_I, N_IN = 1024, 256, 1024
# The old f–I command omitted --n-in and used the simulator's 784-channel default.
# Preserve this distinction; unifying the protocols is a separate scientific change.
FI_N_IN = 784
SEED = 42
COBA_INPUT_RATE_HZ, PING_INPUT_RATE_HZ = 5, 45
CELLS = ("coba", "ping")
FI_EI = {"coba": "0", "ping": "1.5"}
FI_RATES_HZ = [2, 5, 10, 20, 40, 70, 100]
F_GAMMA_BAND_HZ = (5.0, 150.0)
BIOPHYSICS = {
    "E_L_mV": -65.0,
    "E_E_mV": 0.0,
    "E_I_mV": -80.0,
    "g_L_E_uS": 0.05,
    "g_L_I_uS": 0.10,
    "threshold_mV": -50.0,
    "reset_mV": -65.0,
    "tau_ampa_ms": 2.0,
    "tau_gaba_ms": 9.0,
    "C_m_E_nF": 1.0,
    "C_m_I_nF": 0.5,
    "refractory_E_ms": 3.0,
    "refractory_I_ms": 1.5,
}


def _args(cell: str, rate: int, n_in: int, smoke: bool) -> list[str]:
    return [
        "sim",
        "--model",
        "ping",
        "--input",
        "synthetic-spikes",
        "--n-hidden",
        str(N_E),
        "--n-inh",
        str(N_I),
        "--n-in",
        str(n_in),
        "--ei-strength",
        FI_EI[cell],
        "--w-in",
        "1.5",
        "0.3",
        "--w-in-initial-zero-fraction",
        "0.95",
        "--input-rate",
        str(rate),
        "--t-ms",
        "200" if smoke else "400",
        "--dt",
        str(DT_MS),
        "--seed",
        str(SEED),
        "--tau-gaba",
        str(BIOPHYSICS["tau_gaba_ms"]),
    ]


def raster_args(
    cell: str, out_dir: Path | None = None, *, smoke: bool = False
) -> list[str]:
    rate = COBA_INPUT_RATE_HZ if cell == "coba" else PING_INPUT_RATE_HZ
    args = _args(cell, rate, N_IN, smoke) + [
        "--output-fields",
        "spk_e",
        "spk_i",
        "v_e_selected",
        "ge_e_selected",
        "gi_e_selected",
        "v_i_selected",
        "ge_i_selected",
        "e_trace_index",
        "i_trace_index",
        "has_gi_e",
    ]
    if out_dir is not None:
        args += ["--out-dir", str(out_dir)]
    return args


def fi_args(
    cell: str, rate_hz: int, out_dir: Path | None = None, *, smoke: bool = False
) -> list[str]:
    args = _args(cell, rate_hz, FI_N_IN, smoke) + [
        "--recording-mode",
        "spikes",
        "--output-fields",
        "spk_e_count",
        "spk_i_count",
    ]
    if out_dir is not None:
        args += ["--out-dir", str(out_dir)]
    return args


def drive_provenance(*, smoke: bool = False) -> dict:
    def point(args):
        def value(flag):
            return args[args.index(flag) + 1]

        return {
            "input": value("--input"),
            "input_rate_hz": float(value("--input-rate")),
            "ei_strength": float(value("--ei-strength")),
            "t_ms": float(value("--t-ms")),
            "dt_ms": float(value("--dt")),
            "n_in": int(value("--n-in")),
            "seed": int(value("--seed")),
            "scientific_args": args,
        }

    return {
        "raster_operating_points": {
            cell: point(raster_args(cell, smoke=smoke)) for cell in CELLS
        },
        "fi_sweep": {
            "input": "synthetic-spikes",
            "input_rates_hz": list(FI_RATES_HZ),
            "ei_strength_by_cell": {
                cell: float(value) for cell, value in FI_EI.items()
            },
            "t_ms": 200 if smoke else 400,
            "dt_ms": DT_MS,
            "n_in": FI_N_IN,
            "seed": SEED,
        },
    }


def configuration(*, smoke: bool = False) -> dict:
    return {
        "schema": "exp023.recipe/v1",
        "profile": "smoke" if smoke else "production",
        "model": "ping",
        "cells": list(CELLS),
        "n_e": N_E,
        "n_i": N_I,
        "seed": SEED,
        "trials_per_condition": 1,
        "initial_voltage_mV": -65.0,
        "initial_conductance_uS": 0.0,
        "input_weight_parent_mean": 1.5,
        "input_weight_parent_sd": 0.3,
        "input_initial_zero_fraction": 0.95,
        "recurrent_initial_zero_fraction": 0.0,
        "ei_ratio": 2.0,
        "integration": "exponential_euler",
        "drive": drive_provenance(smoke=smoke),
        "biophysics": dict(BIOPHYSICS),
    }


def simulations(*, smoke: bool = False):
    for cell in CELLS:
        yield f"scope/{cell}", raster_args(cell, smoke=smoke)
    for cell in CELLS:
        for rate in FI_RATES_HZ:
            yield f"fi/{cell}__r{rate}", fi_args(cell, rate, smoke=smoke)
