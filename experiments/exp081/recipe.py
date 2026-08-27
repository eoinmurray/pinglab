"""Committed physical definitions and analytical filter; no execution on import."""

from __future__ import annotations

import hashlib

import numpy as np

SLUG = "exp081"
PRESENTATION_MS = 200.0
DT_MS = 0.1
N_TIMESTEPS = int(round(PRESENTATION_MS / DT_MS))
PROBES_US = (0.6, 1.2, 2.4)
INPUT_RATES_HZ = np.linspace(0.0, 25.0, 101)
MOMENT_DRAWS = 512
DISTRIBUTION_RATES_HZ = (0.25, 3.0, 25.0)
FREQUENCY_RESPONSE_RATES_HZ = (0.25, 3.0, 25.0)
NOMINAL_PROBE_US = 1.2
FREQUENCY_PLOT_BOUNDS_HZ = (0.1, 200.0)
DISTRIBUTION_DRAWS = 4096
SEED = 81
FREQUENCY_BOUNDS_HZ = (1e-4, 1e6)
FREQUENCY_GRID_POINTS = 16385
COARSE_GRID_POINTS = 8193

PARAMETERS = {
    "C_m_nF": 1.0,
    "g_L_uS": 0.05,
    "E_L_mV": -65.0,
    "E_e_mV": 0.0,
    "tau_ampa_ms": 2.0,
}


def stable_seed(*parts: int) -> int:
    digest = hashlib.sha256(
        ":".join(str(part) for part in (SEED, *parts)).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def configuration(*, smoke: bool = False) -> dict:
    """Snapshot the committed recipe, including the explicit smoke profile."""
    return {
        "schema": "exp081.recipe/v1",
        "profile": "smoke" if smoke else "full",
        "presentation_ms": PRESENTATION_MS,
        "dt_ms": DT_MS,
        "probes_uS": list(PROBES_US),
        "input_rate_grid_hz": INPUT_RATES_HZ.tolist(),
        "pixel_intensity": 1.0,
        "distribution_rates_hz": list(DISTRIBUTION_RATES_HZ),
        "frequency_response_rates_hz": list(FREQUENCY_RESPONSE_RATES_HZ),
        "frequency_plot_bounds_hz": list(FREQUENCY_PLOT_BOUNDS_HZ),
        "frequency_plot_points": 1400,
        "nominal_probe_uS": NOMINAL_PROBE_US,
        "moment_draws": 32 if smoke else MOMENT_DRAWS,
        "distribution_draws": 128 if smoke else DISTRIBUTION_DRAWS,
        "frequency_bounds_hz": list(FREQUENCY_BOUNDS_HZ),
        "fine_grid_points": 1025 if smoke else FREQUENCY_GRID_POINTS,
        "coarse_grid_points": 513 if smoke else COARSE_GRID_POINTS,
        "membrane": dict(PARAMETERS),
        "seed": SEED,
        "moment_seed": stable_seed(1),
        "distribution_seed": stable_seed(2),
        "sd_ddof": 1,
        "histogram_bins": 60,
        "encoder": "bernoulli_per_timestep",
        "dtype": "float32",
        "voltage_update": "exact_frozen_conductance_after_event",
        "feature": "mean_post_update_voltage_minus_rest",
    }


def linear_operating_point(
    input_rate_hz: np.ndarray | float,
    probe_uS: np.ndarray | float,
    *,
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config if config is not None else configuration()
    rate = np.asarray(input_rate_hz, dtype=np.float64)
    probe = np.asarray(probe_uS, dtype=np.float64)
    mean_g = rate * probe * cfg["membrane"]["tau_ampa_ms"] / 1000.0
    mean_v = (
        cfg["membrane"]["g_L_uS"] * cfg["membrane"]["E_L_mV"]
        + mean_g * cfg["membrane"]["E_e_mV"]
    ) / (cfg["membrane"]["g_L_uS"] + mean_g)
    return mean_g, mean_v


def synapse_membrane_transfer(
    frequency_hz: np.ndarray,
    input_rate_hz: np.ndarray | float,
    probe_uS: np.ndarray | float,
    *,
    config: dict | None = None,
) -> np.ndarray:
    cfg = config if config is not None else configuration()
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    omega = 2.0 * np.pi * frequency / 1000.0
    mean_g, mean_v = linear_operating_point(input_rate_hz, probe_uS, config=cfg)
    synapse = np.asarray(probe_uS) / (1j * omega + 1.0 / cfg["membrane"]["tau_ampa_ms"])
    membrane = (cfg["membrane"]["E_e_mV"] - mean_v) / (
        1j * omega * cfg["membrane"]["C_m_nF"] + cfg["membrane"]["g_L_uS"] + mean_g
    )
    return synapse * membrane


def complete_transfer(
    frequency_hz: np.ndarray,
    input_rate_hz: np.ndarray | float,
    probe_uS: np.ndarray | float,
    *,
    config: dict | None = None,
) -> np.ndarray:
    cfg = config if config is not None else configuration()
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    omega = 2.0 * np.pi * frequency / 1000.0
    argument = omega * cfg["presentation_ms"] / 2.0
    averaging = np.exp(-1j * argument) * np.sinc(argument / np.pi)
    return averaging * synapse_membrane_transfer(
        frequency_hz, input_rate_hz, probe_uS, config=cfg
    )


def predicted_variance(
    input_rates_hz: np.ndarray,
    probe_uS: np.ndarray,
    *,
    grid_points: int = FREQUENCY_GRID_POINTS,
    config: dict | None = None,
) -> np.ndarray:
    cfg = config if config is not None else configuration()
    rates = np.asarray(input_rates_hz, dtype=np.float64).reshape(-1)
    probe = np.asarray(probe_uS, dtype=np.float64).reshape(-1)
    if rates.shape != probe.shape:
        raise ValueError("input_rates_hz and probe_uS must have matching shapes")
    result = np.zeros_like(rates)
    frequencies = np.geomspace(*cfg["frequency_bounds_hz"], grid_points)
    for start in range(0, rates.size, 128):
        indices = np.arange(start, min(start + 128, rates.size))
        positive = indices[rates[indices] > 0]
        if positive.size == 0:
            continue
        transfer = complete_transfer(
            frequencies[None, :],
            rates[positive, None],
            probe[positive, None],
            config=cfg,
        )
        integrand = np.abs(transfer) ** 2 * rates[positive, None] / 1000.0
        integral = np.trapezoid(integrand, frequencies, axis=1)
        dc = (
            np.abs(
                complete_transfer(
                    np.asarray([[0.0]]),
                    rates[positive, None],
                    probe[positive, None],
                    config=cfg,
                )[:, 0]
            )
            ** 2
        )
        low_tail = dc * rates[positive] / 1000.0 * cfg["frequency_bounds_hz"][0]
        result[positive] = 2.0 / 1000.0 * (integral + low_tail)
    return result
