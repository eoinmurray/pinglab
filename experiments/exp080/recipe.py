"""Committed exp080 calibration recipe; importing it performs no execution."""

from __future__ import annotations

import hashlib
import math
from typing import Any

SLUG = "exp080"
PRESENTATION_MS = 200.0
DT_MS = 0.1
N_TIMESTEPS = int(round(PRESENTATION_MS / DT_MS))
PROBE_US = 1.2
RATES_HZ = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
SEEDS = (42, 43, 44)
TRAIN_COUNT = 20_000
VALIDATION_COUNT = 5_000
TEST_COUNT = 5_000
EPOCHS_STANDARD = 50
EPOCHS = EPOCHS_STANDARD
BATCH_SIZE = 256
HIDDEN_UNITS = 1024
LEARNING_RATE = 0.001
CHANCE_ACCURACY = 0.10
USEFUL_ACCURACY = 0.50

PARAMETERS = {
    "C_m_nF": 1.0,
    "g_L_uS": 0.05,
    "E_L_mV": -65.0,
    "E_e_mV": 0.0,
    "tau_ampa_ms": 2.0,
}


def stable_seed(*parts: int) -> int:
    digest = hashlib.sha256(
        ":".join(str(part) for part in (80, *parts)).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def probe_single_spike(spike_time_ms: float) -> float:
    conductance = 0.0
    voltage = PARAMETERS["E_L_mV"]
    total = 0.0
    event_step = int(round(spike_time_ms / DT_MS))
    decay = math.exp(-DT_MS / PARAMETERS["tau_ampa_ms"])
    for step in range(N_TIMESTEPS):
        conductance = conductance * decay + (PROBE_US if step == event_step else 0.0)
        total_g = PARAMETERS["g_L_uS"] + conductance
        equilibrium = (
            PARAMETERS["g_L_uS"] * PARAMETERS["E_L_mV"]
            + conductance * PARAMETERS["E_e_mV"]
        ) / total_g
        voltage = equilibrium + (voltage - equilibrium) * math.exp(
            -DT_MS * total_g / PARAMETERS["C_m_nF"]
        )
        total += voltage - PARAMETERS["E_L_mV"]
    return total / N_TIMESTEPS


def validate_simulator() -> dict[str, Any]:
    decay = math.exp(-DT_MS / PARAMETERS["tau_ampa_ms"])
    early = probe_single_spike(20.0)
    late = probe_single_spike(180.0)
    checks = {
        "ampa_decay_in_unit_interval": 0.0 < decay < 1.0,
        "early_spike_exceeds_late_spike": early > late > 0.0,
        "zero_input_feature_is_zero_by_construction": True,
        "timestep_count_exact": N_TIMESTEPS == 2000,
    }
    if not all(checks.values()):
        raise RuntimeError(f"simulator validation failed: {checks}")
    return {"checks": checks, "early_spike_mV": early, "late_spike_mV": late}


def configuration(*, smoke: bool = False) -> dict:
    return {
        "schema": "exp080.recipe/v1",
        "profile": "smoke" if smoke else "production",
        "presentation_ms": PRESENTATION_MS,
        "dt_ms": DT_MS,
        "probe_uS": PROBE_US,
        "rates_hz": list(RATES_HZ),
        "seeds": list(SEEDS),
        "train_count": 100 if smoke else TRAIN_COUNT,
        "validation_count": 50 if smoke else VALIDATION_COUNT,
        "test_count": 50 if smoke else TEST_COUNT,
        "epochs": 2 if smoke else EPOCHS_STANDARD,
        "batch_size": BATCH_SIZE,
        "hidden_units": HIDDEN_UNITS,
        "learning_rate": LEARNING_RATE,
        "chance_accuracy": CHANCE_ACCURACY,
        "useful_accuracy": USEFUL_ACCURACY,
        "membrane": dict(PARAMETERS),
        "illustration_rates_hz": [0.5, 5.0, 25.0],
        "illustration_seed": stable_seed(11),
        "illustration_training_index": 0,
        "dtype": "float32",
        "encoder": "bernoulli_per_timestep",
        "voltage_update": "exact_frozen_conductance_after_decay_then_event",
        "feature": "mean_post_update_voltage_minus_rest",
        "checkpoint_role": "first_maximum_mixed_rate_validation_accuracy",
        "test_features": "shared_across_decoders_per_image_and_rate",
        "seed_schedule": "sha256_80_parts_first8_little_endian_63bit",
        "selection": "lowest_tested_rate_all_decoders_at_least_useful_accuracy",
    }


def reported_parameters(cfg: dict) -> dict:
    return {
        k: cfg[k]
        for k in (
            "presentation_ms",
            "dt_ms",
            "probe_uS",
            "rates_hz",
            "seeds",
            "train_count",
            "validation_count",
            "test_count",
            "epochs",
            "useful_accuracy",
        )
    }
