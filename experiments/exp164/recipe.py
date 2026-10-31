"""Settings for the two constant-cue digit-routing excerpts."""

import math

SLUG = "exp164"
SCHEMA = "exp164.analysis/v1"
PATHS = ("A_E_to_C_E", "B_E_to_C_E", "C_I_to_E")


def configuration(
    *,
    source_configuration,
    end_ms=1100.0,
    window_ms=(650.0, 1100.0),
    closeup_ms=(900.0, 1050.0),
):
    cfg = dict(
        schema=SCHEMA,
        display_end_ms=float(end_ms),
        window_ms=list(map(float, window_ms)),
        closeup_ms=list(map(float, closeup_ms)),
        rate_smoothing_sd_ms=1.0,
        min_peak_distance_ms=15.0,
        min_prominence_hz=0.05,
        prominence_fraction=0.1,
        reference_start_ms=500.0,
        low_inhibition_fraction=0.25,
        conductance_method="analytical linear filter of saved spikes; native snapshot weights and planner delays/decay; zero initial state; float64 means, not native conductance recordings",
    )
    times = [cfg["display_end_ms"], *cfg["window_ms"], *cfg["closeup_ms"]]
    if (
        len(cfg["window_ms"]) != 2
        or len(cfg["closeup_ms"]) != 2
        or not all(math.isfinite(t) for t in times)
    ):
        raise ValueError("finite two-endpoint windows are required")
    a, b = cfg["window_ms"]
    c, d = cfg["closeup_ms"]
    if (
        not 0 < a < b <= cfg["display_end_ms"] < source_configuration["switch_ms"]
        or not a <= c < d <= b
    ):
        raise ValueError(
            "measurement/close-up windows must precede the switch and nest inside the display"
        )
    if (
        source_configuration["encoding"] != "rates"
        or source_configuration["network"]["populations"]["E"]["size"] != 64
    ):
        raise ValueError("expected the fixed 8×8 rate encoder")
    return cfg
