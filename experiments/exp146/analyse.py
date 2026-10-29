"""Validate an explicit Task 1 compute run and measure its voltage trajectory.

Run analyse.py --source COMPUTE_ID. Saved traces are checked against input,
gate and crossing contracts and an independent scalar Euler implementation.
The absent toolbox reference is reported explicitly, never counted as a pass.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import argparse
import shutil

import numpy as np
from experiments.exp146.recipe import validate_configuration
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from scipy.special import expit


def analyse(*, source, run_id=None):
    computed = source_run(
        REPO / ".pingstore", source, experiment="exp146", stage="compute"
    )
    cfg = computed.record["execution"]["configuration"]
    validate_configuration(cfg)
    with np.load(computed.file("traces.npz")) as data:
        validate_traces(data, cfg)
        numbers = compare_reference(data, cfg)
    numbers["timing"] = load_json(computed.file("timing.json"))
    numbers["duration_ms"] = cfg["duration_ms"]
    numbers["dt_ms"] = cfg["dt_ms"]
    with stage_run(
        REPO,
        "exp146",
        "analyse",
        inputs={"compute": computed},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        write_json_atomic(run.export / "results.json", numbers)
        shutil.copyfile(computed.file("traces.npz"), run.export / "traces.npz")
    return run.run_id


def euler_reference(cfg, current):
    """Return post-step states, using scalar NumPy/SciPy arithmetic only."""
    voltage = sodium = potassium = cfg["initial_voltage_mv"]
    trace = np.empty((len(current), 3), dtype=np.float64)
    for index, injected in enumerate(current):
        sodium += cfg["dt_ms"] * (voltage - sodium) / cfg["tau_na_ms"]
        potassium += cfg["dt_ms"] * (voltage - potassium) / cfg["tau_k_ms"]
        g_na = cfg["g_na_us"] * expit(
            cfg["gamma_na_per_mv"] * (sodium - cfg["bias_na_mv"])
        )
        g_k = cfg["g_k_us"] * expit(
            cfg["gamma_k_per_mv"] * (potassium - cfg["bias_k_mv"])
        )
        total = injected - cfg["leak_us"] * (voltage - cfg["leak_reversal_mv"])
        total -= g_na * (voltage - cfg["e_na_mv"])
        total -= g_k * (voltage - cfg["e_k_mv"])
        voltage += cfg["dt_ms"] * total / cfg["capacitance_nf"]
        trace[index] = voltage, sodium, potassium
    return trace


def crossings(voltage, cfg):
    previous = np.r_[cfg["initial_voltage_mv"], voltage[:-1]]
    return (previous < cfg["spike_threshold_mv"]) & (
        voltage >= cfg["spike_threshold_mv"]
    )


def validate_traces(data, cfg):
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    required = (
        "time_ms",
        "input_time_ms",
        "voltage_mv",
        "z_na_mv",
        "z_k_mv",
        "g_na_us",
        "g_k_us",
        "current_na",
        "spikes",
    )
    for key in required:
        if (
            data[key].shape != (steps,)
            or data[key].dtype != np.float64
            or not np.isfinite(data[key]).all()
        ):
            raise ValueError(f"invalid float64 trace: {key}")
    np.testing.assert_allclose(
        data["time_ms"], (np.arange(steps) + 1) * cfg["dt_ms"], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        data["input_time_ms"], np.arange(steps) * cfg["dt_ms"], rtol=0, atol=1e-12
    )
    expected = np.zeros(steps)
    expected[round(cfg["pulse_start_ms"] / cfg["dt_ms"]) :] = cfg["pulse_current_na"]
    np.testing.assert_array_equal(data["current_na"], expected)
    np.testing.assert_array_equal(data["spikes"], crossings(data["voltage_mv"], cfg))
    for channel in ("na", "k"):
        expected = cfg[f"g_{channel}_us"] * expit(
            cfg[f"gamma_{channel}_per_mv"]
            * (data[f"z_{channel}_mv"] - cfg[f"bias_{channel}_mv"])
        )
        np.testing.assert_allclose(
            data[f"g_{channel}_us"], expected, rtol=1e-12, atol=1e-14
        )


def compare_reference(data, cfg):
    reference = euler_reference(cfg, data["current_na"])
    voltage_error = float(np.max(np.abs(data["voltage_mv"] - reference[:, 0])))
    times = data["time_ms"][data["spikes"].astype(bool)]
    reference_times = data["time_ms"][crossings(reference[:, 0], cfg)]
    same_count = len(times) == len(reference_times)
    spike_error = (
        float(np.max(np.abs(times - reference_times)))
        if same_count and len(times)
        else (0.0 if same_count else None)
    )
    passed = (
        same_count
        and voltage_error < cfg["validation"]["voltage_atol_mv"]
        and spike_error < cfg["validation"]["spike_atol_ms"]
    )
    if not passed:
        raise ValueError("independent Euler comparison failed")
    return dict(
        comparison="Independent scalar forward-Euler equations; not Toolbox reference",
        max_voltage_difference_mv=voltage_error,
        max_gate_difference_mv=float(
            np.max(
                np.abs(
                    np.column_stack((data["z_na_mv"], data["z_k_mv"]))
                    - reference[:, 1:]
                )
            )
        ),
        spike_time_difference_ms=spike_error,
        spike_count=int(len(times)),
        reference_spike_count=int(len(reference_times)),
        independent_euler_passed=bool(passed),
        toolbox_reference_validated=False,
        spike_times_ms=times.tolist(),
        voltage_min_mv=float(data["voltage_mv"].min()),
        voltage_max_mv=float(data["voltage_mv"].max()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
