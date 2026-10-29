"""Validate one explicit HCO compute run and measure its spike ordering.

Run analyse.py --source COMPUTE_ID. Check full float64 trajectories against an
independent gate-first Euler calculation and retain measured numbers separately
from the immutable source. Toolbox reference agreement remains untested.
"""

import sys
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import argparse
import shutil

import numpy as np
from experiments.exp147.recipe import validate_configuration
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from scipy.special import expit


def analyse(*, source, run_id=None):
    computed = source_run(
        REPO / ".pingstore", source, experiment="exp147", stage="compute"
    )
    cfg = computed.record["execution"]["configuration"]
    validate_configuration(cfg)
    with np.load(computed.file("traces.npz")) as data:
        numbers = validate_and_measure(data, cfg)
    numbers["timing"] = load_json(computed.file("timing.json"))
    with stage_run(
        REPO,
        "exp147",
        "analyse",
        inputs={"compute": computed},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        write_json_atomic(run.export / "results.json", numbers)
        shutil.copyfile(computed.file("traces.npz"), run.export / "traces.npz")
    return run.run_id


def current_waveforms(cfg):
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    current = np.zeros((steps, 2))
    for cell, name in enumerate(("n1", "n2")):
        schedule = cfg["currents"][name]
        current[round(schedule["drive_start_ms"] / cfg["dt_ms"]) :, cell] = schedule[
            "drive_na"
        ]
        start = round(schedule["kick_start_ms"] / cfg["dt_ms"])
        stop = round(schedule["kick_stop_ms"] / cfg["dt_ms"])
        current[start:stop, cell] = schedule["kick_na"]
    return current


def reference(cfg):
    neuron, synapse, dt = cfg["neuron"], cfg["synapse"], cfg["dt_ms"]
    voltage = np.full(2, neuron["initial_voltage_mv"])
    sodium, potassium = voltage.copy(), voltage.copy()
    # Gate index is the source cell: index zero is n1→n2.
    gate = np.full(2, synapse["initial_gate_mv"])
    current = current_waveforms(cfg)
    values = np.empty((len(current), 2, 5))
    for step, injected in enumerate(current):
        gate += dt / synapse["tau_ms"] * (voltage - gate)
        conductance = synapse["g_max_us"] * expit(
            synapse["gamma_per_mv"] * (gate - synapse["bias_mv"])
        )
        sodium += dt / neuron["tau_na_ms"] * (voltage - sodium)
        potassium += dt / neuron["tau_k_ms"] * (voltage - potassium)
        g_na = neuron["g_na_us"] * expit(
            neuron["gamma_na_per_mv"] * (sodium - neuron["bias_na_mv"])
        )
        g_k = neuron["g_k_us"] * expit(
            neuron["gamma_k_per_mv"] * (potassium - neuron["bias_k_mv"])
        )
        outward = neuron["leak_us"] * (voltage - neuron["leak_reversal_mv"])
        outward += g_na * (voltage - neuron["e_na_mv"]) + g_k * (
            voltage - neuron["e_k_mv"]
        )
        outward += conductance[::-1] * (voltage - neuron["synapse_reversal_mv"])
        voltage += dt / neuron["capacitance_nf"] * (injected - outward)
        values[step] = np.column_stack((voltage, sodium, potassium, gate, conductance))
    return values


def crossings(voltage, cfg):
    previous = np.r_[cfg["neuron"]["initial_voltage_mv"], voltage[:-1]]
    threshold = cfg["neuron"]["spike_threshold_mv"]
    return (previous < threshold) & (voltage >= threshold)


def validate_and_measure(data, cfg):
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    expected_keys = {"time_ms", "input_time_ms"}
    for name in ("n1", "n2"):
        expected_keys.update(
            f"{name}_{suffix}"
            for suffix in ("voltage_mv", "spikes", "z_na_mv", "z_k_mv", "current_na")
        )
    for edge in ("n1_to_n2", "n2_to_n1"):
        expected_keys.update((f"{edge}_gate_mv", f"{edge}_conductance_us"))
    if set(data) != expected_keys:
        raise ValueError("unexpected HCO trace inventory")
    for name in expected_keys:
        if (
            data[name].shape != (steps,)
            or data[name].dtype != np.float64
            or not np.isfinite(data[name]).all()
        ):
            raise ValueError(f"invalid float64 trace: {name}")
    np.testing.assert_array_equal(
        data["time_ms"], (np.arange(steps) + 1) * cfg["dt_ms"]
    )
    np.testing.assert_array_equal(
        data["input_time_ms"], np.arange(steps) * cfg["dt_ms"]
    )
    currents, expected = current_waveforms(cfg), reference(cfg)
    summary = dict(neurons={}, toolbox_reference_validated=False)
    events = []
    for index, (name, edge) in enumerate((("n1", "n1_to_n2"), ("n2", "n2_to_n1"))):
        np.testing.assert_array_equal(data[f"{name}_current_na"], currents[:, index])
        spikes = crossings(data[f"{name}_voltage_mv"], cfg)
        np.testing.assert_array_equal(data[f"{name}_spikes"], spikes)
        times = data["time_ms"][spikes]
        reference_times = data["time_ms"][crossings(expected[:, index, 0], cfg)]
        if len(times) != len(reference_times):
            raise ValueError(f"{name}: independent reference spike count differs")
        spike_error = (
            float(np.max(np.abs(times - reference_times))) if len(times) else 0.0
        )
        voltage_error = float(
            np.max(np.abs(data[f"{name}_voltage_mv"] - expected[:, index, 0]))
        )
        if (
            voltage_error >= cfg["validation"]["voltage_atol_mv"]
            or spike_error >= cfg["validation"]["spike_atol_ms"]
        ):
            raise ValueError(f"{name}: independent reference comparison failed")
        for suffix, column in (
            (f"{name}_z_na_mv", 1),
            (f"{name}_z_k_mv", 2),
            (f"{edge}_gate_mv", 3),
            (f"{edge}_conductance_us", 4),
        ):
            np.testing.assert_allclose(
                data[suffix], expected[:, index, column], rtol=1e-9, atol=1e-8
            )
        summary["neurons"][name] = dict(
            spike_count=int(len(times)),
            spike_times_ms=times.tolist(),
            max_voltage_difference_mv=voltage_error,
            max_spike_time_difference_ms=spike_error,
        )
        events.extend((float(time), name) for time in times if time >= 10.0)
    events.sort()
    labels = [name for _, name in events]
    simultaneous = any(a[0] == b[0] for a, b in pairwise(events))
    summary.update(
        independent_euler_passed=True,
        alternating_after_10ms=len(labels) >= 2
        and not simultaneous
        and all(a != b for a, b in pairwise(labels)),
        events_after_10ms=len(labels),
        duration_ms=cfg["duration_ms"],
        dt_ms=cfg["dt_ms"],
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
