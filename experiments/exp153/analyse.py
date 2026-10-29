"""Measure the matched locking map and a selected coupling-onset trajectory.

Each drive has an uncoupled calibration; decaying or irregular rhythms cannot
qualify as locked. The selected map condition supplies both E/I rate traces
and continuous observed-peak phase for a 1,500 ms onset display.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp153 import recipe
from experiments.exp153.compute import ancestry
from pingstore.library import load_json, stage_run, write_json_atomic
from scipy.signal import find_peaks


def read_trajectory(computed, condition, cfg):
    saved = load_json(computed.file(condition["label"] + "-trajectory.json"))
    model = load_json(computed.file(condition["label"] + "-model.json"))
    duration, interval = (
        cfg["duration_ms"],
        cfg["sample_interval_ms"],
    )
    expected = np.append(np.arange(0, duration, interval), duration)
    time = np.asarray(saved["time_ms"], dtype=float)
    state = {
        name: np.asarray(values, dtype=float) for name, values in saved["state"].items()
    }
    if (
        saved.get("schema") != "snnlab.meanfield.trajectory/v1"
        or not np.array_equal(time, expected)
        or set(state) != set(model["state_order"])
        or any(
            v.shape != time.shape or not np.isfinite(v).all() for v in state.values()
        )
    ):
        raise ValueError("invalid native mean-field trajectory")
    if any(np.min(v) < -1e-7 for v in state.values()):
        raise ValueError("trajectory leaves nonnegative rate/conductance domain")
    metadata = saved["integration"]
    if metadata["delay_mode"] != "explicit" or metadata["solver"] != "method_of_steps":
        raise ValueError("requires explicit-delay method-of-steps evidence")
    if metadata["parameters"] != {
        name: row["value"] for name, row in model["parameters"].items()
    }:
        raise ValueError("trajectory parameters differ from saved model")
    parameters = metadata["parameters"]
    source = cfg["network"]
    for name, incoming in cfg["incoming_weights_us"].items():
        expected_weight = (
            incoming * condition["coupling_us"] / source["base_coupling_us"]
            if "_K_E" in name
            else incoming * cfg.get("reduction_projection_scales", {}).get(name, 1.0)
        )
        if not np.isclose(
            parameters[name + ".coupling"], expected_weight, rtol=1e-12, atol=1e-14
        ):
            raise ValueError("trajectory coupling differs from its recorded condition")
    for name, rate in recipe.input_rates(source, condition["difference_hz"]).items():
        if parameters[name + ".rate"] != rate:
            raise ValueError("trajectory input differs from its recorded condition")
    if (
        metadata["duration_ms"] != duration
        or metadata["sample_interval_ms"] != interval
    ):
        raise ValueError("trajectory timing differs from recorded settings")
    solver = cfg["solver"]
    for field, expected_value in (
        ("method", solver["method"]),
        ("rtol", solver["rtol"]),
        ("atol", solver["atol"]),
        ("max_step_ms", solver["max_step_ms"]),
    ):
        if metadata[field] != expected_value:
            raise ValueError("trajectory solver differs from recorded settings")
    return time, state


def rhythm(time, rate, window, settings):
    """Use absolute amplitude before peak phase, excluding numerical ripples."""
    selected = (time >= window[0]) & (time < window[1])
    t, y = time[selected], rate[selected]
    if len(t) < 3:
        raise ValueError("rhythm window needs at least three observations")
    amplitude = float(np.ptp(y))
    mean = float(np.mean(y))
    minimum = max(
        settings["min_amplitude_hz"], settings.get("min_relative_amplitude", 0) * mean
    )
    peaks, _ = find_peaks(
        y,
        distance=max(
            1, round(settings.get("min_peak_distance_ms", 15) / (time[1] - time[0]))
        ),
        prominence=max(
            0.1 * settings["min_amplitude_hz"],
            settings.get("prominence_fraction", 0.1) * amplitude,
        ),
    )
    intervals = np.diff(t[peaks])
    frequency = float(1000 / intervals.mean()) if len(intervals) else None
    cv = float(intervals.std() / intervals.mean()) if len(intervals) else None
    band = settings["frequency_band_hz"]
    failures = []
    if amplitude < minimum:
        failures.append("insufficient amplitude")
    if len(peaks) < settings["min_peaks"]:
        failures.append("too few peaks")
    if frequency is None or (band is not None and not band[0] <= frequency <= band[1]):
        failures.append("frequency unavailable or outside requested band")
    if cv is None or cv > settings["max_interval_cv"]:
        failures.append("irregular or unavailable peak intervals")
    summary = dict(
        valid=not failures,
        reasons=failures,
        mean_hz=mean,
        amplitude_hz=amplitude,
        required_amplitude_hz=minimum,
        peaks=len(peaks),
        frequency_hz=frequency,
        interval_cv=cv,
        window_ms=list(window),
    )
    phase = np.full(time.shape, np.nan)
    # Phase can be displayed through a coupling-induced frequency transition.
    # Regularity and sustained-amplitude checks still gate locking decisions.
    if amplitude >= minimum and len(peaks) >= 2:
        phase = np.interp(
            time, t[peaks], 2 * np.pi * np.arange(len(peaks)), left=np.nan, right=np.nan
        )
    return summary, phase


def gate_metrics(time, state, cfg):
    gate = cfg["gate"]
    circuits = {}
    for circuit in ("A", "B"):
        rate = state[f"PING_{circuit}_E.rate"]
        early, late = [
            rhythm(time, rate, window, gate)[0] for window in gate["windows_ms"]
        ]
        ratio = (
            late["amplitude_hz"] / early["amplitude_hz"]
            if early["amplitude_hz"]
            else None
        )
        sustained = ratio is not None and ratio >= gate["min_late_to_early_amplitude"]
        circuits[circuit] = dict(
            passed=bool(early["valid"] and late["valid"] and sustained),
            early=early,
            late=late,
            late_to_early_amplitude=ratio,
            sustained_amplitude=bool(sustained),
        )
    return dict(
        gate_passed=all(row["passed"] for row in circuits.values()), circuits=circuits
    )


def pair_metrics(time, state, cfg):
    settings = cfg["phase"]
    phases, rhythms = {}, {}
    for circuit in ("A", "B"):
        rhythms[circuit], phases[circuit] = rhythm(
            time,
            state[f"PING_{circuit}_E.rate"],
            [settings["start_ms"], cfg["duration_ms"]],
            settings,
        )
    selected = np.isfinite(phases["A"]) & np.isfinite(phases["B"])
    sustained = gate_metrics(time, state, cfg)
    valid = (
        sustained["gate_passed"]
        and all(row["valid"] for row in rhythms.values())
        and selected.sum() >= 100
    )
    result = dict(
        valid=bool(valid),
        locked=False,
        concentration=None,
        drift_hz=None,
        phase_span_cycles=None,
        mean_phase_cycles=None,
        rhythms=rhythms,
        sustained=sustained,
    )
    if valid:
        phase = phases["A"][selected] - phases["B"][selected]
        drift = float(np.polyfit(time[selected] / 1000, phase / (2 * np.pi), 1)[0])
        circular = np.exp(1j * phase).mean()
        concentration = float(abs(circular))
        span = float(np.ptp(phase) / (2 * np.pi))
        result.update(
            drift_hz=drift,
            concentration=concentration,
            phase_span_cycles=span,
            mean_phase_cycles=float(np.angle(circular) / (2 * np.pi)),
            locked=bool(
                abs(drift) < settings["max_abs_drift_hz"]
                and concentration > settings["min_concentration"]
                and span < settings["max_phase_span_cycles"]
            ),
        )
    return result


def relative_phase_trace(time, state, cfg):
    """Unwrapped observed-peak phase; invalid traces remain explicitly empty."""
    phases = {
        circuit: rhythm(
            time,
            state[f"PING_{circuit}_E.rate"],
            [0, cfg["duration_ms"]],
            cfg["phase"],
        )[1]
        for circuit in ("A", "B")
    }
    difference = (phases["A"] - phases["B"]) / (2 * np.pi)
    selected = np.isfinite(difference) & (time <= cfg["duration_ms"])
    return dict(
        time_ms=time[selected].tolist(),
        relative_phase_cycles=difference[selected].tolist(),
    ), dict(
        display_window_ms=[0, cfg["duration_ms"]],
        onset_ms=cfg["network"]["onset_ms"],
        definition="Unwrapped E-rate phase A minus B in cycles, with phase zero at each circuit's first detected peak; no wrapping, smoothing, resetting or extrapolation",
    )


def assemble_map(metrics, cfg):
    """Calibrate columns with valid uncoupled rhythms, preserving invalid cells."""
    expected = recipe.conditions(cfg)
    if [
        (row["label"], row["difference_hz"], row["coupling_us"]) for row in metrics
    ] != [(row["label"], row["difference_hz"], row["coupling_us"]) for row in expected]:
        raise ValueError("incomplete or reordered grid coverage")
    differences = cfg["sweep"]["differences_hz"]
    couplings = cfg["sweep"]["couplings_us"]
    shape = (len(couplings), len(differences))
    arrays = dict(
        concentration=np.full(shape, np.nan),
        valid=np.zeros(shape, dtype=bool),
        locked=np.zeros(shape, dtype=bool),
        detuning_hz=np.full(len(differences), np.nan),
        input_difference_hz=np.asarray(differences),
        coupling_us=np.asarray(couplings),
    )
    baselines = {
        row["difference_hz"]: row for row in metrics if row["coupling_us"] == 0
    }
    cells = []
    for column, difference in enumerate(differences):
        baseline = baselines[difference]
        if baseline["valid"]:
            arrays["detuning_hz"][column] = (
                baseline["rhythms"]["A"]["frequency_hz"]
                - baseline["rhythms"]["B"]["frequency_hz"]
            )
    for metric in metrics:
        row = couplings.index(metric["coupling_us"])
        column = differences.index(metric["difference_hz"])
        calibrated = bool(np.isfinite(arrays["detuning_hz"][column]))
        valid = bool(metric["valid"] and calibrated)
        locked = bool(metric["locked"] and valid)
        arrays["valid"][row, column] = valid
        arrays["locked"][row, column] = locked
        if valid:
            arrays["concentration"][row, column] = metric["concentration"]
        cells.append(
            dict(
                **{
                    key: value
                    for key, value in metric.items()
                    if key not in ("valid", "locked")
                },
                rhythm_valid=metric["valid"],
                baseline_valid=calibrated,
                valid=valid,
                locked=locked,
                uncoupled_detuning_hz=float(arrays["detuning_hz"][column])
                if calibrated
                else None,
            )
        )
    return arrays, cells


def diagnostic(time, state, cfg):
    """Use the recorded display and measurement windows on the selected map trace."""
    example = cfg["example"]
    settings = recipe.analysis_configuration(cfg)["onset_analysis"]
    short = copy.deepcopy(cfg)
    short["duration_ms"] = example["display_window_ms"][1]
    short["phase"]["start_ms"] = example["phase_window_ms"][0]
    short["phase"]["min_peaks"] = settings["min_peaks"]
    short["gate"]["windows_ms"] = settings["gate_windows_ms"]
    short["gate"]["min_peaks"] = settings["min_peaks"]
    trace, details = relative_phase_trace(time, state, short)
    frequencies = {
        circuit: {
            label: rhythm(
                time, state[f"PING_{circuit}_E.rate"], window, short["phase"]
            )[0]
            for label, window in example["frequency_windows_ms"].items()
        }
        for circuit in ("A", "B")
    }
    visible = (time >= example["display_window_ms"][0]) & (time <= short["duration_ms"])
    rates = dict(time_ms=time[visible])
    rates.update(
        {
            f"{circuit}_{population}": state[f"PING_{circuit}_{population}.rate"][
                visible
            ]
            for circuit in ("A", "B")
            for population in ("E", "I")
        }
    )
    return (
        dict(
            **example,
            phase_trace=details,
            frequencies=frequencies,
            coupled=pair_metrics(time, state, short),
        ),
        trace,
        rates,
    )


def analyse(*, source, run_id=None):
    ancestors = ancestry(REPO, source)
    computed = ancestors[source]
    if (
        computed.record["experiment"] != recipe.SLUG
        or computed.record["stage"] != "compute"
    ):
        raise ValueError("requires an exp153 compute source")
    cfg = computed.record["execution"]["configuration"]
    if cfg["schema"] != recipe.SCHEMA or cfg["protocol"] not in (
        "matched-grid",
        "onset-only",
    ):
        raise ValueError("requires the matched 143-condition grid protocol")
    cfg = recipe.analysis_configuration(cfg)
    conditions = json.loads(computed.file("conditions.json").read_text())
    if conditions != recipe.conditions(cfg):
        raise ValueError("incomplete or reordered grid coverage")
    metrics = []
    for condition in conditions:
        time, state = read_trajectory(computed, condition, cfg)
        if cfg["protocol"] == "matched-grid":
            metrics.append(dict(**condition, **pair_metrics(time, state, cfg)))
        if (condition["difference_hz"], condition["coupling_us"]) == (
            cfg["example"]["difference_hz"],
            cfg["example"]["coupling_us"],
        ):
            onset, trace, rates = diagnostic(time, state, cfg)
            onset["label"] = condition["label"]
    arrays, cells = assemble_map(metrics, cfg) if metrics else ({}, [])
    values = arrays["concentration"][arrays["valid"]] if metrics else []
    numbers = dict(
        schema="exp153.analysis/v3"
        if cfg["protocol"] == "onset-only"
        else "exp153.analysis/v2",
        configuration=cfg,
        cells=cells,
        n_conditions=len(cells),
        n_valid=int(arrays["valid"].sum()) if metrics else 0,
        n_locked=int(arrays["locked"].sum()) if metrics else 0,
        max_plv=float(values.max()) if len(values) else None,
        onset=onset,
    )
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs=dict(compute=computed),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        write_json_atomic(run.export / "results.json", numbers)
        write_json_atomic(run.export / "onset-phase.json", trace)
        if metrics:
            np.savez_compressed(run.export / "metrics.npz", **arrays)
        np.savez_compressed(run.export / "onset-rates.npz", **rates)
        for item in ancestors.values():
            item.check_unchanged()
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
