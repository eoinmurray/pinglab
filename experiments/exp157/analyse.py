"""Validate the native evidence needed for Figure 1.

Check the E-only topology, exact nine-window controls, autonomous rhythms,
measured ramp-end rates and numerical refinement. Peak measurements also supply
Figure 1's gamma and phase estimators; no extra result figures are analysed.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import numpy as np
from experiments.exp157 import compute, recipe
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks


def read_trajectory(source, label, cfg):
    with gzip.open(source.file(label + ".json.gz"), "rt") as handle:
        saved = json.load(handle)
    time = np.asarray(saved["time_ms"], dtype=float)
    state = {
        name: np.asarray(values, dtype=float) for name, values in saved["state"].items()
    }
    model = load_json(source.file("model.json"))
    duration = (
        cfg["frequency_calibration"]["duration_ms"]
        if label == "autonomous"
        else cfg["duration_ms"]
    )
    expected = np.arange(
        0, duration + cfg["sample_interval_ms"] / 2, cfg["sample_interval_ms"]
    )
    if saved["schema"] != "snnlab.meanfield.trajectory/v1" or not np.array_equal(
        time, expected
    ):
        raise ValueError("invalid trajectory schema or observation grid")
    if set(state) != set(model["state_order"]) or any(
        values.shape != time.shape
        or not np.isfinite(values).all()
        or values.min() < -1e-7
        for values in state.values()
    ):
        raise ValueError("invalid, nonfinite or negative native states")
    metadata = saved["integration"]
    scale = 10 if label == "refined" else 1
    expected_solver = dict(
        method=cfg["solver"]["method"],
        rtol=cfg["solver"]["rtol"] / scale,
        atol=cfg["solver"]["atol"] / scale,
        max_step_ms=cfg["solver"]["max_step_ms"] / (2 if label == "refined" else 1),
    )
    if any(metadata[k] != v for k, v in expected_solver.items()):
        raise ValueError("native solver differs from configuration")
    if metadata["initial"] != dict.fromkeys(state, 0.0) or any(
        values[0] != 0 for values in state.values()
    ):
        raise ValueError("all rates and conductances must start at zero")
    expected_controls = {
        name: dict(
            kind="PiecewiseLinear",
            times_ms=list(control.times),
            values=list(control.values),
        )
        for name, control in recipe.drive_controls(cfg).items()
    }
    if label == "autonomous":
        expected_controls = {}
    if metadata["controls"] != expected_controls:
        raise ValueError("recorded input controls differ from nine-window protocol")
    defaults = {name: row["value"] for name, row in model["parameters"].items()}
    expected_parameters = dict(defaults)
    if label in ("uncoupled", "autonomous"):
        expected_parameters.update(
            {
                name: 0.0
                for name in defaults
                if name.startswith(("A_E_to_C_", "B_E_to_C_"))
                and name.endswith(".gain")
            }
        )
    if label == "autonomous":
        expected_parameters.update(
            {
                f"X_{c}.rate": cfg[
                    {"A": "a_drive_hz", "B": "b_hold_drive_hz", "C": "c_drive_hz"}[c]
                ]
                for c in recipe.CIRCUITS
            }
        )
    if metadata["parameters"] != expected_parameters:
        raise ValueError("fixed mean-field model parameters changed")
    couplings = {
        name.removesuffix(".coupling"): value
        for name, value in defaults.items()
        if name.endswith(".coupling")
    }
    if couplings != recipe.incoming_weights(cfg):
        raise ValueError("native coupling topology or values differ from E-only recipe")
    for circuit in recipe.CIRCUITS:
        for kind in ("E", "I"):
            if (
                defaults[f"{circuit}_{kind}.tau_rate"]
                != cfg["tau_rate_ms"][circuit][kind]
            ):
                raise ValueError(
                    "native rate relaxation differs from resolved configuration"
                )
    return time, state, metadata


def interpeak_frequency(time, rate, settings):
    """Return reciprocal cycle intervals at midpoints, without rhythm gating.

    Local prominence rejects small shoulders and numerical ripples. A local
    amplitude floor splits inactive stretches; regularity, gamma-band and
    persistence gates do not suppress measured cycle frequencies.
    """
    dt = float(np.diff(time).mean())
    width = int(round(settings["frequency_window_ms"] / dt)) | 1
    span = maximum_filter1d(rate, width) - minimum_filter1d(rate, width)
    floor = np.maximum(
        settings["min_amplitude_hz"],
        settings["min_relative_amplitude"] * uniform_filter1d(rate, width),
    )
    active = span >= floor
    indices, _ = find_peaks(
        rate,
        distance=max(1, int(1000 / settings["gamma_band_hz"][1] / dt * 0.8)),
        prominence=np.maximum(0.05, 0.1 * span),
    )
    indices = indices[active[indices]]
    peaks = time[indices]
    frequency = 1000 / np.diff(peaks)
    for k, (left, right) in enumerate(zip(indices[:-1], indices[1:], strict=True)):
        if not active[left : right + 1].all():
            frequency[k] = np.nan
    return (peaks[:-1] + peaks[1:]) / 2, frequency


def peak_phase(time, rate, settings):
    amplitude = float(np.ptp(rate))
    minimum = max(
        settings["min_amplitude_hz"],
        settings["min_relative_amplitude"] * float(np.mean(rate)),
    )
    distance = max(
        1, int(1000 / settings["gamma_band_hz"][1] / np.diff(time).mean() * 0.8)
    )
    indices, _ = find_peaks(
        rate, distance=distance, prominence=max(0.05, amplitude * 0.1)
    )
    peaks = time[indices]
    intervals = np.diff(peaks)
    frequency = float(1000 / np.mean(intervals)) if len(intervals) else None
    cv = float(np.std(intervals) / np.mean(intervals)) if len(intervals) else None
    # A full-window amplitude alone can mislabel a decaying transient as gamma.
    midpoint = len(rate) // 2
    early, late = float(np.ptp(rate[:midpoint])), float(np.ptp(rate[midpoint:]))
    sustained = late / early if early > 0 else 0.0
    gamma = bool(
        amplitude >= minimum
        and len(peaks) >= settings["min_peaks"]
        and frequency is not None
        and settings["gamma_band_hz"][0] <= frequency <= settings["gamma_band_hz"][1]
        and cv <= settings["max_interval_cv"]
        and sustained >= settings["min_late_early_ratio"]
    )
    metrics = dict(
        mean_hz=float(np.mean(rate)),
        min_hz=float(rate.min()),
        max_hz=float(rate.max()),
        amplitude_hz=amplitude,
        frequency_hz=frequency,
        interval_cv=cv,
        peaks=len(peaks),
        sustained_ratio=sustained,
        gamma=gamma,
    )
    phase = (
        np.interp(time, peaks, np.arange(len(peaks)), left=np.nan, right=np.nan)
        if amplitude >= minimum and len(peaks) >= 2
        else np.full(time.shape, np.nan)
    )
    return metrics, phase, peaks


def window_rhythms(time, state, cfg):
    rows = []
    for window in recipe.windows(cfg):
        selected = (time >= window["start_ms"] + cfg["analysis"]["hold_discard_ms"]) & (
            time < window["stop_ms"]
        )
        rates = {
            name.removesuffix(".rate"): peak_phase(
                time[selected], values[selected], cfg["analysis"]
            )[0]
            for name, values in state.items()
            if name.endswith(".rate")
        }
        rows.append(dict(**window, rates=rates))
    return rows


def numerical_check(time, state, refined_state, cfg):
    rows = window_rhythms(time, state, cfg)
    refined = window_rhythms(time, refined_state, cfg)
    changes = [
        abs(
            row["rates"][name]["frequency_hz"]
            - comparison["rates"][name]["frequency_hz"]
        )
        for row, comparison in zip(rows, refined, strict=True)
        for name, result in row["rates"].items()
        if result["gamma"] and comparison["rates"][name]["gamma"]
    ]
    maximum = max(changes, default=0)
    before_hold = time <= cfg["target"]["ramp_end_ms"]
    hold = (time > cfg["target"]["ramp_end_ms"]) & (time < cfg["duration_ms"])

    def normalized_error(mask):
        return max(
            float(np.max(np.abs(values[mask] - refined_state[name][mask])))
            / max(1.0, float(np.max(np.abs(values[mask]))))
            for name, values in state.items()
        )

    relative_error = normalized_error(np.ones(time.shape, dtype=bool))
    ramp_error = normalized_error(before_hold)
    hold_mean_error = max(
        abs(float(np.mean(values[hold])) - float(np.mean(refined_state[name][hold])))
        / max(1.0, abs(float(np.mean(values[hold]))))
        for name, values in state.items()
    )
    same = all(
        row["rates"][name]["gamma"] == comparison["rates"][name]["gamma"]
        for row, comparison in zip(rows, refined, strict=True)
        for name in row["rates"]
    )
    # The added hold can amplify tiny trajectory differences in irregular C
    # activity. Preserve its waveform discrepancy explicitly; validate its mean
    # rates and rhythm gate rather than hiding sensitivity behind a larger bound.
    if maximum > 0.1 or ramp_error > 0.01 or hold_mean_error > 0.01 or not same:
        raise ValueError(
            "numerical refinement changes the protocol or held-rate evidence"
        )
    return dict(
        max_frequency_change_hz=maximum,
        max_normalized_state_error=relative_error,
        before_hold_normalized_state_error=ramp_error,
        hold_max_mean_rate_error=hold_mean_error,
        hold_waveform_converged=normalized_error(hold) <= 0.01,
        same_rhythm_gates=same,
    )


def measure_autonomous(time, state, cfg):
    settings = cfg["frequency_calibration"]
    selected = (time >= settings["measure_start_ms"]) & (time < settings["duration_ms"])
    metrics = {}
    for circuit in recipe.CIRCUITS:
        for kind in ("E", "I"):
            name = f"{circuit}_{kind}"
            metrics[name], _, _ = peak_phase(
                time[selected], state[name + ".rate"][selected], cfg["analysis"]
            )
    return metrics


def analyse(source, *, run_id=None):
    upstream = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="compute"
    )
    cfg = upstream.record["execution"]["configuration"]
    if cfg.get("protocol") != "nine-500ms-windows-e-only":
        raise ValueError("requires the nine-window E-only protocol")
    time, state, _ = read_trajectory(upstream, "protocol", cfg)
    _, refined_state, _ = read_trajectory(upstream, "refined", cfg)
    # The uncoupled trajectory supplies Figure 1's dashed C frequency reference.
    read_trajectory(upstream, "uncoupled", cfg)
    from types import SimpleNamespace

    target = compute.terminal_rates(SimpleNamespace(time_ms=time, state=state), cfg)
    target["relative_error"] = abs(
        target["b_to_a_ratio"] / cfg["target"]["b_to_a_e_rate"] - 1
    )
    held = (time >= cfg["duration_ms"] - cfg["target"]["terminal_average_ms"]) & (
        time < cfg["duration_ms"]
    )
    hold = dict(
        a_e_mean_hz=float(np.mean(state["A_E.rate"][held])),
        b_e_mean_hz=float(np.mean(state["B_E.rate"][held])),
    )
    hold["b_to_a_ratio"] = hold["b_e_mean_hz"] / hold["a_e_mean_hz"]
    verification = numerical_check(time, state, refined_state, cfg)
    autonomous_time, autonomous_state, _ = read_trajectory(upstream, "autonomous", cfg)
    isolated = measure_autonomous(autonomous_time, autonomous_state, cfg)
    if cfg.get("schema") == "exp157.recipe/v6":
        for circuit in recipe.CIRCUITS:
            measured = isolated[circuit + "_E"]
            error = abs(
                measured["frequency_hz"]
                - cfg["frequency_calibration"]["targets_hz"][circuit]
            )
            if (
                not measured["gamma"]
                or error > cfg["frequency_calibration"]["tolerance_hz"]
            ):
                raise ValueError(f"{circuit}: autonomous frequency calibration failed")
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs=dict(compute=upstream),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                configuration=cfg,
                windows=window_rhythms(time, state, cfg),
                target=target,
                verification=verification,
                autonomous=isolated,
                hold=hold,
            ),
        )
        upstream.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
