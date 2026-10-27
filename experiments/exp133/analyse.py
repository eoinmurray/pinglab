"""Measure frequency detuning and finite-time locking in the reported protocols.

Run with --source identifying a completed compute run. Validate its retained
inputs and parameters, then save the map metrics or onset phase trace for present.py."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp133 import recipe
from experiments.exp133.custom import evidence, inputs, sender_receiver
from pingstore.library import write_json_atomic
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def rhythm(spikes, *, settling_ms=recipe.SETTLING_MS):
    """Detect volleys and interpolate continuous phase only between observed peaks."""
    rate = gaussian_filter1d(
        spikes.mean(axis=1) * 1000 / recipe.DT_MS, 1 / recipe.DT_MS
    )
    peaks, _ = find_peaks(
        rate, distance=round(15 / recipe.DT_MS), prominence=0.1 * rate.max()
    )
    time = np.arange(len(rate)) * recipe.DT_MS
    phase = (
        np.interp(
            time,
            peaks * recipe.DT_MS,
            2 * np.pi * np.arange(len(peaks)),
            left=np.nan,
            right=np.nan,
        )
        if len(peaks) >= 2
        else np.full(len(rate), np.nan)
    )
    selected = peaks[peaks * recipe.DT_MS >= settling_ms]
    intervals_ms = np.diff(selected) * recipe.DT_MS
    frequency = float(1000 / intervals_ms.mean()) if len(intervals_ms) else None
    cv = float(intervals_ms.std() / intervals_ms.mean()) if len(intervals_ms) else None
    valid = (
        len(selected) >= 20
        and frequency is not None
        and 30 <= frequency <= 80
        and cv is not None
        and cv <= 0.2
    )
    return phase, dict(
        frequency_hz=frequency, iei_cv=cv, volleys=len(selected), valid=valid
    )


def phase_metrics(time_ms, difference, *, valid_rhythm=True):
    """Descriptive finite-window criteria; concentration alone cannot establish locking."""
    drift = float(np.polyfit(time_ms / 1000, difference / (2 * np.pi), 1)[0])
    concentration = float(abs(np.mean(np.exp(1j * difference))))
    span = float(np.ptp(difference) / (2 * np.pi))
    locked = (
        valid_rhythm
        and abs(drift) < recipe.MAX_DRIFT_HZ
        and concentration > recipe.MIN_CONCENTRATION
        and span < recipe.MAX_PHASE_SPAN_CYCLES
    )
    return dict(
        valid=bool(valid_rhythm),
        drift_hz=drift,
        concentration=concentration,
        phase_span_cycles=span,
        locked=bool(locked),
    )


def measure_pair(a, b):
    phase_a, summary_a = rhythm(a)
    phase_b, summary_b = rhythm(b)
    time = np.arange(len(a)) * recipe.DT_MS
    selected = (
        np.isfinite(phase_a) & np.isfinite(phase_b) & (time >= recipe.SETTLING_MS)
    )
    valid_rhythm = summary_a["valid"] and summary_b["valid"]
    if selected.sum() < 100:
        metrics = dict(
            valid=False,
            drift_hz=None,
            concentration=None,
            phase_span_cycles=None,
            locked=False,
        )
    else:
        metrics = phase_metrics(
            time[selected], (phase_a - phase_b)[selected], valid_rhythm=valid_rhythm
        )
    return dict(**metrics, network_a=summary_a, network_b=summary_b), {
        "time_ms": time[selected],
        "phase_rad": (phase_a - phase_b)[selected],
    }


def measure(compute):
    steps = round((recipe.T_MS - recipe.COUPLING_ONSET_MS) / recipe.DT_MS)
    shape = (len(recipe.K_VALUES_US), len(recipe.INPUT_DIFFERENCES_HZ))
    arrays = {key: np.full(shape, np.nan) for key in ("concentration",)}
    arrays.update(valid=np.zeros(shape, dtype=bool), locked=np.zeros(shape, dtype=bool))
    summaries = []
    frequencies = np.full((2, shape[1]), np.nan)
    for row, k in enumerate(recipe.K_VALUES_US):
        recording = evidence.recording(compute, recipe.label(k), steps)
        for column, input_difference in enumerate(recipe.INPUT_DIFFERENCES_HZ):
            summary, _ = measure_pair(
                recording["A_E"][:, column], recording["B_E"][:, column]
            )
            summaries.append(
                dict(
                    coupling_us=float(k),
                    input_difference_hz=float(input_difference),
                    **summary,
                )
            )
            for key in arrays:
                value = summary[key]
                arrays[key][row, column] = np.nan if value is None else value
            if row == 0:
                for index, name in enumerate(("network_a", "network_b")):
                    value = summary[name]["frequency_hz"]
                    frequencies[index, column] = np.nan if value is None else value
    arrays["uncoupled_frequency_a_hz"] = frequencies[0]
    arrays["uncoupled_frequency_b_hz"] = frequencies[1]
    arrays["detuning_hz"] = frequencies[0] - frequencies[1]
    arrays["coupling_us"] = recipe.K_VALUES_US.copy()
    arrays["input_difference_hz"] = recipe.INPUT_DIFFERENCES_HZ.copy()
    # Intrinsic detuning requires two valid uncoupled rhythms, even if coupling regularizes a cell.
    baseline_valid = arrays["valid"][0].copy() & np.isfinite(arrays["detuning_hz"])
    arrays["valid"] &= baseline_valid[None, :]
    arrays["locked"] &= arrays["valid"]
    for index, summary in enumerate(summaries):
        row, column = divmod(index, shape[1])
        summary.update(
            valid=bool(arrays["valid"][row, column]),
            locked=bool(arrays["locked"][row, column]),
            uncoupled_detuning_hz=float(arrays["detuning_hz"][column])
            if baseline_valid[column]
            else None,
        )
    return arrays, summaries


def raster_frequency_windows(spikes):
    result = {}
    for name in ("S", "R"):
        result[name] = {}
        duration_ms = len(spikes[f"{name}_E"]) * recipe.DT_MS
        for label, start, end in (
            ("first", 0, 500),
            ("last", duration_ms - 500, duration_ms),
        ):
            segment = spikes[f"{name}_E"][
                round(start / recipe.DT_MS) : round(end / recipe.DT_MS), 0
            ]
            _, summary = rhythm(segment, settling_ms=0)
            frequency = summary["frequency_hz"]
            summary["valid"] = bool(
                frequency is not None
                and summary["volleys"] >= 5
                and 30 <= frequency <= 80
                and summary["iei_cv"] <= 0.2
            )
            result[name][label] = dict(start_ms=start, end_ms=end, **summary)
    return result


def analyse_sender_receiver(compute, *, run_id=None):
    cfg = sender_receiver.compute_contract(compute)
    spikes = sender_receiver.recording(compute)
    time = np.arange(len(spikes["S_E"])) * recipe.DT_MS
    phases = {name: rhythm(spikes[f"{name}_E"][:, 0])[0] for name in ("S", "R")}
    relative = (phases["S"] - phases["R"]) / (2 * np.pi)
    windows = {}
    for name, start, end in (
        ("uncoupled", sender_receiver.BASELINE_START_MS, sender_receiver.ONSET_MS),
        (
            "coupled",
            sender_receiver.ONSET_MS + recipe.SETTLING_MS,
            sender_receiver.DURATION_MS,
        ),
    ):
        selected = np.isfinite(relative) & (time >= start) & (time < end)
        if selected.sum() < 100:
            raise ValueError("insufficient common phase-valid samples")
        metrics = phase_metrics(time[selected], relative[selected] * 2 * np.pi)
        windows[name] = dict(
            start_ms=start,
            end_ms=end,
            **{
                key: metrics[key]
                for key in ("drift_hz", "concentration", "phase_span_cycles")
            },
        )
    baseline = {
        name: rhythm(
            spikes[f"{name}_E"][: round(sender_receiver.ONSET_MS / recipe.DT_MS), 0],
            settling_ms=sender_receiver.BASELINE_START_MS,
        )[1]
        for name in ("S", "R")
    }
    for summary in baseline.values():
        summary.pop("valid")
    result = dict(
        schema="exp133.sender-receiver-analysis/v1",
        configuration=cfg,
        baseline=baseline,
        raster_frequencies=raster_frequency_windows(spikes),
        uncoupled_detuning_hz=baseline["S"]["frequency_hz"]
        - baseline["R"]["frequency_hz"],
        windows=windows,
        interpretation="one finite sender–receiver trajectory; no seed robustness or asymptotic locking inferred",
    )
    with inputs.execution(
        REPO, "analyse", sources={"compute": compute}, run_id=run_id, configuration=cfg
    ) as run:
        np.savez_compressed(
            run.export / "relative-phase.npz", time_ms=time, phase_cycles=relative
        )
        write_json_atomic(run.export / "results.json", result)
    return run.run_id


def analyse(identity, *, run_id=None):
    compute = inputs.source(REPO, identity, "compute")
    if (
        compute.record["execution"]["configuration"].get("schema")
        == sender_receiver.SCHEMA
    ):
        return analyse_sender_receiver(compute, run_id=run_id)
    cfg = evidence.compute_contract(compute)
    with inputs.execution(
        REPO, "analyse", sources={"compute": compute}, run_id=run_id
    ) as run:
        arrays, summaries = measure(compute)
        np.savez_compressed(run.export / "metrics.npz", **arrays)
        write_json_atomic(
            run.export / "results.json",
            dict(
                schema="exp133.analysis/v1",
                configuration=cfg,
                cells=summaries,
                n_conditions=len(summaries),
                n_valid=int(arrays["valid"].sum()),
                n_locked=int(arrays["locked"].sum()),
                detuning_range_hz=[
                    float(np.nanmin(arrays["detuning_hz"])),
                    float(np.nanmax(arrays["detuning_hz"])),
                ],
                analysis_window_ms=[
                    recipe.COUPLING_ONSET_MS + recipe.SETTLING_MS,
                    recipe.T_MS,
                ],
                interpretation="finite-time threshold map from one paired input realization; no inferred continuous boundary, significance test or across-seed reliability",
            ),
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
