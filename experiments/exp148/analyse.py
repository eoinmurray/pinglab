"""Measure the three-window trajectory from one explicit compute run.

Run analyse.py --source COMPUTE_ID. Validate immutable weights, then save window rates and local hold measurements.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse

import numpy as np
from experiments.exp148 import recipe
from experiments.exp148.recipe import windows
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from scipy.signal import find_peaks
from snnlab import analysis


def analyse(*, source, mean_field=None, run_id=None):
    computed = source_run(
        REPO / ".pingstore", source, experiment="exp148", stage="compute"
    )
    cfg = computed.record["execution"]["configuration"]
    audit = load_json(computed.file("simulation/verification.json"))
    if not audit["native_modulation"] or not audit["stored_weights_unchanged"]:
        raise ValueError("simulation failed immutable-weight verification")
    with np.load(computed.file("simulation/events.npz")) as events:
        numbers = measure(events, cfg)
        numbers["hold_dynamics"] = measure_hold_dynamics(events, cfg)
    if cfg.get("record_state"):
        with np.load(computed.file("simulation/state-traces.npz")) as traces:
            rows = []
            for interval in numbers["hold_dynamics"]["intervals"]:
                left, right = interval["start_ms"], interval["end_ms"]
                mask = (traces["time_ms"] >= left) & (traces["time_ms"] < right)
                row = dict(start_ms=left, end_ms=right)
                for key in traces.files:
                    if key not in {"time_ms", "input_spikes"}:
                        values = traces[key][mask]
                        row[key] = dict(
                            mean=float(values.mean()), maximum=float(values.max())
                        )
                row["measured_source_rate_hz"] = float(
                    traces["input_spikes"][mask].sum()
                    / cfg["n_external"]
                    / ((right - left) / 1000)
                )
                rows.append(row)
            numbers["hold_dynamics"]["state_intervals"] = rows
    inputs = dict(compute=computed)
    if mean_field is not None:
        model = source_run(
            REPO / ".pingstore", mean_field, experiment="exp148", stage="compute"
        )
        if model.record["inputs"].get("spiking") != computed.reference:
            raise ValueError("mean-field model uses different spiking evidence")
        inputs["mean_field"] = model
        saved = load_json(model.file("mean-field.json"))
        numbers["mean_field"] = saved
        with np.load(model.file("mean-field.npz")) as trajectory:
            model_windows = []
            for name, start, end in windows(cfg):
                selected = (trajectory["time_ms"] >= start) & (
                    trajectory["time_ms"] < end
                )
                values = trajectory["state"][selected, :2]
                row = dict(
                    window=name,
                    mean_rate_hz=values.mean(axis=0).tolist(),
                    peak_to_peak_rate_hz=np.ptp(values, axis=0).tolist(),
                )
                model_windows.append(row)
            numbers["mean_field"]["protocol_windows"] = model_windows
    with stage_run(
        REPO,
        "exp148",
        "analyse",
        inputs=inputs,
        configuration=cfg,
        run_id=run_id,
    ) as run:
        if "hold_dynamics" in numbers:
            run.record["execution"]["hold_analysis_configuration"] = numbers[
                "hold_dynamics"
            ]["method"]
        write_json_atomic(run.export / "results.json", numbers)
    print(run.run_id, flush=True)
    return run.run_id


def measure_hold_dynamics(events, cfg):
    """Resolve changes within the hold instead of assuming one stationary rhythm.

    Peaks are detected independently in consecutive 100 ms intervals, with local
    prominence. The resulting intervals are descriptive and lose boundary peaks.
    """
    method = recipe.hold_analysis_configuration()
    start, end = windows(cfg)[-1][1:]
    width = method["bin_ms"]
    edges = np.arange(start, end + width / 2, width)
    counts = {p: np.histogram(events[p + "_time_ms"], edges)[0] for p in ("E", "I")}
    rows = []
    for left in np.arange(start, end, method["interval_ms"]):
        right = min(left + method["interval_ms"], end)
        mask = (edges[:-1] >= left) & (edges[:-1] < right)
        row = dict(start_ms=float(left), end_ms=float(right))
        for p in ("E", "I"):
            values = counts[p][mask]
            peaks, _ = find_peaks(
                values,
                distance=round(method["minimum_separation_ms"] / width),
                prominence=float(values.max()) * method["prominence_fraction"],
            )
            times = edges[:-1][mask][peaks] + width / 2
            row[p] = dict(
                mean_rate_hz=float(
                    values.sum() / cfg["n_" + p.lower()] / ((right - left) / 1000)
                ),
                maximum_bin_fraction=float(values.max() / cfg["n_" + p.lower()]),
                peak_times_ms=times.tolist(),
                intervals_ms=np.diff(times).tolist(),
                median_interval_ms=float(np.median(np.diff(times)))
                if len(times) > 1
                else None,
            )
        rows.append(row)
    return dict(
        method=method,
        intervals=rows,
        time_ms=(edges[:-1] + width / 2).tolist(),
        counts={p: values.tolist() for p, values in counts.items()},
        interpretation="Local peaks measure pulse spacing and concentration; not a mechanism test",
    )


def window_spectra(events, cfg):
    """PSD of per-neuron population rates in each of the three windows.

    Use the recorded population bin width and one full-window Hann periodogram.
    Explicit half-open selection prevents neighbouring windows entering the PSD.
    """
    width = cfg["analysis"]["population_bin_ms"]
    spectra = {}
    for name, start, end in windows(cfg):
        bins = round((end - start) / width)
        if not np.isclose(bins * width, end - start):
            raise ValueError("spectral bins must exactly cover the window")
        edges = np.linspace(start, end, bins + 1)
        spectra[name] = {}
        for population in ("E", "I"):
            times = np.asarray(events[population + "_time_ms"])
            counts, _ = np.histogram(times[(times >= start) & (times < end)], edges)
            rate = counts / cfg["n_" + population.lower()] / (width / 1000)
            spectra[name][population] = analysis.power_spectrum(
                rate,
                width,
                method="fft",
                detrend="constant",
                window="hann",
                scaling="density",
            )
    return spectra


def validate_events(events, cfg):
    for population in ("E", "I"):
        times = np.asarray(events[population + "_time_ms"])
        cells = np.asarray(events[population + "_cell"])
        if (
            times.ndim != 1
            or cells.shape != times.shape
            or cells.dtype.kind not in "iu"
        ):
            raise ValueError("events require matching time and integer-cell vectors")
        if (
            not np.isfinite(times).all()
            or np.any(times < 0)
            or np.any(times >= cfg["duration_ms"])
        ):
            raise ValueError("spike time outside the simulation")
        if np.any(np.diff(times) < 0) or not np.allclose(
            times / cfg["dt_ms"], np.rint(times / cfg["dt_ms"]), rtol=0, atol=1e-8
        ):
            raise ValueError(
                "spike times must be ordered and lie on the simulation grid"
            )
        if np.any(cells < 0) or np.any(cells >= cfg["n_" + population.lower()]):
            raise ValueError("spike cell outside its population")
        if len(times) > 1:
            same_time = times[1:] == times[:-1]
            if np.any(same_time & (cells[1:] <= cells[:-1])):
                raise ValueError(
                    "simultaneous events must contain distinct sorted cells"
                )


def measure(events, cfg):
    validate_events(events, cfg)
    rows = []
    for label, start, end in windows(cfg):
        row = dict(window=label, start_ms=start, end_ms=end)
        for population in ("E", "I"):
            times = events[population + "_time_ms"]
            count = int(((times >= start) & (times < end)).sum())
            row[population + "_spikes"] = count
            row[population + "_rate_hz"] = (
                count / cfg["n_" + population.lower()] / ((end - start) / 1000)
            )
        rows.append(row)
    start, end = windows(cfg)[-1][1:]
    analysis = cfg["analysis"]
    width = analysis["population_bin_ms"]
    edges = np.arange(start, end + width / 2, width)
    cycles = {}
    for population in ("E", "I"):
        counts, _ = np.histogram(events[population + "_time_ms"], bins=edges)
        peaks, _ = find_peaks(
            counts,
            distance=max(1, round(analysis["peak_separation_ms"] / width)),
            prominence=counts.max() * analysis["peak_prominence_fraction"],
        )
        times = edges[peaks] + width / 2
        period = float(np.median(np.diff(times))) if len(times) > 1 else None
        cycles[population] = dict(
            peak_times_ms=times.tolist(),
            peak_count=len(times),
            median_cycle_ms=period,
            frequency_hz=1000 / period if period is not None else None,
        )
    return dict(
        windows=rows,
        hold_cycles=cycles,
        interpretation="Population oscillations measured; AI classification and PING mechanism untested",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--mean-field",
        help="Explicit mean-field compute run tied to the same spiking source",
    )
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
