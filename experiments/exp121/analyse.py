"""Measure stable-correct decision times and inhibitory-volley intervals."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import numpy as np
from experiments.exp121 import recipe as R
from pingstore.stages import source_run, stage_run
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def decisions(source, export, study="input"):
    prefix = "" if study == "input" else f"{study}--"
    with np.load(source.file(prefix + "recording.npz")) as saved:
        controls = saved["input_rates_hz" if study == "input" else "control_values"]
        spikes, labels, indices = (
            saved[k] for k in ("output_spikes", "labels", "indices")
        )
    times = (
        np.arange(round(R.CONFIG["duration_ms"] / R.CONFIG["dt_ms"]) + 1)
        * R.CONFIG["dt_ms"]
    )
    stable = np.full((len(controls), len(labels)), np.nan)
    ties = np.zeros(len(controls), dtype=int)
    curves = []
    for k in range(len(controls)):
        counts = np.concatenate(
            [
                np.zeros((len(labels), 1, 10), dtype=np.int64),
                np.cumsum(spikes[k], axis=1, dtype=np.int64),
            ],
            axis=1,
        )
        unique = (counts == counts.max(axis=2, keepdims=True)).sum(axis=2) == 1
        correct = (counts.argmax(axis=2) == labels[:, None]) & unique
        ties[k] = (~unique[:, -1]).sum()
        for j, row in enumerate(correct):
            if row[-1]:
                stable[k, j] = times[np.flatnonzero(~row)[-1] + 1]
        curves.append((stable[k, :, None] <= times[None, :]).mean(axis=0) * 100)
    curves = np.asarray(curves)
    baseline_ok = np.isfinite(stable[0])
    summary = []
    for k, scale in enumerate(controls):
        good = np.isfinite(stable[k])
        common = good & baseline_ok
        delta = stable[k, common] - stable[0, common]
        summary.append(
            dict(
                control_value=float(scale),
                stable_correct_by_400ms=int(good.sum()),
                unresolved_ties_at_400ms=int(ties[k]),
                failed_or_tied=int((~good).sum()),
                common_successes=int(common.sum()),
                rescued=int((good & ~baseline_ok).sum()),
                lost=int((~good & baseline_ok).sum()),
                median_paired_delta_ms=float(np.median(delta)) if len(delta) else None,
                earlier=int((delta < 0).sum()),
                later=int((delta > 0).sum()),
                same_time=int((delta == 0).sum()),
            )
        )
    (export / (prefix + "decisions.json")).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    bins = np.arange(
        0,
        R.CONFIG["duration_ms"] + R.CONFIG["histogram_bin_ms"],
        R.CONFIG["histogram_bin_ms"],
    )
    histograms = np.array(
        [np.histogram(row[np.isfinite(row)], bins=bins)[0] for row in stable]
    )
    np.savez_compressed(
        export / (prefix + "measurements.npz"),
        histogram_edges_ms=bins,
        histogram_counts=histograms,
        test_indices=indices,
        labels=labels,
        control_values=controls,
        time_ms=times,
        stable_correct_ms=stable,
        stable_correct_percent=curves,
    )


def rhythm(source, export, study="input"):
    prefix = "" if study == "input" else f"{study}--"
    with np.load(source.file(prefix + "recording.npz")) as saved:
        counts = saved["i_population_spike_counts"]
        rates = saved["input_rates_hz" if study == "input" else "control_values"]
    cfg = R.CONFIG
    conditions, trials, steps = counts.shape
    steps_per_bin = round(cfg["bin_ms"] / cfg["dt_ms"])
    population_rate = (
        counts.reshape(conditions, trials, steps // steps_per_bin, steps_per_bin).sum(
            axis=-1
        )
        * (1000 / cfg["bin_ms"])
        / cfg["n_i"]
    )
    smooth = gaussian_filter1d(
        population_rate, cfg["smoothing_sigma_ms"] / cfg["bin_ms"], axis=-1
    )
    arrays = dict(control_values=rates)
    summary = {}
    for threshold in cfg["sensitivity_thresholds_hz"]:
        frequency = np.full((conditions, trials), np.nan)
        cv = np.full((conditions, trials), np.nan)
        for k in range(conditions):
            for j in range(trials):
                peaks, _ = find_peaks(
                    smooth[k, j],
                    height=threshold,
                    prominence=threshold,
                    distance=cfg["minimum_peak_distance_ms"] / cfg["bin_ms"],
                )
                intervals = np.diff(peaks) * cfg["bin_ms"]
                if len(peaks) >= cfg["minimum_peaks"]:
                    frequency[k, j] = 1000 / intervals.mean()
                    cv[k, j] = intervals.std(ddof=cfg["cv_ddof"]) / intervals.mean()
        arrays[f"frequency_{threshold}"] = frequency
        arrays[f"cv_{threshold}"] = cv
        for name, values in (("frequency", frequency), ("cv", cv)):
            quartiles = np.full((3, conditions), np.nan)
            for k, row in enumerate(values):
                valid = row[np.isfinite(row)]
                if len(valid):
                    quartiles[:, k] = np.percentile(valid, [25, 50, 75])
            arrays[f"{name}_quartiles_{threshold}"] = quartiles
        summary[str(threshold)] = [
            dict(
                control_value=float(rate),
                valid_trials=int(np.isfinite(frequency[k]).sum()),
                median_frequency_hz=(
                    float(arrays[f"frequency_quartiles_{threshold}"][1, k])
                    if np.isfinite(frequency[k]).any()
                    else None
                ),
                median_cv=(
                    float(arrays[f"cv_quartiles_{threshold}"][1, k])
                    if np.isfinite(cv[k]).any()
                    else None
                ),
            )
            for k, rate in enumerate(rates)
        ]
    np.savez_compressed(export / (prefix + "rhythm-measurements.npz"), **arrays)
    (export / (prefix + "rhythm-results.json")).write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary["25"], indent=2))


def analyse(identity, internal_identity=None):
    source = source_run(
        ROOT / ".pingstore", identity, stage="compute", experiment="exp121"
    )
    inputs = {"compute": source}
    study_sources = {}
    for number, internal_id in enumerate(internal_identity or (), start=1):
        internal = source_run(
            ROOT / ".pingstore", internal_id, stage="compute", experiment="exp121"
        )
        if internal.record["inputs"]["input_study"] != source.reference:
            raise ValueError("Internal study does not reuse the named input study")
        inputs[f"internal_{number}"] = internal
        for study in internal.record["execution"]["configuration"]["studies"]:
            if study not in R.INTERNAL_STUDIES or study in study_sources:
                raise ValueError(f"Unknown or duplicate study: {study}")
            study_sources[study] = internal
    studies = tuple(study for study in R.INTERNAL_STUDIES if study in study_sources)
    with stage_run(
        ROOT,
        "exp121",
        "analyse",
        inputs=inputs,
        configuration=dict(
            R.CONFIG,
            studies=("input", *studies),
        ),
    ) as run:
        decisions(source, run.export)
        rhythm(source, run.export)
        for study in studies:
            decisions(study_sources[study], run.export, study)
            rhythm(study_sources[study], run.export, study)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--internal-source", action="append")
    args = parser.parse_args()
    analyse(args.source, args.internal_source)
