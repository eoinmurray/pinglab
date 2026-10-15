"""Trial-level sustained decisions and I-volley frequency from a shared sweep."""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from experiments.exp123 import recipe
from experiments.exp123.analyse import volley_completion_times, volley_times

FIELDS = ("image_index", "seed", "tau_gaba_ms", "frequency_hz", "decision_ms", "completed_volleys")


def measure_image(image_index, directory, cfg, measurement):
    unit = Path(directory)
    with np.load(unit / "input.npz", allow_pickle=False) as data:
        label = int(data["label"])
    ns = len(cfg["seeds"])
    dt = cfg["dt_ms"]
    rows = []
    for tau in cfg["tau_gaba_ms"]:
        with np.load(unit / f"tau{tau:g}--spikes.npz", allow_pickle=False) as data:
            inhibitory, output = data["spk_i"], data["spk_out"]
        counts = np.concatenate((np.zeros((1, ns, 10), dtype=np.int32),
                                 np.cumsum(output, axis=0, dtype=np.int32)))
        rivals = np.delete(counts, label, axis=-1).max(axis=-1)
        unique_correct = counts[:, :, label] > rivals
        decisions = []
        peaks_all, valid_all, completions_all = [], [], []
        for b in range(ns):
            # The zero-count initial state always ties, so failures is nonempty.
            last_failure = np.flatnonzero(~unique_correct[:, b])[-1]
            first_stable = int(last_failure + 1)
            decisions.append(first_stable * dt if first_stable < len(counts) else np.nan)
            peaks, valid, _ = volley_times(inhibitory[:, b], dt, measurement)
            completions = volley_completion_times(inhibitory[:, b], peaks, dt, measurement)
            peaks_all.append(peaks)
            valid_all.append(valid)
            completions_all.append(completions)
        for b, seed in enumerate(cfg["seeds"]):
            peaks, completions = peaks_all[b], completions_all[b]
            frequency = 1000 / np.diff(peaks).mean() if valid_all[b] and len(peaks) > 1 else np.nan
            decision = decisions[b]
            mapped = np.nan
            if (valid_all[b] and np.isfinite(decision) and len(completions)
                    and decision <= completions[-1]):
                count = int(np.searchsorted(completions, decision, side="left") + 1)
                coverage = sum(v and len(c) >= count for v, c in zip(valid_all, completions_all)) / ns
                if coverage >= .8:
                    mapped = count
            rows.append((image_index, seed, tau, frequency, decision, mapped))
    return np.asarray(rows, dtype=np.float64)


def summarise(arrays):
    frequency = arrays["frequency_hz"]
    has_frequency = np.isfinite(frequency)
    stable = has_frequency & np.isfinite(arrays["decision_ms"])
    mapped = has_frequency & np.isfinite(arrays["completed_volleys"])
    if not stable.any() or not mapped.any():
        raise ValueError("Pooled decision figure requires eligible time and volley observations")
    fmax = max(5, np.ceil(frequency[has_frequency].max() / 5) * 5)
    frequency_edges = np.arange(0, fmax + 5, 5)
    centres = (frequency_edges[:-1] + frequency_edges[1:]) / 2
    membership = np.digitize(frequency, frequency_edges) - 1
    medians = {}
    for mask, key in [(stable, "decision_ms"), (mapped, "completed_volleys")]:
        values = []
        for j in range(len(centres)):
            observations = arrays[key][mask & (membership == j)]
            values.append(float(np.median(observations)) if len(observations) else None)
        medians[key] = values
    time_edges = np.arange(0, 205, 5)
    max_volley = int(np.nanmax(arrays["completed_volleys"][mapped]))
    volley_edges = np.arange(.5, max_volley + 1.5, 1)
    time_histogram = np.histogram2d(arrays["decision_ms"][stable], frequency[stable],
                                    bins=(time_edges, frequency_edges))[0]
    volley_histogram = np.histogram2d(arrays["completed_volleys"][mapped], frequency[mapped],
                                      bins=(volley_edges, frequency_edges))[0]
    return dict(
        schema="exp123.pooled.decisions/v1",
        presentations=len(frequency), missing_frequency=int((~has_frequency).sum()),
        stable_with_frequency=int(stable.sum()), mapped_with_frequency=int(mapped.sum()),
        total_stable=int(np.isfinite(arrays["decision_ms"]).sum()),
        frequency_bin_edges_hz=frequency_edges.tolist(), frequency_bin_centres_hz=centres.tolist(),
        time_bin_edges_ms=time_edges.tolist(), volley_bin_edges=volley_edges.tolist(),
        time_counts=time_histogram.astype(int).tolist(), volley_counts=volley_histogram.astype(int).tolist(),
        median_elapsed_time_ms=medians["decision_ms"], median_completed_volleys=medians["completed_volleys"],
        frequency_bin_total=np.histogram(frequency[has_frequency], frequency_edges)[0].tolist(),
        frequency_bin_stable=np.histogram(frequency[stable], frequency_edges)[0].tolist(),
        frequency_bin_mapped=np.histogram(frequency[mapped], frequency_edges)[0].tolist(),
        measurement=dict(
            decision="earliest completed update at which the true class is the unique cumulative-output-count winner and remains so through 200 ms; ties interrupt stability",
            frequency="1000 / mean successive accepted I-volley intervals in ms, per trial over the full presentation; no spectral-peak estimator",
            completed_volleys="first trough completion at or after stable decision, one-based; accepted I sequence, decision within bounded completions, >=80% draw coverage at that count within image/decay group",
            aggregation="pool eligible trials over all images, draws and decay settings; black curves are medians within 5 Hz frequency bins; these are paired repeats, not independent training replicates",
            exclusions="time panel excludes undefined frequency and no sustained correct decision; volley panel also excludes missing or coverage-ineligible completion mappings",
        ),
    )


def validate_illustrations(arrays, report):
    lookup = {(int(i), int(seed), tau): j for j, (i, seed, tau) in enumerate(zip(
        arrays["image_index"], arrays["seed"], arrays["tau_gaba_ms"]))}
    for row in report["rows"]:
        j = lookup[(row["image_index"], row["seed"], row["tau_gaba_ms"])]
        for key, old in [("decision_ms", "sustained_decision_ms"),
                         ("completed_volleys", "sustained_decision_completed_volley_count")]:
            expected = row[old]
            matches = np.isnan(arrays[key][j]) if expected is None else np.isclose(
                arrays[key][j], expected, rtol=0, atol=1e-10)
            if not matches:
                raise ValueError(f"Pooled measurement disagrees with illustrative trial {j}: {key}")


def measure_pooled_decisions(source, run, cfg, workers=4):
    measurement = recipe.measurement()
    completed = {}
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = {executor.submit(measure_image, index,
                    str(source.unit(recipe.recording_unit(index, recipe.RATE_HZ, "benchmark"))),
                    cfg, measurement): index for index in cfg["image_indices"]}
        for future in as_completed(pending):
            completed[pending[future]] = future.result()
            if len(completed) % 50 == 0:
                print(f"Pooled decisions measured for {len(completed)}/{len(pending)} images", flush=True)
    rows = np.concatenate([completed[index] for index in cfg["image_indices"]])
    arrays = {key: rows[:, j] for j, key in enumerate(FIELDS)}
    illustrations = json.loads((run.export / "results.json").read_text())
    validate_illustrations(arrays, illustrations)
    np.savez_compressed(run.export / "pooled_decisions.npz", **arrays)
    report = summarise(arrays)
    report["design"] = {key: cfg[key] for key in (
        "image_indices", "seeds", "tau_gaba_ms", "dt_ms", "duration_ms", "input_rate_hz")}
    (run.export / "pooled_decisions.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
