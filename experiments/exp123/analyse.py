"""Measure one explicit shared-sweep compute run without upstream execution.

Run analyse.py --source COMPUTE_RUN with optional --workers and --reuse-analysis.
The custom estimators retain trajectories, dynamics, costs and decision coverage
in a new Pingstore analysis run; inputs are validated by identity and digest.
"""

import sys
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import argparse
import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from experiments.exp123 import recipe
from pingstore.library import source_run, stage_run
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, find_peaks
from scipy.special import softmax
from snnlab import analysis


def analyse(identity, workers=1, reuse_identity=None):
    if workers < 1:
        raise ValueError("workers must be positive")
    source = source_run(
        REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    if source.record["execution"]["configuration"].get("schema") != "exp123.compute/v3":
        raise ValueError("expected the uniform shared-sweep compute protocol")
    reuse = (
        source_run(
            REPO / ".pingstore", reuse_identity, stage="analyse", experiment=recipe.SLUG
        )
        if reuse_identity
        else None
    )
    return analyse_shared(source, workers=workers, reuse=reuse)


def _bootstrap_accuracy_curves(
    *,
    labels,
    measurement,
    ni,
    nt,
    ns,
    grids,
    correct,
    cfg,
):
    """Summarise the retained decisions with the registered bootstrap draws."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(measurement["bootstrap_seed"])
    weights = np.zeros((1000, ni), dtype=float)
    for digit in range(10):
        positions = np.flatnonzero(labels == digit)
        draws = rng.choice(positions, (1000, len(positions)), replace=True)
        for b in range(1000):
            weights[b] += np.bincount(draws[b], minlength=ni)
    curves = {}
    arrays = dict(
        image_indices=cfg["image_indices"],
        labels=labels,
        seeds=cfg["seeds"],
        tau_gaba_ms=cfg["tau_gaba_ms"],
    )
    for key, grid in grids.items():
        values = correct[key]
        count = np.isfinite(values).sum(axis=2)
        per_image = np.divide(
            np.nansum(values, axis=2),
            count,
            out=np.full((ni, nt, len(grid)), np.nan),
            where=count > 0,
        )
        means, lows, highs = [], [], []
        for ti in range(nt):
            available = np.isfinite(per_image[:, ti])
            denominator = available.sum(axis=0)
            mean = np.divide(
                np.nansum(per_image[:, ti], axis=0),
                denominator,
                out=np.full(len(grid), np.nan),
                where=denominator > 0,
            )
            numer = np.einsum(
                "bi,ig->bg",
                weights,
                np.nan_to_num(per_image[:, ti]),
                optimize=False,
            )
            denom = np.einsum(
                "bi,ig->bg", weights, available.astype(float), optimize=False
            )
            bootstrap = np.divide(
                numer, denom, out=np.full_like(numer, np.nan), where=denom > 0
            )
            low, high = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
            for g in np.flatnonzero(denominator):
                low[g], high[g] = np.nanquantile(bootstrap[:, g], [0.025, 0.975])
            means.append(mean)
            lows.append(low)
            highs.append(high)
        coverage = np.isfinite(values[:, 0]).sum(axis=(0, 1)) / (ni * ns)
        curves[key] = dict(
            grid=grid.tolist(),
            accuracy=[
                [float(v) if np.isfinite(v) else None for v in row] for row in means
            ],
            ci_low=[
                [float(v) if np.isfinite(v) else None for v in row] for row in lows
            ],
            ci_high=[
                [float(v) if np.isfinite(v) else None for v in row] for row in highs
            ],
            trial_coverage=coverage.tolist(),
            images_available=np.isfinite(per_image[:, 0]).sum(axis=0).tolist(),
        )
        arrays[f"{key}_grid"] = grid
        arrays[f"{key}_correct"] = values
        arrays[f"{key}_per_image_accuracy"] = per_image
    return curves, arrays


def measure_accuracy(source, run, cfg):
    shared = cfg.get("shared_sweep", False)
    measurement = recipe.measurement()
    measurement.update(
        schema="exp123.accuracy.measurement/v1",
        accuracy_prediction="instantaneous cumulative-count argmax; lowest class index wins ties",
        time_grid_ms=list(range(201)),
        cycle_grid=np.arange(0, 8.01, 0.25).tolist(),
        input_spike_grid=list(range(0, 1501, 25)),
        coordinate_sampling="last completed output update at query time; input counts sampled at first budget crossing",
        coverage="within each image/seed, include a coordinate only if available in all three settings; no extrapolation",
        aggregation="mean eligible draws within image, then equal mean over eligible images",
        bootstrap="1000 paired stratified image resamples; 50 images per digit with replacement; percentile 95% intervals",
        bootstrap_seed=12345,
    )
    grids = {
        "time": np.array(measurement["time_grid_ms"], dtype=float),
        "cycles": np.array(measurement["cycle_grid"]),
        "input": np.array(measurement["input_spike_grid"]),
    }
    if shared:
        grids["network"] = np.arange(0, 30001, 100, dtype=float)
        measurement.update(
            network_spike_grid=grids["network"].tolist(),
            network_cost="E + I + output spikes; excludes input",
            network_sampling="first completed update reaching each budget, including simultaneous-spike overshoot; only trials reaching it in every decay setting",
            latency_aggregation="mean over all eligible image/draw pairs at each network budget",
        )
    ni, nt, ns = len(cfg["image_indices"]), len(cfg["tau_gaba_ms"]), len(cfg["seeds"])
    correct = {
        key: np.full((ni, nt, ns, len(grid)), np.nan, dtype=np.float32)
        for key, grid in grids.items()
    }
    labels = []
    time_cost = np.zeros((nt, len(grids["time"])), dtype=float)
    network_latency_sum = np.zeros((nt, len(grids.get("network", []))), dtype=float)
    network_latency_count = np.zeros_like(network_latency_sum)
    valid_rows = []
    dt = cfg["dt_ms"]
    for j, image_index in enumerate(cfg["image_indices"]):
        unit = source.unit(
            recipe.recording_unit(image_index, recipe.RATE_HZ, "benchmark")
        )
        with np.load(unit / "input.npz", allow_pickle=False) as data:
            label = int(data["label"])
            input_counts = np.concatenate(
                (
                    np.zeros((1, ns), dtype=np.int32),
                    np.cumsum(data["spikes"].sum(axis=2), axis=0, dtype=np.int32),
                )
            )
        labels.append(label)
        outputs, peaks_all, valid_all, network_counts = [], [], [], []
        for ti, tau in enumerate(cfg["tau_gaba_ms"]):
            with np.load(unit / f"tau{tau:g}--spikes.npz", allow_pickle=False) as data:
                output_spikes, inhibitory_spikes = data["spk_out"], data["spk_i"]
                cumulative = np.concatenate(
                    (
                        np.zeros((1, ns, 10), dtype=np.int32),
                        np.cumsum(output_spikes, axis=0, dtype=np.int32),
                    )
                )
                outputs.append(cumulative.argmax(axis=-1) == label)
                if shared:
                    counts = np.concatenate(
                        (
                            np.zeros((1, ns), dtype=np.int64),
                            np.cumsum(
                                data["spk_e"].sum(axis=2)
                                + inhibitory_spikes.sum(axis=2)
                                + output_spikes.sum(axis=2),
                                axis=0,
                                dtype=np.int64,
                            ),
                        )
                    )
                    network_counts.append(counts)
                    time_cost[ti] += counts[
                        np.rint(grids["time"] / dt).astype(int)
                    ].mean(axis=1)
                peaks, accepted = [], []
                for b in range(ns):
                    p, valid, participation = volley_times(
                        inhibitory_spikes[:, b], dt, measurement
                    )
                    peaks.append(p)
                    accepted.append(valid)
                    valid_rows.append(
                        dict(
                            image_index=image_index,
                            seed=cfg["seeds"][b],
                            tau_gaba_ms=tau,
                            valid=valid,
                            volleys=len(p),
                            i_participation_pct=participation,
                        )
                    )
                peaks_all.append(peaks)
                valid_all.append(accepted)
        for b in range(ns):
            if shared:
                reaches_network = grids["network"] <= min(
                    c[-1, b] for c in network_counts
                )
            reaches_input = grids["input"] <= input_counts[-1, b]
            input_steps = np.searchsorted(
                input_counts[:, b], grids["input"][reaches_input], side="left"
            )
            cycle_ok = all(v[b] for v in valid_all)
            end = min(len(p[b]) - 1 for p in peaks_all) if cycle_ok else -1
            reaches_cycle = grids["cycles"] <= end
            for ti in range(nt):
                correct["time"][j, ti, b] = outputs[ti][
                    np.rint(grids["time"] / dt).astype(int), b
                ]
                correct["input"][j, ti, b, reaches_input] = outputs[ti][input_steps, b]
                if shared:
                    network_steps = np.searchsorted(
                        network_counts[ti][:, b],
                        grids["network"][reaches_network],
                        side="left",
                    )
                    correct["network"][j, ti, b, reaches_network] = outputs[ti][
                        network_steps, b
                    ]
                    network_latency_sum[ti, reaches_network] += network_steps * dt
                    network_latency_count[ti, reaches_network] += 1
                if cycle_ok:
                    peaks = peaks_all[ti][b]
                    query = np.interp(
                        grids["cycles"][reaches_cycle], np.arange(len(peaks)), peaks
                    )
                    steps = np.floor(query / dt + 1e-8).astype(int)
                    correct["cycles"][j, ti, b, reaches_cycle] = outputs[ti][steps, b]
        if (j + 1) % 100 == 0:
            print(f"Accuracy measured for {j + 1}/{ni} images", flush=True)
    curves, arrays = _bootstrap_accuracy_curves(
        labels=labels,
        measurement=measurement,
        ni=ni,
        nt=nt,
        ns=ns,
        grids=grids,
        correct=correct,
        cfg=cfg,
    )
    np.savez_compressed(run.export / "accuracy.npz", **arrays)
    report = dict(
        schema="exp123.accuracy.analysis/v1",
        design={
            key: cfg[key]
            for key in (
                "protocol",
                "image_indices",
                "image_selection",
                "image_selection_seed",
                "partition",
                "seeds",
                "generator_seed_rule",
                "tau_gaba_ms",
                "dt_ms",
                "duration_ms",
                "burn_in_ms",
                "input_rate_hz",
                "biological_defaults",
            )
        },
        measurement={
            key: value for key, value in measurement.items() if key != "script_sha256"
        },
        curves=curves,
        volley_diagnostics=valid_rows,
        presentations=ni * nt * ns,
        final_time_accuracy=[c[-1] for c in curves["time"]["accuracy"]],
    )
    if shared:
        report["mean_network_spikes_by_time"] = (time_cost / ni).tolist()
        latency = np.divide(
            network_latency_sum,
            network_latency_count,
            out=np.full_like(network_latency_sum, np.nan),
            where=network_latency_count > 0,
        )
        report["mean_network_budget_latency_ms"] = [
            [float(v) if np.isfinite(v) else None for v in row] for row in latency
        ]
    (run.export / "accuracy.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def dynamics(e, i, cfg):
    dt = cfg["dt_ms"]
    counts = (
        analysis.binned_spike_counts(
            e, dt, bin_ms=cfg["population_bin_ms"], rounding="exact", trailing="error"
        )["counts"]
        .sum(axis=1)
        .astype(float)
    )
    smooth = gaussian_filter1d(counts, sigma=2.0 / cfg["population_bin_ms"])
    e_spectrum = analysis.power_spectrum(
        e.sum(axis=1),
        dt,
        method="welch",
        center=True,
        nperseg=len(e),
        window="hann",
        scaling="density",
        detrend=False,
    )
    freq, psd = e_spectrum["frequencies_hz"], e_spectrum["power"]
    i_counts = (
        analysis.binned_spike_counts(
            i, dt, bin_ms=cfg["population_bin_ms"], rounding="exact", trailing="error"
        )["counts"]
        .sum(axis=1)
        .astype(float)
    )
    psd_i = analysis.power_spectrum(
        i.sum(axis=1),
        dt,
        method="welch",
        center=True,
        nperseg=len(i),
        window="hann",
        scaling="density",
        detrend=False,
    )["power"]
    peak_result = analysis.spectral_peak(
        freq,
        psd,
        cfg["frequency_search_hz"],
        interpolation="parabolic",
        interpolation_boundary="spectrum",
        clamp_band=False,
    )
    mask = (freq >= cfg["frequency_search_hz"][0]) & (
        freq <= cfg["frequency_search_hz"][1]
    )
    idx = np.flatnonzero(mask)[np.argmax(psd[mask])]
    candidate = float(freq[idx])
    demean = counts - counts.mean()
    ac = correlate(demean, demean, mode="full", method="fft")[len(counts) - 1 :]
    ac = ac / ac[0] if ac[0] > 0 else np.full_like(ac, np.nan)
    i_demean = i_counts - i_counts.mean()
    ac_i = correlate(i_demean, i_demean, mode="full", method="fft")[len(i_counts) - 1 :]
    ac_i = ac_i / ac_i[0] if ac_i[0] > 0 else np.full_like(ac_i, np.nan)
    lags = np.arange(len(ac)) * cfg["population_bin_ms"]
    period = 1000 / candidate
    inhibitory_counts = i.sum(axis=1).astype(float)
    inhibitory_smooth = gaussian_filter1d(
        inhibitory_counts, sigma=cfg["volley_smoothing_ms"] / dt
    )
    volleys, _ = find_peaks(
        inhibitory_smooth,
        distance=round(cfg["volley_minimum_spacing_ms"] / dt),
        prominence=max(
            cfg["volley_minimum_prominence_spikes"],
            cfg["volley_relative_prominence"] * inhibitory_smooth.max(),
        ),
    )
    # Midpoints bracket each interior inhibitory volley and exclude partial edge cycles.
    boundary = analysis.cycle_boundaries(volleys, len(e), policy="midpoints")["edges"][
        1:-1
    ]
    if len(boundary) < 2:
        boundary = np.empty(0, dtype=np.int64)
    e_cycle_counts = analysis.cycle_spike_counts(e, boundary)["counts"]
    i_cycle_counts = analysis.cycle_spike_counts(i, boundary)["counts"]
    cycles = [(int(a), int(b)) for a, b in pairwise(boundary)]
    intervals = np.diff(volleys) * dt
    cv = (
        float(intervals.std() / intervals.mean())
        if len(intervals) > 1 and intervals.mean() > 0
        else None
    )
    participation_i = (100 * (i_cycle_counts > 0).mean(axis=1)).tolist()
    valid = bool(
        len(cycles) >= cfg["minimum_cycles"]
        and participation_i
        and np.mean(participation_i) >= cfg["minimum_mean_i_participation_pct"]
    )
    cycle_frequency = 1000 / intervals.mean() if valid else None
    period = 1000 / cycle_frequency if valid else 1000 / candidate
    peak_mask = (lags >= 0.75 * period) & (lags <= 1.25 * period)
    trough_mask = (lags >= 0.25 * period) & (lags <= 0.75 * period)
    peak = (
        float(np.max(ac[peak_mask]))
        if np.isfinite(ac).all() and peak_mask.any() and trough_mask.any()
        else None
    )
    trough = float(np.min(ac[trough_mask])) if peak is not None else None
    prominence = peak - trough if peak is not None else None
    pe = (100 * (e_cycle_counts > 0).mean(axis=1)).tolist() if valid else []
    pi = participation_i if valid else []
    costs = (
        (e_cycle_counts.sum(axis=1) + i_cycle_counts.sum(axis=1)).tolist()
        if valid
        else []
    )
    result = dict(
        candidate_frequency_hz=candidate if counts.any() else None,
        psd_frequency_hz=peak_result["frequency_hz"],
        volley_frequency_hz=float(cycle_frequency) if valid else None,
        valid_cycles=valid,
        ac_peak=peak,
        rhythmicity_ac_prominence=prominence,
        cycle_interval_cv=cv,
        accepted_cycles=len(cycles) if valid else 0,
        participation_e_pct=float(np.mean(pe)) if pe else None,
        participation_i_pct=float(np.mean(pi)) if pi else None,
        core_spikes_per_cycle=float(np.mean(costs)) if costs else None,
        participation_e_cycles_pct=pe,
        participation_i_cycles_pct=pi,
        core_spikes_cycles=costs,
        cycle_boundaries_steps=cycles if valid else [],
    )
    result["volley_intervals_ms"] = intervals.tolist()
    rhythm = analysis.rhythmicity_metrics(e, dt, max_lag_ms=100.0, bin_ms=1.0)
    rhythm_lags, rhythm_ac = rhythm["ac_lags"], rhythm["ac"]
    iei_lags, iei_counts = rhythm["iei_lags"], rhythm["iei_counts"]
    helper_scalars = analysis.rhythmicity_scalars(
        rhythm_lags, rhythm_ac, iei_lags, iei_counts, bin_ms=1.0
    )
    result.update(
        {
            f"rhythmicity_{name}": float(value)
            if value is not None and np.isfinite(value)
            else None
            for name, value in helper_scalars.items()
        }
    )
    return result, dict(
        counts=counts,
        smooth=smooth,
        frequency_hz=freq,
        psd=psd,
        psd_i=psd_i,
        counts_i=i_counts,
        ac_lags_ms=lags[:251],
        autocorrelation=ac[:251],
        autocorrelation_i=ac_i[:251],
        boundaries_steps=boundary,
        inhibitory_smooth=inhibitory_smooth,
        volley_peaks_steps=volleys,
        rhythmicity_ac_lags_ms=rhythm_lags,
        rhythmicity_autocorrelogram=rhythm_ac,
        rhythmicity_iei_lags_ms=iei_lags,
        rhythmicity_iei_counts=iei_counts,
    )


FIELDS = (
    "image_index",
    "seed",
    "tau_gaba_ms",
    "frequency_hz",
    "decision_ms",
    "completed_volleys",
)


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
        counts = np.concatenate(
            (
                np.zeros((1, ns, 10), dtype=np.int32),
                np.cumsum(output, axis=0, dtype=np.int32),
            )
        )
        rivals = np.delete(counts, label, axis=-1).max(axis=-1)
        unique_correct = counts[:, :, label] > rivals
        decisions = []
        peaks_all, valid_all, completions_all = [], [], []
        for b in range(ns):
            # The zero-count initial state always ties, so failures is nonempty.
            last_failure = np.flatnonzero(~unique_correct[:, b])[-1]
            first_stable = int(last_failure + 1)
            decisions.append(
                first_stable * dt if first_stable < len(counts) else np.nan
            )
            peaks, valid, _ = volley_times(inhibitory[:, b], dt, measurement)
            completions = volley_completion_times(
                inhibitory[:, b], peaks, dt, measurement
            )
            peaks_all.append(peaks)
            valid_all.append(valid)
            completions_all.append(completions)
        for b, seed in enumerate(cfg["seeds"]):
            peaks, completions = peaks_all[b], completions_all[b]
            frequency = (
                1000 / np.diff(peaks).mean()
                if valid_all[b] and len(peaks) > 1
                else np.nan
            )
            decision = decisions[b]
            mapped = np.nan
            if (
                valid_all[b]
                and np.isfinite(decision)
                and len(completions)
                and decision <= completions[-1]
            ):
                count = int(np.searchsorted(completions, decision, side="left") + 1)
                coverage = (
                    sum(
                        v and len(c) >= count
                        for v, c in zip(valid_all, completions_all, strict=True)
                    )
                    / ns
                )
                if coverage >= 0.8:
                    mapped = count
            rows.append((image_index, seed, tau, frequency, decision, mapped))
    return np.asarray(rows, dtype=np.float64)


def summarise(arrays):
    frequency = arrays["frequency_hz"]
    has_frequency = np.isfinite(frequency)
    stable = has_frequency & np.isfinite(arrays["decision_ms"])
    mapped = has_frequency & np.isfinite(arrays["completed_volleys"])
    if not stable.any() or not mapped.any():
        raise ValueError(
            "Pooled decision figure requires eligible time and volley observations"
        )
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
    volley_edges = np.arange(0.5, max_volley + 1.5, 1)
    time_histogram = np.histogram2d(
        arrays["decision_ms"][stable],
        frequency[stable],
        bins=(time_edges, frequency_edges),
    )[0]
    volley_histogram = np.histogram2d(
        arrays["completed_volleys"][mapped],
        frequency[mapped],
        bins=(volley_edges, frequency_edges),
    )[0]
    return dict(
        schema="exp123.pooled.decisions/v1",
        presentations=len(frequency),
        missing_frequency=int((~has_frequency).sum()),
        stable_with_frequency=int(stable.sum()),
        mapped_with_frequency=int(mapped.sum()),
        total_stable=int(np.isfinite(arrays["decision_ms"]).sum()),
        frequency_bin_edges_hz=frequency_edges.tolist(),
        frequency_bin_centres_hz=centres.tolist(),
        time_bin_edges_ms=time_edges.tolist(),
        volley_bin_edges=volley_edges.tolist(),
        time_counts=time_histogram.astype(int).tolist(),
        volley_counts=volley_histogram.astype(int).tolist(),
        median_elapsed_time_ms=medians["decision_ms"],
        median_completed_volleys=medians["completed_volleys"],
        frequency_bin_total=np.histogram(frequency[has_frequency], frequency_edges)[
            0
        ].tolist(),
        frequency_bin_stable=np.histogram(frequency[stable], frequency_edges)[
            0
        ].tolist(),
        frequency_bin_mapped=np.histogram(frequency[mapped], frequency_edges)[
            0
        ].tolist(),
        measurement=dict(
            decision="earliest completed update at which the true class is the unique cumulative-output-count winner and remains so through 200 ms; ties interrupt stability",
            frequency="1000 / mean successive accepted I-volley intervals in ms, per trial over the full presentation; no spectral-peak estimator",
            completed_volleys="first trough completion at or after stable decision, one-based; accepted I sequence, decision within bounded completions, >=80% draw coverage at that count within image/decay group",
            aggregation="pool eligible trials over all images, draws and decay settings; black curves are medians within 5 Hz frequency bins; these are paired repeats, not independent training replicates",
            exclusions="time panel excludes undefined frequency and no sustained correct decision; volley panel also excludes missing or coverage-ineligible completion mappings",
        ),
    )


def validate_illustrations(arrays, report):
    lookup = {
        (int(i), int(seed), tau): j
        for j, (i, seed, tau) in enumerate(
            zip(
                arrays["image_index"],
                arrays["seed"],
                arrays["tau_gaba_ms"],
                strict=True,
            )
        )
    }
    for row in report["rows"]:
        j = lookup[(row["image_index"], row["seed"], row["tau_gaba_ms"])]
        for key, old in [
            ("decision_ms", "sustained_decision_ms"),
            ("completed_volleys", "sustained_decision_completed_volley_count"),
        ]:
            expected = row[old]
            matches = (
                np.isnan(arrays[key][j])
                if expected is None
                else np.isclose(arrays[key][j], expected, rtol=0, atol=1e-10)
            )
            if not matches:
                raise ValueError(
                    f"Pooled measurement disagrees with illustrative trial {j}: {key}"
                )


def measure_pooled_decisions(source, run, cfg, workers=4):
    measurement = recipe.measurement()
    completed = {}
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        pending = {
            executor.submit(
                measure_image,
                index,
                str(
                    source.unit(
                        recipe.recording_unit(index, recipe.RATE_HZ, "benchmark")
                    )
                ),
                cfg,
                measurement,
            ): index
            for index in cfg["image_indices"]
        }
        for future in as_completed(pending):
            completed[pending[future]] = future.result()
            if len(completed) % 50 == 0:
                print(
                    f"Pooled decisions measured for {len(completed)}/{len(pending)} images",
                    flush=True,
                )
    rows = np.concatenate([completed[index] for index in cfg["image_indices"]])
    arrays = {key: rows[:, j] for j, key in enumerate(FIELDS)}
    illustrations = json.loads((run.export / "results.json").read_text())
    validate_illustrations(arrays, illustrations)
    np.savez_compressed(run.export / "pooled_decisions.npz", **arrays)
    report = summarise(arrays)
    report["design"] = {
        key: cfg[key]
        for key in (
            "image_indices",
            "seeds",
            "tau_gaba_ms",
            "dt_ms",
            "duration_ms",
            "input_rate_hz",
        )
    }
    (run.export / "pooled_decisions.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def condition_dynamics(condition, cfg, settings, units):
    """Measure one decay setting without changing image or draw aggregation."""
    tau = condition["tau_gaba_ms"]
    values = {
        key: []
        for key in (
            "participation_e_pct",
            "e_spikes_per_second",
            "rhythmicity_contrast",
        )
    }
    power_sum = None
    accepted = 0
    # Equal draw counts make this pooled mean equal to mean-within-image,
    # followed by an equal mean over images. Undefined participation is
    # handled separately by first averaging accepted draws within image.
    for image_index in cfg["image_indices"]:
        unit = Path(units[image_index])
        participation = []
        with np.load(unit / f"tau{tau:g}--spikes.npz", allow_pickle=False) as data:
            excitatory, inhibitory = data["spk_e"], data["spk_i"]
            for b in range(len(cfg["seeds"])):
                e, i = excitatory[:, b], inhibitory[:, b]
                metrics, diagnostic = dynamics(e, i, settings)
                accepted += int(metrics["valid_cycles"])
                if metrics["participation_e_pct"] is not None:
                    participation.append(metrics["participation_e_pct"])
                values["e_spikes_per_second"].append(
                    float(e.sum() / (cfg["duration_ms"] / 1000))
                )
                contrast = metrics["rhythmicity_contrast"]
                if contrast is not None:
                    values["rhythmicity_contrast"].append(contrast)
                power = diagnostic["psd"]
                if power_sum is None:
                    power_sum = np.zeros_like(power)
                power_sum += power
                frequency = diagnostic["frequency_hz"]
        if participation:
            values["participation_e_pct"].append(float(np.mean(participation)))
    peak = analysis.spectral_peak(
        frequency,
        power_sum,
        settings["frequency_search_hz"],
        interpolation="parabolic",
        interpolation_boundary="spectrum",
        clamp_band=False,
    )["frequency_hz"]
    summary = dict(
        condition,
        psd_frequency_hz=peak,
        accepted_draws=accepted,
        total_draws=len(cfg["image_indices"]) * len(cfg["seeds"]),
        participation_images=len(values["participation_e_pct"]),
    )
    for key, samples in values.items():
        summary[key] = float(np.mean(samples)) if samples else None
        summary[f"{key}_n"] = len(samples)
    return summary, dict(condition, frequency_hz=peak)


def measure_dynamics(source, run, cfg, workers=1):
    settings = recipe.dynamics_configuration(cfg)
    conditions = [
        dict(
            condition_id=f"tau{j:02d}",
            sweep="tau",
            condition_index=j,
            axis_value=tau,
            tau_gaba_ms=tau,
        )
        for j, tau in enumerate(cfg["tau_gaba_ms"])
    ]
    units = {
        index: str(
            source.unit(recipe.recording_unit(index, recipe.RATE_HZ, "benchmark"))
        )
        for index in cfg["image_indices"]
    }
    measured = {}

    def retain(result):
        summary, psd_summary = result
        measured[summary["condition_index"]] = result
        (run.export / f"dynamics-{summary['condition_id']}.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        print(
            f"Dynamics measured at GABA {summary['tau_gaba_ms']:g} ms "
            f"({len(measured)}/{len(conditions)} settings)",
            flush=True,
        )

    if workers == 1:
        for condition in conditions:
            retain(condition_dynamics(condition, cfg, settings, units))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            futures = [
                executor.submit(condition_dynamics, condition, cfg, settings, units)
                for condition in conditions
            ]
            for future in as_completed(futures):
                retain(future.result())
    ordered = [measured[j] for j in range(len(conditions))]
    summaries = [result[0] for result in ordered]
    psd_summaries = [result[1] for result in ordered]
    report = dict(
        schema="exp123.overview.analysis/v1",
        design=dict(
            cfg,
            conditions=conditions,
            aggregation="peak of pooled PSD; E spike rate averages all equally sampled draws/images; participation averages accepted draws within image then eligible images; contrast averages finite draws",
            frequency_search_hz=settings["frequency_search_hz"],
        ),
        summaries=summaries,
        psd_summaries=psd_summaries,
        rows=summaries,
    )
    (run.export / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def measure_overview_rasters(source, run, definition):
    """Export sparse plotting events for the three existing illustrative rasters."""
    cfg = definition["sweep"]
    index = definition["illustrative_image_indices"][0]
    unit = source.unit(recipe.recording_unit(index, recipe.RATE_HZ, "benchmark"))
    events = dict(
        tau_gaba_ms=np.array(recipe.TAUS_MS),
        dt_ms=cfg["dt_ms"],
        image_index=index,
        seed=cfg["seeds"][0],
    )
    for j, tau in enumerate(recipe.TAUS_MS):
        with np.load(unit / f"tau{tau:g}--spikes.npz", allow_pickle=False) as spikes:
            for key in ("spk_e", "spk_i"):
                step, neuron = np.nonzero(spikes[key][:, 0])
                events[f"tau{j}--{key}--step"] = step
                events[f"tau{j}--{key}--neuron"] = neuron
    np.savez_compressed(run.export / "overview_rasters.npz", **events)


def analyse_shared(source, workers=1, reuse=None):
    definition = source.record["execution"]["configuration"]
    cfg = dict(definition["sweep"], shared_sweep=True)
    inputs = {"compute": source}
    if reuse is not None:
        if (
            reuse.record["execution"]["configuration"].get("schema")
            != "exp123.shared.measurement/v1"
            or reuse.record["inputs"].get("compute") != source.reference
        ):
            raise ValueError(
                "Reused shared measurements must reference this exact compute payload"
            )
        inputs["previous_analysis"] = reuse
    with stage_run(
        recipe.REPO,
        recipe.SLUG,
        "analyse",
        inputs=inputs,
        configuration=dict(
            schema="exp123.shared.measurement/v1",
            sweep=cfg,
            illustrative_image_indices=definition["illustrative_image_indices"],
            dynamics_workers=workers,
            reused_measurements="trajectory, accuracy and dynamics exports from explicit previous analysis"
            if reuse is not None
            else None,
            measures=[
                "trajectories",
                "matched_accuracy",
                "network_budget_latency",
                "dynamics",
                "deadline_cost",
                "pooled_sustained_decisions",
                "target_accuracy_frontier",
                "overview_raster_events",
            ],
            target_accuracy_pct=recipe.TARGET_ACCURACY_PCT,
        ),
    ) as run:
        if reuse is None:
            examples = dict(cfg, image_indices=definition["illustrative_image_indices"])
            measure_trajectories(source, run, examples)
            accuracy = measure_accuracy(source, run, cfg)
            measure_dynamics(source, run, cfg, workers=workers)
        else:
            shutil.copytree(reuse.directory / "export", run.export, dirs_exist_ok=True)
            accuracy = json.loads(reuse.file("accuracy.json").read_text())

        if reuse is None or not (run.export / "pooled_decisions.npz").exists():
            measure_pooled_decisions(source, run, cfg, workers=workers)
        rows = []
        for ti, tau in enumerate(cfg["tau_gaba_ms"]):
            for j, time in enumerate(accuracy["curves"]["time"]["grid"]):
                rows.append(
                    dict(
                        input_rate_hz=cfg["input_rate_hz"],
                        tau_gaba_ms=tau,
                        deadline_ms=time,
                        accuracy=accuracy["curves"]["time"]["accuracy"][ti][j],
                        mean_network_spikes=accuracy["mean_network_spikes_by_time"][ti][
                            j
                        ],
                    )
                )
        tradeoffs = dict(
            schema="exp123.shared.tradeoffs/v1",
            design=dict(
                tau_gaba_ms=cfg["tau_gaba_ms"],
                input_rates_hz=[cfg["input_rate_hz"]],
                deadlines_ms=[20, 40, 80, 120, 200],
            ),
            rows=rows,
            measurement="all 500 images and ten paired encodings; no trial selection at elapsed-time coordinates; mean cumulative E+I+output spikes excluding input",
        )
        (run.export / "tradeoffs.json").write_text(
            json.dumps(tradeoffs, indent=2, allow_nan=False) + "\n"
        )

        with np.load(run.export / "pooled_decisions.npz", allow_pickle=False) as trials:
            target = measure_target_accuracy(
                tradeoffs, trials, recipe.TARGET_ACCURACY_PCT
            )
        (run.export / "target_accuracy.json").write_text(
            json.dumps(target, indent=2, allow_nan=False) + "\n"
        )
        measure_overview_rasters(source, run, definition)
        return run.run_id


def measure_target_accuracy(tradeoffs, trials, target_pct):
    """Select empirical first crossings, then filter the cost/time frontier."""
    settings = []
    for tau in tradeoffs["design"]["tau_gaba_ms"]:
        frequencies = trials["frequency_hz"][trials["tau_gaba_ms"] == tau]
        accepted = frequencies[np.isfinite(frequencies)]
        if not len(accepted):
            raise ValueError(f"No accepted I-volley frequency at decay {tau:g}")
        settings.append(
            dict(
                tau_gaba_ms=tau,
                mean_frequency_hz=float(accepted.mean()),
                accepted_frequency_trials=len(accepted),
                total_trials=len(frequencies),
            )
        )
    points = []
    for setting in settings:
        reached = [
            r
            for r in tradeoffs["rows"]
            if r["tau_gaba_ms"] == setting["tau_gaba_ms"]
            and r["accuracy"] * 100 >= target_pct
        ]
        if reached:
            points.append(
                dict(
                    min(reached, key=lambda r: r["deadline_ms"]),
                    **{
                        key: value
                        for key, value in setting.items()
                        if key != "tau_gaba_ms"
                    },
                )
            )
    for a in points:
        a["on_frontier"] = not any(
            b["mean_network_spikes"] <= a["mean_network_spikes"]
            and b["deadline_ms"] <= a["deadline_ms"]
            and (
                b["mean_network_spikes"] < a["mean_network_spikes"]
                or b["deadline_ms"] < a["deadline_ms"]
            )
            for b in points
        )
    return dict(
        schema="exp123.target.accuracy/v1",
        target_accuracy_pct=target_pct,
        frequency_by_setting=settings,
        points=points,
        frontier=sorted(
            [p for p in points if p["on_frontier"]],
            key=lambda p: p["mean_network_spikes"],
        ),
        unreachable_tau_gaba_ms=[
            s["tau_gaba_ms"]
            for s in settings
            if not any(p["tau_gaba_ms"] == s["tau_gaba_ms"] for p in points)
        ],
        measurement=dict(
            crossing="first measured 1 ms deadline with population accuracy >= target; no interpolation; accuracy need not stay above target",
            cost="mean cumulative E+I+output spikes excluding input at the common crossing deadline; all equally sampled images and draws, including incorrect trials",
            frontier="no other first-crossing setting has both no greater mean spikes and no greater elapsed time, with at least one strict improvement; only frontier settings displayed",
            frequency="mean accepted per-trial I-volley interval frequency over full 200 ms; missing frequencies excluded from colour summary only",
            limitations="one frozen network; target and frontier evaluated on the same test sample; no individual stopping rule, physical energy estimate or uncertainty intervals",
        ),
    )


def volley_times(spikes, dt, cfg):
    smoothed = gaussian_filter1d(
        spikes.sum(axis=1).astype(float), cfg["volley_smoothing_ms"] / dt
    )
    peaks, _ = find_peaks(
        smoothed,
        distance=round(cfg["volley_spacing_ms"] / dt),
        prominence=max(
            cfg["volley_minimum_prominence"],
            cfg["volley_relative_prominence"] * smoothed.max(),
        ),
    )
    edges = analysis.cycle_boundaries(peaks, len(spikes), policy="midpoints")["edges"][
        1:-1
    ]
    if len(edges) < 2:
        edges = np.empty(0, dtype=np.int64)
    counts = analysis.cycle_spike_counts(spikes, edges)["counts"]
    participation = float(100 * (counts > 0).mean()) if counts.size else None
    valid = bool(
        len(counts) >= 1 and participation >= cfg["minimum_i_participation_pct"]
    )
    return (peaks + 1) * dt, valid, participation


def volley_completion_times(spikes, peaks_ms, dt, cfg):
    """Define each completed volley by the first trough before the next peak."""
    smoothed = gaussian_filter1d(
        spikes.sum(axis=1).astype(float), cfg["volley_smoothing_ms"] / dt
    )
    peaks = np.rint(peaks_ms / dt).astype(int) - 1
    troughs = [
        left + 1 + int(np.argmin(smoothed[left + 1 : right]))
        for left, right in pairwise(peaks)
    ]
    return (np.asarray(troughs, dtype=int) + 1) * dt


def _record_trajectory_trials(
    *,
    cfg,
    unit,
    measurement,
    dt,
    label,
    all_counts,
    all_true,
    all_max,
    all_pred,
    phase_valid,
    diagnostic,
    times,
    rows,
    image_index,
    all_peaks,
    all_completions,
):
    """Accumulate per-seed scores, volley timings and spectral diagnostics."""
    for ti, tau in enumerate(cfg["tau_gaba_ms"]):
        with np.load(unit / f"tau{tau:g}--spikes.npz", allow_pickle=False) as data:
            i, out, e = [data[key].copy() for key in ("spk_i", "spk_out", "spk_e")]
        cumulative = np.concatenate(
            (
                np.zeros((1, out.shape[1], out.shape[2]), dtype=np.int32),
                np.cumsum(out, axis=0, dtype=np.int32),
            ),
            axis=0,
        )
        scores = softmax(cumulative / measurement["softmax_temperature"], axis=-1)
        all_counts.append(cumulative.transpose(1, 0, 2))
        all_true.append(scores[:, :, label].T)
        all_max.append(scores.max(axis=-1).T)
        all_pred.append(cumulative.argmax(axis=-1).T.astype(np.uint8))
        trial_peaks = []
        trial_completions = []
        powers = []
        for b, seed in enumerate(cfg["seeds"]):
            peaks, valid, participation = volley_times(i[:, b], dt, measurement)
            trial_peaks.append(peaks)
            completions = volley_completion_times(i[:, b], peaks, dt, measurement)
            trial_completions.append(completions)
            phase_valid.append(valid)
            rivals = np.delete(cumulative[:, b], label, axis=-1).max(axis=-1)
            unique_correct = cumulative[:, b, label] > rivals
            failures = np.flatnonzero(~unique_correct)
            decision_index = int(failures[-1] + 1)
            decision_ms = (
                float(times[decision_index]) if decision_index < len(times) else None
            )
            decision_cycles = None
            decision_volley_count = None
            decision_completed_volley_count = None
            if (
                decision_ms is not None
                and valid
                and len(completions)
                and decision_ms <= completions[-1]
            ):
                decision_completed_volley_count = int(
                    np.searchsorted(completions, decision_ms, side="left") + 1
                )
            if (
                decision_ms is not None
                and valid
                and peaks[0] <= decision_ms <= peaks[-1]
            ):
                decision_cycles = float(
                    np.interp(decision_ms, peaks, np.arange(len(peaks)))
                )
                decision_volley_count = int(
                    np.searchsorted(peaks, decision_ms, side="right")
                )
            diagnostic[f"tau{ti}-seed{seed}--volley_times_ms"] = peaks
            diagnostic[f"tau{ti}-seed{seed}--volley_completion_times_ms"] = completions
            spectrum = analysis.power_spectrum(
                e[:, b].sum(axis=1),
                dt,
                center=True,
                detrend=False,
                nperseg=len(e),
                window="hann",
                scaling="density",
            )
            powers.append(spectrum["power"])
            f = analysis.spectral_peak(
                spectrum["frequencies_hz"],
                spectrum["power"],
                [5.0, 150.0],
                interpolation="parabolic",
                interpolation_boundary="spectrum",
                clamp_band=False,
            )
            hits = np.flatnonzero(
                scores[:, b].max(axis=-1) >= measurement["saturation_threshold"]
            )
            rows.append(
                dict(
                    image_index=image_index,
                    label=label,
                    tau_gaba_ms=tau,
                    seed=seed,
                    sustained_decision_ms=decision_ms,
                    sustained_decision_cycles=decision_cycles,
                    sustained_decision_volley_count=decision_volley_count,
                    sustained_decision_completed_volley_count=decision_completed_volley_count,
                    final_prediction=int(cumulative[-1, b].argmax()),
                    final_correct=bool(cumulative[-1, b].argmax() == label),
                    final_true_score=float(scores[-1, b, label]),
                    final_max_score=float(scores[-1, b].max()),
                    first_max_score_099_ms=float(times[hits[0]]) if len(hits) else None,
                    score_099_before_first_volley=bool(
                        len(hits) and len(peaks) and times[hits[0]] < peaks[0]
                    ),
                    volleys=len(peaks),
                    valid_cycle_coordinate=valid,
                    i_participation_pct=participation,
                    psd_frequency_hz=f["frequency_hz"],
                )
            )
        all_peaks.append(trial_peaks)
        all_completions.append(trial_completions)
        diagnostic[f"tau{ti}--mean_psd"] = np.mean(powers, axis=0)
        diagnostic["psd_frequencies_hz"] = spectrum["frequencies_hz"]


def _measure_image_trajectories(
    *, source, run, cfg, shared, measurement, dt, times, image_reports, rows
):
    """Measure one image across the registered time constants and seeds."""
    for image_index in cfg["image_indices"]:
        unit = source.unit(
            recipe.recording_unit(
                image_index, recipe.RATE_HZ, "benchmark" if shared else "trajectories"
            )
        )
        with np.load(unit / "input.npz", allow_pickle=False) as data:
            label = int(data["label"])
        all_counts = []
        all_true = []
        all_max = []
        all_pred = []
        all_peaks = []
        all_completions = []
        phase_valid = []
        diagnostic = {}
        _record_trajectory_trials(
            cfg=cfg,
            unit=unit,
            measurement=measurement,
            dt=dt,
            label=label,
            all_counts=all_counts,
            all_true=all_true,
            all_max=all_max,
            all_pred=all_pred,
            phase_valid=phase_valid,
            diagnostic=diagnostic,
            times=times,
            rows=rows,
            image_index=image_index,
            all_peaks=all_peaks,
            all_completions=all_completions,
        )
        all_counts = np.asarray(all_counts)
        all_true = np.asarray(all_true)
        all_max = np.asarray(all_max)
        all_pred = np.asarray(all_pred)
        valid_matrix = np.asarray(phase_valid).reshape(len(cfg["tau_gaba_ms"]), -1)
        available = bool(valid_matrix.any()) if shared else all(phase_valid)
        end = (
            (
                max(
                    len(p) - 1
                    for ti, trial in enumerate(all_peaks)
                    for b, p in enumerate(trial)
                    if valid_matrix[ti, b]
                )
                if shared
                else min(len(p) - 1 for trial in all_peaks for p in trial)
            )
            if available
            else 0
        )
        grid = (
            np.arange(round(end / measurement["phase_grid_step"]) + 1)
            * measurement["phase_grid_step"]
            if available
            else np.empty(0)
        )
        for row in rows:
            if row["image_index"] == image_index:
                count = row["sustained_decision_completed_volley_count"]
                if not available or (count is not None and count > end):
                    row["sustained_decision_completed_volley_count"] = None
        phase = np.full((len(cfg["tau_gaba_ms"]), len(cfg["seeds"]), len(grid)), np.nan)
        volley_grid = np.arange(1, end + 1) if available else np.empty(0, dtype=int)
        volley_scores = np.full(
            (len(cfg["tau_gaba_ms"]), len(cfg["seeds"]), len(volley_grid)), np.nan
        )
        if available:
            for ti, trial in enumerate(all_peaks):
                for b, peaks in enumerate(trial):
                    if shared and not valid_matrix[ti, b]:
                        continue
                    reach = grid <= len(peaks) - 1
                    query_times = np.interp(grid[reach], np.arange(len(peaks)), peaks)
                    phase[ti, b, reach] = np.interp(query_times, times, all_true[ti, b])
                    completions = all_completions[ti][b][:end]
                    indices = np.searchsorted(times, completions, side="right") - 1
                    volley_scores[ti, b, : len(indices)] = all_true[ti, b, indices]
        coverage = np.isfinite(volley_scores).mean(axis=1)
        if shared:
            for row in rows:
                if row["image_index"] != image_index:
                    continue
                count = row["sustained_decision_completed_volley_count"]
                ti = list(cfg["tau_gaba_ms"]).index(row["tau_gaba_ms"])
                if count is not None and coverage[ti, count - 1] < 0.8:
                    row["sustained_decision_completed_volley_count"] = None
        destination = run.export / f"image{image_index:05d}"
        destination.mkdir()
        np.savez_compressed(
            destination / "trajectories.npz",
            time_ms=times,
            true_score=all_true,
            max_score=all_max,
            predicted_class=all_pred,
            cumulative_counts=all_counts,
            cycle_position=grid,
            true_score_by_cycle=phase,
            completed_volley_count=volley_grid,
            true_score_by_completed_volley=volley_scores,
            completed_volley_coverage=coverage,
            image_index=image_index,
            label=label,
            tau_gaba_ms=cfg["tau_gaba_ms"],
            seeds=cfg["seeds"],
        )
        np.savez_compressed(destination / "volleys.npz", **diagnostic)
        image_reports.append(
            dict(
                image_index=image_index,
                label=label,
                cycle_coordinate_available=available,
                common_complete_intervals=end if available else None,
            )
        )


def measure_trajectories(source, run, cfg):
    shared = cfg.get("shared_sweep", False)
    measurement = recipe.measurement()
    dt = cfg["dt_ms"]
    times = np.arange(round(cfg["duration_ms"] / dt) + 1) * dt
    image_reports = []
    rows = []
    _measure_image_trajectories(
        source=source,
        run=run,
        cfg=cfg,
        shared=shared,
        measurement=measurement,
        dt=dt,
        times=times,
        image_reports=image_reports,
        rows=rows,
    )
    aggregate = []
    for tau in cfg["tau_gaba_ms"]:
        selected = [r for r in rows if r["tau_gaba_ms"] == tau]
        saturated = [
            r["first_max_score_099_ms"]
            for r in selected
            if r["first_max_score_099_ms"] is not None
        ]
        aggregate.append(
            dict(
                tau_gaba_ms=tau,
                presentations=len(selected),
                final_correct_fraction=float(
                    np.mean([r["final_correct"] for r in selected])
                ),
                final_max_score_099_fraction=float(
                    np.mean([r["final_max_score"] >= 0.99 for r in selected])
                ),
                median_first_max_score_099_ms=float(np.median(saturated))
                if saturated
                else None,
                first_max_score_099_before_first_volley_fraction=float(
                    np.mean([r["score_099_before_first_volley"] for r in selected])
                ),
            )
        )
    decisions = []
    for image_index in cfg["image_indices"]:
        for tau in cfg["tau_gaba_ms"]:
            selected = [
                r
                for r in rows
                if r["image_index"] == image_index and r["tau_gaba_ms"] == tau
            ]
            summary = dict(
                image_index=image_index,
                label=selected[0]["label"],
                tau_gaba_ms=tau,
                trials=len(selected),
            )
            for coordinate in (
                "ms",
                "cycles",
                "volley_count",
                "completed_volley_count",
            ):
                values = [
                    r[f"sustained_decision_{coordinate}"]
                    for r in selected
                    if r[f"sustained_decision_{coordinate}"] is not None
                ]
                summary[f"n_{coordinate}"] = len(values)
                summary[f"quartiles_{coordinate}"] = (
                    np.quantile(values, [0.25, 0.5, 0.75]).tolist() if values else None
                )
            decisions.append(summary)
    report = dict(
        schema="exp123.analysis/v1",
        image_reports=image_reports,
        rows=rows,
        aggregate=aggregate,
        decisions=decisions,
        design={
            key: cfg[key]
            for key in (
                "image_indices",
                "seeds",
                "tau_gaba_ms",
                "dt_ms",
                "duration_ms",
                "input_rate_hz",
            )
        },
        measurement={
            key: value for key, value in measurement.items() if key != "script_sha256"
        },
    )
    if shared:
        report["measurement"].update(
            phase_coverage="within each decay setting, available accepted draws only; display completed-volley scores at >=80% draw coverage; no extrapolation",
            decision_completed_volley_coverage="accepted sequence, in-range completion and >=80% draw coverage at that count within its decay setting",
        )
    (run.export / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reuse-analysis")
    args = parser.parse_args()
    print(analyse(args.source, args.workers, args.reuse_analysis))
