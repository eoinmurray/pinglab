"""Recompute comparable native metrics from an explicitly selected compute run."""

import argparse
from importlib.metadata import version
from pathlib import Path
import sys
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from snnlab import analysis
from experiments.exp168.recipe import SLUG, CONDITIONS
from pingstore.library import source_run, stage_run, write_json_atomic


def raster_from_events(times, cells, cfg, pop):
    start, end = cfg["measurement_ms"]
    selected = (times >= start) & (times < end)
    matrix = np.zeros(
        (round((end - start) / cfg["dt_ms"]), cfg["n_" + pop.lower()]), dtype=bool
    )
    steps = np.rint((times[selected] - start) / cfg["dt_ms"]).astype(np.int64)
    indices = cells[selected].astype(np.int64)
    if (
        np.any(steps < 0)
        or np.any(steps >= len(matrix))
        or np.any(indices < 0)
        or np.any(indices >= matrix.shape[1])
    ):
        raise ValueError("spike event outside recorded dimensions")
    matrix[steps, indices] = True
    if int(matrix.sum()) != int(selected.sum()):
        raise ValueError("duplicate spike events")
    return matrix, selected


def measure(path, cfg):
    start, end = cfg["measurement_ms"]
    duration = (end - start) / 1000
    out = {}
    curves = {}
    samples = []
    with np.load(path, allow_pickle=False) as z:
        for pop in ("E", "I"):
            times, cells = z[pop + "_time_ms"], z[pop + "_cell"]
            raster, mask = raster_from_events(times, cells, cfg, pop)
            native = analysis.rhythmicity_metrics(
                raster, cfg["dt_ms"], max_lag_ms=100.0, bin_ms=1.0
            )
            cv = analysis.isi_cv(
                raster, cfg["dt_ms"], min_spikes=20, ddof=0, aggregation="median"
            )
            out[pop + "_rate_hz"] = int(mask.sum()) / cfg["n_" + pop.lower()] / duration
            out[pop + "_rhythmicity"] = native["contrast"]
            out[pop + "_lobe_to_trough"] = native["lobe_to_trough"]
            out[pop + "_median_cv"] = cv["aggregate"]
            out[pop + "_cv_cells"] = int(cv["valid"].sum())
            for key in ("ac_lags", "ac", "iei_lags", "iei_counts"):
                curves[pop + "_" + key] = native[key]
            width = round(1.0 / cfg["dt_ms"])
            samples.append(
                raster[:, :200].reshape(round(end - start), width, 200).any(axis=1)
            )
            del raster
            counts = np.histogram(times[mask], np.arange(start, end + 0.25, 0.5))[0]
            rate = counts / cfg["n_" + pop.lower()] / 0.0005
            spectrum = analysis.power_spectrum(
                rate - rate.mean(),
                0.5,
                method="fft",
                detrend="constant",
                window="hann",
                scaling="density",
            )
            f, p = spectrum["frequencies_hz"], spectrum["power"]
            peak = analysis.spectral_peak(f, p, (5.0, 150.0))
            bg = np.median(p[(f >= 100) & (f <= 300)])
            out[pop + "_peak_hz"] = peak["frequency_hz"]
            out[pop + "_peak_contrast"] = (
                float(peak["peak_power"] / bg)
                if peak["peak_power"] is not None and bg > 0
                else None
            )
            out[pop + "_max_bin_fraction"] = float(
                counts.max() / cfg["n_" + pop.lower()]
            )
    correlation = analysis.spike_count_correlations(
        np.concatenate(samples, axis=1).astype(float),
        1.0,
        bin_ms=20.0,
        aggregation="mean",
    )
    out["mean_correlation"] = correlation["aggregate"]
    pairs = correlation["correlations"][np.triu_indices(400, 1)]
    out["correlation_sd"] = float(np.std(pairs[np.isfinite(pairs)]))
    return {
        k: float(v) if isinstance(v, np.floating) else v for k, v in out.items()
    }, curves


def analyse(source, *, run_id=None):
    started = perf_counter()
    upstream = source_run(REPO / ".pingstore", source, experiment=SLUG, stage="compute")
    configurations = upstream.record["execution"]["configuration"]["conditions"]
    settings = {
        "method": "snnlab.analysis.rhythmicity_metrics",
        "bin_ms": 1.0,
        "max_lag_ms": 100.0,
        "min_spikes_for_cv": 20,
        "correlation_cells_per_population": 200,
        "correlation_bin_ms": 20.0,
    }
    metrics, curves = {}, {}
    for label in CONDITIONS:
        metrics[label], values = measure(
            upstream.file(label, "events.npz"), configurations[label]
        )
        curves.update({label + "_" + key: value for key, value in values.items()})
    with stage_run(
        REPO,
        SLUG,
        "analyse",
        inputs={"compute": upstream},
        configuration=settings,
        run_id=run_id,
    ) as run:
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                schema="exp168.analysis/v1",
                configurations=configurations,
                metrics=metrics,
                analysis=settings,
                snnlab_version=version("snnlab"),
                evidence=upstream.record["execution"].get(
                    "evidence", {"kind": "native"}
                ),
                limitations=[
                    "One selected network/input seed (43), two-second windows.",
                    "AI is an AI-like candidate: its original maximum-I-bin screen was 3.3%, above the preset 3% cutoff.",
                    "The selected AI and PING states differ in three independent external-drive settings.",
                ],
            ),
        )
        np.savez_compressed(run.export / "curves.npz", **curves)
        run.record["execution"]["elapsed_s"] = perf_counter() - started
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
