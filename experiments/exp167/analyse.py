"""Analyse explicit native events, or carry forward declared historical metrics."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp167.recipe import SLUG
from pingstore.library import load_json, source_run, stage_run, write_json_atomic


def metrics(path, label):
    from snnlab import analysis

    z = np.load(path)
    start, end = 50.0, 1050.0
    out = {"condition": label}
    for population, n in (("E", 4000), ("I", 1000)):
        times = z[population + "_time_ms"]
        cells = z[population + "_cell"]
        mask = (times >= start) & (times < end)
        selected = times[mask]
        out[population + "_rate_hz"] = len(selected) / n / ((end - start) / 1000)
        # Per-cell CV
        cvs = []
        for cell in range(n):
            spikes = np.sort(selected[cells[mask] == cell])
            if len(spikes) >= 20:
                isi = np.diff(spikes)
                cvs.append(isi.std() / isi.mean())
        out[population + "_median_cv"] = float(np.median(cvs)) if cvs else None
        out[population + "_cv_cells"] = len(cvs)
        # Spectral peak and contrast
        edges = np.arange(start, end + 0.25, 0.5)
        counts = np.histogram(selected, edges)[0]
        rate = counts / n / (0.5 / 1000)
        spectrum = analysis.power_spectrum(
            rate - rate.mean(),
            0.5,
            method="fft",
            detrend="constant",
            window="hann",
            scaling="density",
        )
        freqs, power = spectrum["frequencies_hz"], spectrum["power"]
        peak = analysis.spectral_peak(freqs, power, (5.0, 150.0))
        background = np.median(power[(freqs >= 100) & (freqs <= 300)])
        out[population + "_peak_hz"] = float(peak["frequency_hz"])
        out[population + "_peak_contrast"] = float(peak["peak_power"] / background)
        # Volley concentration
        out[population + "_max_bin_fraction"] = float(counts.max() / n)
    # Rhythmicity via pooled autocorrelogram
    for population in ("E", "I"):
        times = z[population + "_time_ms"]
        cells = z[population + "_cell"]
        mask = (times >= start) & (times < end)
        selected = times[mask]
        e1 = np.arange(start, end + 1.0, 1.0)
        counts = np.histogram(selected, e1)[0].astype(float)
        ac = np.correlate(counts, counts, mode="full")[
            len(counts) - 1 : len(counts) + 99
        ]
        ac = ac / (len(counts) - np.arange(100))
        smoothed = np.convolve(ac, np.array([1, 1, 1]) / 3, mode="same")
        trough = None
        for i in range(2, len(smoothed) - 1):
            if smoothed[i] <= smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
                trough = i
                break
        lobe = np.max(smoothed[:trough]) if trough else None
        out[population + "_rhythmicity"] = (
            float((lobe - smoothed[trough]) / (lobe + smoothed[trough]))
            if trough and lobe
            else None
        )
    # Pairwise correlations (20 ms bins, 1 ms step)
    combined = []
    for population in ("E", "I"):
        times = z[population + "_time_ms"]
        cells = z[population + "_cell"]
        mask = (times >= start) & (times < end)
        selected_cells = np.linspace(
            0, 200 - 1, min(200, {"E": 4000, "I": 1000}[population]), dtype=int
        )
        raster = np.zeros((int((end - start) / 1.0), len(selected_cells)), dtype=bool)
        for j, cell in enumerate(selected_cells):
            spikes = times[mask & (cells == cell)]
            for s in spikes:
                idx = int((s - start) / 1.0)
                if 0 <= idx < raster.shape[0]:
                    raster[idx, j] = True
        combined.append(raster)
    combined = np.concatenate(combined, axis=1)
    result = analysis.spike_count_correlations(
        combined.astype(float), 1.0, bin_ms=20.0, aggregation="mean"
    )
    out["mean_correlation"] = result["aggregate"]
    out["correlation_sd"] = float(
        np.std(
            result["correlations"][np.triu_indices(len(result["correlations"]), 1)][
                np.isfinite(
                    result["correlations"][
                        np.triu_indices(len(result["correlations"]), 1)
                    ]
                )
            ]
        )
    )
    return out


def validate_metrics(values):
    if set(values) != {"AI", "PING"}:
        raise ValueError("expected AI and PING metrics")
    expected = {"condition", "mean_correlation", "correlation_sd"} | {
        p + "_" + name
        for p in ("E", "I")
        for name in (
            "rate_hz",
            "median_cv",
            "cv_cells",
            "peak_hz",
            "peak_contrast",
            "max_bin_fraction",
            "rhythmicity",
        )
    }
    for label, row in values.items():
        if set(row) != expected or row["condition"] != label:
            raise ValueError("unexpected metric schema")
        for key, value in row.items():
            if key == "condition":
                continue
            if value is None:
                if (
                    not key.endswith("median_cv")
                    or row[key.replace("median_cv", "cv_cells")] != 0
                ):
                    raise ValueError("unexpected missing metric")
            elif not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError("nonfinite metric")
    return values


def analyse(source, *, run_id=None):
    started = perf_counter()
    upstream = source_run(REPO / ".pingstore", source, experiment=SLUG, stage="compute")
    cfg = upstream.record["execution"]["configuration"]
    imported = (
        upstream.record["execution"].get("evidence", {}).get("kind")
        == "historical-scratch-import"
    )
    values = (
        load_json(upstream.file("metrics.json"))
        if imported
        else {
            label: metrics(upstream.file(f"{label}/events.npz"), label)
            for label in cfg["conditions"]
        }
    )
    validate_metrics(values)
    with stage_run(
        REPO,
        SLUG,
        "analyse",
        inputs={"compute": upstream},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                configuration=cfg,
                metrics=values,
                evidence=upstream.record["execution"].get(
                    "evidence", {"kind": "native"}
                ),
                metrics_recomputed=not imported,
            ),
        )
        run.record["execution"]["elapsed_s"] = perf_counter() - started
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
