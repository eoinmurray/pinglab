"""Measure spectra, autocorrelation, complete-cycle participation and cost."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[2] / "tools"),
]
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, find_peaks
from snnlab import analysis
from experiments.exp122 import recipe
from pingstore.stages import source_run, stage_run


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
    e_cycle_counts = analysis.cycle_spike_counts(e, boundary)["counts"]
    i_cycle_counts = analysis.cycle_spike_counts(i, boundary)["counts"]
    cycles = [(int(a), int(b)) for a, b in zip(boundary[:-1], boundary[1:])]
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


def analyse(identity):
    source = source_run(
        recipe.REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    cfg = recipe.analysis_configuration(source.record["execution"]["configuration"])
    start = round(cfg["transient_ms"] / cfg["dt_ms"])
    rows = []
    with stage_run(
        recipe.REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute": source},
        configuration=cfg,
    ) as run:
        for condition in cfg["conditions"]:
            with np.load(
                source.file(f"{condition['condition_id']}--spikes.npz"),
                allow_pickle=False,
            ) as data:
                for b, seed in enumerate(cfg["seeds"]):
                    e, i, out = [data[k][:, b] for k in ("spk_e", "spk_i", "spk_out")]
                    metrics, diag = dynamics(e[start:], i[start:], cfg)
                    metrics.update(
                        **condition,
                        seed=seed,
                        core_spikes_full=int(e.sum() + i.sum()),
                        total_e_spikes_full=int(e.sum()),
                        e_spikes_per_second=float(
                            e.sum() / (cfg["duration_ms"] / 1000)
                        ),
                        output_spikes_full=int(out.sum()),
                        total_network_spikes_full=int(e.sum() + i.sum() + out.sum()),
                        core_spikes_window=int(e[start:].sum() + i[start:].sum()),
                        total_network_spikes_window=int(
                            e[start:].sum() + i[start:].sum() + out[start:].sum()
                        ),
                        e_rate_hz=analysis.firing_rate(e[start:], cfg["dt_ms"]),
                        i_rate_hz=analysis.firing_rate(i[start:], cfg["dt_ms"]),
                    )
                    rows.append(metrics)
                    np.savez_compressed(
                        run.export
                        / f"{condition['condition_id']}-seed{seed}--diagnostics.npz",
                        **diag,
                    )
        psd_summaries = []
        for condition in cfg["conditions"]:
            powers = []
            for seed in cfg["seeds"]:
                with np.load(
                    run.export
                    / f"{condition['condition_id']}-seed{seed}--diagnostics.npz"
                ) as diag:
                    powers.append(diag["psd"].copy())
                    freq = diag["frequency_hz"].copy()
            mean_psd = np.mean(powers, axis=0)
            psd_summaries.append(
                dict(
                    **condition,
                    frequency_hz=analysis.spectral_peak(
                        freq,
                        mean_psd,
                        cfg["frequency_search_hz"],
                        interpolation="parabolic",
                        interpolation_boundary="spectrum",
                        clamp_band=False,
                    )["frequency_hz"],
                )
            )
        summaries = []
        for j, condition in enumerate(cfg["conditions"]):
            selected = [
                row for row in rows if row["condition_id"] == condition["condition_id"]
            ]
            summary = dict(
                **condition,
                psd_frequency_hz=psd_summaries[j]["frequency_hz"],
                accepted_cycles=[row["accepted_cycles"] for row in selected],
            )
            for key in (
                "participation_e_pct",
                "e_spikes_per_second",
                "rhythmicity_contrast",
            ):
                values = [row[key] for row in selected if row[key] is not None]
                summary[key] = float(np.mean(values)) if values else None
            summaries.append(summary)
        (run.export / "metrics.json").write_text(
            json.dumps(
                dict(
                    schema="exp122.analysis/v2",
                    design={
                        key: cfg[key]
                        for key in (
                            "image_index",
                            "image_label",
                            "partition",
                            "seeds",
                            "tau_gaba_ms",
                            "leak_scales",
                            "capacitance_scales",
                            "conditions",
                            "dt_ms",
                            "duration_ms",
                            "transient_ms",
                            "population_bin_ms",
                            "frequency_search_hz",
                            "input_rate_hz",
                            "training_tau_gaba_ms",
                            "biological_defaults",
                            "output_lif",
                            "volley_smoothing_ms",
                            "volley_minimum_spacing_ms",
                            "volley_relative_prominence",
                            "volley_minimum_prominence_spikes",
                            "minimum_cycles",
                            "minimum_mean_i_participation_pct",
                        )
                    },
                    rows=rows,
                    psd_summaries=psd_summaries,
                    summaries=summaries,
                ),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        scalar_keys = [k for k, v in rows[0].items() if not isinstance(v, list)]
        with (run.export / "metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=scalar_keys)
            writer.writeheader()
            writer.writerows({k: row[k] for k in scalar_keys} for row in rows)
        identity = run.run_id
    print(
        f"Identifiable cycle sequences: {sum(row['valid_cycles'] for row in rows)}/{len(rows)}",
        flush=True,
    )
    print(identity, flush=True)
    return identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    analyse(parser.parse_args().source)
