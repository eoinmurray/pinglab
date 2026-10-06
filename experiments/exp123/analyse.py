"""Cumulative-spike softmax scores and I-volley cycle coordinates."""

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[2] / "tools"),
]
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.special import softmax
from snnlab import analysis
from experiments.exp123 import recipe
from pingstore.stages import source_run, stage_run


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


def analyse(identity, added_rates_identity=None):
    source = source_run(
        recipe.REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    cfg = source.record["execution"]["configuration"]
    if cfg.get("schema") != "exp123.compute/v1":
        raise ValueError("unsupported compute recipe")
    if added_rates_identity is not None:
        from experiments.exp123.pareto import analyse_pareto

        return analyse_pareto(source, added_rates_identity)
    if cfg.get("protocol") == "accuracy":
        from experiments.exp123.accuracy import analyse_accuracy

        return analyse_accuracy(source)
    measurement = recipe.measurement()
    dt = cfg["dt_ms"]
    times = np.arange(round(cfg["duration_ms"] / dt) + 1) * dt
    image_reports = []
    rows = []
    with stage_run(
        recipe.REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute": source},
        configuration=measurement,
    ) as run:
        for image_index in cfg["image_indices"]:
            unit = source.unit(f"image{image_index:05d}")
            with np.load(unit / "input.npz", allow_pickle=False) as data:
                label = int(data["label"])
            all_counts = []
            all_true = []
            all_max = []
            all_pred = []
            all_peaks = []
            phase_valid = []
            diagnostic = {}
            for ti, tau in enumerate(cfg["tau_gaba_ms"]):
                with np.load(
                    unit / f"tau{tau:g}--spikes.npz", allow_pickle=False
                ) as data:
                    i, out, e = [
                        data[key].copy() for key in ("spk_i", "spk_out", "spk_e")
                    ]
                cumulative = np.concatenate(
                    (
                        np.zeros((1, out.shape[1], out.shape[2]), dtype=np.int32),
                        np.cumsum(out, axis=0, dtype=np.int32),
                    ),
                    axis=0,
                )
                scores = softmax(
                    cumulative / measurement["softmax_temperature"], axis=-1
                )
                all_counts.append(cumulative.transpose(1, 0, 2))
                all_true.append(scores[:, :, label].T)
                all_max.append(scores.max(axis=-1).T)
                all_pred.append(cumulative.argmax(axis=-1).T.astype(np.uint8))
                trial_peaks = []
                powers = []
                for b, seed in enumerate(cfg["seeds"]):
                    peaks, valid, participation = volley_times(i[:, b], dt, measurement)
                    trial_peaks.append(peaks)
                    phase_valid.append(valid)
                    rivals = np.delete(cumulative[:, b], label, axis=-1).max(axis=-1)
                    unique_correct = cumulative[:, b, label] > rivals
                    failures = np.flatnonzero(~unique_correct)
                    decision_index = int(failures[-1] + 1)
                    decision_ms = (
                        float(times[decision_index])
                        if decision_index < len(times)
                        else None
                    )
                    decision_cycles = None
                    if (
                        decision_ms is not None
                        and valid
                        and peaks[0] <= decision_ms <= peaks[-1]
                    ):
                        decision_cycles = float(
                            np.interp(decision_ms, peaks, np.arange(len(peaks)))
                        )
                    diagnostic[f"tau{ti}-seed{seed}--volley_times_ms"] = peaks
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
                            final_prediction=int(cumulative[-1, b].argmax()),
                            final_correct=bool(cumulative[-1, b].argmax() == label),
                            final_true_score=float(scores[-1, b, label]),
                            final_max_score=float(scores[-1, b].max()),
                            first_max_score_099_ms=float(times[hits[0]])
                            if len(hits)
                            else None,
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
                diagnostic[f"tau{ti}--mean_psd"] = np.mean(powers, axis=0)
                diagnostic["psd_frequencies_hz"] = spectrum["frequencies_hz"]
            all_counts = np.asarray(all_counts)
            all_true = np.asarray(all_true)
            all_max = np.asarray(all_max)
            all_pred = np.asarray(all_pred)
            available = all(phase_valid)
            end = (
                min(len(p) - 1 for trial in all_peaks for p in trial)
                if available
                else 0
            )
            grid = (
                np.arange(round(end / measurement["phase_grid_step"]) + 1)
                * measurement["phase_grid_step"]
                if available
                else np.empty(0)
            )
            phase = np.full(
                (len(cfg["tau_gaba_ms"]), len(cfg["seeds"]), len(grid)), np.nan
            )
            if available:
                for ti, trial in enumerate(all_peaks):
                    for b, peaks in enumerate(trial):
                        query_times = np.interp(grid, np.arange(len(peaks)), peaks)
                        phase[ti, b] = np.interp(query_times, times, all_true[ti, b])
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
                for coordinate in ("ms", "cycles"):
                    values = [
                        r[f"sustained_decision_{coordinate}"]
                        for r in selected
                        if r[f"sustained_decision_{coordinate}"] is not None
                    ]
                    summary[f"n_{coordinate}"] = len(values)
                    summary[f"quartiles_{coordinate}"] = (
                        np.quantile(values, [0.25, 0.5, 0.75]).tolist()
                        if values
                        else None
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
                key: value
                for key, value in measurement.items()
                if key != "script_sha256"
            },
        )
        (run.export / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        result = run.run_id
    print(
        f"Cycle coordinates available for {sum(r['cycle_coordinate_available'] for r in image_reports)}/{len(image_reports)} images",
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--added-rates-source")
    args = parser.parse_args()
    analyse(args.source, args.added_rates_source)
