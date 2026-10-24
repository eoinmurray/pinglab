"""Measure retained inhibitory-decay sweep evidence; never run inference or plot.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse

import numpy as np
from experiments.exp041 import recipe
from experiments.exp041.custom import evidence, inputs
from experiments.exp041.recipe import F_GAMMA_BAND_HZ
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from scipy import signal as sp_signal

MEASUREMENT = {
    "schema": "exp041.measurement/v1",
    "spectrum": "Welch per trial; demean; full-trial segment; density scaling; average PSD across trials",
    "gamma_peak": "5-150 Hz argmax of mean PSD with parabolic offset clamped to +/- half a bin",
    "fit_points": "six tau_GABA means across seeds; equal weight per condition",
    "fit_models": ["affine", "through_origin"],
    "r_squared": "centred total sum of squares for both fits",
    "uncertainty": "seed SEM (ddof=1)",
    "history_partition": "validation",
    "trial_peak_histogram": "diagnostic only; pooled across seeds; 1 Hz bins",
}


def analyse(identity, *, run_id=None):
    compute = inputs.source(REPO, identity, "compute")
    cfg = inputs.configuration(compute)
    ref = compute.record["inputs"]["bank"]
    bank = inputs.source(
        REPO, ref["run_id"], "compute", experiment="exp022", reference=ref
    )
    contract = evidence.training_contract(bank.export)
    checkpoints = evidence.checkpoints(bank.export, contract)
    retained = load_json(compute.export / "evidence.json")
    if retained != {
        "schema": "exp041.compute/v1",
        "config": cfg,
        "training_contract": evidence.retained_contract(
            retained.get("training_contract"), contract, bank.export
        ),
        "checkpoint_provenance": checkpoints,
    }:
        raise PingstoreError(
            "compute evidence disagrees with its pinned bank or recipe"
        )
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": compute, "bank": bank},
        run_id=run_id,
        configuration=MEASUREMENT,
    ) as run:
        rows, rasters, arrays = [], [], {}
        common = contract["common"]
        for cell in contract["cells"]:
            directory = compute.unit("infer", cell["cell_name"])
            row = evidence.measurement(
                directory / "metrics.json", cell, common, cfg["evaluation_samples"]
            )
            traces = evidence.population_traces(
                directory / "pop_traces.npz", common, cfg["evaluation_samples"]
            )
            row.update(spectrum(traces, common["dt"]))
            evidence.finite(row["f_gamma_hz"], "mean-PSD gamma peak")
            rows.append(row)
            if cell["seed"] != cfg["raster"]["seed"]:
                continue
            raw = evidence.snapshot(
                compute.file("snapshot", cell["cell_name"], "recording.npz"),
                common["dt"],
                common,
            )
            e, i = raw["spk_e"], raw["spk_i"]
            rng = np.random.default_rng(cfg["raster"]["selection_seed"])
            e_idx = np.sort(
                rng.choice(e.shape[1], cfg["raster"]["n_e_plot"], replace=False)
            )
            i_idx = np.sort(
                rng.choice(i.shape[1], cfg["raster"]["n_i_plot"], replace=False)
            )
            arrays[cell["cell_name"] + "__e"] = e[:, e_idx].astype(bool)
            arrays[cell["cell_name"] + "__i"] = i[:, i_idx].astype(bool)
            rasters.append(
                {
                    **cell,
                    "label": raw["label"],
                    "dt_ms": common["dt"],
                    "t_ms": common["t_ms"],
                    "e_indices": e_idx.tolist(),
                    "i_indices": i_idx.tolist(),
                    "e_rate_hz": float(e.sum() / (e.shape[1] * common["t_ms"] / 1000)),
                    "i_rate_hz": float(i.sum() / (i.shape[1] * common["t_ms"] / 1000)),
                }
            )
        aggregate = summarize(rows)
        fit = fit_law(aggregate)
        if any(v is None or not np.isfinite(v) for v in fit.values()):
            raise PingstoreError(
                "undefined rate-frequency fit; inspect retained evidence"
            )
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp041.analysis/v1",
                "recipe": cfg,
                "measurement": MEASUREMENT,
                "config": {
                    "dataset": "mnist",
                    "tau_gaba_sweep_ms": cfg["tau_gaba_sweep_ms"],
                    "seeds": cfg["seeds"],
                    "f_gamma_band_hz": cfg["f_gamma_band_hz"],
                    "max_samples": common["max_samples"],
                    "evaluation_samples": cfg["evaluation_samples"],
                    "epochs": common["epochs"],
                    "t_ms": common["t_ms"],
                    "dt": common["dt"],
                    "training_contract": contract,
                },
                "checkpoint_policy": recipe.CHECKPOINT_POLICY,
                "checkpoint_provenance": checkpoints,
                "git_sha_train": load_json(
                    bank.file(contract["cells"][0]["cell_name"], "config.json")
                ).get("git_sha"),
                "results": [
                    {
                        k: v
                        for k, v in r.items()
                        if k not in ("freqs_hz", "psd", "per_trial_peaks_hz")
                    }
                    for r in rows
                ],
                "fit": fit,
                "aggregate": aggregate,
                "rasters": rasters,
                "curves": evidence.histories(bank.export, contract),
            },
        )
        write_json_atomic(run.export / "spectra.json", {"rows": rows})
        np.savez_compressed(run.export / "rasters.npz", **arrays)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp041 compute ID")
    parser.add_argument("--run-id", help="unused source-neutral reservation")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except PingstoreError as exc:
        parser.exit(1, f"exp041 analyse: {exc}\n")


def _peak_with_parabolic(psd: np.ndarray, freqs: np.ndarray) -> float:
    """Locate the gamma-band peak with parabolic sub-bin interpolation.

    Returns NaN if the PSD is flat in the gamma band. Welch with
    nperseg = T_steps gives Δf = fs/nperseg (5 Hz at fs=10000, T=200ms)
    — too coarse on its own across six τ_GABA values. Parabolic
    interpolation through (k-1, k, k+1) recovers the analytic peak
    location with error O((Δf)^3) when the peak is well-isolated.
    """
    band_mask = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
    if not band_mask.any() or psd[band_mask].max() <= 0:
        return float("nan")
    in_band = np.where(band_mask)[0]
    peak_local = int(psd[in_band].argmax())
    peak_idx = int(in_band[peak_local])
    if not (0 < peak_idx < len(psd) - 1):
        return float(freqs[peak_idx])
    y0 = float(psd[peak_idx - 1])
    y1 = float(psd[peak_idx])
    y2 = float(psd[peak_idx + 1])
    denom = y0 - 2.0 * y1 + y2
    offset = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    offset = max(-0.5, min(0.5, offset))
    df = float(freqs[1] - freqs[0])
    return float(freqs[peak_idx]) + offset * df


def spectrum(traces, dt_ms):
    """Per-trial Welch density, then peak of the mean PSD; not median peaks."""
    psds = []
    for tr in traces:
        freqs, power = sp_signal.welch(
            tr - tr.mean(),
            fs=1000.0 / dt_ms,
            nperseg=traces[0].size,
            scaling="density",
            detrend=False,
        )
        psds.append(power)
    mean = np.mean(np.stack(psds, axis=0), axis=0)
    per_trial = [_peak_with_parabolic(p, freqs) for p in psds]
    return {
        "f_gamma_hz": _peak_with_parabolic(mean, freqs),
        "freqs_hz": freqs.tolist(),
        "psd": mean.tolist(),
        "per_trial_peaks_hz": [float(x) for x in per_trial if np.isfinite(x)],
    }


def summarize(rows):
    """Six condition means and seed SEMs; retain per-trial diagnostic statistics."""
    aggregate = []
    for tau in sorted({r["tau_gaba_ms"] for r in rows}):
        sub = [r for r in rows if r["tau_gaba_ms"] == tau]
        item = {"tau_gaba_ms": tau}
        for key in ("f_gamma_hz", "e_rate_hz", "acc"):
            values = [r[key] for r in sub]
            item[key] = {
                "mean": float(np.mean(values)),
                "sem": float(np.std(values, ddof=1) / np.sqrt(len(sub)))
                if len(sub) > 1
                else 0.0,
            }
        item["freqs_hz"] = sub[0]["freqs_hz"]
        item["psd"] = np.mean(np.stack([r["psd"] for r in sub]), axis=0).tolist()
        # The PSD panel historically marked the un-interpolated peak of the seed-mean spectrum.
        freqs = np.array(item["freqs_hz"])
        power = np.array(item["psd"])
        band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
        idx = power[band].argmax()
        item["psd_marker"] = {
            "frequency_hz": float(freqs[band][idx]),
            "power": float(power[band][idx]),
        }
        peaks = np.array([p for row in sub for p in row["per_trial_peaks_hz"]])
        bins = np.arange(F_GAMMA_BAND_HZ[0], F_GAMMA_BAND_HZ[1] + 1.0, 1.0)
        counts, edges = np.histogram(peaks, bins=bins)
        item["trial_peaks"] = {
            "counts": counts.tolist(),
            "bins_hz": edges.tolist(),
            "median_hz": float(np.median(peaks)) if peaks.size else None,
            "iqr_hz": float(np.percentile(peaks, 75) - np.percentile(peaks, 25))
            if peaks.size
            else None,
        }
        aggregate.append(item)
    return aggregate


def fit_law(aggregate):
    """Retained least-squares fits on the six seed means, with centred R squared."""
    fg_arr = np.array([row["f_gamma_hz"]["mean"] for row in aggregate])
    er_arr = np.array([row["e_rate_hz"]["mean"] for row in aggregate])
    slope_aff, intercept_aff = np.polyfit(fg_arr, er_arr, 1)
    p_fit, a_fit = float(slope_aff), float(intercept_aff)
    ss_tot = float(np.sum((er_arr - er_arr.mean()) ** 2))
    ss_res_aff = float(np.sum((er_arr - (p_fit * fg_arr + a_fit)) ** 2))
    p0 = float(np.sum(fg_arr * er_arr) / np.sum(fg_arr**2))
    ss_res_0 = float(np.sum((er_arr - p0 * fg_arr) ** 2))
    return {
        "p_affine": p_fit,
        "a_affine": a_fit,
        "r2_affine": 1.0 - ss_res_aff / ss_tot if ss_tot > 0 else None,
        "p_origin": p0,
        "r2_origin": 1.0 - ss_res_0 / ss_tot if ss_tot > 0 else None,
    }


if __name__ == "__main__":
    main()
