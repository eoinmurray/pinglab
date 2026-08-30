"""Measure retained inhibitory-decay sweep evidence; never run inference or plot."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp041 import evidence, inputs, measurements, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic

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
        "training_contract": contract,
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
            row.update(measurements.spectrum(traces, common["dt"]))
            evidence.finite(row["f_gamma_hz"], "mean-PSD gamma peak")
            rows.append(row)
            if cell["seed"] != cfg["raster"]["seed"]:
                continue
            raw = evidence.snapshot(
                compute.file("snapshot", cell["cell_name"], "snapshot.npz"),
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
        aggregate = measurements.summarize(rows)
        fit = measurements.fit_law(aggregate)
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


if __name__ == "__main__":
    main()
