"""Measure explicitly selected exp083 recordings; never simulate or publish."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp083 import evidence, inputs, measurements, recipe
from experiments.helpers.gamma_frequency import estimate_gamma_from_raster
from pingstore.contracts import PingstoreError, write_json_atomic


def analyse(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "compute")
    original, graph, manifest = evidence.compute_payload(source)
    with inputs.execution(
        REPO, "analyse", sources={"compute": source}, run_id=run_id
    ) as run:
        (run.export / "spectra").mkdir()
        (run.export / "rasters").mkdir()
        summaries = []
        for condition in recipe.conditions():
            rate = condition["input_rate_hz"]
            arrays = evidence.recording(source.export / condition["file"])
            e, i = arrays["e_spikes"], arrays["i_spikes"]
            estimate = estimate_gamma_from_raster(
                e, dt_ms=recipe.DT_MS, config=recipe.FREQUENCY_CONFIG
            )
            rows = measurements._trial_rows(rate, e, i, estimate)
            summaries.append(measurements.summarize_condition(rate, rows))
            np.savez_compressed(
                run.export / f"spectra/rate-{rate:g}.npz",
                frequencies_hz=estimate.frequencies_hz,
                mean_psd=estimate.mean_psd,
            )
            if rate in recipe.REPRESENTATIVE_RATES_HZ:
                e_t, e_cells = np.nonzero(e[:, recipe.DISPLAY_TRIAL])
                i_t, i_cells = np.nonzero(i[:, recipe.DISPLAY_TRIAL])
                np.savez_compressed(
                    run.export / f"rasters/rate-{rate:g}.npz",
                    e_t=e_t,
                    e_cells=e_cells,
                    i_t=i_t,
                    i_cells=i_cells,
                )
        write_json_atomic(run.export / "network-graph.json", graph)
        write_json_atomic(run.export / "network-manifest.json", manifest)
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp083.analysis/v1",
                "recipe": recipe.configuration(),
                "question": "Does the default SNNLANG PING component contain a reproducible gamma regime as homogeneous Poisson drive increases?",
                "config": recipe.SCALE,
                "frequency_analysis": recipe.FREQUENCY_CONFIG.json(),
                "representative_rates_hz": list(recipe.REPRESENTATIVE_RATES_HZ),
                "graph": original["graph"],
                "conditions": summaries,
                "rasters": evidence.display_entries(
                    "rasters", recipe.REPRESENTATIVE_RATES_HZ
                ),
                "spectra": evidence.display_entries("spectra", recipe.INPUT_RATES_HZ),
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp083 analyse: {exc}\n")


if __name__ == "__main__":
    main()
