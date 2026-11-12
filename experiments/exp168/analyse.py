"""Analyse the pinned EE/II-on and EE/II-off native runs."""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp168 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic
from snnlab import analysis


def _origin(run_id, reference):
    if reference is None:
        return source_run(REPO / ".pingstore", run_id)
    return source_run(
        REPO / ".pingstore",
        run_id,
        reference={
            "run_id": run_id,
        },
    )


def raster_from_events(times, cells, cfg, population):
    start, end = cfg["measurement_ms"]
    selected = (times >= start) & (times < end)
    matrix = np.zeros(
        (round((end - start) / cfg["dt_ms"]), cfg[f"n_{population.lower()}"]),
        dtype=bool,
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
    duration_s = (end - start) / 1000
    out = {}
    samples = []
    with np.load(path, allow_pickle=False) as events:
        for population in ("E", "I"):
            times = events[population + "_time_ms"]
            cells = events[population + "_cell"]
            raster, mask = raster_from_events(times, cells, cfg, population)
            native = analysis.rhythmicity_metrics(
                raster, cfg["dt_ms"], max_lag_ms=100.0, bin_ms=1.0
            )
            cv = analysis.isi_cv(
                raster, cfg["dt_ms"], min_spikes=20, ddof=0, aggregation="median"
            )
            size = cfg[f"n_{population.lower()}"]
            out[population + "_rate_hz"] = int(mask.sum()) / size / duration_s
            out[population + "_rhythmicity"] = native["contrast"]
            out[population + "_median_cv"] = cv["aggregate"]
            out[population + "_cv_cells"] = int(cv["valid"].sum())
            width = round(1.0 / cfg["dt_ms"])
            samples.append(
                raster[:, :200].reshape(round(end - start), width, 200).any(axis=1)
            )
            del raster
    correlation = analysis.spike_count_correlations(
        np.concatenate(samples, axis=1).astype(float),
        1.0,
        bin_ms=20.0,
        aggregation="mean",
    )
    out["mean_correlation"] = correlation["aggregate"]
    return {k: float(v) if isinstance(v, np.floating) else v for k, v in out.items()}


def analyse(*, run_id=None):
    started = perf_counter()
    origins = {
        "on": _origin("exp168-r029-compute", None),
        "off": _origin(
            recipe.SOURCES["off"]["run_id"],
            recipe.SOURCES["off"]["payload_digest"],
        ),
    }
    configurations = {}
    metrics = {}
    for recurrence, origin in origins.items():
        source_configurations = origin.details["execution"]["configuration"][
            "conditions"
        ]
        configurations[recurrence] = source_configurations
        metrics[recurrence] = {
            state: measure(
                origin.file(state, "events.npz"), source_configurations[state]
            )
            for state in recipe.CONDITIONS
        }
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs={"compute_on": origins["on"], "compute_off": origins["off"]},
        configuration={
            "method": "snnlab.analysis.rhythmicity_metrics",
            "bin_ms": 1.0,
            "max_lag_ms": 100.0,
            "min_spikes_for_cv": 20,
            "correlation_cells_per_population": 200,
            "correlation_bin_ms": 20.0,
        },
        run_id=run_id,
    ) as run:
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                schema="exp168.analysis.both/v1",
                configurations=configurations,
                metrics=metrics,
                analysis={
                    "bin_ms": 1.0,
                    "max_lag_ms": 100.0,
                    "min_spikes_for_cv": 20,
                    "correlation_cells_per_population": 200,
                    "correlation_bin_ms": 20.0,
                },
                snnlab_version=version("snnlab"),
                evidence={"kind": "native"},
                limitations=[
                    "Each condition is one selected network/input realization.",
                    "AI and PING differ in three external-drive settings.",
                ],
            ),
        )
        run.details["execution"]["elapsed_s"] = perf_counter() - started
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
