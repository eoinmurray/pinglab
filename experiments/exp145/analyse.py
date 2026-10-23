"""Measure AI diagnostics after the configured burn-in in a saved probe.

Run analyse.py --source COMPUTE_ID to validate an explicit compute run and
write rates, CV coverage, pair correlations, population scaling and signed-current
cancellation curves. Burn-in is excluded from every reported measurement.
Sampled native conductances and voltages remain available in the source run.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp145.custom.metrics import (
    current_cancellation,
    pair_correlations,
    population_scaling,
)
from experiments.exp145.custom.recording import read_recording
from experiments.exp145.recipe import configuration
from pingstore.contracts import write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab import analysis


def measure(spikes, cfg):
    cv = analysis.isi_cv(
        spikes,
        cfg["dt_ms"],
        min_spikes=cfg["analysis"]["min_cv_spikes"],
        aggregation="median",
    )
    rates = analysis.per_neuron_firing_rates(spikes, cfg["dt_ms"])["rates_hz"]
    return dict(
        rate_hz=float(rates.mean()),
        active_fraction=float((rates > 0).mean()),
        median_cv=cv["aggregate"],
        cv_eligible=int(cv["valid"].sum()),
        cells=spikes.shape[1],
    ), dict(
        cv=cv["cv"],
    )


def analyse(args):
    source = source_run(
        REPO / ".pingstore", args.source, experiment="exp145", stage="compute"
    )
    cfg = source.record["execution"]["configuration"]
    # Older compute runs retain the bin size for the now-removed rate spectrum.
    cfg = {
        **cfg,
        "analysis": {k: v for k, v in cfg["analysis"].items() if k != "rate_bin_ms"},
    }
    expected = configuration(
        seed=cfg["seed"],
        device=cfg["device"],
        n_e=cfg["n_e"],
        n_i=cfg["n_i"],
        n_external=cfg["n_external"],
        measurement_ms=cfg["duration_ms"] - cfg["burn_in_ms"],
    )
    # The slide step is an analysis choice absent from older compute recipes.
    cfg["analysis"].setdefault(
        "correlation_step_ms", expected["analysis"]["correlation_step_ms"]
    )
    if cfg != expected:
        raise ValueError("unsupported source recipe")
    with stage_run(
        REPO,
        "exp145",
        "analyse",
        inputs={"compute": source},
        configuration=cfg,
        run_id=args.run_id,
    ) as run:
        recording = read_recording(source.unit("recording"), run.scratch / "reader")
        summaries, traces, measured = {}, {}, {}
        start = round(cfg["burn_in_ms"] / cfg["dt_ms"])
        for population in ("E", "I"):
            spikes = recording.signals[f"{population}.spikes"]
            info = recording.info[f"{population}.spikes"]
            steps = round(cfg["duration_ms"] / cfg["dt_ms"])
            if (
                spikes.shape != (steps, cfg[f"n_{population.lower()}"])
                or not np.isin(spikes, (0, 1)).all()
                or info.axes != ("time", "cell")
                or info.dt_ms != cfg["dt_ms"]
                or info.step_indices != tuple(range(steps))
            ):
                raise ValueError("invalid full-window spike recording")
            measured[population] = spikes[start:]
            summaries[population], values = measure(measured[population], cfg)
            scaling, scale_arrays = population_scaling(
                measured[population],
                cfg,
                seed=cfg["seed"]
                + cfg["analysis"]["subsample_seed_offset"]
                + (population == "I"),
            )
            summaries[population]["scaling"] = scaling
            values.update(scale_arrays)
            traces.update(
                {f"{population}_{key}": value for key, value in values.items()}
            )
        correlation_summary, correlation_arrays = pair_correlations(measured, cfg)
        population = cfg["analysis"]["current_population"]
        reference = recording.info[f"{population}.voltage"]
        for signal in [f"{population}.voltage"] + [
            f"{source}_to_{population}.conductance" for source in ("E", "I", "X")
        ]:
            info = recording.info[signal]
            values = recording.signals[signal]
            if (
                info.axes != ("time", "cell")
                or info.dt_ms != cfg["dt_ms"]
                or info.step_indices != reference.step_indices
                or info.cell_ids != reference.cell_ids
                or values.shape != (steps, len(reference.cell_ids))
                or not np.isfinite(values).all()
            ):
                raise ValueError("unaligned sampled current reconstruction signals")
        current_summary, current_arrays = current_cancellation(recording.signals, cfg)
        traces.update(correlation_arrays)
        traces.update(current_arrays)
        np.savez_compressed(run.export / "measurements.npz", **traces)
        write_json_atomic(
            run.export / "results.json",
            dict(
                populations=summaries,
                burn_in_ms=cfg["burn_in_ms"],
                window_ms=cfg["duration_ms"] - cfg["burn_in_ms"],
                correlations=correlation_summary,
                current_cancellation=current_summary,
                correlation_step_ms=cfg["analysis"]["correlation_step_ms"],
                correlation_estimator="Pearson counts in complete sliding windows; excludes constant cells",
            ),
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    print(analyse(parser.parse_args()))


if __name__ == "__main__":
    main()
