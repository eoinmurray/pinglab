"""Measure saved E/I activity without executing the network.

Invoke analyse.py --source <compute-run-id>. The source must be a completed
exp144 activity run with a supported recorded recipe. Rates, ISI variability,
spike-count correlations and power spectra use native SNNLab estimators.

Statistics exclude the recipe's burn-in; a separate onset rate uses the first
200 ms from rest. Native coordinate-aware recordings remain in the compute run. This stage writes
per-condition measurements and results.json to a separate Pingstore run.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp144.recipe import validate_configuration
from pingstore.contracts import write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab import analysis
from snnlab.sim import RecordingData


def validate_spikes(spikes, cfg, population):
    """Reject truncated or non-binary recordings before measuring them."""
    expected = (
        round(cfg["duration_ms"] / cfg["dt_ms"]),
        cfg[f"n_{population.lower()}"],
    )
    if spikes.shape != expected or not np.isin(spikes, (0, 1)).all():
        raise ValueError(f"invalid {population} spike recording")


def measure(spikes, cfg):
    dt = cfg["dt_ms"]
    rates = analysis.per_neuron_firing_rates(spikes, dt)["rates_hz"]
    cv = analysis.isi_cv(
        spikes, dt, min_spikes=cfg["analysis"]["min_cv_spikes"], aggregation="median"
    )
    corrs = {}
    cells = np.linspace(
        0,
        spikes.shape[1] - 1,
        min(cfg["analysis"]["correlation_cells"], spikes.shape[1]),
        dtype=int,
    )
    for width in cfg["analysis"]["correlation_bins_ms"]:
        r = analysis.spike_count_correlations(
            spikes, dt, bin_ms=width, cells=cells, aggregation="mean"
        )
        corrs[str(width)] = r["aggregate"]
    trace = analysis.population_rate_trace(
        spikes, dt, bin_ms=cfg["analysis"]["rate_bin_ms"]
    )
    spec = analysis.power_spectrum(
        trace["rate_hz"],
        cfg["analysis"]["rate_bin_ms"],
        center=True,
        nperseg=min(cfg["analysis"]["psd_max_segment_bins"], len(trace["rate_hz"])),
    )
    rhythm = analysis.rhythmicity_metrics(spikes, dt)
    return dict(
        rate_hz=float(rates.mean()),
        active_fraction=float((rates > 0).mean()),
        median_isi_cv=cv["aggregate"],
        cv_eligible=int(cv["valid"].sum()),
        correlation=corrs,
        rhythmicity_contrast=rhythm["contrast"],
    ), dict(
        rate_hz=trace["rate_hz"],
        frequencies_hz=spec["frequencies_hz"],
        power=spec["power"],
        cv=cv["cv"],
    )


def analyse(args):
    source = source_run(
        REPO / ".pingstore", args.source, experiment="exp144", stage="compute"
    )
    cfg = source.record["execution"]["configuration"]
    validate_configuration(cfg)
    with stage_run(
        REPO,
        "exp144",
        "analyse",
        inputs={"compute": source},
        configuration=cfg,
        run_id=args.run_id,
    ) as run:
        rows = []
        for case in cfg["cases"]:
            for seed in cfg["seeds"]:
                row = dict(case=case, seed=seed, populations={})
                arrays = {}
                recording = RecordingData.read(
                    source.unit(case["id"], f"seed-{seed}", "recording")
                ).select_batch(0)
                for population in ("E", "I"):
                    signal = f"{population}.spikes"
                    full = recording.signals[signal]
                    info = recording.info[signal]
                    validate_spikes(full, cfg, population)
                    if (
                        info.axes != ("time", "cell")
                        or info.dt_ms != cfg["dt_ms"]
                        or info.step_indices != tuple(range(len(full)))
                    ):
                        raise ValueError("unexpected native recording coordinates")
                    summary, traces = measure(
                        full[round(cfg["burn_in_ms"] / cfg["dt_ms"]) :], cfg
                    )
                    summary["first_200ms_rate_hz"] = float(
                        analysis.per_neuron_firing_rates(
                            full[: round(cfg["analysis"]["onset_ms"] / cfg["dt_ms"])],
                            cfg["dt_ms"],
                        )["rates_hz"].mean()
                    )
                    row["populations"][population] = summary
                    arrays.update(
                        {f"{population}_{key}": value for key, value in traces.items()}
                    )
                del recording
                rows.append(row)
                dest = run.export / case["id"] / f"seed-{seed}"
                dest.mkdir(parents=True)
                np.savez_compressed(dest / "measurements.npz", **arrays)
                print(case["id"], seed, json.dumps(row["populations"]), flush=True)
        write_json_atomic(
            run.export / "results.json",
            dict(schema="exp144.analysis/v1", rows=rows),
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    print(analyse(parser.parse_args()))


if __name__ == "__main__":
    main()
