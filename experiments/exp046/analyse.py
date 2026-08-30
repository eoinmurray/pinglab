"""Count cycles from retained spikes and explicit exp041 frequencies; never simulate."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp046 import evidence, inputs, measurements, recipe
from pingstore.contracts import PingstoreError, write_json_atomic


def analyse(identity, frequency_id, *, run_id=None):
    compute = inputs.source(REPO, identity, "compute")
    cfg, bank, contract, checkpoints = inputs.compute_evidence(REPO, compute)
    frequencies = inputs.source(REPO, frequency_id, "analyse", experiment="exp041")
    values = inputs.frequency_evidence(REPO, frequencies, bank, cfg, checkpoints)
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": compute, "bank": bank, "frequencies": frequencies},
        run_id=run_id,
        configuration=measurements.MEASUREMENT,
    ) as run:
        rows = []
        for cell in contract["cells"]:
            directory = compute.unit("infer", cell["cell_name"])
            common = contract["common"]
            scalar = evidence.measurement(
                directory / "metrics.json", cell, common, cfg["evaluation_samples"]
            )
            raster, rates = evidence.recordings(
                directory, common, cfg["evaluation_samples"]
            )
            row = measurements.measure(
                raster,
                rates,
                scalar["acc"],
                cell["tau_gaba_ms"],
                common["dt"],
                values[(cell["tau_gaba_ms"], cell["seed"])],
            )
            rows.append({**row, "seed": cell["seed"]})
        result = {
            "schema": "exp046.analysis/v1",
            "recipe": cfg,
            "measurement": measurements.MEASUREMENT,
            "checkpoint_policy": recipe.CHECKPOINT_POLICY,
            "checkpoint_provenance": checkpoints,
            "config": {
                "tau_gabas_ms": cfg["tau_gaba_sweep_ms"],
                "seeds": cfg["seeds"],
                "evaluation_samples": cfg["evaluation_samples"],
                "exp041_source": "ping__tg{N}__seed{S}",
                "exp041_training_epochs": contract["common"]["epochs"],
                "dt_ms": contract["common"]["dt"],
            },
            "results": rows,
            **measurements.summarize(rows),
        }
        write_json_atomic(run.export / "results.json", result)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp046 compute run")
    parser.add_argument(
        "--frequency-source", required=True, help="completed exp041 analysis run"
    )
    parser.add_argument("--run-id", help="unused source-neutral reservation")
    args = parser.parse_args()
    try:
        analyse(args.source, args.frequency_source, run_id=args.run_id)
    except (PingstoreError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"exp046 analyse: {exc}\n")


if __name__ == "__main__":
    main()
