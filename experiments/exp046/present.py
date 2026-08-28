"""Render saved exp046 measurements; never count, fit, simulate or publish."""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp046 import inputs, measurements, plots
from experiments.helpers import theme
from experiments.helpers.fmt import format_duration
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def present(identity, *, run_id=None):
    analysis = inputs.source(REPO, identity, "analyse")
    refs = analysis.record["inputs"]
    if set(refs) != {"compute", "bank", "frequencies"}:
        raise PingstoreError(
            "exp046 analysis must pin compute, bank and exp041 frequencies"
        )
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    cfg, bank, _, checkpoints = inputs.compute_evidence(REPO, compute)
    frequencies = inputs.source(
        REPO,
        refs["frequencies"]["run_id"],
        "analyse",
        experiment="exp041",
        reference=refs["frequencies"],
    )
    if refs["bank"] != bank.reference:
        raise PingstoreError("exp046 analysis and compute bank pins differ")
    inputs.frequency_evidence(REPO, frequencies, bank, cfg, checkpoints)
    result = load_json(analysis.export / "results.json")
    if (
        result.get("schema") != "exp046.analysis/v1"
        or result.get("recipe") != cfg
        or result.get("measurement") != measurements.MEASUREMENT
        or result.get("checkpoint_provenance") != checkpoints
        or analysis.record["execution"].get("configuration") != measurements.MEASUREMENT
    ):
        raise PingstoreError("inconsistent exp046 analysis payload")
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": analysis},
        run_id=run_id,
        configuration={
            "schema": "exp046.presentation/v1",
            "legacy_reference_slope": 0.20,
            "scientific_reference_review": "deferred by author",
        },
    ) as run:
        theme.set_paper_mode(True)
        plots.plot_distribution(
            result["per_tau"], run.export / "spikes_per_cycle_distribution"
        )
        plots.plot_ceiling_vs_fgamma(
            result["results"], run.export / "ceiling_vs_fgamma"
        )
        duration = time.monotonic() - started
        write_json_atomic(
            run.export / "numbers.json",
            {
                **result,
                "run_id": run.run_id,
                "notebook_run_id": run.run_id,
                "duration_s": duration,
                "duration": format_duration(duration),
                "git_sha": run.record["provenance"]["git_commit"],
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp046 analysis run")
    parser.add_argument("--run-id", help="unused source-neutral reservation")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"exp046 present: {exc}\n")


if __name__ == "__main__":
    main()
