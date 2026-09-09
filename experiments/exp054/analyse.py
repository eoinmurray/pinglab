"""Measure explicit full recordings and saved theory; never simulate or publish."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp033.measurements import spiking_medians
from experiments.exp054 import evidence, inputs, measurements, recipe
from experiments.exp054 import theory as theory_inputs
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def analyse(identity, frequency_source, *, theory_source=None, run_id=None):
    source = inputs.source(REPO, identity, "compute")
    cfg = evidence.compute_contract(source)
    frequencies = inputs.source(REPO, frequency_source, "analyse", experiment="exp041")
    imported = source.record["execution"]["operation"] == "historical-import"
    theory = None
    if imported:
        if set(source.record["inputs"]) != {"mean_field", "frequencies"}:
            raise PingstoreError(
                "exp054 historical import must pin mean-field and frequency evidence"
            )
        if source.record["inputs"]["frequencies"] != frequencies.reference:
            raise PingstoreError("exp054 import pins a different frequency source")
        ref = source.record["inputs"]["mean_field"]
        theory = inputs.source(
            REPO, ref["run_id"], "compute", experiment="exp033", reference=ref
        )
    elif source.record["inputs"]:
        raise PingstoreError("native exp054 compute must not have upstream inputs")
    sources = {"compute": source, "frequencies": frequencies}
    stage_cfg = cfg
    if theory_source is not None:
        if imported:
            raise PingstoreError("theory refresh requires native preserved spikes")
        refreshed, theory_cfg, refreshed_mf = theory_inputs.source(
            REPO, theory_source, frequencies
        )
        stage_cfg = recipe.refresh_configuration(cfg, theory_cfg)
        sources["theory"] = refreshed
    with inputs.execution(
        REPO,
        "analyse",
        sources=sources,
        run_id=run_id,
        configuration=stage_cfg,
    ) as run:
        coords = measurements.recordings(source, cfg)
        numbers = measurements.summary(coords, cfg)
        if theory_source is not None:
            mf = refreshed_mf
            run.record["theory_refresh"] = {
                "spiking_probes_reused": len(recipe.jobs(cfg)),
                "embedded_compute_theory_used": False,
                "theory_measurements_reused": True,
            }
            with (run.directory / "README.md").open("a") as history:
                history.write(
                    "\nReanalysed all preserved spike probes. Used the explicitly pinned exp033 analysis for theory; the spike source's embedded historical theory was not used. No simulation or theory solve ran.\n"
                )
        elif imported:
            mf, provenance = evidence.retained_mean_field(theory, frequencies)
            original = load_json(source.export / "historical-numbers.json")
            provenance["empirical_recheck"] = evidence.compare_retained_numbers(
                numbers, original
            )
            numbers = {
                key: original[key] for key in ("config", "grid", "rate_invariance")
            }
            for i, row in enumerate(coords["grid"]):
                for j, cell in enumerate(row):
                    cell["contrast"] = numbers["grid"]["contrast"][i][j]
            for role in ("private", "shared"):
                for row, value in zip(
                    coords[role + "_null"],
                    numbers["rate_invariance"][role + "_scan"]["contrast"],
                    strict=True,
                ):
                    row["contrast"] = value
            run.record["historical_analysis"] = provenance
        else:
            mf = measurements.mean_field(evidence.read(source.export), cfg)
            mf["spiking_exp041"] = {
                str(k): v
                for k, v in spiking_medians(
                    load_json(frequencies.export / "results.json")
                ).items()
            }
        coords.update(mean_field=mf, schema="exp054.analysis/v1", recipe=stage_cfg)
        numbers["mean_field"] = {k: v for k, v in mf.items() if k != "sweep"}
        write_json_atomic(run.export / "results.json", numbers)
        evidence.write(run.export, coords)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp054 compute run")
    parser.add_argument(
        "--frequency-source", required=True, help="completed exp041 analysis run"
    )
    parser.add_argument("--run-id", help="fresh v4 identity reserved before dispatch")
    parser.add_argument(
        "--theory-source",
        help="completed exp033 analysis; replaces embedded theory only",
    )
    args = parser.parse_args()
    analyse(
        args.source,
        args.frequency_source,
        theory_source=args.theory_source,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
