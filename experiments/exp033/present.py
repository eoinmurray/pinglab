"""Render saved measurements and coordinates; never solve or measure the model."""

import argparse
import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp033 import evidence, inputs, plots
from pingstore.contracts import (
    PingstoreError,
    load_json,
    write_json_atomic,
)


def article_numbers(numbers):
    """Qualify criterion labels without changing results or decisions."""
    result = copy.deepcopy(numbers)
    labels = (
        "Reference 4D Hopf in the gamma band",
        "Sampled onset is consistent with supercriticality",
        "Tested two-rate QSS reduction rings down",
        "Three variables suffice in the tested QSS ring family",
    )
    if len(result["success_criteria"]) != len(labels):
        raise PingstoreError("unexpected exp033 scientific criteria")
    for criterion, label in zip(result["success_criteria"], labels, strict=True):
        criterion["label"] = label
    return result


def present(identity, *, run_id=None):
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    if set(analysis.record["inputs"]) != {"compute", "frequencies"}:
        raise PingstoreError("exp033 analysis must pin compute and exp041 frequencies")
    upstreams = {}
    for role, experiment, stage in (
        ("compute", "exp033", "compute"),
        ("frequencies", "exp041", "analyse"),
    ):
        ref = analysis.record["inputs"][role]
        upstreams[role] = inputs.source(
            REPO, ref["run_id"], stage, experiment=experiment, reference=ref
        )
    if inputs.configuration(upstreams["compute"]) != cfg:
        raise PingstoreError("analysis and compute disagree on the theory recipe")
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": analysis},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        numbers = load_json(analysis.export / "results.json")
        if numbers["config"] != {key: cfg.get(key) for key in numbers["config"]}:
            raise PingstoreError("analysis numbers disagree with the theory recipe")
        coords = evidence.read(analysis.export)
        result = numbers["results"]
        h, crit = result["hopf"], result["criticality"]
        freq = result["frequency_vs_tau_gaba"]
        mf, meas = (
            freq["mean_field"],
            {float(k): v for k, v in freq["spiking_exp041"].items()},
        )
        if h:
            plots.plot_limit_cycle(
                coords["cycle"], run.export / "limit_cycle.svg", run.run_id
            )
            plots.plot_timeseries(
                coords["waveform"], run.export / "timeseries.svg", run.run_id
            )
            plots.plot_phase_planes(
                coords["phase"], run.export / "phase_planes.svg", run.run_id
            )
            plots.plot_reduction_ladder(
                h,
                result["reductions"]["three_d_qss"],
                coords["ladder"],
                run.export / "reduction_ladder.svg",
                run.run_id,
            )
            plots.fig_bifurcation_compound(
                coords["sweep"],
                h,
                crit,
                mf,
                meas,
                run.export / "bifurcation_compound.svg",
                run.run_id,
            )
        plots.plot_sigma_sensitivity(
            result["sigma_sensitivity"],
            run.export / "sigma_sensitivity.svg",
            run.run_id,
        )
        write_json_atomic(run.export / "numbers.json", article_numbers(numbers))
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp033 v4 analysis run"
    )
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
