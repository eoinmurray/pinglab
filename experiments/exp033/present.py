"""Render saved measurements and coordinates; never solve or measure the model."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp033 import appearance, evidence, inputs, plots
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)


def present(identity, *, run_id=None):
    analysis = inputs.source(REPO, identity, "analyse")
    inputs.configuration(analysis)
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
    with inputs.execution(
        REPO, "present", sources={"analysis": analysis}, run_id=run_id
    ) as run:
        numbers = load_json(analysis.export / "results.json")
        coords = evidence.read(analysis.export)
        result = numbers["results"]
        h, crit = result["hopf"], result["criticality"]
        freq = result["frequency_vs_tau_gaba"]
        mf, meas = (
            freq["mean_field"],
            {float(k): v for k, v in freq["spiking_exp041"].items()},
        )
        if h:
            if "retained_figures" in coords:
                retained = coords["retained_figures"]
                if set(retained) != set(evidence.CARRY):
                    raise PingstoreError("historical figure selection is incomplete")
                source = upstreams["compute"]
                if source.record.get("historical_import", {}).get(
                    "carry_forward_figures"
                ) != list(evidence.CARRY):
                    raise PingstoreError("unapproved historical figure source")
                for name, item in retained.items():
                    if (
                        item["source"] != source.reference
                        or item["path"] != "retained-figures/" + name
                    ):
                        raise PingstoreError("historical figure pin or path differs")
                    path = source.export / item["path"]
                    if file_sha256(path) != item["sha256"]:
                        raise PingstoreError("historical figure checksum differs")
                    labels = {
                        "timeseries.svg": (
                            ("A", 39, 18), ("B", 39, 127),
                            ("C", 39, 236), ("D", 39, 345),
                        ),
                        "phase_planes.svg": (
                            ("A", 39, 18), ("B", 304, 18), ("C", 569, 18),
                            ("D", 39, 249), ("E", 304, 249), ("F", 569, 249),
                        ),
                    }.get(name, ())
                    operations = appearance.historical_svg(
                        path,
                        run.export / name,
                        move_legend=name == "reduction_ladder.svg",
                        panel_labels=labels,
                    )
                    run.record.setdefault("figure_edits", {})[name] = {
                        "source_sha256": item["sha256"],
                        "output_sha256": file_sha256(run.export / name),
                        "operations": operations,
                    }
                run.record["retained_figure_sources"] = retained
            else:
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
        write_json_atomic(
            run.export / "numbers.json", appearance.article_numbers(numbers)
        )
        run.record["presentation_revision"] = {
            "scientific_measurements_changed": False,
            "changes": [
                "qualify criticality and QSS criterion labels",
                "absolute-Hz sensitivity axis; remove internal figure labels",
                "historical SVG display edits with source and output hashes",
            ],
        }
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
