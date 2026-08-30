"""Draw the four measured/topology figures from saved analysis; never simulate."""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.exp086 import evidence, inputs, plots, recipe
from experiments.helpers.fmt import format_duration
from pingstore.contracts import PingstoreError, write_json_atomic
from snnlang import load_bundle


def present(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "analyse")
    refs = source.record["inputs"]
    if set(refs) != {"compute"}:
        raise PingstoreError("exp086 analysis must pin its compute run")
    pin = refs["compute"]
    compute = inputs.source(REPO, pin["run_id"], "compute", reference=pin)
    cfg = evidence.compute_contract(compute)
    result, trajectories = evidence.analysis(source, cfg)
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration={
            "schema": "exp086.presentation/v2",
            "figures": list(recipe.FIGURES),
            "omitted_theory_schematics": "author-approved; originals unavailable",
            "display_window_ms": recipe.DISPLAY_WINDOW_MS,
        },
    ) as run:
        strong = next(row for row in trajectories if row["k"] == max(recipe.K_VALUES))
        uncoupled = next(row for row in trajectories if row["k"] == 0.0)
        intermediate = next(
            row
            for row in trajectories
            if row["k"] == result["selected_intermediate"]["k"]
        )
        bundle = load_bundle(
            compute.unit("branches", recipe.label(strong["k"]), "network.bundle")
        )
        bundle.visualise(
            run.export / "network.svg", view="circuit", expand_groups=recipe.PING_GROUPS
        )
        plots.plot_uncoupled(uncoupled, run.export / "uncoupled.png")
        plots.plot_coupling_regimes(
            strong,
            intermediate,
            uncoupled,
            run.export / "coupling_regimes_measured.png",
        )
        plots.plot_intermittent_attraction(
            intermediate, run.export / "intermittent_attraction_measured.png"
        )
        duration = time.monotonic() - started
        write_json_atomic(run.export / "protocol.json", result)
        write_json_atomic(
            run.export / "numbers.json",
            {
                **result,
                "run_id": run.run_id,
                "duration_s": duration,
                "duration": format_duration(duration),
                "git_sha": run.record["provenance"]["git_commit"],
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="explicit exp086 analysis ID")
    parser.add_argument("--run-id", help="unused v3 identity reserved before dispatch")
    args = parser.parse_args()
    try:
        present(
            args.source,
            run_id=args.run_id,
        )
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"exp086 present: {exc}\n")


if __name__ == "__main__":
    main()
