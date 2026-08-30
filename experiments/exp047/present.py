"""Draw saved exp047 analysis; never simulate, aggregate or publish."""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp047 import evidence, inputs, plots, recipe
from experiments.helpers import theme
from experiments.helpers.fmt import format_duration
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def analysis_source(repo, identity, reference=None):
    source = inputs.source(repo, identity, "analyse", reference=reference)
    refs = source.record["inputs"]
    if (
        set(refs) != {"compute"}
        or source.record["execution"].get("configuration") != recipe.MEASUREMENT
    ):
        raise PingstoreError(
            "exp047 analysis must pin compute and its measurement contract"
        )
    pin = refs["compute"]
    compute = inputs.source(repo, pin["run_id"], "compute", reference=pin)
    cfg = evidence.compute_contract(compute)
    result = evidence.analysis(load_json(source.export / "results.json"), cfg)
    return source, result


def present(identity, *, run_id=None):
    source, result = analysis_source(REPO, identity)
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration={"schema": "exp047.presentation/v1", "figure_run_stamps": False},
    ) as run:
        theme.set_paper_mode(True)
        plots.plot_controls(
            result["summary"], result["config"], run.export / "pool_size_controls"
        )
        duration = time.monotonic() - started
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
    parser.add_argument(
        "--source", required=True, help="explicit completed exp047 analysis ID"
    )
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp047 present: {exc}\n")


if __name__ == "__main__":
    main()
