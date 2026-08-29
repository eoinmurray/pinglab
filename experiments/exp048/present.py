"""Draw explicit saved analysis; never launch inference, analysis or publication."""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp048 import evidence, inputs, plots, recipe
from experiments.exp048.analyse import MEASUREMENT
from experiments.helpers import theme
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def present(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "analyse")
    if source.record["execution"].get("configuration") == evidence.ANALYSIS:
        return evidence.present_retained(REPO, source, run_id=run_id)
    refs = source.record["inputs"]
    if set(refs) != {"compute", "bank"}:
        raise PingstoreError("analysis must pin compute and bank")
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    bank, contract = inputs.compute_evidence(REPO, compute)
    result = load_json(source.export / "results.json")
    if (
        refs["bank"] != bank.reference
        or result.get("schema") != "exp048.analysis/v1"
        or result.get("measurement") != MEASUREMENT
        or result.get("config") != recipe.configuration()
        or result.get("checkpoint_provenance") != contract["checkpoints"]
        or source.record["execution"].get("configuration") != MEASUREMENT
    ):
        raise PingstoreError("analysis recipe, measurements or checkpoint pins differ")
    evidence.analysis_rows(result)
    figures = {}
    for job in recipe.jobs()[:2]:
        kind = job["kind"]
        data = evidence.load_arrays(source.export / f"{kind}.npz")
        evidence.analysis_figure(
            data, job, result["headline" if kind == "headline" else "varying_headline"]
        )
        figures[kind] = {k: a.item() if a.ndim == 0 else a for k, a in data.items()}
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration={"schema": "exp048.presentation/v1"},
    ) as run:
        theme.set_paper_mode(True)
        plots.plot_headline_stream(
            figures["headline"], run.export / "headline_stream", run.run_id
        )
        plots.plot_varying_headline_stream(
            figures["varying"], run.export / "varying_headline_stream", run.run_id
        )
        plots.plot_acc_vs_tau(
            result["tau_sweep_agg"], run.export / "acc_vs_tau", run.run_id
        )
        plots.plot_grid_and_rate(
            result["grid_sweep_agg"],
            result["encoding_rate_psychometric"]["curve"],
            run.export / "acc_grid_tau_rate",
            run.run_id,
        )
        write_json_atomic(
            run.export / "numbers.json",
            {
                **result,
                "notebook_run_id": run.run_id,
                "run_id": run.run_id,
                "duration_s": round(time.monotonic() - started, 1),
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp048 present: {exc}\n")


if __name__ == "__main__":
    main()
