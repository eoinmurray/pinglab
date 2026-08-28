"""Render saved exp025 analysis; never simulate, aggregate or publish."""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp025 import inputs, plots, recipe
from experiments.exp025.analyse import MEASUREMENT
from experiments.helpers import theme
from experiments.helpers.fmt import format_duration
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def present(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "analyse")
    refs = source.record["inputs"]
    if set(refs) != {"compute", "bank"}:
        raise PingstoreError("analysis must pin compute and bank")
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    cfg, bank, contract = inputs.compute_evidence(REPO, compute)
    result = load_json(source.export / "results.json")
    if (
        refs["bank"] != bank.reference
        or result.get("schema") != "exp025.analysis/v1"
        or result.get("recipe") != cfg
        or result.get("measurement") != MEASUREMENT
        or source.record["execution"].get("configuration") != MEASUREMENT
    ):
        raise PingstoreError("analysis evidence or bank pin differs")
    checkpoints = [
        c for group in result["training_sources"].values() for c in group["checkpoints"]
    ]
    names = {
        c["cell_name"]
        for c in recipe.bank_cells()
        if c["group"] == "shared_tr02" or c["seed"] in cfg["low_w_in_seeds"]
    }
    expected = [c for c in contract["checkpoints"] if c["training_cell"] in names]
    if sorted(checkpoints, key=lambda c: c["training_cell"]) != sorted(
        expected, key=lambda c: c["training_cell"]
    ):
        raise PingstoreError("analysis checkpoint evidence differs")
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration={
            "schema": "exp025.presentation/v1",
            "scientific_review": "deferred",
        },
    ) as run:
        theme.set_paper_mode(True)
        out = run.export
        rid = run.run_id
        plots.plot_rate_target_p_fgamma(
            result["rate_target_p_fgamma"], out / "theta_p_fgamma", rid
        )
        for model in ("coba", "ping"):
            plots.render_raster(
                source.export / f"raster__{model}.npz",
                out / f"raster__{model}",
                f"{model} — trained network, MNIST digit 0, 400 ms",
            )
        plots.fig_results_compound(
            result["frontier_statistics"],
            result["plot_data"]["baseline"],
            source.export / "raster__coba.npz",
            source.export / "raster__ping.npz",
            out / "results_compound",
            rid,
        )
        plots.plot_low_w_in(
            result["low_w_in_sweep"],
            result["plot_data"]["low_w_in"],
            out / "low_w_in_sweep",
            rid,
        )
        plots.plot_w_in_scale_sweep(
            result["w_in_scale_sweep"],
            result["plot_data"]["scale_crossing"],
            out / "w_in_scale_sweep",
            rid,
        )
        plots.plot_w_in_scale_sweep_vs_rate(
            result["w_in_scale_sweep"], out / "w_in_scale_sweep_vs_rate", rid
        )
        duration = time.monotonic() - started
        write_json_atomic(
            out / "numbers.json",
            {
                **result,
                "run_id": rid,
                "notebook_run_id": rid,
                "duration_s": duration,
                "duration": format_duration(duration),
                "git_sha": run.record["provenance"]["git_commit"],
            },
        )
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        present(a.source, run_id=a.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        p.exit(1, f"exp025 present: {exc}\n")


if __name__ == "__main__":
    main()
