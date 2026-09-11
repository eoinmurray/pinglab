"""Render saved exp025 analysis; never simulate, aggregate or publish."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp025 import inputs, plots
from experiments.exp025.analyse import MEASUREMENT, MEASUREMENT_V1
from experiments.helpers import theme
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def present(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "analyse")
    refs = source.record["inputs"]
    if set(refs) != {"compute", "bank"}:
        raise PingstoreError("analysis must pin compute and bank")
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    cfg, bank, _ = inputs.compute_evidence(REPO, compute)
    result = load_json(source.export / "results.json")
    analysis_schema = result.get("schema")
    expected_measurement = (
        MEASUREMENT_V1 if analysis_schema == "exp025.analysis/v1" else MEASUREMENT
    )
    if (
        refs["bank"] != bank.reference
        or analysis_schema not in ("exp025.analysis/v1", "exp025.analysis/v2")
        or result.get("recipe") != cfg
        or source.record["execution"].get("configuration") != expected_measurement
        or (
            analysis_schema == "exp025.analysis/v1"
            and result.get("measurement") != MEASUREMENT_V1
        )
    ):
        raise PingstoreError("analysis evidence or bank pin differs")
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration={
            "schema": "exp025.presentation/v2",
            "scientific_review": "deferred",
        },
    ) as run:
        theme.set_paper_mode(True)
        out = run.export
        plots.plot_rate_target_p_fgamma(
            result["rate_target_p_fgamma"], out / "theta_p_fgamma"
        )
        plots.fig_results_compound(
            result["frontier_statistics"],
            result["plot_data"]["baseline"],
            source.export / "raster__coba.npz",
            source.export / "raster__ping.npz",
            out / "results_compound",
        )
        plots.plot_low_w_in(
            result["low_w_in_sweep"],
            result["plot_data"]["low_w_in"],
            out / "low_w_in_sweep",
        )
        plots.plot_w_in_scale_sweep(
            result["w_in_scale_sweep"],
            result["plot_data"]["scale_crossing"],
            out / "w_in_scale_sweep",
        )
        plots.plot_w_in_scale_sweep_vs_rate(
            result["w_in_scale_sweep"], out / "w_in_scale_sweep_vs_rate"
        )
        write_json_atomic(out / "numbers.json", result)
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
