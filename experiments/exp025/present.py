"""Render saved exp025 analysis; never simulate, aggregate or publish.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp025.custom.network_artifacts import copy_circuits
from experiments.exp025.custom import inputs, plots
from experiments import theme
from pingstore.contracts import PingstoreError, write_json_atomic


def present(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "analyse")
    result = inputs.analysis_evidence(REPO, source)
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
        write_json_atomic(
            out / "numbers.json", {**(result), "circuit_files": copy_circuits(run)}
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
