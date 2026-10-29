"""Aggregate explicit completed computation; never simulate, draw or publish.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import argparse

import numpy as np
from experiments.exp047 import recipe
from experiments.exp047.custom import evidence, inputs
from pingstore.library import PingstoreError, write_json_atomic


def analyse(identity, *, run_id=None):
    compute = inputs.source(REPO, identity, "compute")
    cfg = evidence.compute_contract(compute)
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": compute},
        run_id=run_id,
        configuration=recipe.MEASUREMENT,
    ) as run:
        rows = evidence.rows(compute.export, cfg)
        result = analyse_rows(rows, cfg)
        evidence.analysis(result, cfg)
        write_json_atomic(run.export / "results.json", result)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="explicit completed exp047 compute ID"
    )
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp047 analyse: {exc}\n")


def summarise(rows):
    result = {}
    for key in ("r_e_hz", "r_i_hz"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_sd"] = float(values.std(ddof=1))
    return result


def analyse_rows(rows, cfg):
    raw, summary = (
        {"fixed_total": {}, "fixed_synapse": {}},
        {"fixed_total": {}, "fixed_synapse": {}},
    )
    for control, level, n_i, g_ie in recipe.conditions(cfg):
        selected = [rows[recipe.job(n_i, g_ie, seed)["id"]] for seed in cfg["seeds"]]
        raw[control].setdefault(level, {})[str(n_i)] = selected
        summary[control].setdefault(level, {})[str(n_i)] = summarise(selected)
    return {
        "schema": "exp047.analysis/v1",
        "recipe": cfg,
        "measurement": recipe.MEASUREMENT,
        "config": recipe.report_config(cfg),
        "definition": recipe.DEFINITION,
        "raw": raw,
        "summary": summary,
    }


if __name__ == "__main__":
    main()
