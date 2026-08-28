"""Render saved exp038 analysis; never simulate, aggregate or publish."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp038 import inputs, plots, recipe
from experiments.exp038.analyse import MEASUREMENT
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
        or result.get("schema") != "exp038.analysis/v1"
        or result.get("recipe") != cfg
        or result.get("measurement") != MEASUREMENT
        or source.record["execution"].get("configuration") != MEASUREMENT
        or result.get("checkpoint_policy") != recipe.CHECKPOINT_POLICY
        or result.get("checkpoint_provenance") != contract["checkpoints"]
    ):
        raise PingstoreError("analysis evidence or bank pin differs")
    jobs = [j for j in recipe.jobs(cfg) if "sample_index" in j]
    if result.get("rasters") != [
        {"job": j, "file": f"raster-{i:03d}.npz"} for i, j in enumerate(jobs)
    ]:
        raise PingstoreError("analysis raster grid differs")
    for key, count in (
        ("baseline_results", 36),
        ("frontier_summary", 12),
        ("ei_sweep", len(cfg["ei_strengths"]) * 3),
        ("ei_sweep_summary", len(cfg["ei_strengths"])),
        ("fi_sweep_uniform", len(cfg["uniform_rates"]) * 2),
    ):
        if not isinstance(result.get(key), list) or len(result[key]) != count:
            raise PingstoreError(f"analysis grid incomplete: {key}")
    rates, ei = [], []
    for entry in result["rasters"]:
        with np.load(source.export / entry["file"], allow_pickle=False) as raw:
            data = {
                k: np.array(raw[k]) if raw[k].ndim else raw[k].item() for k in raw.files
            }
        (rates if entry["job"]["kind"] == "rate_raster" else ei).append(data)
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration={
            "schema": "exp038.presentation/v1",
            "labels": "recorded image class; bidirectional loop strength; explicit population-mean sum",
        },
    ) as run:
        theme.set_paper_mode(True)
        out, rid = run.export, run.run_id
        plots.plot_rate_rasters(rates, out / "rate_rasters__ping", rid)
        plots.plot_fi_curve(rates, out / "fi_curve__ping", rid)
        plots.plot_fi_curve_uniform(
            result["plot_data"]["uniform"], out / "fi_curve_uniform", rid
        )
        plots.plot_ei_rasters(ei, out / "ei_rasters", rid)
        plots.fig_loop_transfer_compound(
            result["ei_sweep_summary"],
            ei[0],
            ei[-1],
            out / "loop_transfer_compound",
            rid,
        )
        duration = time.monotonic() - started
        write_json_atomic(
            out / "numbers.json",
            {
                **{key: value for key, value in result.items() if key != "rasters"},
                "illustrative_labels": {
                    "rate_rasters": [sample["label"] for sample in rates],
                    "ei_rasters": [sample["label"] for sample in ei],
                },
                "run_id": rid,
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
        p.exit(1, f"exp038 present: {exc}\n")


if __name__ == "__main__":
    main()
