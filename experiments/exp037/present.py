"""Render saved exp037 analysis; never simulate, aggregate or publish."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp037 import inputs, plots, recipe
from experiments.exp037.analyse import MEASUREMENT
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
        or result.get("schema") != "exp037.analysis/v1"
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
    expected_grid = [
        {k: j[k] for k in ("model", "seed", "mode", "level")}
        for j in recipe.jobs(cfg)
        if j["kind"] == "sweep"
    ]
    if [
        {k: r.get(k) for k in ("model", "seed", "mode", "level")}
        for r in result.get("perturbation", [])
    ] != expected_grid:
        raise PingstoreError("analysis perturbation grid incomplete")
    if (
        len(result.get("baseline_results", [])) != 36
        or len(result.get("frontier_summary", [])) != 12
        or len(result.get("perturbation_summary", []))
        != len(expected_grid) // len(cfg["seeds"])
    ):
        raise PingstoreError("analysis summary grid incomplete")
    rasters = []
    for entry in result["rasters"]:
        with np.load(source.export / entry["file"], allow_pickle=False) as raw:
            rasters.append(
                {
                    k: np.array(raw[k]) if raw[k].ndim else raw[k].item()
                    for k in raw.files
                }
            )
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration={
            "schema": "exp037.presentation/v2",
            "labels": "Bernoulli deletion/insertion; final-epoch reference E normalization",
            "range": "complete saved perturbation grid",
            "figure_run_stamps": False,
        },
    ) as run:
        theme.set_paper_mode(True)
        out, rid = run.export, run.run_id
        plots.plot_perturbation_curves(
            result["plot_data"], out / "perturbation_curves", rid
        )
        for model in recipe.MODELS:
            for mode in ("drop", "add"):
                samples = [
                    r for r in rasters if r["model"] == model and r["mode"] == mode
                ]
                plots.plot_perturbation_rasters(
                    samples,
                    out / f"perturb_rasters__{mode}__{model}",
                    rid,
                    level_fmt="p(drop) = {level:.1f}"
                    if mode == "drop"
                    else "r(add) = {level:g} Hz",
                    title="",
                )
        duration = time.monotonic() - started
        write_json_atomic(
            out / "numbers.json",
            {
                **{key: value for key, value in result.items() if key != "rasters"},
                "illustrative_labels": [r["label"] for r in rasters],
                "notebook_run_id": rid,
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
        p.exit(1, f"exp037 present: {exc}\n")


if __name__ == "__main__":
    main()
