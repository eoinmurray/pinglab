"""Render saved exp049 analysis; never simulate, aggregate or publish."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp049 import inputs, plots, recipe
from experiments.exp049.analyse import MEASUREMENT
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
        or result.get("schema") != "exp049.analysis/v1"
        or result.get("recipe") != cfg
        or result.get("measurement") != MEASUREMENT
        or source.record["execution"].get("configuration") != MEASUREMENT
        or result.get("checkpoint_policy") != recipe.CHECKPOINT_POLICY
        or result.get("checkpoint_provenance") != contract["checkpoints"]
    ):
        raise PingstoreError("analysis evidence or bank pin differs")
    expected = [
        {"condition": c, "seed": seed}
        for c in recipe.COND_ORDER
        for seed in recipe.SEEDS
    ]
    if [
        {k: r.get(k) for k in ("condition", "seed")} for r in result.get("summary", [])
    ] != expected:
        raise PingstoreError("analysis summary grid incomplete")
    if result.get("rasters") != [
        {"condition": c, "file": f"raster-{c}.npz"} for c in recipe.COND_ORDER
    ]:
        raise PingstoreError("analysis raster grid differs")
    for key in ("cards", "weights", "attractor", "trajectories"):
        if set(result.get("plot_data", {}).get(key, {})) != set(recipe.COND_ORDER):
            raise PingstoreError("analysis plot grid incomplete")
    rasters = {}
    for entry in result["rasters"]:
        with np.load(source.export / entry["file"], allow_pickle=False) as raw:
            rasters[entry["condition"]] = {
                k: np.array(raw[k]) if raw[k].ndim else raw[k].item() for k in raw.files
            }
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration={
            "schema": "exp049.presentation/v1",
            "labels": "official-test endpoints; validation trajectories; reference-image contrast; no inferred basin boundary",
        },
    ) as run:
        theme.set_paper_mode(True)
        out, rid = run.export, run.run_id
        data = result["plot_data"]
        for cond in recipe.COND_ORDER:
            plots.plot_condition_card(
                cond, data["cards"][cond], rasters[cond], out / ("card__" + cond), rid
            )
            plots.plot_weight_matrices(
                cond, data["weights"][cond], out / ("weights__" + cond), rid
            )
        plots.fig_attractor(data["attractor"], out / "attractor_ei", rid)
        plots.fig_training_curves(data, out / "training_curves", rid)
        plots.fig_phase_portrait(data, out / "phase_portrait", rid)
        plots.fig_acc_rate_trajectory(data, out / "acc_rate_trajectory", rid)
        duration = time.monotonic() - started
        write_json_atomic(
            out / "numbers.json",
            {
                **{key: value for key, value in result.items() if key != "rasters"},
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
        p.exit(1, f"exp049 present: {exc}\n")


if __name__ == "__main__":
    main()
