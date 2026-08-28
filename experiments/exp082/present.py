"""Render explicit saved analysis and its pinned recordings; no upstream execution."""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp082 import evidence, inputs, plots, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def present(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "analyse")
    if set(source.record["inputs"]) != {"compute", "bank"}:
        raise PingstoreError("analysis must pin compute and bank")
    pin = source.record["inputs"]["compute"]
    compute = inputs.source(REPO, pin["run_id"], "compute", reference=pin)
    cfg, bank, _ = inputs.compute_evidence(REPO, compute)
    if source.record["inputs"]["bank"] != bank.reference:
        raise PingstoreError("analysis bank differs from compute ancestry")
    result = load_json(source.export / "numbers.json")
    expected = [
        {k: j[k] for k in ("seed", "duration_ms", "rate_hz")} for j in recipe.jobs(cfg)
    ]
    if (
        result.get("schema") != "exp082.analysis/v1"
        or [
            {k: r.get(k) for k in ("seed", "duration_ms", "rate_hz")}
            for r in result.get("grid_per_seed", [])
        ]
        != expected
    ):
        raise PingstoreError("analysis grid incomplete or inconsistent")
    arrays = evidence.arrays(source.export / "display.npz")
    streams = {}
    for name in ("matched", "variable"):
        raw, _ = evidence.stream(compute.export, name)
        streams[name] = {**raw, **result[name + "_stream"]}
    index = result["single_trial_segment_index"]
    matched = streams["matched"]
    if (
        type(index) is not int
        or not 0 <= index < 5
        or matched["correct"][index] != 1
        or any(matched["correct"][:index])
    ):
        raise PingstoreError("analysis explanatory-trial selection differs")
    start, stop = matched["boundaries"][index : index + 2]
    streams["single_trial"] = {
        **result["single_trial"],
        "pixels": matched["pixels"][index : index + 1],
        **{k: matched[k][start:stop] for k in ("spikes_e", "spikes_i", "spikes_out")},
    }
    for name, stream in streams.items():
        for key in ("probabilities", "counts", "final_counts"):
            stream[key] = arrays[name + "_" + key]
        n = len(stream["spikes_out"])
        if (
            stream["probabilities"].shape != (n, 10)
            or stream["counts"].shape != (n, 10)
            or stream["final_counts"].shape != (10,)
        ):
            raise PingstoreError("analysis display dimensions differ")
        if not np.isfinite(stream["probabilities"]).all():
            raise PingstoreError("nonfinite display values")
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration={"schema": "exp082.presentation/v1"},
    ) as run:
        out, rid = run.export, run.run_id
        plots.plot_single_trial(streams["single_trial"], out / "single_trial.png", rid)
        plots.plot_single_trial_transition(
            streams["single_trial"], out / "single_trial_transition.png", rid
        )
        plots.plot_stream(streams["matched"], out / "matched_stream.png", rid)
        plots.plot_variable_headline(
            streams["variable"], out / "variable_stream.png", rid
        )
        plots.plot_psychometric(
            result["plot_data"], out / "psychometric_200ms.svg", rid
        )
        plots.plot_duration_rate_summary(
            result["plot_data"], out / "duration_rate_summary.png", rid
        )
        plots.plot_design(out / "shared_design_schematic.svg")
        write_json_atomic(out / "numbers.json", {**result, "run_id": rid})
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        present(a.source, run_id=a.run_id)
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        p.exit(1, f"exp082 present: {exc}\n")


if __name__ == "__main__":
    main()
