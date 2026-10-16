"""Keep the synchrony figures and export signal-readout measurements."""

import shutil

from experiments.exp136 import recipe
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab.lang import load_bundle


def present_attention(repo, identity, *, run_id=None):
    analysis = source_run(
        repo / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    result = load_json(analysis.file("results.json"))
    cfg = result["configuration"]
    if result.get("schema") != "exp136.attention-analysis/v1":
        raise ValueError("requires a signal-routing analysis")
    pin = analysis.record["inputs"]["compute"]
    compute = source_run(
        repo / ".pingstore",
        pin["run_id"],
        stage="compute",
        experiment=recipe.SLUG,
        reference=pin,
    )
    baseline = source_run(
        repo / ".pingstore",
        cfg["baseline_presentation"],
        stage="present",
        experiment=recipe.SLUG,
    )
    with stage_run(
        repo,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "synchrony_reference": baseline},
        configuration=dict(kind="held-out signal decoding with synchrony reference"),
        run_id=run_id,
    ) as run:
        for condition in ("coupled", "zero"):
            for suffix in ("svg", "png"):
                name = f"continuous-{condition}.{suffix}"
                shutil.copy2(baseline.file(name), run.export / name)
        bundle = load_bundle(
            compute.restore_bundle("network.bundle", run.scratch / "network.bundle")
        )
        for suffix in ("svg", "png"):
            bundle.visualise(
                run.export / f"network-circuit.{suffix}", view="circuit", scale=2
            )
        write_json_atomic(
            run.export / "numbers.json",
            dict(attention=result, synchrony=load_json(baseline.file("numbers.json"))),
        )
    return run.run_id
