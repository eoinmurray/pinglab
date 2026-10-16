"""Receiver-drive/coupling refinement after a fixed readout audit."""

import numpy as np
import torch
from experiments.exp136 import recipe
from experiments.exp136.attention_analyse import measure_attention
from experiments.exp136.attention_compute import execute_attention
from experiments.exp136.readout_audit import reconstruction_quality
from experiments.exp136.routing_search import scaled_parameters
from experiments.helpers.network_artifacts import snapshot_parameters
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab.sim import ExecutionSpec, build
from threadpoolctl import threadpool_limits


def compute_refinement(repo, identity, selection, *, confirmation=False, run_id=None):
    torch.set_num_threads(1)
    if selection is None:
        raise ValueError("receiver refinement requires an explicit analysis selection")
    parameters = source_run(
        repo / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    selected = source_run(
        repo / ".pingstore", selection, stage="analyse", experiment=recipe.SLUG
    )
    result = load_json(selected.file("results.json"))
    ancestry_pin = selected.record["inputs"]["compute"]
    ancestry = source_run(
        repo / ".pingstore",
        ancestry_pin["run_id"],
        stage="compute",
        experiment=recipe.SLUG,
        reference=ancestry_pin,
    )
    if ancestry.record["inputs"]["parameters"] != parameters.reference:
        raise ValueError("parameter source differs from the selected wiring")
    if confirmation:
        if (
            result.get("schema") != "exp136.receiver-analysis/v1"
            or result["level"] != "refine"
        ):
            raise ValueError("confirmation requires a receiver-refinement selection")
        definitions = [
            dict(result["selected"]["network"], candidate="refined-confirmation")
        ]
        readout = result["readout"]
    else:
        if result.get("schema") != "exp136.readout-audit-analysis/v1":
            raise ValueError("receiver search requires an explicit readout audit")
        definitions = recipe.receiver_candidates(result["network"])
        readout = result["selected"]["readout"]
    datasets = [
        recipe.receiver_configuration(n, readout, confirmation=confirmation)
        for n in definitions
    ]
    cfg = dict(
        schema="exp136.receiver-refine/v1",
        level="confirm" if confirmation else "refine",
        candidates=definitions,
        datasets=datasets,
        readout=readout,
        baseline_presentation="exp136-r082-present",
        selection_rule="readable senders >= .1, positive cue gains, then minimum cued receiver R2, minimum cue gain, minimum local R2, earlier setting wins ties",
    )
    values = snapshot_parameters(parameters, recipe.PARAMETER_UNIT, "start")
    bundle = recipe.author_network()
    model = build(
        ExecutionSpec(
            kind="build", graph=bundle.graph, seed=recipe.NETWORK_SEED, device="cpu"
        )
    ).model
    if set(values) != set(model.parameter_map()):
        raise ValueError("parameter source does not cover the fixed circuit")
    with stage_run(
        repo,
        recipe.SLUG,
        "compute",
        inputs=dict(parameters=parameters, selection=selected),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        bundle.write(run.export / "network.bundle", visualise=True)
        for index, (candidate, dataset) in enumerate(zip(definitions, datasets)):
            print(f"Receiver {index + 1}/{len(definitions)}: {candidate}", flush=True)
            execute_attention(
                run, model, bundle, scaled_parameters(values, dataset), dataset
            )
        run.record["execution"]["paired_parameters_and_inputs_verified"] = True
    return run.run_id


def analyse_refinement(repo, identity, *, run_id=None):
    source = source_run(
        repo / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    cfg = source.record["execution"]["configuration"]
    if cfg.get("schema") != "exp136.receiver-refine/v1" or not source.record[
        "execution"
    ].get("paired_parameters_and_inputs_verified"):
        raise ValueError("requires verified receiver-refinement compute evidence")
    with (
        stage_run(
            repo,
            recipe.SLUG,
            "analyse",
            inputs=dict(compute=source),
            configuration=dict(
                level=cfg["level"], selection_rule=cfg["selection_rule"]
            ),
            run_id=run_id,
        ) as run,
        threadpool_limits(limits=1),
    ):
        rows = []
        best = None
        for network, dataset in zip(cfg["candidates"], cfg["datasets"]):
            result, models, predictions = measure_attention(source, dataset)
            if cfg["level"] == "refine":
                result["samples"]["validation"] = result["samples"].pop("test")
            row = dict(
                network=network, quality=reconstruction_quality(result), result=result
            )
            rows.append(row)
            if best is None or tuple(row["quality"]) > tuple(best["quality"]):
                best = row
                np.savez_compressed(run.export / "decoders.npz", **models)
                np.savez_compressed(run.export / "predictions.npz", **predictions)
            print(f"{network['candidate']}: {row['quality']}", flush=True)
        write_json_atomic(
            run.export / "results.json",
            dict(
                schema="exp136.receiver-analysis/v1",
                level=cfg["level"],
                candidates=rows,
                selected=best,
                readout=cfg["readout"],
                baseline_presentation=cfg["baseline_presentation"],
                selection_rule=cfg["selection_rule"],
            ),
        )
    return run.run_id
