"""Explicit coarse/fine routing searches and independent confirmation stages."""

import numpy as np
import torch
from experiments.exp136 import recipe
from experiments.exp136.attention_analyse import measure_attention
from experiments.exp136.attention_compute import execute_attention
from experiments.helpers.network_artifacts import snapshot_parameters
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab.sim import ExecutionSpec, build


def scaled_parameters(values, cfg):
    """Preserve realised masks/local weights; scale only fixed pathway totals."""
    result = {k: v.copy() for k, v in values.items()}
    for sender in ("S1", "S2"):
        for pop in ("E", "I"):
            key = f"{sender}_E_to_R_{pop}.weight"
            if not np.allclose(values[key].sum(axis=0), recipe.FORWARD_US):
                raise ValueError("source must contain the Fig. 1–2 pathway totals")
            result[key] *= cfg["forward_strength_us"] / recipe.FORWARD_US
        key = f"cue_{sender}_E_to_{sender}_E.weight"
        if not np.allclose(values[key].sum(axis=0), recipe.CUE_TO_E_US):
            raise ValueError("source must contain the Fig. 1–2 cue totals")
        result[key] *= cfg["cue_to_e_strength_us"] / recipe.CUE_TO_E_US
    return result


def candidate_quality(result):
    """Lexicographic selection prevents apparently selective but unreadable signals."""
    local = min(v["mean_r2"] for v in result["local_encoding"].values())
    scores = np.asarray(result["conditions"]["coupled"]["mean_r2"])
    gain = np.diag(scores) - np.diag(scores[::-1])
    readable = local >= 0.1
    return (
        int(readable),
        int(readable and min(np.diag(scores)) > 0),
        float(min(gain)) if readable else float(local),
        float(min(np.diag(scores))),
        float(local),
    )


def compute_search(repo, identity, level, selection=None, *, run_id=None):
    torch.set_num_threads(1)
    parameters = source_run(
        repo / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    inputs = {"parameters": parameters}
    if level != "coarse":
        if selection is None:
            raise ValueError(
                "fine/confirm requires --selection naming an explicit analyse run"
            )
        selected = source_run(
            repo / ".pingstore", selection, stage="analyse", experiment=recipe.SLUG
        )
        previous = load_json(selected.file("results.json"))
        if previous.get("schema") != "exp136.routing-search-analysis/v1":
            raise ValueError("selection must be a completed routing search analysis")
        expected = "coarse" if level == "fine" else "fine"
        if previous["level"] != expected:
            raise ValueError(f"{level} requires a {expected} selection")
        if selected.record["inputs"]["parameters"] != parameters.reference:
            raise ValueError("parameter ancestry differs from the search")
        inputs["selection"] = selected
        centre = previous["selected"]["candidate"]
    else:
        centre = None
    definitions = (
        [dict(centre, candidate="confirmation")]
        if level == "confirm"
        else recipe.routing_candidates(level, centre)
    )
    datasets = [
        recipe.routing_configuration(c, confirmation=level == "confirm")
        for c in definitions
    ]
    cfg = dict(
        schema="exp136.routing-search/v1",
        level=level,
        candidates=definitions,
        datasets=datasets,
        baseline_presentation="exp136-r075-present",
        selection_rule="lexicographic: both local R2 >= .1; positive cued R scores; minimum same-signal cue gain; minimum cued R2; minimum local R2; earlier candidate wins ties",
        search_axes=recipe.SEARCH_AXES,
    )
    values = snapshot_parameters(parameters, recipe.PARAMETER_UNIT, "start")
    bundle = recipe.author_network()
    model = build(
        ExecutionSpec(
            kind="build", graph=bundle.graph, seed=recipe.NETWORK_SEED, device="cpu"
        )
    ).model
    if set(model.parameter_map()) != set(values):
        raise ValueError("retained parameters do not cover the fixed circuit")
    with stage_run(
        repo, recipe.SLUG, "compute", inputs=inputs, configuration=cfg, run_id=run_id
    ) as run:
        bundle.write(run.export / "network.bundle", visualise=True)
        for index, (candidate, dataset) in enumerate(zip(definitions, datasets)):
            print(
                f"{level} candidate {index + 1}/{len(definitions)}: {candidate}",
                flush=True,
            )
            execute_attention(
                run, model, bundle, scaled_parameters(values, dataset), dataset
            )
        run.record["execution"]["paired_parameters_and_inputs_verified"] = True
    return run.run_id


def analyse_search(repo, identity, *, run_id=None):
    source = source_run(
        repo / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    cfg = source.record["execution"]["configuration"]
    if cfg.get("schema") != "exp136.routing-search/v1" or not source.record[
        "execution"
    ].get("paired_parameters_and_inputs_verified"):
        raise ValueError("requires verified routing-search compute evidence")
    pin = source.record["inputs"]["parameters"]
    parameters = source_run(
        repo / ".pingstore",
        pin["run_id"],
        stage="compute",
        experiment=recipe.SLUG,
        reference=pin,
    )
    inputs = dict(compute=source, parameters=parameters)
    with stage_run(
        repo,
        recipe.SLUG,
        "analyse",
        inputs=inputs,
        configuration=dict(level=cfg["level"], selection_rule=cfg["selection_rule"]),
        run_id=run_id,
    ) as run:
        rows = []
        for candidate, dataset in zip(cfg["candidates"], cfg["datasets"]):
            result, models, predictions = measure_attention(source, dataset)
            row = dict(
                candidate=candidate, quality=candidate_quality(result), result=result
            )
            rows.append(row)
            prefix = candidate["candidate"]
            np.savez_compressed(run.export / f"{prefix}--decoders.npz", **models)
            np.savez_compressed(
                run.export / f"{prefix}--predictions.npz", **predictions
            )
            print(
                f"{prefix}: local {[round(v['mean_r2'], 3) for v in result['local_encoding'].values()]}; cue gain {result['conditions']['coupled']['mean_cue_gain']}",
                flush=True,
            )
        ranked = sorted(rows, key=lambda r: tuple(r["quality"]), reverse=True)
        output = dict(
            schema="exp136.routing-search-analysis/v1",
            level=cfg["level"],
            candidates=rows,
            selected=ranked[0],
            selection_rule=cfg["selection_rule"],
            baseline_presentation=cfg["baseline_presentation"],
        )
        write_json_atomic(run.export / "results.json", output)
    return run.run_id
