"""Select a readout on exposed development trials, never on the next test set."""

from copy import deepcopy

import numpy as np
from experiments.exp136 import recipe
from experiments.exp136.attention_analyse import attention_data, measure_attention
from pingstore.contracts import write_json_atomic
from pingstore.stages import source_run, stage_run
from threadpoolctl import threadpool_limits


def reconstruction_quality(result):
    """Require readable senders and both cue benefits, then optimise reconstruction."""
    local = min(v["mean_r2"] for v in result["local_encoding"].values())
    c = result["conditions"]["coupled"]
    score = min(np.diag(np.asarray(c["mean_r2"])))
    gain = min(c["mean_cue_gain"])
    readable = local >= 0.1
    return (
        int(readable),
        int(readable and gain > 0),
        float(score),
        float(gain),
        float(local),
    )


def audit_configuration(dataset, readout):
    cfg = deepcopy(dataset)
    cfg["decoder"].update(readout)
    cfg["selection"] = dict(
        split="development validation; previous test trials already inspected",
        local_min_r2=0.1,
        rule="readable senders, positive cue gains, then minimum cued receiver R2",
    )
    cfg["limitation"] = (
        "readout selection on exposed development trials; requires new test sequences"
    )
    return cfg


def analyse_readouts(repo, identity, *, run_id=None):
    source = source_run(
        repo / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    raw = source.record["execution"]["configuration"]
    if raw.get("schema") != "exp136.routing-search/v1" or raw["level"] != "confirm":
        raise ValueError("audit requires the explicit earlier confirmation compute run")
    if len(raw["datasets"]) != 1:
        raise ValueError("audit holds exactly one network setting fixed")
    candidates = recipe.readout_candidates()
    configuration = dict(
        schema="exp136.readout-audit/v1",
        candidates=candidates,
        development_split="previous test pairs are now validation, not untouched tests",
        selection_rule="readable senders >= .1 and positive cue gain for both, then minimum cued R2, minimum cue gain, minimum local R2, earlier readout wins ties",
    )
    with (
        stage_run(
            repo,
            recipe.SLUG,
            "analyse",
            inputs=dict(compute=source),
            configuration=configuration,
            run_id=run_id,
        ) as run,
        threadpool_limits(limits=1),
    ):
        rows = []
        data = None
        feature_key = None
        best = None
        for candidate in candidates:
            cfg = audit_configuration(raw["datasets"][0], candidate)
            key = (candidate["feature_mode"], candidate["bin_ms"])
            if key != feature_key:
                data = attention_data(source, cfg)
                feature_key = key
            result, models, predictions = measure_attention(source, cfg, data=data)
            result["samples"]["validation"] = result["samples"].pop("test")
            row = dict(
                readout=candidate, quality=reconstruction_quality(result), result=result
            )
            rows.append(row)
            if best is None or tuple(row["quality"]) > tuple(best["quality"]):
                best = row
                np.savez_compressed(run.export / "decoders.npz", **models)
                np.savez_compressed(run.export / "predictions.npz", **predictions)
            print(
                f"{candidate['readout']}: min cued R2={row['quality'][2]:.3f}, min cue gain={row['quality'][3]:.3f}",
                flush=True,
            )
        write_json_atomic(
            run.export / "results.json",
            dict(
                schema="exp136.readout-audit-analysis/v1",
                candidates=rows,
                selected=best,
                network=raw["candidates"][0],
                selection_rule=configuration["selection_rule"],
                baseline_presentation="exp136-r082-present",
                requires_fresh_test=True,
            ),
        )
    return run.run_id
