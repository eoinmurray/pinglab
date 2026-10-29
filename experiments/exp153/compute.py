"""Initialize local reference weights and integrate an explicit delayed closure.

All 143 grid trajectories share initial history and closure settings; each
input difference has its own uncoupled calibration. Each invocation creates an
immutable Pingstore run; analysis and presentation are invoked separately.
"""

import argparse
import os
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp153 import recipe
from experiments.exp153.custom.initialization import capture_start
from hpc.concurrent_compute import run_active_workers
from pingstore.library import file_sha256, source_run, stage_run, write_json_atomic
from snnlab import lang
from snnlab import meanfield as mf
from snnlab.sim import Performance, build


def ancestry(repo, identity, *, reference=None):
    """Validate an explicit chain confined to this experiment's own stages."""
    found, visiting = {}, set()

    def visit(name, pin):
        if not name.startswith(recipe.SLUG + "-"):
            raise ValueError("stage ancestry must stay within this experiment")
        if name in visiting:
            raise ValueError("cyclic source lineage")
        if name in found:
            if pin is not None and found[name].reference != pin:
                raise ValueError("conflicting source pins")
            return
        item = source_run(
            repo / ".pingstore", name, experiment=recipe.SLUG, reference=pin
        )
        visiting.add(name)
        for upstream in item.record["inputs"].values():
            visit(upstream["run_id"], upstream)
        visiting.remove(name)
        found[name] = item

    visit(identity, reference)
    return found


def mean_incoming_weights(bundle, parameters):
    """Average native initialized source-target matrix sums over target neurons."""
    targets = {p["id"]: p["size"] for p in bundle.graph["populations"]}
    weights = {}
    for projection in bundle.graph["projections"]:
        name = projection["id"]
        matrix = parameters[name + ".weight"].detach().cpu().numpy()
        count = targets[projection["target"].rsplit(".", 1)[0]]
        if (
            matrix.ndim != 2
            or matrix.shape[1] != count
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("invalid native source-target matrix")
        weights[name] = float(matrix.sum(dtype=np.float64) / count)
    return weights


def integrate(model, cfg):
    initial = recipe.initial_state(model, cfg)
    solver = cfg["solver"]
    return mf.integrate(
        model,
        duration=cfg["duration_ms"],
        initial=initial,
        history=mf.ConstantHistory(initial),
        sample_interval=cfg["sample_interval_ms"],
        method=solver["method"],
        rtol=solver["rtol"],
        atol=solver["atol"],
        max_step=solver["max_step_ms"],
    )


def run_conditions(directory, item, worker_index, cfg):
    """Shared serial/parallel worker; each condition has disjoint exports."""
    import torch

    torch.set_num_threads(1)
    export = directory / "export"
    bundle = lang.load_bundle(export / "network.bundle")
    for condition in item["conditions"]:
        model = recipe.build_model(
            bundle,
            cfg,
            difference_hz=condition["difference_hz"],
            coupling_us=condition["coupling_us"],
        )
        label = condition["label"]
        model.write(export / f"{label}-model.json")
        integrate(model, cfg).write(export / f"{label}-trajectory.json")
        print(
            f"Completed {label}: drive difference {condition['difference_hz']:g} Hz, coupling {condition['coupling_us']:g} µS",
            flush=True,
        )
    return len(item["conditions"])


def condition_inventory(directory, item, worker_index):
    export = directory / "export"
    return {
        f"{condition['label']}-{kind}.json": file_sha256(
            export / f"{condition['label']}-{kind}.json"
        )
        for condition in item["conditions"]
        for kind in ("model", "trajectory")
    }


def compute(
    *,
    difference_hz=-4.0,
    coupling_us=0.09,
    parallel=1,
    run_id=None,
    onset_only=False,
    tau_rate_e_ms=5.0,
    tau_rate_i_ms=5.0,
    sensory_gain=9.2,
    recurrent_ee_us=0.4,
    recurrent_ii_us=0.8,
    local_feedback_scale=1.5,
    cross_delay_ms=2.0,
    cross_tau_ms=0.5,
):
    if not 1 <= parallel <= (os.cpu_count() or 1):
        raise ValueError("parallel workers must fit available logical CPUs")
    cfg = recipe.configuration(
        difference_hz=difference_hz,
        coupling_us=coupling_us,
        onset_only=onset_only,
        tau_rate_e_ms=tau_rate_e_ms,
        tau_rate_i_ms=tau_rate_i_ms,
        sensory_gain=sensory_gain,
        recurrent_ee_us=recurrent_ee_us,
        recurrent_ii_us=recurrent_ii_us,
        local_feedback_scale=local_feedback_scale,
        cross_delay_ms=cross_delay_ms,
        cross_tau_ms=cross_tau_ms,
    )
    bundle = recipe.author_network(cfg)
    initialized = build(
        graph=bundle.graph,
        seed=cfg["network"]["network_seed"],
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    cfg = recipe.configuration(
        difference_hz=difference_hz,
        coupling_us=coupling_us,
        incoming_weights_us=mean_incoming_weights(bundle, initialized.parameters),
        onset_only=onset_only,
        tau_rate_e_ms=tau_rate_e_ms,
        tau_rate_i_ms=tau_rate_i_ms,
        sensory_gain=sensory_gain,
        recurrent_ee_us=recurrent_ee_us,
        recurrent_ii_us=recurrent_ii_us,
        local_feedback_scale=local_feedback_scale,
        cross_delay_ms=cross_delay_ms,
        cross_tau_ms=cross_tau_ms,
    )
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        inputs={},
        configuration=cfg,
        run_id=run_id,
    ) as run:
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        capture_start(bundle, initialized, cfg, run.export / recipe.SNAPSHOT)
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={"network-initialization": ["start"]}
        )
        run.record["execution"].update(
            backend="snnlab.meanfield method-of-steps DDE",
            snnlab_version=version("snnlab"),
        )
        index = recipe.conditions(cfg)
        write_json_atomic(run.export / "conditions.json", index)
        count = min(parallel, len(index))
        items = [
            dict(
                id=f"cpu-{worker}",
                device="cpu",
                threads=1,
                conditions=index[worker::count],
            )
            for worker in range(count)
        ]
        run.record["execution"]["parallel_workers"] = count
        if count == 1:
            run_conditions(run.directory, items[0], 0, cfg)
        else:
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ[name] = "1"
            run_active_workers(
                run=run,
                items=items,
                worker=run_conditions,
                arguments=(cfg,),
                inventory=condition_inventory,
            )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difference-hz", type=float, default=-4.0)
    parser.add_argument("--coupling-us", type=float, default=0.09)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--onset-only", action="store_true")
    parser.add_argument("--tau-rate-e-ms", type=float, default=5.0)
    parser.add_argument("--tau-rate-i-ms", type=float, default=5.0)
    parser.add_argument("--sensory-gain", type=float, default=9.2)
    parser.add_argument("--recurrent-ee-us", type=float, default=0.4)
    parser.add_argument("--recurrent-ii-us", type=float, default=0.8)
    parser.add_argument("--local-feedback-scale", type=float, default=1.5)
    parser.add_argument("--cross-delay-ms", type=float, default=2.0)
    parser.add_argument("--cross-tau-ms", type=float, default=0.5)
    compute(**vars(parser.parse_args()))
